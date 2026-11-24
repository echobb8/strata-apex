#import "strata_metal_cache.h"

#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

struct StrataMetalReflectEntry {
    std::string name;
    uint32_t    max_threads_per_tg;
    uint32_t    simd_width;
    uint32_t    static_tg_mem;
    bool        from_archive;
    double      build_ms;
};

struct StrataMetalPsoCacheImpl {
    std::shared_mutex                    archive_mu;
    std::mutex                           stats_mu;
    std::chrono::steady_clock::time_point t_open;
    std::vector<StrataMetalReflectEntry> entries;
    uint32_t                             archive_added_count = 0U;
    bool                                 reflect_verbose     = false;
    std::string                          cache_dir;
};

namespace {

std::atomic<uint64_t> g_tmp_file_seq{0ULL};

bool env_is_enabled(const char* name) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0);
}

bool env_is_verbose_reflect(const char* name) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return false;
    return (std::strcmp(v, "2") == 0 ||
            std::strcmp(v, "all") == 0 ||
            std::strcmp(v, "ALL") == 0 ||
            std::strcmp(v, "verbose") == 0);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - t0).count();
}

void sha256_update_bytes(CC_SHA256_CTX* ctx, const void* data, size_t len) {
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    size_t rem = len;
    while (rem > 0U) {
        const CC_LONG chunk = static_cast<CC_LONG>(std::min<size_t>(rem, 0x40000000ULL));
        CC_SHA256_Update(ctx, ptr, chunk);
        ptr += chunk;
        rem -= static_cast<size_t>(chunk);
    }
}

void sha256_update_field(CC_SHA256_CTX* ctx, const void* data, size_t len) {
    const uint64_t len64 = static_cast<uint64_t>(len);
    CC_SHA256_Update(ctx, &len64, static_cast<CC_LONG>(sizeof(len64)));
    if (data && len > 0U) {
        sha256_update_bytes(ctx, data, len);
    }
}

std::string query_macos_build_string() {
    char buf[128];
    std::memset(buf, 0, sizeof(buf));
    size_t sz = sizeof(buf) - 1U;
    if (sysctlbyname("kern.osversion", buf, &sz, nullptr, 0U) == 0 && buf[0] != '\0') {
        return std::string(buf);
    }
    return std::string("unknown_os");
}

std::string sanitize_tag(const char* raw) {
    if (!raw || raw[0] == '\0') return std::string("strata");
    std::string out;
    out.reserve(32U);
    for (size_t i = 0U; raw[i] != '\0' && out.size() < 48U; ++i) {
        const unsigned char c = static_cast<unsigned char>(raw[i]);
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == '-' || c == '.') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? std::string("strata") : out;
}

NSError* make_strata_error(const char* msg) {
    NSString* ns_msg = [NSString stringWithUTF8String:(msg ? msg : "unknown error")];
    return [NSError errorWithDomain:@"StrataMetalPsoCache"
                               code:1
                           userInfo:@{NSLocalizedDescriptionKey: ns_msg}];
}

} // namespace

StrataMetalPsoCache* strata_metal_cache_open(
    id<MTLDevice>  device,
    id<MTLLibrary> library,
    const char*    engine_tag,
    const char*    msl_src,
    size_t         msl_len,
    uint64_t       compile_salt
) {
    if (!device || !library) {
        return nullptr;
    }

    auto* cache = new StrataMetalPsoCache();
    cache->device                 = device;
    cache->library                = library;
    cache->archive                = nil;
    cache->cache_hit              = false;
    cache->dirty                  = false;
    cache->disabled               = env_is_enabled("STRATA_METAL_CACHE_DISABLE");
    cache->reflect_enabled        = env_is_enabled("STRATA_METAL_REFLECT");
    cache->pso_count              = 0U;
    cache->archive_hit_count      = 0U;
    cache->archive_miss_count     = 0U;
    cache->spill_risk_count       = 0U;
    cache->min_max_threads_per_tg = 0U;
    cache->elapsed_ms             = 0.0;
    cache->serialize_ms           = 0.0;
    cache->impl                   = new StrataMetalPsoCacheImpl();
    cache->impl->t_open           = std::chrono::steady_clock::now();
    cache->impl->reflect_verbose  = env_is_verbose_reflect("STRATA_METAL_REFLECT");

    const std::string tag = sanitize_tag(engine_tag);
    std::snprintf(cache->engine_tag, sizeof(cache->engine_tag), "%s", tag.c_str());

    // Resolve cache directory (STRATA_METAL_CACHE_DIR overrides default)
    const char* env_dir = std::getenv("STRATA_METAL_CACHE_DIR");
    std::string dir = (env_dir && env_dir[0] != '\0')
        ? std::string(env_dir)
        : std::string(STRATA_METAL_CACHE_DEFAULT_DIR);
    while (dir.size() > 1U && dir.back() == '/') {
        dir.pop_back();
    }
    cache->impl->cache_dir = dir;

    // Compute deterministic SHA-256 key over:
    //   magic + engine_tag + msl_source_bytes + compile_salt (langVersion/mathMode/fpFuncs)
    //   + [device name] + macOS build string (kern.osversion)
    CC_SHA256_CTX sha;
    CC_SHA256_Init(&sha);
    static constexpr char kMagic[] = "STRATA_MTL_BINARCH_V1";
    sha256_update_field(&sha, kMagic, sizeof(kMagic) - 1U);
    sha256_update_field(&sha, tag.data(), tag.size());
    sha256_update_field(&sha, msl_src, msl_src ? msl_len : 0U);
    sha256_update_field(&sha, &compile_salt, sizeof(compile_salt));

    NSString* dev_name_ns = [device name];
    const char* dev_name  = dev_name_ns ? [dev_name_ns UTF8String] : "unknown_gpu";
    sha256_update_field(&sha, dev_name, std::strlen(dev_name));

    const std::string os_ver = query_macos_build_string();
    sha256_update_field(&sha, os_ver.data(), os_ver.size());

    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &sha);

    static constexpr char kHex[] = "0123456789abcdef";
    for (size_t i = 0U; i < 8U; ++i) {
        cache->key_hex16[2U * i]      = kHex[(digest[i] >> 4U) & 0x0FU];
        cache->key_hex16[2U * i + 1U] = kHex[digest[i] & 0x0FU];
    }
    cache->key_hex16[16] = '\0';

    std::snprintf(cache->cache_path, sizeof(cache->cache_path),
                  "%s/%s_%s.metallib_bin",
                  dir.c_str(), cache->engine_tag, cache->key_hex16);

    if (cache->disabled) {
        return cache;
    }

    @autoreleasepool {
        // Attempt warm load from existing disk archive
        struct stat st;
        if (::stat(cache->cache_path, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
            MTLBinaryArchiveDescriptor* bdesc = [[MTLBinaryArchiveDescriptor alloc] init];
            bdesc.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:cache->cache_path]];
            NSError* load_err = nil;
            id<MTLBinaryArchive> loaded = [device newBinaryArchiveWithDescriptor:bdesc error:&load_err];
            if (loaded != nil) {
                cache->archive   = loaded;
                cache->cache_hit = true;
                cache->dirty     = false;
            }
        }

        // Cold miss or incompatible/corrupt archive: create fresh empty archive
        if (cache->archive == nil) {
            MTLBinaryArchiveDescriptor* bdesc = [[MTLBinaryArchiveDescriptor alloc] init];
            bdesc.url = nil;
            NSError* create_err = nil;
            cache->archive   = [device newBinaryArchiveWithDescriptor:bdesc error:&create_err];
            cache->cache_hit = false;
            cache->dirty     = (cache->archive != nil);
        }
    }

    return cache;
}

id<MTLComputePipelineState> strata_metal_cache_make_pso_labeled(
    StrataMetalPsoCache*       cache,
    const char*                fn_name,
    const char*                label_tag,
    MTLFunctionConstantValues* const_values,
    NSError* __autoreleasing*  err_out
) {
    if (!cache || !cache->device || !cache->library || !fn_name || fn_name[0] == '\0') {
        if (err_out) *err_out = make_strata_error("Invalid arguments to strata_metal_cache_make_pso");
        return nil;
    }

    const auto t_pso = std::chrono::steady_clock::now();

    NSString* ns_name = [NSString stringWithUTF8String:fn_name];
    if (!ns_name) {
        if (err_out) *err_out = make_strata_error("Invalid UTF-8 function name");
        return nil;
    }

    NSError* local_err = nil;
    id<MTLFunction> fn = nil;
    if (const_values != nil) {
        fn = [cache->library newFunctionWithName:ns_name constantValues:const_values error:&local_err];
    } else {
        fn = [cache->library newFunctionWithName:ns_name];
    }
    if (!fn) {
        if (err_out) {
            if (local_err != nil) {
                *err_out = local_err;
            } else {
                char msg[256];
                std::snprintf(msg, sizeof(msg), "Metal compute function '%s' not found in library", fn_name);
                *err_out = make_strata_error(msg);
            }
        }
        return nil;
    }

    MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = fn;

    id<MTLComputePipelineState> pso = nil;
    bool from_archive = false;

    if (cache->cache_hit && cache->archive != nil && !cache->disabled) {
        desc.binaryArchives = @[cache->archive];
        NSError* arch_err = nil;
        {
            std::shared_lock<std::shared_mutex> slk(cache->impl->archive_mu);
            pso = [cache->device newComputePipelineStateWithDescriptor:desc
                                                               options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                            reflection:nil
                                                                 error:&arch_err];
        }
        if (pso != nil) {
            from_archive = true;
        } else {
            // Specific PSO was absent from archive -> compile normally & record into archive
            desc.binaryArchives = nil;
            local_err = nil;
            pso = [cache->device newComputePipelineStateWithDescriptor:desc
                                                               options:MTLPipelineOptionNone
                                                            reflection:nil
                                                                 error:&local_err];
            if (pso != nil) {
                std::unique_lock<std::shared_mutex> ulk(cache->impl->archive_mu);
                NSError* add_err = nil;
                if ([cache->archive addComputePipelineFunctionsWithDescriptor:desc error:&add_err]) {
                    cache->dirty = true;
                    cache->impl->archive_added_count++;
                }
            }
        }
    } else {
        pso = [cache->device newComputePipelineStateWithDescriptor:desc
                                                           options:MTLPipelineOptionNone
                                                        reflection:nil
                                                             error:&local_err];
        if (pso != nil && cache->archive != nil && !cache->disabled) {
            std::unique_lock<std::shared_mutex> ulk(cache->impl->archive_mu);
            NSError* add_err = nil;
            if ([cache->archive addComputePipelineFunctionsWithDescriptor:desc error:&add_err]) {
                cache->dirty = true;
                cache->impl->archive_added_count++;
            }
        }
    }

    if (!pso) {
        if (err_out) {
            *err_out = local_err ? local_err : make_strata_error("Compute pipeline compilation failed");
        }
        return nil;
    }

    const uint32_t max_tg   = static_cast<uint32_t>(pso.maxTotalThreadsPerThreadgroup);
    const uint32_t simd_w   = static_cast<uint32_t>(pso.threadExecutionWidth);
    const uint32_t tg_mem   = static_cast<uint32_t>(pso.staticThreadgroupMemoryLength);
    const double   build_ms = ms_since(t_pso);

    {
        std::lock_guard<std::mutex> lk(cache->impl->stats_mu);
        cache->pso_count++;
        if (from_archive) {
            cache->archive_hit_count++;
        } else {
            cache->archive_miss_count++;
        }
        if (max_tg < 1024U) {
            cache->spill_risk_count++;
        }
        if (cache->min_max_threads_per_tg == 0U || max_tg < cache->min_max_threads_per_tg) {
            cache->min_max_threads_per_tg = max_tg;
        }
        if (cache->reflect_enabled) {
            std::string display_name(fn_name);
            if (label_tag && label_tag[0] != '\0') {
                display_name.push_back('[');
                display_name.append(label_tag);
                display_name.push_back(']');
            }
            cache->impl->entries.push_back(StrataMetalReflectEntry{
                std::move(display_name),
                max_tg,
                simd_w,
                tg_mem,
                from_archive,
                build_ms
            });
        }
    }

    if (err_out) *err_out = nil;
    return pso;
}

id<MTLComputePipelineState> strata_metal_cache_make_pso(
    StrataMetalPsoCache*       cache,
    const char*                fn_name,
    MTLFunctionConstantValues* const_values,
    NSError* __autoreleasing*  err_out
) {
    return strata_metal_cache_make_pso_labeled(cache, fn_name, nullptr, const_values, err_out);
}

id<MTLComputePipelineState> strata_metal_cache_make_pso_const_u32(
    StrataMetalPsoCache*      cache,
    const char*               fn_name,
    uint32_t                  const_index,
    uint32_t                  const_value,
    NSError* __autoreleasing* err_out
) {
    MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
    uint32_t v = const_value;
    [fcv setConstantValue:&v type:MTLDataTypeUInt atIndex:static_cast<NSUInteger>(const_index)];

    char tag[32];
    if (const_index == 0U) {
        std::snprintf(tag, sizeof(tag), "M=%u", const_value);
    } else {
        std::snprintf(tag, sizeof(tag), "c%u=%u", const_index, const_value);
    }
    return strata_metal_cache_make_pso_labeled(cache, fn_name, tag, fcv, err_out);
}

bool strata_metal_cache_finish(StrataMetalPsoCache* cache) {
    if (!cache || !cache->impl) {
        return false;
    }

    bool ok = true;
    std::unique_lock<std::shared_mutex> ulk(cache->impl->archive_mu);
    std::lock_guard<std::mutex> slk(cache->impl->stats_mu);

    // If an archive loaded from disk served 0 PSOs out of >0 requested, report MISS accurately.
    if (cache->cache_hit && cache->pso_count > 0U && cache->archive_hit_count == 0U) {
        cache->cache_hit = false;
    }

    if (!cache->disabled &&
        cache->dirty &&
        cache->archive != nil &&
        cache->impl->archive_added_count > 0U) {
        const auto t_ser = std::chrono::steady_clock::now();
        @autoreleasepool {
            NSString* dir_ns = [NSString stringWithUTF8String:cache->impl->cache_dir.c_str()];
            NSError* dir_err = nil;
            BOOL dir_ok = [[NSFileManager defaultManager]
                createDirectoryAtPath:dir_ns
          withIntermediateDirectories:YES
                           attributes:nil
                                error:&dir_err];
            if (!dir_ok) {
                std::fprintf(stderr,
                             "[strata_metal_cache] Warning: failed to create cache dir '%s': %s\n",
                             cache->impl->cache_dir.c_str(),
                             dir_err ? [[dir_err localizedDescription] UTF8String] : "unknown");
                ok = false;
            } else {
                const uint64_t seq = g_tmp_file_seq.fetch_add(1ULL, std::memory_order_relaxed);
                char tmp_path[640];
                std::snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d.%llu",
                              cache->cache_path,
                              static_cast<int>(::getpid()),
                              static_cast<unsigned long long>(seq));
                NSURL* tmp_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:tmp_path]];
                NSError* ser_err = nil;
                if ([cache->archive serializeToURL:tmp_url error:&ser_err]) {
                    if (::rename(tmp_path, cache->cache_path) == 0) {
                        cache->dirty = false;
                    } else {
                        const int rc_errno = errno;
                        ::unlink(tmp_path);
                        std::fprintf(stderr,
                                     "[strata_metal_cache] Warning: atomic rename '%s' -> '%s' failed (%s)\n",
                                     tmp_path, cache->cache_path, std::strerror(rc_errno));
                        ok = false;
                    }
                } else {
                    ::unlink(tmp_path);
                    std::fprintf(stderr,
                                 "[strata_metal_cache] Warning: archive serialize failed for '%s': %s\n",
                                 cache->cache_path,
                                 ser_err ? [[ser_err localizedDescription] UTF8String] : "unknown");
                    ok = false;
                }
            }
        }
        cache->serialize_ms = ms_since(t_ser);
    }

    cache->elapsed_ms = ms_since(cache->impl->t_open);

    if (cache->reflect_enabled) {
        const char* status_str = cache->disabled
            ? "DISABLED"
            : (cache->cache_hit ? "HIT" : "MISS");
        std::fprintf(stderr,
                     "[strata_metal_reflect] engine=%s cache=%s psos=%u (archive_hit=%u miss=%u) "
                     "spill_risk(<1024)=%u/%u load_ms=%.2f (ser_ms=%.2f) file=%s\n",
                     cache->engine_tag,
                     status_str,
                     cache->pso_count,
                     cache->archive_hit_count,
                     cache->archive_miss_count,
                     cache->spill_risk_count,
                     cache->pso_count,
                     cache->elapsed_ms,
                     cache->serialize_ms,
                     cache->cache_path);

        auto& entries = cache->impl->entries;
        std::sort(entries.begin(), entries.end(),
                  [](const StrataMetalReflectEntry& a, const StrataMetalReflectEntry& b) {
                      if (a.max_threads_per_tg != b.max_threads_per_tg) {
                          return a.max_threads_per_tg < b.max_threads_per_tg;
                      }
                      if (a.static_tg_mem != b.static_tg_mem) {
                          return a.static_tg_mem > b.static_tg_mem;
                      }
                      return a.name < b.name;
                  });

        uint32_t printed = 0U;
        for (const auto& e : entries) {
            if (e.max_threads_per_tg < 1024U) {
                std::fprintf(stderr,
                             "  [SPILL-RISK] %-48s maxTG=%4u  simd=%2u  tgMem=%6u B  (%s %.2f ms)\n",
                             e.name.c_str(),
                             e.max_threads_per_tg,
                             e.simd_width,
                             e.static_tg_mem,
                             e.from_archive ? "HIT " : "MISS",
                             e.build_ms);
                printed++;
            } else if (cache->impl->reflect_verbose) {
                std::fprintf(stderr,
                             "  [OK        ] %-48s maxTG=%4u  simd=%2u  tgMem=%6u B  (%s %.2f ms)\n",
                             e.name.c_str(),
                             e.max_threads_per_tg,
                             e.simd_width,
                             e.static_tg_mem,
                             e.from_archive ? "HIT " : "MISS",
                             e.build_ms);
                printed++;
            }
        }
        if (printed == 0U) {
            std::fprintf(stderr,
                         "  (0 PSOs below 1024 maxTotalThreadsPerThreadgroup — zero register-pressure warnings)\n");
        }
        std::fflush(stderr);
    }

    return ok;
}

void strata_metal_cache_destroy(StrataMetalPsoCache* cache) {
    if (!cache) return;
    delete cache->impl;
    cache->impl    = nullptr;
    cache->archive = nil;
    cache->library = nil;
    cache->device  = nil;
    delete cache;
}
