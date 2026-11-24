/*
 * STRATA-M5 v2.0 — Step 3: Regime II Frontier MoE (`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`)
 * `Glm5NextForConditionalGeneration` (`320B` Total / `18B` Active, `43` Shards = `169.22 GiB` > `128 GiB` UMA)
 *
 * File: src/model/strata_glm53_convert.cpp
 *
 * Implements:
 *   1. Pipelined Shard-by-Shard Hugging Face Downloader (`curl -fSL --retry 5 --retry-delay 2` +
 *      4-stream HTTP/2 range acceleration into a single `F_NOCACHE` `.staging/<shard>.part` file, overlapping
 *      background shard `i + 1` download with shard `i` conversion, `< 8 GiB` peak `.staging/` disk,
 *      crash-safe `.convert_done_shards` resume manifest, and immediate `unlink` after each shard).
 *   2. Multi-Threaded ARM NEON Streaming Converter (`F_NOCACHE` `pread`/`pwrite`, `< 256 MiB` Peak RSS):
 *        - Disaggregates `114,160` safetensors tensors (`347` `model.visual.*` skipped + `111,172` main
 *          text + `2,641` Layer-45 NextN MTP) in any shard order with zero cross-shard RAM buffering:
 *            * `spine.strata` (`5.3611 GiB`, `16 KiB` VM-page aligned): `embed_tokens`, `final_norm`,
 *              `lm_head`, dense layers `0..2`, `45L` `mHC` (`fn` BF16->F32, `base`/`scale` F32),
 *              `34L` `KDA` (`lin_qkv_proj`, `lin_fbg_a_proj`, `f_b`, `g_b`, `o_proj`, `qkv_conv1d`,
 *              `A_log`, `dt_bias`, `o_norm`), `11L + 1L MTP` `DSA NoPE MLA` (unpacking `kv_b_proj` via
 *              `sg53_unpack_kv_b_proj_head()` into lossless Q4 `mla_kv_b_k`/`mla_kv_b_v` AND absorbed
 *              `FP16` `mla_w_uk` `[64*512, 256]` + `mla_w_uv` `[64*256, 512]`), `Glm5NextIndexer`
 *              (incl. `rows_logical=32 -> rows_padded=64` zero-padded `weights_proj`, `LayerNorm`
 *              `k_norm.{weight,bias}`, `compress_{ape,gate}`), `FP16 [288, 4096]` `moe_router_gate`,
 *              `FP32 [288]` `moe_e_score_bias`, `shared_experts`, and Layer-45 `MTP` spine.
 *            * `experts_nvme.strata` (`163.2656 GiB` = `12,384 x 13.5000 MiB`, `256 KiB` NVMe LBA stripe
 *              & `16 KiB` VM-page aligned): all `43 x 288` routed experts (`U32` weights + NEON
 *              `BF16 -> IEEE-754 F16` scales/biases).
 *        - Preserves raw `T5LayerNorm` weights (`NO +1.0f` on any norm!) and verifies `0` NaN/Inf across
 *          every converted norm, `mHC`, `KDA`, `MLA`, indexer, router, and scale/bias tensor.
 *   3. Zero-Copy `16 KiB` `mmap` Spine + `N = 8` Parallel `F_NOCACHE` NVMe FD Bundle Loader.
 *   4. Official `154,880`-Vocab `GLM-5.3-Flash` Byte-Level BPE Tokenizer (`\p{N}{1,3}`, `\p{L}+`,
 *      `ignore_merges = true`, atomic matching of all `36` `added_tokens` `154820..154855`) &
 *      `chat_template.jinja` Turn-1 / Turn-2+ Formatter.
 */

#include "../../include/strata_glm53.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

extern char** environ;

namespace {

/* ============================================================================
 * 1. Low-Level IEEE-754 BF16/FP16/FP32 NEON Kernels, NaN/Inf Checkers & I/O
 * ============================================================================ */

static inline float bf16_to_f32_scalar(uint16_t bf) {
    return sg53_bf16_to_f32(bf);
}

static inline float fp16_to_f32_fast(uint16_t h) {
#if defined(__aarch64__)
    float16x4_t v = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(v), 0);
#else
    return sg53_f16_to_f32(h);
#endif
}

static inline uint16_t f32_to_fp16_fast(float f) {
    if (f > 65504.0f) f = 65504.0f;
    else if (f < -65504.0f) f = -65504.0f;
#if defined(__aarch64__)
    float16x4_t h = vcvt_f16_f32(vdupq_n_f32(f));
    return vget_lane_u16(vreinterpret_u16_f16(h), 0);
#else
    return sg53_f32_to_f16(f);
#endif
}

static inline bool bf16_is_finite(uint16_t bf) {
    return (bf & 0x7F80u) != 0x7F80u;
}

static inline bool fp16_is_finite(uint16_t h) {
    return (h & 0x7C00u) != 0x7C00u;
}

static bool convert_bf16_to_fp16_span_checked(const uint16_t* src_bf16, uint16_t* dst_fp16, size_t n) {
    size_t i = 0;
    bool all_finite = true;
#if defined(__aarch64__)
    const uint16x8_t  v_exp_mask = vdupq_n_u16(0x7F80u);
    const float32x4_t v_max      = vdupq_n_f32(65504.0f);
    const float32x4_t v_min      = vdupq_n_f32(-65504.0f);
    uint16x8_t bad_acc = vdupq_n_u16(0);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t bf = vld1q_u16(src_bf16 + i);
        bad_acc = vorrq_u16(bad_acc, vceqq_u16(vandq_u16(bf, v_exp_mask), v_exp_mask));
        float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
        float32x4_t hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
        lo = vminq_f32(v_max, vmaxq_f32(v_min, lo));
        hi = vminq_f32(v_max, vmaxq_f32(v_min, hi));
        float16x8_t h8 = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
        vst1q_u16(dst_fp16 + i, vreinterpretq_u16_f16(h8));
    }
    if (vmaxvq_u16(bad_acc) != 0u) all_finite = false;
#endif
    for (; i < n; ++i) {
        if (!bf16_is_finite(src_bf16[i])) all_finite = false;
        dst_fp16[i] = f32_to_fp16_fast(bf16_to_f32_scalar(src_bf16[i]));
    }
    return all_finite;
}

static bool convert_f32_to_fp16_span_checked(const float* src_f32, uint16_t* dst_fp16, size_t n) {
    size_t i = 0;
    bool all_finite = true;
#if defined(__aarch64__)
    const float32x4_t v_max = vdupq_n_f32(65504.0f);
    const float32x4_t v_min = vdupq_n_f32(-65504.0f);
    for (; i + 8 <= n; i += 8) {
        float32x4_t f0 = vld1q_f32(src_f32 + i);
        float32x4_t f1 = vld1q_f32(src_f32 + i + 4);
        for (int lane = 0; lane < 4; ++lane) {
            if (!std::isfinite(src_f32[i + lane]) || !std::isfinite(src_f32[i + 4 + lane])) {
                all_finite = false;
            }
        }
        float32x4_t lo = vminq_f32(v_max, vmaxq_f32(v_min, f0));
        float32x4_t hi = vminq_f32(v_max, vmaxq_f32(v_min, f1));
        float16x8_t h8 = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
        vst1q_u16(dst_fp16 + i, vreinterpretq_u16_f16(h8));
    }
#endif
    for (; i < n; ++i) {
        if (!std::isfinite(src_f32[i])) all_finite = false;
        dst_fp16[i] = f32_to_fp16_fast(src_f32[i]);
    }
    return all_finite;
}

static bool convert_bf16_to_f32_span_checked(const uint16_t* src_bf16, float* dst_f32, size_t n) {
    size_t i = 0;
    bool all_finite = true;
#if defined(__aarch64__)
    const uint16x8_t v_exp_mask = vdupq_n_u16(0x7F80u);
    uint16x8_t bad_acc = vdupq_n_u16(0);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t bf = vld1q_u16(src_bf16 + i);
        bad_acc = vorrq_u16(bad_acc, vceqq_u16(vandq_u16(bf, v_exp_mask), v_exp_mask));
        float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
        float32x4_t hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
        vst1q_f32(dst_f32 + i,     lo);
        vst1q_f32(dst_f32 + i + 4, hi);
    }
    if (vmaxvq_u16(bad_acc) != 0u) all_finite = false;
#endif
    for (; i < n; ++i) {
        if (!bf16_is_finite(src_bf16[i])) all_finite = false;
        dst_f32[i] = bf16_to_f32_scalar(src_bf16[i]);
    }
    return all_finite;
}

static bool check_f16_span_finite(const uint16_t* p, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    const uint16x8_t v_exp_mask = vdupq_n_u16(0x7C00u);
    uint16x8_t bad_acc = vdupq_n_u16(0);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(p + i);
        bad_acc = vorrq_u16(bad_acc, vceqq_u16(vandq_u16(h, v_exp_mask), v_exp_mask));
    }
    if (vmaxvq_u16(bad_acc) != 0u) return false;
#endif
    for (; i < n; ++i) {
        if (!fp16_is_finite(p[i])) return false;
    }
    return true;
}

static bool check_f32_span_finite(const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

/* Fallback group-64 affine Q4 quantizer for synthetic unquantized BF16/F16/F32 test checkpoints */
static inline void quantize_group_f32(
    const float* src_f32,
    uint32_t     g_len,
    uint32_t*    dst_u32,
    uint16_t*    dst_scale,
    uint16_t*    dst_bias
) {
    float w_min = src_f32[0];
    float w_max = src_f32[0];
    for (uint32_t i = 1; i < g_len; ++i) {
        float x = src_f32[i];
        if (x < w_min) w_min = x;
        if (x > w_max) w_max = x;
    }
    const uint16_t bias_h  = f32_to_fp16_fast(w_min);
    const uint16_t scale_h = f32_to_fp16_fast((w_max - w_min) * (1.0f / 15.0f));
    *dst_scale = scale_h;
    *dst_bias  = bias_h;

    const float scale_eff = fp16_to_f32_fast(scale_h);
    const float bias_eff  = fp16_to_f32_fast(bias_h);
    const uint32_t n_u32  = g_len >> 3u;
    if (!(scale_eff > 0.0f)) {
        std::memset(dst_u32, 0, n_u32 * sizeof(uint32_t));
        return;
    }
    const float inv_s = 1.0f / scale_eff;
    for (uint32_t u = 0; u < n_u32; ++u) {
        uint32_t packed = 0;
        for (uint32_t j = 0; j < 8; ++j) {
            float qf = std::nearbyintf((src_f32[u * 8 + j] - bias_eff) * inv_s);
            if (qf < 0.0f) qf = 0.0f;
            else if (qf > 15.0f) qf = 15.0f;
            packed |= (static_cast<uint32_t>(qf) & 0xFu) << (4u * j);
        }
        dst_u32[u] = packed;
    }
}

static double current_peak_rss_gib() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
#if defined(__APPLE__)
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
#else
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
#endif
}

static bool pread_exact(int fd, void* buf, size_t count, uint64_t offset) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < count) {
        ssize_t r = pread(fd, p + done, count - done, static_cast<off_t>(offset + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false;
        done += static_cast<size_t>(r);
    }
    return true;
}

static bool pwrite_exact(int fd, const void* buf, size_t count, uint64_t offset) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t done = 0;
    while (done < count) {
        ssize_t w = pwrite(fd, p + done, count - done, static_cast<off_t>(offset + done));
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (w == 0) return false;
        done += static_cast<size_t>(w);
    }
    return true;
}

static bool pwrite_zeros(int fd, uint64_t offset, uint64_t count) {
    if (count == 0) return true;
    uint8_t zbuf[65536];
    std::memset(zbuf, 0, sizeof(zbuf));
    uint64_t rem = count;
    uint64_t cur = offset;
    while (rem > 0) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(sizeof(zbuf), rem));
        if (!pwrite_exact(fd, zbuf, chunk, cur)) return false;
        cur += chunk;
        rem -= chunk;
    }
    return true;
}

static bool file_exists_nonempty(const std::string& path, uint64_t* size_out = nullptr) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) {
        return false;
    }
    if (size_out) *size_out = static_cast<uint64_t>(st.st_size);
    return true;
}

static bool is_directory_path(const std::string& path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

static bool mkdirs(const std::string& dir) {
    if (dir.empty()) return false;
    std::string cur;
    if (dir[0] == '/') cur = "/";
    size_t i = (dir[0] == '/') ? 1 : 0;
    while (i <= dir.size()) {
        size_t slash = dir.find('/', i);
        std::string part = (slash == std::string::npos) ? dir.substr(i) : dir.substr(i, slash - i);
        if (!part.empty()) {
            if (!cur.empty() && cur.back() != '/') cur.push_back('/');
            cur += part;
            struct stat st;
            if (stat(cur.c_str(), &st) != 0) {
                if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
            } else if (!S_ISDIR(st.st_mode)) {
                return false;
            }
        }
        if (slash == std::string::npos) break;
        i = slash + 1;
    }
    return true;
}

static bool read_whole_file(const std::string& path, std::string& out) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close(fd);
        return false;
    }
    out.resize(static_cast<size_t>(st.st_size));
    bool ok = out.empty() ? true : pread_exact(fd, out.data(), out.size(), 0);
    close(fd);
    return ok;
}

static bool copy_file_if_exists(const std::string& src, const std::string& dst, bool required) {
    if (src == dst) return file_exists_nonempty(src) || !required;
    struct stat st_src, st_dst;
    if (stat(src.c_str(), &st_src) == 0 && stat(dst.c_str(), &st_dst) == 0 &&
        st_src.st_dev == st_dst.st_dev && st_src.st_ino == st_dst.st_ino) {
        return true;
    }
    int in_fd = open(src.c_str(), O_RDONLY);
    if (in_fd < 0) return !required;
    if (fstat(in_fd, &st_src) != 0) {
        close(in_fd);
        return false;
    }
    std::string tmp_dst = dst + ".part";
    int out_fd = open(tmp_dst.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (out_fd < 0) {
        close(in_fd);
        return false;
    }
    std::vector<uint8_t> buf(1024 * 1024);
    uint64_t off = 0;
    uint64_t total = static_cast<uint64_t>(st_src.st_size);
    bool ok = true;
    while (off < total) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(buf.size(), total - off));
        if (!pread_exact(in_fd, buf.data(), chunk, off) ||
            !pwrite_exact(out_fd, buf.data(), chunk, off)) {
            ok = false;
            break;
        }
        off += chunk;
    }
    close(out_fd);
    close(in_fd);
    if (!ok || rename(tmp_dst.c_str(), dst.c_str()) != 0) {
        unlink(tmp_dst.c_str());
        return false;
    }
    return true;
}

static uint64_t measure_dir_bytes(const std::string& dir_path) {
    DIR* d = opendir(dir_path.c_str());
    if (!d) return 0ULL;
    uint64_t sum = 0ULL;
    while (struct dirent* de = readdir(d)) {
        if (std::strcmp(de->d_name, ".") == 0 || std::strcmp(de->d_name, "..") == 0) continue;
        std::string p = dir_path + "/" + de->d_name;
        struct stat st;
        if (stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
            sum += static_cast<uint64_t>(st.st_size);
        }
    }
    closedir(d);
    return sum;
}

static void try_preallocate_apfs(int fd, uint64_t length_bytes) {
#if defined(__APPLE__) && defined(F_PREALLOCATE)
    if (fd < 0 || length_bytes == 0) return;
    fstore_t fst;
    std::memset(&fst, 0, sizeof(fst));
    fst.fst_flags   = F_ALLOCATECONTIG | F_ALLOCATEALL;
    fst.fst_posmode = F_PEOFPOSMODE;
    fst.fst_offset  = 0;
    fst.fst_length  = static_cast<off_t>(length_bytes);
    if (fcntl(fd, F_PREALLOCATE, &fst) != 0) {
        fst.fst_flags = F_ALLOCATEALL;
        (void)fcntl(fd, F_PREALLOCATE, &fst);
    }
#else
    (void)fd;
    (void)length_bytes;
#endif
}

/*
 * Single-stream atomic `curl -fSL --retry 5 --retry-delay 2` downloader (used for metadata
 * files and as a resilient fallback for shard downloads).
 */
static bool is_safe_shard_basename(const std::string& name) {
    if (name.empty() || name.size() > 255U || name == "." || name == ".." || name[0] == '-') return false;
    for (unsigned char c : name) {
        if (c == '/' || c == '\\' || c == '[' || c == ']' || c == '{' || c == '}' ||
            c < 0x20U || c == 0x7FU) return false;
    }
    return true;
}

static bool url_allows_hf_token(const std::string& url) {
    std::string lo = url;
    for (char& c : lo) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    }
    if (lo.rfind("https://", 0U) != 0U) return false;
    const size_t host_start = 8U;
    size_t host_end = lo.find_first_of(":/?#", host_start);
    const std::string host = lo.substr(host_start, (host_end == std::string::npos) ? std::string::npos : (host_end - host_start));
    if (host == "huggingface.co" ||
        (host.size() > 15U && host.compare(host.size() - 15U, 15U, ".huggingface.co") == 0)) {
        return true;
    }
    const char* ep = std::getenv("HF_ENDPOINT");
    if (ep && ep[0] != '\0') {
        std::string ep_lo = ep;
        for (char& c : ep_lo) {
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
        }
        if (ep_lo.rfind("https://", 0U) == 0U) {
            size_t e_end = ep_lo.find_first_of(":/?#", 8U);
            const std::string ep_host = ep_lo.substr(8U, (e_end == std::string::npos) ? std::string::npos : (e_end - 8U));
            if (!ep_host.empty() && host == ep_host) return true;
        }
    }
    return false;
}

static std::string make_curl_auth_config(const std::string& url, const char* hf_token) {
    if (!hf_token || hf_token[0] == '\0' || !url_allows_hf_token(url)) return {};
    std::string tok;
    for (const char* p = hf_token; *p; ++p) {
        if (*p == '\r' || *p == '\n' || *p == '"' || *p == '\\') continue;
        tok.push_back(*p);
    }
    if (tok.empty()) return {};
    return "header = \"Authorization: Bearer " + tok + "\"\n";
}

static void write_all_close_fd(int fd, const std::string& data) {
    size_t woff = 0U;
    while (woff < data.size()) {
        ssize_t wn = write(fd, data.data() + woff, data.size() - woff);
        if (wn < 0 && errno == EINTR) continue;
        if (wn <= 0) break;
        woff += static_cast<size_t>(wn);
    }
    close(fd);
}

static bool curl_download_single(
    const std::string& url,
    const std::string& dst_path,
    uint64_t*          bytes_out,
    double*            elapsed_out,
    const char*        hf_token_override = nullptr
) {
    auto t0 = std::chrono::steady_clock::now();
    std::string part_path = dst_path + ".part";
    const char* hf_token = (hf_token_override && hf_token_override[0] != '\0')
                               ? hf_token_override
                               : std::getenv("HF_TOKEN");
    const std::string auth_cfg = make_curl_auth_config(url, hf_token);

    for (int outer_try = 0; outer_try < 6; ++outer_try) {
        if (outer_try == 0 || outer_try >= 3) {
            unlink(part_path.c_str());
        }
        std::vector<const char*> argv;
        argv.push_back("curl");
        argv.push_back("--globoff");
        argv.push_back("-fsSL");
        argv.push_back("-C");
        argv.push_back("-");
        argv.push_back("--retry");
        argv.push_back("5");
        argv.push_back("--retry-delay");
        argv.push_back("2");
        argv.push_back("--retry-all-errors");
        argv.push_back("--connect-timeout");
        argv.push_back("30");
        argv.push_back("--speed-limit");
        argv.push_back("65536");
        argv.push_back("--speed-time");
        argv.push_back("45");
        int cfg_pipe[2] = {-1, -1};
        posix_spawn_file_actions_t actions;
        bool use_actions = false;
        if (!auth_cfg.empty() && pipe(cfg_pipe) == 0) {
            posix_spawn_file_actions_init(&actions);
            posix_spawn_file_actions_addclose(&actions, cfg_pipe[1]);
            posix_spawn_file_actions_adddup2(&actions, cfg_pipe[0], STDIN_FILENO);
            posix_spawn_file_actions_addclose(&actions, cfg_pipe[0]);
            use_actions = true;
            argv.push_back("-K");
            argv.push_back("-");
        }
        argv.push_back("-o");
        argv.push_back(part_path.c_str());
        argv.push_back(url.c_str());
        argv.push_back(nullptr);

        pid_t pid = -1;
        int rc = posix_spawn(&pid, "/usr/bin/curl", use_actions ? &actions : nullptr, nullptr,
                             const_cast<char* const*>(argv.data()), environ);
        if (rc != 0) {
            rc = posix_spawnp(&pid, "curl", use_actions ? &actions : nullptr, nullptr,
                              const_cast<char* const*>(argv.data()), environ);
        }
        if (use_actions) {
            posix_spawn_file_actions_destroy(&actions);
            close(cfg_pipe[0]);
            if (rc == 0 && pid > 0) write_all_close_fd(cfg_pipe[1], auth_cfg);
            else close(cfg_pipe[1]);
        }
        if (rc == 0 && pid > 0) {
            int status = 0;
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            uint64_t sz = 0;
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0 && file_exists_nonempty(part_path, &sz)) {
                if (rename(part_path.c_str(), dst_path.c_str()) == 0) {
                    if (bytes_out) *bytes_out = sz;
                    if (elapsed_out) {
                        *elapsed_out = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
                    }
                    return true;
                }
            }
        }
        if (outer_try >= 2) {
            unlink(part_path.c_str());
        }
        if (outer_try + 1 < 6) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    unlink(part_path.c_str());
    return false;
}

/*
 * Probes 1 byte (`Range: bytes=0-0`) via `curl -fsSL` to confirm HTTP 206 Partial Content
 * support, extract the exact total byte length from `Content-Range: bytes 0-0/<total>`,
 * and capture the effective CDN URL for direct parallel range transfers.
 */
static bool probe_http_range_size(
    const std::string& url,
    const std::string& auth_cfg,
    uint64_t*          total_bytes_out,
    std::string*       effective_url_out
) {
    int pipefd[2] = {-1, -1};
    if (pipe(pipefd) != 0) return false;
    int cfg_pipe[2] = {-1, -1};
    const bool send_auth = !auth_cfg.empty() && (pipe(cfg_pipe) == 0);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    if (send_auth) {
        posix_spawn_file_actions_addclose(&actions, cfg_pipe[1]);
        posix_spawn_file_actions_adddup2(&actions, cfg_pipe[0], STDIN_FILENO);
        posix_spawn_file_actions_addclose(&actions, cfg_pipe[0]);
    }

    std::vector<const char*> argv;
    argv.push_back("curl");
    argv.push_back("--globoff");
    argv.push_back("-fsSL");
    argv.push_back("--retry");
    argv.push_back("5");
    argv.push_back("--retry-delay");
    argv.push_back("2");
    argv.push_back("--connect-timeout");
    argv.push_back("20");
    argv.push_back("-r");
    argv.push_back("0-0");
    argv.push_back("-D");
    argv.push_back("-");
    argv.push_back("-o");
    argv.push_back("/dev/null");
    argv.push_back("-w");
    argv.push_back("\nSTRATA_EFF_URL:%{url_effective}\n");
    if (send_auth) {
        argv.push_back("-K");
        argv.push_back("-");
    }
    argv.push_back(url.c_str());
    argv.push_back(nullptr);

    pid_t pid = -1;
    int rc = posix_spawn(&pid, "/usr/bin/curl", &actions, nullptr,
                         const_cast<char* const*>(argv.data()), environ);
    if (rc != 0) {
        rc = posix_spawnp(&pid, "curl", &actions, nullptr,
                          const_cast<char* const*>(argv.data()), environ);
    }
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[1]);
    if (send_auth) {
        close(cfg_pipe[0]);
        if (rc == 0 && pid > 0) write_all_close_fd(cfg_pipe[1], auth_cfg);
        else close(cfg_pipe[1]);
    }

    if (rc != 0 || pid <= 0) {
        close(pipefd[0]);
        return false;
    }

    std::string out;
    char buf[4096];
    while (true) {
        ssize_t n = read(pipefd[0], buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;
        out.append(buf, static_cast<size_t>(n));
    }
    close(pipefd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return false;

    /* Extract last `Content-Range: bytes 0-0/<total>` (case-insensitive) */
    std::string lower = out;
    for (char& c : lower) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    }
    size_t pos = lower.rfind("content-range:");
    if (pos == std::string::npos) return false;
    size_t line_end = lower.find('\n', pos);
    std::string cr_line = lower.substr(pos, (line_end == std::string::npos) ? std::string::npos : (line_end - pos));
    size_t slash = cr_line.find('/');
    if (slash == std::string::npos) return false;
    const char* num_start = cr_line.c_str() + slash + 1;
    char* num_end = nullptr;
    unsigned long long total = std::strtoull(num_start, &num_end, 10);
    if (num_end == num_start || total == 0ULL) return false;

    *total_bytes_out = static_cast<uint64_t>(total);
    *effective_url_out = url;
    const std::string tag = "STRATA_EFF_URL:";
    size_t epos = out.rfind(tag);
    if (epos != std::string::npos) {
        size_t s = epos + tag.size();
        size_t e = out.find_first_of("\r\n", s);
        std::string eu = (e == std::string::npos) ? out.substr(s) : out.substr(s, e - s);
        if (eu.rfind("http://", 0) == 0 || eu.rfind("https://", 0) == 0) {
            *effective_url_out = eu;
        }
    }
    return true;
}

/*
 * High-throughput 4-stream parallel HTTP range downloader writing via `F_NOCACHE` `pwrite`
 * directly into ONE `<dst_path>.part` file (`0` concatenation overhead, `< 4.000 GiB` staging
 * footprint per shard, `< 2 MiB` RAM), with automatic byte-exact resume retries (`--retry-delay 2`)
 * and seamless fallback to `curl_download_single`.
 */
static bool curl_download_shard_fast(
    const std::string& url,
    const std::string& dst_path,
    uint64_t*          bytes_out,
    double*            elapsed_out,
    uint32_t           num_streams = 4u,
    const char*        hf_token_override = nullptr
) {
    auto t0 = std::chrono::steady_clock::now();
    const char* hf_token = (hf_token_override && hf_token_override[0] != '\0')
                               ? hf_token_override
                               : std::getenv("HF_TOKEN");
    const std::string auth_cfg = make_curl_auth_config(url, hf_token);

    const uint32_t kNumStreams = (num_streams == 0u) ? 4u : std::min<uint32_t>(num_streams, 8u);
    if (kNumStreams <= 1u) {
        return curl_download_single(url, dst_path, bytes_out, elapsed_out, hf_token_override);
    }

    uint64_t total_bytes = 0ULL;
    std::string cdn_url;
    if (!probe_http_range_size(url, auth_cfg, &total_bytes, &cdn_url) ||
        total_bytes < 64ULL * 1024ULL * 1024ULL) {
        return curl_download_single(url, dst_path, bytes_out, elapsed_out, hf_token_override);
    }

    std::string part_path = dst_path + ".part";
    unlink(part_path.c_str());
    int pfd = open(part_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (pfd < 0) {
        return curl_download_single(url, dst_path, bytes_out, elapsed_out, hf_token_override);
    }
#if defined(F_NOCACHE)
    (void)fcntl(pfd, F_NOCACHE, 1);
#endif
    try_preallocate_apfs(pfd, total_bytes);
    if (ftruncate(pfd, static_cast<off_t>(total_bytes)) != 0) {
        close(pfd);
        unlink(part_path.c_str());
        return curl_download_single(url, dst_path, bytes_out, elapsed_out, hf_token_override);
    }

    const uint64_t raw_chunk = total_bytes / kNumStreams;
    const uint64_t aligned_chunk = (raw_chunk + STRATA_GLM53_NVME_STRIPE_BYTES - 1ULL) &
                                   ~(STRATA_GLM53_NVME_STRIPE_BYTES - 1ULL);

    std::atomic<bool> range_failed{false};
    auto download_range = [&](uint64_t start_off, uint64_t end_incl) {
        if (start_off > end_incl) return;
        std::vector<uint8_t> io_buf(262144); /* 256 KiB stripe buffer */
        uint64_t cur = start_off;

        for (int attempt = 0; attempt < 6 && cur <= end_incl && !range_failed.load(std::memory_order_relaxed); ++attempt) {
            /* Use direct CDN URL on first 2 tries, original HF URL (fresh redirect) on retries */
            const bool use_cdn = (attempt < 2) && (cdn_url != url);
            const std::string& target_url = use_cdn ? cdn_url : url;
            std::string range_spec = std::to_string(cur) + "-" + std::to_string(end_incl);

            int pipefd[2] = {-1, -1};
            if (pipe(pipefd) != 0) break;
            int cfg_pipe[2] = {-1, -1};
            const bool send_auth = (!use_cdn && !auth_cfg.empty() && pipe(cfg_pipe) == 0);

            posix_spawn_file_actions_t actions;
            posix_spawn_file_actions_init(&actions);
            posix_spawn_file_actions_addclose(&actions, pipefd[0]);
            posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
            posix_spawn_file_actions_addclose(&actions, pipefd[1]);
            if (send_auth) {
                posix_spawn_file_actions_addclose(&actions, cfg_pipe[1]);
                posix_spawn_file_actions_adddup2(&actions, cfg_pipe[0], STDIN_FILENO);
                posix_spawn_file_actions_addclose(&actions, cfg_pipe[0]);
            }

            std::vector<const char*> argv;
            argv.push_back("curl");
            argv.push_back("--globoff");
            argv.push_back("-fsSL");
            argv.push_back("--connect-timeout");
            argv.push_back("30");
            argv.push_back("--speed-limit");
            argv.push_back("65536");
            argv.push_back("--speed-time");
            argv.push_back("45");
            if (send_auth) {
                argv.push_back("-K");
                argv.push_back("-");
            }
            argv.push_back("-r");
            argv.push_back(range_spec.c_str());
            argv.push_back(target_url.c_str());
            argv.push_back(nullptr);

            pid_t pid = -1;
            int rc = posix_spawn(&pid, "/usr/bin/curl", &actions, nullptr,
                                 const_cast<char* const*>(argv.data()), environ);
            if (rc != 0) {
                rc = posix_spawnp(&pid, "curl", &actions, nullptr,
                                  const_cast<char* const*>(argv.data()), environ);
            }
            posix_spawn_file_actions_destroy(&actions);
            close(pipefd[1]);
            if (send_auth) {
                close(cfg_pipe[0]);
                if (rc == 0 && pid > 0) write_all_close_fd(cfg_pipe[1], auth_cfg);
                else close(cfg_pipe[1]);
            }

            if (rc != 0 || pid <= 0) {
                close(pipefd[0]);
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }

            bool write_err = false;
            size_t fill = 0;
            while (!range_failed.load(std::memory_order_relaxed)) {
                ssize_t r = read(pipefd[0], io_buf.data() + fill, io_buf.size() - fill);
                if (r < 0) {
                    if (errno == EINTR) continue;
                    break;
                }
                if (r == 0) break;
                fill += static_cast<size_t>(r);
                if (fill == io_buf.size()) {
                    if (cur + fill > end_incl + 1ULL || !pwrite_exact(pfd, io_buf.data(), fill, cur)) {
                        write_err = true;
                        break;
                    }
                    cur += fill;
                    fill = 0;
                }
            }
            if (!write_err && fill > 0) {
                if (cur + fill > end_incl + 1ULL || !pwrite_exact(pfd, io_buf.data(), fill, cur)) {
                    write_err = true;
                } else {
                    cur += fill;
                }
            }
            close(pipefd[0]);

            int status = 0;
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}

            if (write_err) {
                range_failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0 && cur == end_incl + 1ULL) {
                return;
            }
            if (attempt + 1 < 6) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }

        if (cur != end_incl + 1ULL) {
            range_failed.store(true, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kNumStreams);
    for (uint32_t s = 0; s < kNumStreams; ++s) {
        uint64_t r0 = static_cast<uint64_t>(s) * aligned_chunk;
        if (r0 >= total_bytes) break;
        uint64_t r1 = (s + 1u == kNumStreams)
                          ? (total_bytes - 1ULL)
                          : std::min<uint64_t>(total_bytes - 1ULL, r0 + aligned_chunk - 1ULL);
        threads.emplace_back(download_range, r0, r1);
    }
    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }
    close(pfd);

    uint64_t actual_sz = 0ULL;
    if (!range_failed.load(std::memory_order_relaxed) &&
        file_exists_nonempty(part_path, &actual_sz) && actual_sz == total_bytes &&
        rename(part_path.c_str(), dst_path.c_str()) == 0) {
        if (bytes_out) *bytes_out = total_bytes;
        if (elapsed_out) {
            *elapsed_out = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
        }
        return true;
    }

    unlink(part_path.c_str());
    return curl_download_single(url, dst_path, bytes_out, elapsed_out, hf_token_override);
}

/* ============================================================================
 * 2. Persistent Thread Pool for Multi-GB/s Parallel NEON Conversion
 * ============================================================================ */

class NeonWorkerPool {
public:
    explicit NeonWorkerPool(uint32_t num_threads) {
        if (num_threads == 0) {
            unsigned hc = std::thread::hardware_concurrency();
            num_threads = (hc > 0) ? std::min(hc, 24u) : 8u;
        }
        n_threads_ = num_threads;
        if (n_threads_ <= 1) return;
        workers_.reserve(n_threads_ - 1);
        for (uint32_t t = 1; t < n_threads_; ++t) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~NeonWorkerPool() {
        if (workers_.empty()) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++epoch_;
        }
        cv_start_.notify_all();
        for (auto& th : workers_) {
            if (th.joinable()) th.join();
        }
    }

    uint32_t num_threads() const { return n_threads_; }

    void parallel_for_rows(uint64_t total_rows, uint64_t grain, const std::function<void(uint64_t, uint64_t)>& fn) {
        if (total_rows == 0) return;
        if (n_threads_ <= 1 || total_rows <= grain) {
            fn(0, total_rows);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_             = &fn;
            total_rows_     = total_rows;
            grain_          = std::max<uint64_t>(1ULL, grain);
            next_row_.store(0ULL, std::memory_order_relaxed);
            active_workers_ = static_cast<uint32_t>(workers_.size());
            ++epoch_;
        }
        cv_start_.notify_all();

        run_steal_loop();

        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [this]() { return active_workers_ == 0; });
        fn_ = nullptr;
    }

private:
    void run_steal_loop() {
        const uint64_t tot = total_rows_;
        const uint64_t gr  = grain_;
        while (true) {
            uint64_t r0 = next_row_.fetch_add(gr, std::memory_order_relaxed);
            if (r0 >= tot) break;
            uint64_t r1 = std::min<uint64_t>(tot, r0 + gr);
            (*fn_)(r0, r1);
        }
    }

    void worker_loop() {
        uint64_t seen_epoch = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_start_.wait(lk, [&]() { return stop_ || epoch_ != seen_epoch; });
                if (stop_) return;
                seen_epoch = epoch_;
            }
            run_steal_loop();
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--active_workers_ == 0) {
                    cv_done_.notify_one();
                }
            }
        }
    }

    uint32_t                              n_threads_ = 1;
    std::vector<std::thread>              workers_;
    std::mutex                            mu_;
    std::condition_variable               cv_start_;
    std::condition_variable               cv_done_;
    bool                                  stop_ = false;
    uint64_t                              epoch_ = 0;
    uint32_t                              active_workers_ = 0;
    const std::function<void(uint64_t, uint64_t)>* fn_ = nullptr;
    uint64_t                              total_rows_ = 0;
    uint64_t                              grain_ = 64;
    std::atomic<uint64_t>                 next_row_{0};
};

/* ============================================================================
 * 3. UTF-8 Helpers & Lightweight RFC-8259 JSON Scanner + Small DOM
 * ============================================================================ */

static inline void append_utf8_codepoint(std::string& out, uint32_t cp) {
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0x10FFFFu) {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

static inline uint32_t decode_utf8_one(const uint8_t* s, size_t len, size_t* adv) {
    if (len == 0) {
        *adv = 0;
        return 0;
    }
    uint8_t b0 = s[0];
    if (b0 < 0x80u) {
        *adv = 1;
        return b0;
    }
    if ((b0 & 0xE0u) == 0xC0u && len >= 2 && (s[1] & 0xC0u) == 0x80u) {
        uint32_t cp = ((static_cast<uint32_t>(b0 & 0x1Fu)) << 6) |
                      static_cast<uint32_t>(s[1] & 0x3Fu);
        if (cp >= 0x80u) {
            *adv = 2;
            return cp;
        }
    } else if ((b0 & 0xF0u) == 0xE0u && len >= 3 &&
               (s[1] & 0xC0u) == 0x80u && (s[2] & 0xC0u) == 0x80u) {
        uint32_t cp = ((static_cast<uint32_t>(b0 & 0x0Fu)) << 12) |
                      ((static_cast<uint32_t>(s[1] & 0x3Fu)) << 6) |
                      static_cast<uint32_t>(s[2] & 0x3Fu);
        if (cp >= 0x800u) {
            *adv = 3;
            return cp;
        }
    } else if ((b0 & 0xF8u) == 0xF0u && len >= 4 &&
               (s[1] & 0xC0u) == 0x80u && (s[2] & 0xC0u) == 0x80u && (s[3] & 0xC0u) == 0x80u) {
        uint32_t cp = ((static_cast<uint32_t>(b0 & 0x07u)) << 18) |
                      ((static_cast<uint32_t>(s[1] & 0x3Fu)) << 12) |
                      ((static_cast<uint32_t>(s[2] & 0x3Fu)) << 6) |
                      static_cast<uint32_t>(s[3] & 0x3Fu);
        if (cp >= 0x10000u && cp <= 0x10FFFFu) {
            *adv = 4;
            return cp;
        }
    }
    *adv = 1;
    return 0xFFFDu;
}

struct JsonCursor {
    const char* s;
    size_t      n;
    size_t      p;

    JsonCursor(const char* str, size_t len) : s(str), n(len), p(0) {}

    inline void skip_ws() {
        while (p < n) {
            unsigned char c = static_cast<unsigned char>(s[p]);
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p;
            else break;
        }
    }

    inline bool consume(char expected) {
        skip_ws();
        if (p < n && s[p] == expected) {
            ++p;
            return true;
        }
        return false;
    }

    inline char peek() {
        skip_ws();
        return (p < n) ? s[p] : '\0';
    }

    static inline int hex_val(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    }

    bool parse_hex4(uint32_t& out) {
        if (p + 4 > n) return false;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            int h = hex_val(s[p + i]);
            if (h < 0) return false;
            v = (v << 4) | static_cast<uint32_t>(h);
        }
        p += 4;
        out = v;
        return true;
    }

    bool parse_string(std::string& out) {
        skip_ws();
        if (p >= n || s[p] != '"') return false;
        ++p;
        out.clear();
        while (p < n) {
            const char* start = s + p;
            size_t run = 0;
            while (p + run < n) {
                char c = s[p + run];
                if (c == '"' || c == '\\') break;
                ++run;
            }
            if (run > 0) {
                out.append(start, run);
                p += run;
            }
            if (p >= n) return false;
            char c = s[p++];
            if (c == '"') return true;
            if (p >= n) return false;
            char esc = s[p++];
            switch (esc) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    uint32_t u1 = 0;
                    if (!parse_hex4(u1)) return false;
                    if (u1 >= 0xD800u && u1 <= 0xDBFFu) {
                        if (p + 2 <= n && s[p] == '\\' && s[p + 1] == 'u') {
                            p += 2;
                            uint32_t u2 = 0;
                            if (!parse_hex4(u2)) return false;
                            if (u2 >= 0xDC00u && u2 <= 0xDFFFu) {
                                uint32_t cp = 0x10000u + (((u1 - 0xD800u) << 10) | (u2 - 0xDC00u));
                                append_utf8_codepoint(out, cp);
                                break;
                            }
                        }
                    }
                    append_utf8_codepoint(out, u1);
                    break;
                }
                default:
                    return false;
            }
        }
        return false;
    }

    bool parse_number(double& out_d, int64_t& out_i) {
        skip_ws();
        if (p >= n) return false;
        const char* start = s + p;
        char* end_d = nullptr;
        out_d = std::strtod(start, &end_d);
        if (end_d == start) return false;
        char* end_i = nullptr;
        out_i = std::strtoll(start, &end_i, 10);
        if (end_i != end_d) out_i = static_cast<int64_t>(out_d);
        p = static_cast<size_t>(end_d - s);
        return true;
    }

    bool parse_uint64(uint64_t& out_u) {
        skip_ws();
        if (p >= n) return false;
        const char* start = s + p;
        char* end_u = nullptr;
        unsigned long long v = std::strtoull(start, &end_u, 10);
        if (end_u == start) return false;
        out_u = static_cast<uint64_t>(v);
        p = static_cast<size_t>(end_u - s);
        return true;
    }

    bool parse_bool(bool& out_b) {
        skip_ws();
        if (p + 4 <= n && std::memcmp(s + p, "true", 4) == 0) {
            p += 4;
            out_b = true;
            return true;
        }
        if (p + 5 <= n && std::memcmp(s + p, "false", 5) == 0) {
            p += 5;
            out_b = false;
            return true;
        }
        return false;
    }

    bool skip_value(int depth = 0) {
        if (depth > 128) return false;
        skip_ws();
        if (p >= n) return false;
        char c = s[p];
        if (c == '"') {
            std::string tmp;
            return parse_string(tmp);
        }
        if (c == '{') {
            ++p;
            if (consume('}')) return true;
            std::string key;
            while (true) {
                if (!parse_string(key)) return false;
                if (!consume(':')) return false;
                if (!skip_value(depth + 1)) return false;
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (c == '[') {
            ++p;
            if (consume(']')) return true;
            while (true) {
                if (!skip_value(depth + 1)) return false;
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
        }
        if (c == 't' || c == 'f') {
            bool b = false;
            return parse_bool(b);
        }
        if (c == 'n') {
            if (p + 4 <= n && std::memcmp(s + p, "null", 4) == 0) {
                p += 4;
                return true;
            }
            return false;
        }
        double d = 0.0;
        int64_t iv = 0;
        return parse_number(d, iv);
    }
};

struct SmallJson {
    enum Type { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } type = J_NULL;
    bool b_val = false;
    double d_val = 0.0;
    int64_t i_val = 0;
    std::string s_val;
    std::vector<SmallJson> arr;
    std::map<std::string, SmallJson> obj;

    const SmallJson* get(const std::string& k) const {
        if (type != J_OBJ) return nullptr;
        auto it = obj.find(k);
        return (it == obj.end()) ? nullptr : &it->second;
    }
    int64_t get_i64(const std::string& k, int64_t def) const {
        const SmallJson* v = get(k);
        return (v && v->type == J_NUM) ? v->i_val : def;
    }
    double get_f64(const std::string& k, double def) const {
        const SmallJson* v = get(k);
        return (v && v->type == J_NUM) ? v->d_val : def;
    }
    bool get_bool(const std::string& k, bool def) const {
        const SmallJson* v = get(k);
        return (v && v->type == J_BOOL) ? v->b_val : def;
    }
    std::string get_str(const std::string& k, const std::string& def = "") const {
        const SmallJson* v = get(k);
        return (v && v->type == J_STR) ? v->s_val : def;
    }
};

static bool parse_small_json_rec(JsonCursor& cur, SmallJson& out, int depth = 0) {
    if (depth > 128) return false;
    char c = cur.peek();
    if (c == '\0') return false;
    if (c == '"') {
        out.type = SmallJson::J_STR;
        return cur.parse_string(out.s_val);
    }
    if (c == '{') {
        cur.consume('{');
        out.type = SmallJson::J_OBJ;
        if (cur.consume('}')) return true;
        while (true) {
            std::string k;
            if (!cur.parse_string(k)) return false;
            if (!cur.consume(':')) return false;
            SmallJson child;
            if (!parse_small_json_rec(cur, child, depth + 1)) return false;
            out.obj.emplace(std::move(k), std::move(child));
            if (cur.consume('}')) return true;
            if (!cur.consume(',')) return false;
        }
    }
    if (c == '[') {
        cur.consume('[');
        out.type = SmallJson::J_ARR;
        if (cur.consume(']')) return true;
        while (true) {
            SmallJson elem;
            if (!parse_small_json_rec(cur, elem, depth + 1)) return false;
            out.arr.push_back(std::move(elem));
            if (cur.consume(']')) return true;
            if (!cur.consume(',')) return false;
        }
    }
    if (c == 't' || c == 'f') {
        out.type = SmallJson::J_BOOL;
        return cur.parse_bool(out.b_val);
    }
    if (c == 'n') {
        if (cur.p + 4 <= cur.n && std::memcmp(cur.s + cur.p, "null", 4) == 0) {
            cur.p += 4;
            out.type = SmallJson::J_NULL;
            return true;
        }
        return false;
    }
    out.type = SmallJson::J_NUM;
    return cur.parse_number(out.d_val, out.i_val);
}

static bool parse_small_json(const std::string& text, SmallJson& out) {
    JsonCursor cur(text.data(), text.size());
    return parse_small_json_rec(cur, out);
}

/* ============================================================================
 * 4. Safetensors Header Parser & Disaggregated Spine / NVMe Target Registry
 * ============================================================================ */

enum SafetensorDtype {
    ST_UNKNOWN = 0,
    ST_BF16    = 1,
    ST_F16     = 2,
    ST_F32     = 3,
    ST_I64     = 4,
    ST_U32     = 5,
    ST_F8_E4M3 = 6
};

struct SafetensorEntry {
    std::string           canon_name;
    SafetensorDtype       dtype = ST_UNKNOWN;
    std::vector<uint64_t> shape;
    uint64_t              abs_offset = 0;
    uint64_t              byte_len   = 0;
};

static SafetensorDtype parse_st_dtype(const std::string& s) {
    if (s == "BF16")    return ST_BF16;
    if (s == "F16")     return ST_F16;
    if (s == "F32")     return ST_F32;
    if (s == "I64")     return ST_I64;
    if (s == "U32")     return ST_U32;
    if (s == "F8_E4M3") return ST_F8_E4M3;
    return ST_UNKNOWN;
}

static bool is_vision_tensor_name(const std::string& key) {
    return key.rfind("model.visual.", 0) == 0 ||
           key.rfind("visual.", 0) == 0 ||
           key.rfind("vision_tower.", 0) == 0;
}

static std::string canonicalize_tensor_name(const std::string& key) {
    if (key.rfind("model.language_model.", 0) == 0) {
        return std::string("model.") + key.substr(std::strlen("model.language_model."));
    }
    if (key.rfind("language_model.model.", 0) == 0) {
        return std::string("model.") + key.substr(std::strlen("language_model.model."));
    }
    if (key.rfind("language_model.lm_head.", 0) == 0) {
        return key.substr(std::strlen("language_model."));
    }
    if (key.rfind("language_model.", 0) == 0) {
        std::string sub = key.substr(std::strlen("language_model."));
        if (sub.rfind("layers.", 0) == 0 || sub.rfind("embed_tokens.", 0) == 0 || sub == "norm.weight") {
            return std::string("model.") + sub;
        }
        return sub;
    }
    return key;
}

static bool parse_safetensors_shard_entries(
    int fd,
    uint64_t shard_file_bytes,
    std::vector<SafetensorEntry>& entries_out,
    uint32_t& vision_skipped_out
) {
    entries_out.clear();
    vision_skipped_out = 0;

    uint8_t len_buf[8];
    if (!pread_exact(fd, len_buf, 8, 0)) return false;
    uint64_t hdr_len = 0;
    for (int i = 0; i < 8; ++i) {
        hdr_len |= static_cast<uint64_t>(len_buf[i]) << (8 * i);
    }
    if (hdr_len == 0 || hdr_len > 128ULL * 1024ULL * 1024ULL || 8ULL + hdr_len > shard_file_bytes) {
        return false;
    }
    std::string hdr_json(static_cast<size_t>(hdr_len), '\0');
    if (!pread_exact(fd, hdr_json.data(), static_cast<size_t>(hdr_len), 8)) return false;

    const uint64_t data_base = 8ULL + hdr_len;
    uint64_t max_end_seen = 0ULL;

    JsonCursor cur(hdr_json.data(), hdr_json.size());
    if (!cur.consume('{')) return false;
    if (cur.consume('}')) return true;

    std::string key, prop, dt_str;
    while (true) {
        if (!cur.parse_string(key)) return false;
        if (!cur.consume(':')) return false;
        if (key == "__metadata__") {
            if (!cur.skip_value()) return false;
        } else {
            if (!cur.consume('{')) return false;
            SafetensorEntry ent;
            uint64_t off0 = 0, off1 = 0;
            if (!cur.consume('}')) {
                while (true) {
                    if (!cur.parse_string(prop)) return false;
                    if (!cur.consume(':')) return false;
                    if (prop == "dtype") {
                        if (!cur.parse_string(dt_str)) return false;
                        ent.dtype = parse_st_dtype(dt_str);
                    } else if (prop == "shape") {
                        if (!cur.consume('[')) return false;
                        if (!cur.consume(']')) {
                            while (true) {
                                uint64_t dim = 0;
                                if (!cur.parse_uint64(dim)) return false;
                                ent.shape.push_back(dim);
                                if (cur.consume(']')) break;
                                if (!cur.consume(',')) return false;
                            }
                        }
                    } else if (prop == "data_offsets") {
                        if (!cur.consume('[')) return false;
                        if (!cur.parse_uint64(off0)) return false;
                        if (!cur.consume(',')) return false;
                        if (!cur.parse_uint64(off1)) return false;
                        if (!cur.consume(']')) return false;
                    } else {
                        if (!cur.skip_value()) return false;
                    }
                    if (cur.consume('}')) break;
                    if (!cur.consume(',')) return false;
                }
            }
            if (off1 < off0 || data_base + off1 > shard_file_bytes) return false;
            if (data_base + off1 > max_end_seen) max_end_seen = data_base + off1;

            if (is_vision_tensor_name(key)) {
                ++vision_skipped_out;
            } else {
                ent.canon_name = canonicalize_tensor_name(key);
                ent.abs_offset = data_base + off0;
                ent.byte_len   = off1 - off0;
                entries_out.push_back(std::move(ent));
            }
        }
        if (cur.consume('}')) break;
        if (!cur.consume(',')) return false;
    }

    if (max_end_seen > shard_file_bytes) return false;
    return true;
}

enum DstStore {
    DST_SPINE        = 0,
    DST_EXPERTS_NVME = 1
};

enum TargetKind {
    TGT_SPINE_F32     = 0,
    TGT_SPINE_F16     = 1,
    TGT_Q4_WEIGHT     = 2,
    TGT_Q4_SCALES     = 3,
    TGT_Q4_BIASES     = 4,
    TGT_MLA_KVB_W     = 5,
    TGT_MLA_KVB_S     = 6,
    TGT_MLA_KVB_B     = 7
};

struct TargetSpec {
    uint32_t               id              = 0;
    TargetKind             kind            = TGT_SPINE_F32;
    DstStore               dst_store       = DST_SPINE;
    bool                   is_mtp          = false;
    int32_t                global_exp_idx  = -1;   /* 0 .. 12383 for routed experts, else -1 */
    uint16_t               exp_part_mask   = 0u;   /* bit in 0..8 (or 7-bit mask on unquantized) */
    uint32_t               companion_s_id  = UINT32_MAX;
    uint32_t               companion_b_id  = UINT32_MAX;
    uint32_t               mla_layer_slot  = 0u;   /* 0..44 for main DSA, 45 for MTP */
    uint64_t               tail_zero_off   = 0ULL;
    uint64_t               tail_zero_bytes = 0ULL;
    StrataGlm53QuantMatrix qm{};
    StrataGlm53TensorSlice slice{};
};

class TargetRegistry {
public:
    std::vector<TargetSpec>                   specs;
    std::unordered_map<std::string, uint32_t> name_to_id;

    uint32_t add_spine_f32(const std::string& name, const StrataGlm53TensorSlice& sl, bool is_mtp) {
        TargetSpec s{};
        s.id        = static_cast<uint32_t>(specs.size());
        s.kind      = TGT_SPINE_F32;
        s.dst_store = DST_SPINE;
        s.is_mtp    = is_mtp;
        s.slice     = sl;
        specs.push_back(s);
        name_to_id[name] = s.id;
        return s.id;
    }

    uint32_t add_spine_f16(const std::string& name, const StrataGlm53TensorSlice& sl, bool is_mtp) {
        TargetSpec s{};
        s.id        = static_cast<uint32_t>(specs.size());
        s.kind      = TGT_SPINE_F16;
        s.dst_store = DST_SPINE;
        s.is_mtp    = is_mtp;
        s.slice     = sl;
        specs.push_back(s);
        name_to_id[name] = s.id;
        return s.id;
    }

    void add_q4_triplet(
        const std::string& prefix,
        DstStore dst_store,
        const StrataGlm53QuantMatrix& qm,
        bool is_mtp,
        int32_t global_exp_idx = -1,
        uint32_t exp_proj_slot = 0u, /* 0=gate, 1=up, 2=down */
        uint64_t rec_tail_off  = 0ULL,
        uint64_t rec_tail_len  = 0ULL
    ) {
        const uint32_t wid = static_cast<uint32_t>(specs.size());
        const uint32_t sid = wid + 1u;
        const uint32_t bid = wid + 2u;

        TargetSpec sw{};
        sw.id             = wid;
        sw.kind           = TGT_Q4_WEIGHT;
        sw.dst_store      = dst_store;
        sw.is_mtp         = is_mtp;
        sw.global_exp_idx = global_exp_idx;
        sw.exp_part_mask  = (global_exp_idx >= 0) ? static_cast<uint16_t>(1u << (exp_proj_slot * 3u + 0u)) : 0u;
        sw.companion_s_id = sid;
        sw.companion_b_id = bid;
        sw.qm             = qm;

        TargetSpec ss{};
        ss.id             = sid;
        ss.kind           = TGT_Q4_SCALES;
        ss.dst_store      = dst_store;
        ss.is_mtp         = is_mtp;
        ss.global_exp_idx = global_exp_idx;
        ss.exp_part_mask  = (global_exp_idx >= 0) ? static_cast<uint16_t>(1u << (exp_proj_slot * 3u + 1u)) : 0u;
        ss.qm             = qm;

        TargetSpec sb{};
        sb.id              = bid;
        sb.kind            = TGT_Q4_BIASES;
        sb.dst_store       = dst_store;
        sb.is_mtp          = is_mtp;
        sb.global_exp_idx  = global_exp_idx;
        sb.exp_part_mask   = (global_exp_idx >= 0) ? static_cast<uint16_t>(1u << (exp_proj_slot * 3u + 2u)) : 0u;
        sb.tail_zero_off   = rec_tail_off;
        sb.tail_zero_bytes = rec_tail_len;
        sb.qm              = qm;

        specs.push_back(sw);
        specs.push_back(ss);
        specs.push_back(sb);

        name_to_id[prefix + ".weight"] = wid;
        name_to_id[prefix + ".scales"] = sid;
        name_to_id[prefix + ".biases"] = bid;
    }

    void add_mla_kv_b_triplet(const std::string& prefix, uint32_t mla_layer_slot, bool is_mtp) {
        const uint32_t wid = static_cast<uint32_t>(specs.size());
        const uint32_t sid = wid + 1u;
        const uint32_t bid = wid + 2u;

        TargetSpec sw{};
        sw.id             = wid;
        sw.kind           = TGT_MLA_KVB_W;
        sw.dst_store      = DST_SPINE;
        sw.is_mtp         = is_mtp;
        sw.mla_layer_slot = mla_layer_slot;
        sw.companion_s_id = sid;
        sw.companion_b_id = bid;

        TargetSpec ss{};
        ss.id             = sid;
        ss.kind           = TGT_MLA_KVB_S;
        ss.dst_store      = DST_SPINE;
        ss.is_mtp         = is_mtp;
        ss.mla_layer_slot = mla_layer_slot;

        TargetSpec sb{};
        sb.id             = bid;
        sb.kind           = TGT_MLA_KVB_B;
        sb.dst_store      = DST_SPINE;
        sb.is_mtp         = is_mtp;
        sb.mla_layer_slot = mla_layer_slot;

        specs.push_back(sw);
        specs.push_back(ss);
        specs.push_back(sb);

        name_to_id[prefix + ".weight"] = wid;
        name_to_id[prefix + ".scales"] = sid;
        name_to_id[prefix + ".biases"] = bid;
    }
};

static StrataGlm53TensorSlice make_conv1d_subslice(
    const StrataGlm53TensorSlice& parent,
    uint32_t row_offset,
    uint32_t sub_rows,
    uint32_t kernel_dim
) {
    StrataGlm53TensorSlice s{};
    s.offset     = strata_glm53_slice_row_offset(&parent, row_offset);
    s.elem_count = sub_rows * kernel_dim;
    s.bytes      = static_cast<uint64_t>(s.elem_count) * sizeof(float);
    s.dtype      = STRATA_GLM53_DTYPE_F32;
    s.shape0     = sub_rows;
    s.shape1     = kernel_dim;
    return s;
}

static StrataGlm53QuantMatrix make_expert_proj_qm(
    uint64_t proj_base_off,
    uint64_t weight_bytes,
    uint64_t scales_bytes,
    uint32_t rows,
    uint32_t cols,
    uint32_t gs
) {
    StrataGlm53QuantMatrix m{};
    m.weight_off   = proj_base_off;
    m.weight_bytes = weight_bytes;
    m.scales_off   = proj_base_off + weight_bytes;
    m.scales_bytes = scales_bytes;
    m.biases_off   = proj_base_off + weight_bytes + scales_bytes;
    m.biases_bytes = scales_bytes;
    m.rows_logical = rows;
    m.rows_padded  = (rows + 63u) & ~63u;
    m.cols         = cols;
    m.group_size   = gs;
    return m;
}

static void register_decoder_layer_targets(
    TargetRegistry&               reg,
    const std::string&            lp,
    const StrataGlm53SpineHeader& hdr,
    const StrataGlm53LayerDesc&   ld,
    uint32_t                      mla_slot,
    bool                          is_mtp
) {
    const StrataGlm53Config& cfg = hdr.cfg;

    if (ld.has_mhc) {
        reg.add_spine_f32(lp + "hc_attn_fn",    ld.hc_attn.fn,    is_mtp);
        reg.add_spine_f32(lp + "hc_attn_base",  ld.hc_attn.base,  is_mtp);
        reg.add_spine_f32(lp + "hc_attn_scale", ld.hc_attn.scale, is_mtp);
        reg.add_spine_f32(lp + "hc_ffn_fn",     ld.hc_ffn.fn,     is_mtp);
        reg.add_spine_f32(lp + "hc_ffn_base",   ld.hc_ffn.base,   is_mtp);
        reg.add_spine_f32(lp + "hc_ffn_scale",  ld.hc_ffn.scale,  is_mtp);
    }

    /* CRITICAL: Glm5NextTextRMSNorm is direct T5LayerNorm (NO +1.0f!) */
    reg.add_spine_f32(lp + "input_layernorm.weight",          ld.input_layernorm,          is_mtp);
    reg.add_spine_f32(lp + "post_attention_layernorm.weight", ld.post_attention_layernorm, is_mtp);

    const std::string ap = lp + "self_attn.";
    if (ld.attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) {
        const uint32_t nh      = cfg.linear_num_heads;        /* 64   */
        const uint32_t hd      = cfg.linear_head_dim;         /* 128  */
        const uint32_t qkv_dim = nh * hd;                     /* 8192 */
        const uint32_t k_dim   = cfg.linear_conv_kernel_dim;  /* 4    */

        reg.add_q4_triplet(ap + "q_proj", DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.lin_qkv_proj, 0u,           qkv_dim), is_mtp);
        reg.add_q4_triplet(ap + "k_proj", DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.lin_qkv_proj, qkv_dim,      qkv_dim), is_mtp);
        reg.add_q4_triplet(ap + "v_proj", DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.lin_qkv_proj, 2u * qkv_dim, qkv_dim), is_mtp);

        reg.add_q4_triplet(ap + "f_a_proj", DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.lin_fbg_a_proj, 0u,      hd), is_mtp);
        reg.add_q4_triplet(ap + "b_proj",   DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.lin_fbg_a_proj, hd,      nh), is_mtp);
        reg.add_q4_triplet(ap + "g_a_proj", DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.lin_fbg_a_proj, hd + nh, hd), is_mtp);

        reg.add_q4_triplet(ap + "f_b_proj", DST_SPINE, ld.lin_f_b_proj, is_mtp);
        reg.add_q4_triplet(ap + "g_b_proj", DST_SPINE, ld.lin_g_b_proj, is_mtp);
        reg.add_q4_triplet(ap + "o_proj",   DST_SPINE, ld.lin_o_proj,   is_mtp);

        reg.add_spine_f32(ap + "q_conv1d.weight",
                          make_conv1d_subslice(ld.lin_qkv_conv1d, 0u,           qkv_dim, k_dim), is_mtp);
        reg.add_spine_f32(ap + "k_conv1d.weight",
                          make_conv1d_subslice(ld.lin_qkv_conv1d, qkv_dim,      qkv_dim, k_dim), is_mtp);
        reg.add_spine_f32(ap + "v_conv1d.weight",
                          make_conv1d_subslice(ld.lin_qkv_conv1d, 2u * qkv_dim, qkv_dim, k_dim), is_mtp);

        reg.add_spine_f32(ap + "A_log",         ld.lin_A_log,   is_mtp);
        reg.add_spine_f32(ap + "dt_bias",       ld.lin_dt_bias, is_mtp);
        reg.add_spine_f32(ap + "o_norm.weight", ld.lin_o_norm,  is_mtp);
    } else {
        const uint32_t qr  = cfg.q_lora_rank;  /* 1536 */
        const uint32_t kvr = cfg.kv_lora_rank; /* 512  */

        reg.add_q4_triplet(ap + "q_a_proj", DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.mla_qkv_a_proj, 0u, qr), is_mtp);
        reg.add_q4_triplet(ap + "kv_a_proj_with_mqa", DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.mla_qkv_a_proj, qr, kvr), is_mtp);

        reg.add_spine_f32(ap + "q_a_layernorm.weight",  ld.mla_q_a_layernorm,  is_mtp);
        reg.add_q4_triplet(ap + "q_b_proj", DST_SPINE,  ld.mla_q_b_proj,       is_mtp);
        reg.add_spine_f32(ap + "kv_a_layernorm.weight", ld.mla_kv_a_layernorm, is_mtp);
        reg.add_mla_kv_b_triplet(ap + "kv_b_proj", mla_slot, is_mtp);
        reg.add_q4_triplet(ap + "o_proj",   DST_SPINE,  ld.mla_o_proj,         is_mtp);

        const std::string ip = ap + "indexer.";
        reg.add_q4_triplet(ip + "wq_b",         DST_SPINE, ld.indexer_wq_b,         is_mtp);
        reg.add_q4_triplet(ip + "wk",           DST_SPINE, ld.indexer_wk,           is_mtp);
        reg.add_q4_triplet(ip + "weights_proj", DST_SPINE, ld.indexer_weights_proj, is_mtp);
        reg.add_spine_f32(ip + "k_norm.weight",             ld.indexer_k_norm_weight, is_mtp);
        reg.add_spine_f32(ip + "k_norm.bias",               ld.indexer_k_norm_bias,   is_mtp);
        reg.add_spine_f32(ip + "index_kpool_compress_ape",  ld.indexer_compress_ape,  is_mtp);
        reg.add_spine_f32(ip + "index_kpool_compress_gate", ld.indexer_compress_gate, is_mtp);
    }

    const std::string mp = lp + "mlp.";
    if (ld.mlp_type == STRATA_GLM53_MLP_DENSE) {
        const uint32_t d_ff = cfg.intermediate_size; /* 12288 */
        reg.add_q4_triplet(mp + "gate_proj", DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.dense_gate_up_proj, 0u,   d_ff), is_mtp);
        reg.add_q4_triplet(mp + "up_proj",   DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.dense_gate_up_proj, d_ff, d_ff), is_mtp);
        reg.add_q4_triplet(mp + "down_proj", DST_SPINE, ld.dense_down_proj, is_mtp);
    } else {
        const uint32_t d    = cfg.hidden_size;
        const uint32_t d_mo = cfg.moe_intermediate_size;
        const uint32_t d_sh = cfg.moe_intermediate_size * cfg.n_shared_experts;
        const uint32_t gs   = cfg.quant_group_size;

        reg.add_spine_f16(mp + "gate.weight",                  ld.moe_router_gate,  is_mtp);
        reg.add_spine_f32(mp + "gate.e_score_correction_bias", ld.moe_e_score_bias, is_mtp);

        reg.add_q4_triplet(mp + "shared_experts.gate_proj", DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.shared_gate_up_proj, 0u,   d_sh), is_mtp);
        reg.add_q4_triplet(mp + "shared_experts.up_proj",   DST_SPINE,
                           strata_glm53_q4_row_subslice(&ld.shared_gate_up_proj, d_sh, d_sh), is_mtp);
        reg.add_q4_triplet(mp + "shared_experts.down_proj", DST_SPINE, ld.shared_down_proj, is_mtp);

        const uint32_t moe_l = static_cast<uint32_t>(ld.moe_layer_idx);
        const uint64_t prj_b = hdr.expert_proj_bytes;
        const uint64_t w_b   = hdr.expert_weight_bytes;
        const uint64_t s_b   = hdr.expert_scales_bytes;
        const uint64_t rec_b = hdr.expert_record_bytes;
        const uint64_t used  = 3ULL * prj_b;
        const uint64_t tail  = (rec_b > used) ? (rec_b - used) : 0ULL;

        for (uint32_t e = 0; e < cfg.n_routed_experts; ++e) {
            const int32_t  g_exp = static_cast<int32_t>(moe_l * cfg.n_routed_experts + e);
            const uint64_t e_off = strata_glm53_expert_file_offset(&hdr, moe_l, e);
            const std::string ep = mp + "experts." + std::to_string(e) + ".";

            reg.add_q4_triplet(ep + "gate_proj", DST_EXPERTS_NVME,
                               make_expert_proj_qm(e_off + 0ULL * prj_b, w_b, s_b, d_mo, d, gs),
                               is_mtp, g_exp, 0u);
            reg.add_q4_triplet(ep + "up_proj",   DST_EXPERTS_NVME,
                               make_expert_proj_qm(e_off + 1ULL * prj_b, w_b, s_b, d_mo, d, gs),
                               is_mtp, g_exp, 1u);
            reg.add_q4_triplet(ep + "down_proj", DST_EXPERTS_NVME,
                               make_expert_proj_qm(e_off + 2ULL * prj_b, w_b, s_b, d, d_mo, gs),
                               is_mtp, g_exp, 2u, e_off + used, tail);
        }
    }
}

static void build_target_registry(
    const StrataGlm53SpineHeader& hdr,
    TargetRegistry&               reg
) {
    const StrataGlm53Config& cfg = hdr.cfg;
    reg.specs.reserve(115000);
    reg.name_to_id.reserve(131072);

    reg.add_q4_triplet("model.embed_tokens", DST_SPINE, hdr.embed_tokens, false);
    for (uint32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        register_decoder_layer_targets(
            reg, "model.layers." + std::to_string(l) + ".", hdr, hdr.layers[l], l, false);
    }
    reg.add_spine_f32("model.norm.weight", hdr.final_norm, false);
    if (!cfg.tie_word_embeddings) {
        reg.add_q4_triplet("lm_head", DST_SPINE, hdr.lm_head, false);
    }

    if (hdr.mtp.enabled) {
        const uint32_t mtp_l = hdr.mtp.global_layer_idx; /* 45 */
        const std::string mp = "model.layers." + std::to_string(mtp_l) + ".";
        reg.add_spine_f32(mp + "enorm.weight",            hdr.mtp.enorm,            true);
        reg.add_spine_f32(mp + "hnorm.weight",            hdr.mtp.hnorm,            true);
        reg.add_q4_triplet(mp + "eh_proj", DST_SPINE,     hdr.mtp.eh_proj,          true);
        register_decoder_layer_targets(reg, mp, hdr, hdr.mtp.layer, mtp_l, true);
        reg.add_spine_f32(mp + "shared_head.norm.weight", hdr.mtp.shared_head_norm, true);
    }
}

/* ============================================================================
 * 5. Low-RSS Chunked NEON Shard Converter (`< 256 MiB` Peak RSS)
 * ============================================================================ */

struct ShardConverter {
    int                             spine_fd   = -1;
    int                             experts_fd = -1;
    const StrataGlm53SpineHeader*   spine_hdr  = nullptr;
    NeonWorkerPool*                 pool       = nullptr;
    std::atomic<bool>               nan_inf_detected{false};

    /* Single reusable 72 MiB scratch arena across all 43 shards */
    std::vector<uint8_t>            in_buf;     /* 32 MiB */
    std::vector<uint8_t>            out_w_buf;  /* 32 MiB */
    std::vector<uint8_t>            out_s_buf;  /* 4 MiB  */
    std::vector<uint8_t>            out_b_buf;  /* 4 MiB  */

    ShardConverter(int sfd, int efd, const StrataGlm53SpineHeader* hdr, NeonWorkerPool* wp)
        : spine_fd(sfd), experts_fd(efd), spine_hdr(hdr), pool(wp) {
        in_buf.resize(32ULL * 1024ULL * 1024ULL);
        out_w_buf.resize(32ULL * 1024ULL * 1024ULL);
        out_s_buf.resize(4ULL * 1024ULL * 1024ULL);
        out_b_buf.resize(4ULL * 1024ULL * 1024ULL);
    }

    inline int out_fd_for(DstStore ds) const {
        return (ds == DST_SPINE) ? spine_fd : experts_fd;
    }

    static uint64_t shape_elems(const std::vector<uint64_t>& shape) {
        if (shape.empty()) return 0;
        uint64_t n = 1;
        for (uint64_t d : shape) n *= d;
        return n;
    }

    static uint64_t dtype_bpe(SafetensorDtype dt) {
        switch (dt) {
            case ST_BF16:
            case ST_F16:  return 2ULL;
            case ST_F32:
            case ST_U32:  return 4ULL;
            case ST_I64:  return 8ULL;
            default:      return 0ULL;
        }
    }

    bool convert_spine_f32(int shard_fd, const SafetensorEntry& ent, const TargetSpec& spec) {
        const uint64_t elems = shape_elems(ent.shape);
        const uint64_t bpe   = dtype_bpe(ent.dtype);
        if (elems != spec.slice.elem_count || bpe == 0 || ent.byte_len != elems * bpe) {
            std::fprintf(stderr, "[glm53-stream] ERROR: F32 shape mismatch on '%s' (got %llu, expected %u)\n",
                         ent.canon_name.c_str(), static_cast<unsigned long long>(elems), spec.slice.elem_count);
            return false;
        }

        uint64_t rem_elems = elems;
        uint64_t r_off     = ent.abs_offset;
        uint64_t w_off     = spec.slice.offset;
        const uint64_t max_batch = std::min<uint64_t>(in_buf.size() / bpe, out_w_buf.size() / sizeof(float));

        while (rem_elems > 0) {
            const uint64_t n = std::min<uint64_t>(rem_elems, max_batch);
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(n * bpe), r_off)) return false;

            float* dst = reinterpret_cast<float*>(out_w_buf.data());
            bool finite_ok = true;
            if (ent.dtype == ST_BF16) {
                finite_ok = convert_bf16_to_f32_span_checked(
                    reinterpret_cast<const uint16_t*>(in_buf.data()), dst, static_cast<size_t>(n));
            } else if (ent.dtype == ST_F16) {
                const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
                finite_ok = check_f16_span_finite(s16, static_cast<size_t>(n));
                for (uint64_t i = 0; i < n; ++i) dst[i] = fp16_to_f32_fast(s16[i]);
            } else if (ent.dtype == ST_F32) {
                const float* sf = reinterpret_cast<const float*>(in_buf.data());
                finite_ok = check_f32_span_finite(sf, static_cast<size_t>(n));
                std::memcpy(dst, sf, static_cast<size_t>(n * sizeof(float)));
            } else {
                return false;
            }
            if (!finite_ok) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                std::fprintf(stderr, "[glm53-stream] ERROR: NaN/Inf detected in '%s'\n", ent.canon_name.c_str());
                return false;
            }

            const uint64_t chunk_out = n * sizeof(float);
            if (!pwrite_exact(spine_fd, dst, static_cast<size_t>(chunk_out), w_off)) return false;
            r_off     += n * bpe;
            w_off     += chunk_out;
            rem_elems -= n;
        }

        const uint64_t written = elems * sizeof(float);
        if (spec.slice.bytes > written) {
            if (!pwrite_zeros(spine_fd, spec.slice.offset + written, spec.slice.bytes - written)) return false;
        }
        return true;
    }

    bool convert_spine_f16(int shard_fd, const SafetensorEntry& ent, const TargetSpec& spec) {
        const uint64_t elems = shape_elems(ent.shape);
        const uint64_t bpe   = dtype_bpe(ent.dtype);
        if (elems != spec.slice.elem_count || bpe == 0 || ent.byte_len != elems * bpe) {
            std::fprintf(stderr, "[glm53-stream] ERROR: F16 shape mismatch on '%s' (got %llu, expected %u)\n",
                         ent.canon_name.c_str(), static_cast<unsigned long long>(elems), spec.slice.elem_count);
            return false;
        }

        uint64_t rem_elems = elems;
        uint64_t r_off     = ent.abs_offset;
        uint64_t w_off     = spec.slice.offset;
        const uint64_t max_batch = std::min<uint64_t>(in_buf.size() / bpe, out_w_buf.size() / sizeof(uint16_t));

        while (rem_elems > 0) {
            const uint64_t n = std::min<uint64_t>(rem_elems, max_batch);
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(n * bpe), r_off)) return false;

            uint16_t* dst16 = reinterpret_cast<uint16_t*>(out_w_buf.data());
            std::atomic<bool> batch_finite{true};
            if (ent.dtype == ST_BF16) {
                const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
                pool->parallel_for_rows(n, 32768ULL, [&](uint64_t i0, uint64_t i1) {
                    if (!convert_bf16_to_fp16_span_checked(s16 + i0, dst16 + i0, static_cast<size_t>(i1 - i0))) {
                        batch_finite.store(false, std::memory_order_relaxed);
                    }
                });
            } else if (ent.dtype == ST_F16) {
                const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
                if (!check_f16_span_finite(s16, static_cast<size_t>(n))) batch_finite.store(false);
                std::memcpy(dst16, s16, static_cast<size_t>(n * sizeof(uint16_t)));
            } else if (ent.dtype == ST_F32) {
                const float* sf = reinterpret_cast<const float*>(in_buf.data());
                pool->parallel_for_rows(n, 32768ULL, [&](uint64_t i0, uint64_t i1) {
                    if (!convert_f32_to_fp16_span_checked(sf + i0, dst16 + i0, static_cast<size_t>(i1 - i0))) {
                        batch_finite.store(false, std::memory_order_relaxed);
                    }
                });
            } else {
                return false;
            }
            if (!batch_finite.load(std::memory_order_relaxed)) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                std::fprintf(stderr, "[glm53-stream] ERROR: NaN/Inf detected in '%s'\n", ent.canon_name.c_str());
                return false;
            }

            const uint64_t out_chunk_bytes = n * sizeof(uint16_t);
            if (!pwrite_exact(spine_fd, dst16, static_cast<size_t>(out_chunk_bytes), w_off)) return false;
            r_off     += n * bpe;
            w_off     += out_chunk_bytes;
            rem_elems -= n;
        }

        const uint64_t written = elems * sizeof(uint16_t);
        if (spec.slice.bytes > written) {
            if (!pwrite_zeros(spine_fd, spec.slice.offset + written, spec.slice.bytes - written)) return false;
        }
        return true;
    }

    bool convert_q4_weight(
        int shard_fd,
        const SafetensorEntry& ent,
        const TargetSpec& spec,
        bool* quantized_unpacked_out
    ) {
        if (quantized_unpacked_out) *quantized_unpacked_out = false;
        const int dst_fd    = out_fd_for(spec.dst_store);
        const uint64_t rows = spec.qm.rows_logical;
        const uint64_t cols = spec.qm.cols;

        if (ent.dtype == ST_U32) {
            const uint64_t u32_cols      = cols / 8ULL;
            const uint64_t logical_elems = rows * u32_cols;
            const uint64_t logical_bytes = logical_elems * sizeof(uint32_t);
            if (ent.shape.size() != 2 || ent.shape[0] != rows || ent.shape[1] != u32_cols ||
                ent.byte_len != logical_bytes || logical_bytes > spec.qm.weight_bytes) {
                std::fprintf(stderr, "[glm53-stream] ERROR: U32 Q4 weight mismatch on '%s' (got [%llu, %llu], expected [%llu, %llu])\n",
                             ent.canon_name.c_str(),
                             ent.shape.size() > 0 ? static_cast<unsigned long long>(ent.shape[0]) : 0ULL,
                             ent.shape.size() > 1 ? static_cast<unsigned long long>(ent.shape[1]) : 0ULL,
                             static_cast<unsigned long long>(rows),
                             static_cast<unsigned long long>(u32_cols));
                return false;
            }

            uint64_t rem = logical_bytes;
            uint64_t r_off = ent.abs_offset;
            uint64_t w_off = spec.qm.weight_off;
            while (rem > 0) {
                const size_t chunk = static_cast<size_t>(std::min<uint64_t>(in_buf.size(), rem));
                if (!pread_exact(shard_fd, in_buf.data(), chunk, r_off) ||
                    !pwrite_exact(dst_fd, in_buf.data(), chunk, w_off)) {
                    return false;
                }
                r_off += chunk;
                w_off += chunk;
                rem   -= chunk;
            }
            if (spec.qm.weight_bytes > logical_bytes) {
                if (!pwrite_zeros(dst_fd, spec.qm.weight_off + logical_bytes,
                                  spec.qm.weight_bytes - logical_bytes)) {
                    return false;
                }
            }
            return true;
        }

        /* Fallback path for synthetic unquantized BF16/F16/F32 weight matrices */
        const uint64_t bpe   = dtype_bpe(ent.dtype);
        const uint64_t elems = shape_elems(ent.shape);
        const uint64_t gs    = spec.qm.group_size;
        if (bpe == 0 || gs != 64ULL || (cols % 64ULL) != 0ULL || elems != rows * cols ||
            ent.byte_len != elems * bpe) {
            return false;
        }
        const uint64_t in_row_bytes   = cols * bpe;
        const uint64_t w_row_bytes    = cols / 2ULL;
        const uint64_t groups_per_row = cols / 64ULL;
        const uint64_t sb_row_bytes   = groups_per_row * 2ULL;

        uint64_t max_rows = std::min<uint64_t>(in_buf.size() / in_row_bytes, out_w_buf.size() / w_row_bytes);
        max_rows = std::min<uint64_t>(max_rows, out_s_buf.size() / sb_row_bytes);
        if (max_rows == 0) return false;

        uint64_t row_done = 0;
        while (row_done < rows) {
            const uint64_t batch_rows = std::min<uint64_t>(max_rows, rows - row_done);
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(batch_rows * in_row_bytes),
                             ent.abs_offset + row_done * in_row_bytes)) {
                return false;
            }
            uint8_t*  w_base = out_w_buf.data();
            uint16_t* s_base = reinterpret_cast<uint16_t*>(out_s_buf.data());
            uint16_t* b_base = reinterpret_cast<uint16_t*>(out_b_buf.data());
            const uint8_t* src_raw = in_buf.data();

            pool->parallel_for_rows(batch_rows, 32ULL, [&](uint64_t r0, uint64_t r1) {
                float tmp[64];
                for (uint64_t r = r0; r < r1; ++r) {
                    uint32_t* q_row = reinterpret_cast<uint32_t*>(w_base + r * w_row_bytes);
                    uint16_t* sc_r  = s_base + r * groups_per_row;
                    uint16_t* bi_r  = b_base + r * groups_per_row;
                    for (uint64_t g = 0; g < groups_per_row; ++g) {
                        const uint64_t base_c = r * cols + g * 64ULL;
                        if (ent.dtype == ST_BF16) {
                            const uint16_t* p16 = reinterpret_cast<const uint16_t*>(src_raw) + base_c;
                            for (int j = 0; j < 64; ++j) tmp[j] = bf16_to_f32_scalar(p16[j]);
                        } else if (ent.dtype == ST_F16) {
                            const uint16_t* p16 = reinterpret_cast<const uint16_t*>(src_raw) + base_c;
                            for (int j = 0; j < 64; ++j) tmp[j] = fp16_to_f32_fast(p16[j]);
                        } else {
                            const float* pf = reinterpret_cast<const float*>(src_raw) + base_c;
                            std::memcpy(tmp, pf, 64 * sizeof(float));
                        }
                        quantize_group_f32(tmp, 64u, q_row + g * 8ULL, &sc_r[g], &bi_r[g]);
                    }
                }
            });

            if (!pwrite_exact(dst_fd, w_base, static_cast<size_t>(batch_rows * w_row_bytes),
                              spec.qm.weight_off + row_done * w_row_bytes) ||
                !pwrite_exact(dst_fd, s_base, static_cast<size_t>(batch_rows * sb_row_bytes),
                              spec.qm.scales_off + row_done * sb_row_bytes) ||
                !pwrite_exact(dst_fd, b_base, static_cast<size_t>(batch_rows * sb_row_bytes),
                              spec.qm.biases_off + row_done * sb_row_bytes)) {
                return false;
            }
            row_done += batch_rows;
        }

        const uint64_t w_written  = rows * w_row_bytes;
        const uint64_t sb_written = rows * sb_row_bytes;
        if (spec.qm.weight_bytes > w_written &&
            !pwrite_zeros(dst_fd, spec.qm.weight_off + w_written, spec.qm.weight_bytes - w_written)) {
            return false;
        }
        if (spec.qm.scales_bytes > sb_written &&
            !pwrite_zeros(dst_fd, spec.qm.scales_off + sb_written, spec.qm.scales_bytes - sb_written)) {
            return false;
        }
        if (spec.qm.biases_bytes > sb_written &&
            !pwrite_zeros(dst_fd, spec.qm.biases_off + sb_written, spec.qm.biases_bytes - sb_written)) {
            return false;
        }
        if (quantized_unpacked_out) *quantized_unpacked_out = true;
        return true;
    }

    bool convert_q4_scales_or_biases(int shard_fd, const SafetensorEntry& ent, const TargetSpec& spec) {
        const int      dst_fd        = out_fd_for(spec.dst_store);
        const uint64_t rows          = spec.qm.rows_logical;
        const uint64_t grp_cols      = spec.qm.cols / spec.qm.group_size;
        const uint64_t logical_elems = rows * grp_cols;
        const uint64_t bpe           = dtype_bpe(ent.dtype);
        const uint64_t dst_off       = (spec.kind == TGT_Q4_SCALES) ? spec.qm.scales_off   : spec.qm.biases_off;
        const uint64_t dst_bytes     = (spec.kind == TGT_Q4_SCALES) ? spec.qm.scales_bytes : spec.qm.biases_bytes;

        if (ent.shape.size() != 2 || ent.shape[0] != rows || ent.shape[1] != grp_cols ||
            bpe == 0 || ent.byte_len != logical_elems * bpe ||
            logical_elems * sizeof(uint16_t) > dst_bytes) {
            std::fprintf(stderr, "[glm53-stream] ERROR: Q4 scale/bias mismatch on '%s'\n",
                         ent.canon_name.c_str());
            return false;
        }

        uint64_t rem_elems = logical_elems;
        uint64_t r_off     = ent.abs_offset;
        uint64_t w_off     = dst_off;
        const uint64_t max_batch = std::min<uint64_t>(in_buf.size() / bpe, out_w_buf.size() / sizeof(uint16_t));

        while (rem_elems > 0) {
            const uint64_t n = std::min<uint64_t>(rem_elems, max_batch);
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(n * bpe), r_off)) return false;

            uint16_t* dst16 = reinterpret_cast<uint16_t*>(out_w_buf.data());
            std::atomic<bool> batch_finite{true};
            if (ent.dtype == ST_BF16) {
                const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
                pool->parallel_for_rows(n, 32768ULL, [&](uint64_t i0, uint64_t i1) {
                    if (!convert_bf16_to_fp16_span_checked(s16 + i0, dst16 + i0, static_cast<size_t>(i1 - i0))) {
                        batch_finite.store(false, std::memory_order_relaxed);
                    }
                });
            } else if (ent.dtype == ST_F16) {
                const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
                if (!check_f16_span_finite(s16, static_cast<size_t>(n))) batch_finite.store(false);
                std::memcpy(dst16, s16, static_cast<size_t>(n * sizeof(uint16_t)));
            } else if (ent.dtype == ST_F32) {
                const float* sf = reinterpret_cast<const float*>(in_buf.data());
                pool->parallel_for_rows(n, 32768ULL, [&](uint64_t i0, uint64_t i1) {
                    if (!convert_f32_to_fp16_span_checked(sf + i0, dst16 + i0, static_cast<size_t>(i1 - i0))) {
                        batch_finite.store(false, std::memory_order_relaxed);
                    }
                });
            } else {
                return false;
            }
            if (!batch_finite.load(std::memory_order_relaxed)) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                std::fprintf(stderr, "[glm53-stream] ERROR: NaN/Inf detected in '%s'\n", ent.canon_name.c_str());
                return false;
            }

            const uint64_t chunk_bytes = n * sizeof(uint16_t);
            if (!pwrite_exact(dst_fd, dst16, static_cast<size_t>(chunk_bytes), w_off)) return false;
            r_off     += n * bpe;
            w_off     += chunk_bytes;
            rem_elems -= n;
        }

        const uint64_t written = logical_elems * sizeof(uint16_t);
        if (dst_bytes > written) {
            if (!pwrite_zeros(dst_fd, dst_off + written, dst_bytes - written)) return false;
        }
        if (spec.tail_zero_bytes > 0) {
            if (!pwrite_zeros(dst_fd, spec.tail_zero_off, spec.tail_zero_bytes)) return false;
        }
        return true;
    }

    /*
     * Unpacks `self_attn.kv_b_proj.{weight,scales,biases}` (`[nh * (q_nope + v_dim), kv_lora_rank]`)
     * via `sg53_unpack_kv_b_proj_head()` into lossless Q4 (`mla_kv_b_k`, `mla_kv_b_v`) AND pre-dequantized
     * absorbed FP16 matrices (`mla_w_uk` transposed `[nh * kvr, q_nope]` + `mla_w_uv` `[nh * v_dim, kvr]`).
     */
    bool convert_mla_kv_b_triplet(
        int shard_fd,
        const SafetensorEntry& w_ent,
        const SafetensorEntry* s_ent,
        const SafetensorEntry* b_ent,
        const StrataGlm53LayerDesc& ld
    ) {
        const StrataGlm53Config& cfg = spine_hdr->cfg;
        const uint32_t nh            = cfg.num_attention_heads; /* 64  */
        const uint32_t q_nope        = cfg.qk_nope_head_dim;    /* 256 */
        const uint32_t v_dim         = cfg.v_head_dim;          /* 256 */
        const uint32_t kvr           = cfg.kv_lora_rank;        /* 512 */
        const uint32_t gs            = cfg.quant_group_size;    /* 64  */
        const uint32_t rows_per_head = q_nope + v_dim;          /* 512 */
        const uint32_t total_rows    = nh * rows_per_head;      /* 32768 */
        const uint32_t u32_per_row   = kvr >> 3u;               /* 64  */
        const uint32_t grp_per_row   = kvr / gs;                /* 8   */

        if (!s_ent || !b_ent) {
            std::fprintf(stderr, "[glm53-stream] ERROR: missing kv_b_proj scales/biases for '%s'\n",
                         w_ent.canon_name.c_str());
            return false;
        }

        const uint64_t w_bytes_in = static_cast<uint64_t>(total_rows) * u32_per_row * sizeof(uint32_t);
        const uint64_t s_bytes_in = static_cast<uint64_t>(total_rows) * grp_per_row * sizeof(uint16_t);
        if (w_ent.dtype != ST_U32 || w_ent.byte_len != w_bytes_in ||
            s_ent->dtype != ST_BF16 || s_ent->byte_len != s_bytes_in ||
            b_ent->dtype != ST_BF16 || b_ent->byte_len != s_bytes_in ||
            w_bytes_in + 2ULL * s_bytes_in > in_buf.size()) {
            std::fprintf(stderr, "[glm53-stream] ERROR: kv_b_proj shape/dtype mismatch on '%s'\n",
                         w_ent.canon_name.c_str());
            return false;
        }

        uint8_t* w_in = in_buf.data();
        uint8_t* s_in = w_in + w_bytes_in;
        uint8_t* b_in = s_in + s_bytes_in;
        if (!pread_exact(shard_fd, w_in, static_cast<size_t>(w_bytes_in), w_ent.abs_offset) ||
            !pread_exact(shard_fd, s_in, static_cast<size_t>(s_bytes_in), s_ent->abs_offset) ||
            !pread_exact(shard_fd, b_in, static_cast<size_t>(s_bytes_in), b_ent->abs_offset)) {
            return false;
        }

        const uint16_t* s_bf16_all = reinterpret_cast<const uint16_t*>(s_in);
        const uint16_t* b_bf16_all = reinterpret_cast<const uint16_t*>(b_in);
        const size_t sb_elems_all  = static_cast<size_t>(total_rows) * grp_per_row;
        for (size_t i = 0; i < sb_elems_all; ++i) {
            if (!bf16_is_finite(s_bf16_all[i]) || !bf16_is_finite(b_bf16_all[i])) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                return false;
            }
        }

        /* Process in batches of up to 8 heads inside existing `out_w_buf`/`out_s_buf`/`out_b_buf` (0 extra alloc) */
        const uint64_t kw_head_b = static_cast<uint64_t>(q_nope) * u32_per_row * sizeof(uint32_t);
        const uint64_t vw_head_b = static_cast<uint64_t>(v_dim)  * u32_per_row * sizeof(uint32_t);
        const uint64_t ks_head_b = static_cast<uint64_t>(q_nope) * grp_per_row * sizeof(uint16_t);
        const uint64_t vs_head_b = static_cast<uint64_t>(v_dim)  * grp_per_row * sizeof(uint16_t);
        const uint64_t uk_head_b = static_cast<uint64_t>(kvr)    * q_nope      * sizeof(uint16_t);
        const uint64_t uv_head_b = static_cast<uint64_t>(v_dim)  * kvr         * sizeof(uint16_t);

        const uint64_t per_head_w_arena = kw_head_b + vw_head_b + uk_head_b + uv_head_b;
        const uint32_t max_batch_heads = static_cast<uint32_t>(
            std::min<uint64_t>(nh, std::min<uint64_t>(
                out_w_buf.size() / per_head_w_arena,
                out_s_buf.size() / (ks_head_b + vs_head_b))));
        if (max_batch_heads == 0) return false;

        uint32_t h_done = 0;
        while (h_done < nh) {
            const uint32_t bh = std::min<uint32_t>(max_batch_heads, nh - h_done);

            uint8_t* kw_base = out_w_buf.data();
            uint8_t* vw_base = kw_base + bh * kw_head_b;
            uint8_t* uk_base = vw_base + bh * vw_head_b;
            uint8_t* uv_base = uk_base + bh * uk_head_b;

            uint8_t* ks_base = out_s_buf.data();
            uint8_t* vs_base = ks_base + bh * ks_head_b;

            uint8_t* kb_base = out_b_buf.data();
            uint8_t* vb_base = kb_base + bh * ks_head_b;

            const uint32_t* w_u32_all = reinterpret_cast<const uint32_t*>(w_in);
            pool->parallel_for_rows(bh, 1ULL, [&](uint64_t i0, uint64_t i1) {
                for (uint64_t i = i0; i < i1; ++i) {
                    const uint32_t h = h_done + static_cast<uint32_t>(i);
                    const uint32_t* hw = w_u32_all  + static_cast<size_t>(h) * rows_per_head * u32_per_row;
                    const uint16_t* hs = s_bf16_all + static_cast<size_t>(h) * rows_per_head * grp_per_row;
                    const uint16_t* hb = b_bf16_all + static_cast<size_t>(h) * rows_per_head * grp_per_row;

                    sg53_unpack_kv_b_proj_head(
                        hw, hs, hb,
                        q_nope, v_dim, kvr, gs,
                        reinterpret_cast<uint32_t*>(kw_base + i * kw_head_b),
                        reinterpret_cast<uint16_t*>(ks_base + i * ks_head_b),
                        reinterpret_cast<uint16_t*>(kb_base + i * ks_head_b),
                        reinterpret_cast<uint32_t*>(vw_base + i * vw_head_b),
                        reinterpret_cast<uint16_t*>(vs_base + i * vs_head_b),
                        reinterpret_cast<uint16_t*>(vb_base + i * vs_head_b),
                        reinterpret_cast<uint16_t*>(uk_base + i * uk_head_b),
                        reinterpret_cast<uint16_t*>(uv_base + i * uv_head_b)
                    );
                }
            });

            if (!check_f16_span_finite(reinterpret_cast<const uint16_t*>(uk_base),
                                       static_cast<size_t>(bh * uk_head_b / 2ULL)) ||
                !check_f16_span_finite(reinterpret_cast<const uint16_t*>(uv_base),
                                       static_cast<size_t>(bh * uv_head_b / 2ULL))) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                std::fprintf(stderr, "[glm53-stream] ERROR: NaN/Inf in unpacked mla_w_uk/w_uv on '%s'\n",
                             w_ent.canon_name.c_str());
                return false;
            }

            if (!pwrite_exact(spine_fd, kw_base, static_cast<size_t>(bh * kw_head_b),
                              ld.mla_kv_b_k.weight_off + h_done * kw_head_b) ||
                !pwrite_exact(spine_fd, ks_base, static_cast<size_t>(bh * ks_head_b),
                              ld.mla_kv_b_k.scales_off + h_done * ks_head_b) ||
                !pwrite_exact(spine_fd, kb_base, static_cast<size_t>(bh * ks_head_b),
                              ld.mla_kv_b_k.biases_off + h_done * ks_head_b) ||
                !pwrite_exact(spine_fd, vw_base, static_cast<size_t>(bh * vw_head_b),
                              ld.mla_kv_b_v.weight_off + h_done * vw_head_b) ||
                !pwrite_exact(spine_fd, vs_base, static_cast<size_t>(bh * vs_head_b),
                              ld.mla_kv_b_v.scales_off + h_done * vs_head_b) ||
                !pwrite_exact(spine_fd, vb_base, static_cast<size_t>(bh * vs_head_b),
                              ld.mla_kv_b_v.biases_off + h_done * vs_head_b) ||
                !pwrite_exact(spine_fd, uk_base, static_cast<size_t>(bh * uk_head_b),
                              ld.mla_w_uk.offset + h_done * uk_head_b) ||
                !pwrite_exact(spine_fd, uv_base, static_cast<size_t>(bh * uv_head_b),
                              ld.mla_w_uv.offset + h_done * uv_head_b)) {
                return false;
            }

            h_done += bh;
        }

        /* Zero-fill any padding tails on small synthetic dimensions */
        if (ld.mla_kv_b_k.weight_bytes > nh * kw_head_b &&
            !pwrite_zeros(spine_fd, ld.mla_kv_b_k.weight_off + nh * kw_head_b,
                          ld.mla_kv_b_k.weight_bytes - nh * kw_head_b)) return false;
        if (ld.mla_kv_b_k.scales_bytes > nh * ks_head_b &&
            (!pwrite_zeros(spine_fd, ld.mla_kv_b_k.scales_off + nh * ks_head_b,
                           ld.mla_kv_b_k.scales_bytes - nh * ks_head_b) ||
             !pwrite_zeros(spine_fd, ld.mla_kv_b_k.biases_off + nh * ks_head_b,
                           ld.mla_kv_b_k.biases_bytes - nh * ks_head_b))) return false;
        if (ld.mla_kv_b_v.weight_bytes > nh * vw_head_b &&
            !pwrite_zeros(spine_fd, ld.mla_kv_b_v.weight_off + nh * vw_head_b,
                          ld.mla_kv_b_v.weight_bytes - nh * vw_head_b)) return false;
        if (ld.mla_kv_b_v.scales_bytes > nh * vs_head_b &&
            (!pwrite_zeros(spine_fd, ld.mla_kv_b_v.scales_off + nh * vs_head_b,
                           ld.mla_kv_b_v.scales_bytes - nh * vs_head_b) ||
             !pwrite_zeros(spine_fd, ld.mla_kv_b_v.biases_off + nh * vs_head_b,
                           ld.mla_kv_b_v.biases_bytes - nh * vs_head_b))) return false;
        if (ld.mla_w_uk.bytes > nh * uk_head_b &&
            !pwrite_zeros(spine_fd, ld.mla_w_uk.offset + nh * uk_head_b,
                          ld.mla_w_uk.bytes - nh * uk_head_b)) return false;
        if (ld.mla_w_uv.bytes > nh * uv_head_b &&
            !pwrite_zeros(spine_fd, ld.mla_w_uv.offset + nh * uv_head_b,
                          ld.mla_w_uv.bytes - nh * uv_head_b)) return false;

        return true;
    }

    /*
     * Post-pack on-disk `F_NOCACHE` `pread` NaN/Inf sweep over EVERY dense F32 & F16 slice in
     * `spine.strata` (`0` mmap pages touched -> `0` RSS increase, 100% disk verification even across resumes!).
     */
    bool verify_spine_slices_on_disk() {
        auto check_slice = [&](const StrataGlm53TensorSlice& sl) -> bool {
            if (sl.elem_count == 0 || sl.dtype == STRATA_GLM53_DTYPE_NONE) return true;
            const uint64_t bpe = (sl.dtype == STRATA_GLM53_DTYPE_F32) ? 4ULL : 2ULL;
            uint64_t rem = sl.elem_count;
            uint64_t off = sl.offset;
            const uint64_t max_n = in_buf.size() / bpe;
            while (rem > 0) {
                const uint64_t n = std::min<uint64_t>(rem, max_n);
                if (!pread_exact(spine_fd, in_buf.data(), static_cast<size_t>(n * bpe), off)) return false;
                if (sl.dtype == STRATA_GLM53_DTYPE_F32) {
                    if (!check_f32_span_finite(reinterpret_cast<const float*>(in_buf.data()),
                                               static_cast<size_t>(n))) return false;
                } else if (sl.dtype == STRATA_GLM53_DTYPE_F16) {
                    if (!check_f16_span_finite(reinterpret_cast<const uint16_t*>(in_buf.data()),
                                               static_cast<size_t>(n))) return false;
                }
                off += n * bpe;
                rem -= n;
            }
            return true;
        };

        auto check_layer = [&](const StrataGlm53LayerDesc& ld) -> bool {
            if (ld.has_mhc) {
                if (!check_slice(ld.hc_attn.fn) || !check_slice(ld.hc_attn.base) || !check_slice(ld.hc_attn.scale) ||
                    !check_slice(ld.hc_ffn.fn)  || !check_slice(ld.hc_ffn.base)  || !check_slice(ld.hc_ffn.scale)) {
                    return false;
                }
            }
            if (!check_slice(ld.input_layernorm) || !check_slice(ld.post_attention_layernorm)) return false;
            if (ld.attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) {
                if (!check_slice(ld.lin_qkv_conv1d) || !check_slice(ld.lin_A_log) ||
                    !check_slice(ld.lin_dt_bias)    || !check_slice(ld.lin_o_norm)) return false;
            } else {
                if (!check_slice(ld.mla_q_a_layernorm)     || !check_slice(ld.mla_kv_a_layernorm) ||
                    !check_slice(ld.mla_w_uk)              || !check_slice(ld.mla_w_uv)           ||
                    !check_slice(ld.indexer_k_norm_weight) || !check_slice(ld.indexer_k_norm_bias)||
                    !check_slice(ld.indexer_compress_ape)  || !check_slice(ld.indexer_compress_gate)) return false;
            }
            if (ld.mlp_type == STRATA_GLM53_MLP_SPARSE_MOE) {
                if (!check_slice(ld.moe_router_gate) || !check_slice(ld.moe_e_score_bias)) return false;
            }
            return true;
        };

        for (uint32_t l = 0; l < spine_hdr->cfg.num_hidden_layers; ++l) {
            if (!check_layer(spine_hdr->layers[l])) return false;
        }
        if (!check_slice(spine_hdr->final_norm)) return false;
        if (spine_hdr->mtp.enabled) {
            if (!check_slice(spine_hdr->mtp.enorm) ||
                !check_slice(spine_hdr->mtp.hnorm) ||
                !check_layer(spine_hdr->mtp.layer) ||
                !check_slice(spine_hdr->mtp.shared_head_norm)) return false;
        }
        return !nan_inf_detected.load(std::memory_order_relaxed);
    }
};

/* ============================================================================
 * 6. Official `GLM-5.3-Flash` Byte-Level BPE Tokenizer (`154,880` Vocab)
 *
 * Exact `tokenizer.json` Pre-Tokenizer Split Regex:
 *   `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
 * + GPT-2 `ByteLevel` mapping + `model.ignore_merges = true` + atomic `added_tokens` (`154820..154855`).
 * ============================================================================ */

/* Non-ASCII Unicode General Category L* (620 exact intervals from Unicode UCD) */
static const uint32_t kUnicodeLetterRanges[][2] = {
    {0x00AAu, 0x00AAu}, {0x00B5u, 0x00B5u}, {0x00BAu, 0x00BAu}, {0x00C0u, 0x00D6u}, {0x00D8u, 0x00F6u}, {0x00F8u, 0x02C1u},
    {0x02C6u, 0x02D1u}, {0x02E0u, 0x02E4u}, {0x02ECu, 0x02ECu}, {0x02EEu, 0x02EEu}, {0x0370u, 0x0374u}, {0x0376u, 0x0377u},
    {0x037Au, 0x037Du}, {0x037Fu, 0x037Fu}, {0x0386u, 0x0386u}, {0x0388u, 0x038Au}, {0x038Cu, 0x038Cu}, {0x038Eu, 0x03A1u},
    {0x03A3u, 0x03F5u}, {0x03F7u, 0x0481u}, {0x048Au, 0x052Fu}, {0x0531u, 0x0556u}, {0x0559u, 0x0559u}, {0x0560u, 0x0588u},
    {0x05D0u, 0x05EAu}, {0x05EFu, 0x05F2u}, {0x0620u, 0x064Au}, {0x066Eu, 0x066Fu}, {0x0671u, 0x06D3u}, {0x06D5u, 0x06D5u},
    {0x06E5u, 0x06E6u}, {0x06EEu, 0x06EFu}, {0x06FAu, 0x06FCu}, {0x06FFu, 0x06FFu}, {0x0710u, 0x0710u}, {0x0712u, 0x072Fu},
    {0x074Du, 0x07A5u}, {0x07B1u, 0x07B1u}, {0x07CAu, 0x07EAu}, {0x07F4u, 0x07F5u}, {0x07FAu, 0x07FAu}, {0x0800u, 0x0815u},
    {0x081Au, 0x081Au}, {0x0824u, 0x0824u}, {0x0828u, 0x0828u}, {0x0840u, 0x0858u}, {0x0860u, 0x086Au}, {0x08A0u, 0x08B4u},
    {0x08B6u, 0x08C7u}, {0x0904u, 0x0939u}, {0x093Du, 0x093Du}, {0x0950u, 0x0950u}, {0x0958u, 0x0961u}, {0x0971u, 0x0980u},
    {0x0985u, 0x098Cu}, {0x098Fu, 0x0990u}, {0x0993u, 0x09A8u}, {0x09AAu, 0x09B0u}, {0x09B2u, 0x09B2u}, {0x09B6u, 0x09B9u},
    {0x09BDu, 0x09BDu}, {0x09CEu, 0x09CEu}, {0x09DCu, 0x09DDu}, {0x09DFu, 0x09E1u}, {0x09F0u, 0x09F1u}, {0x09FCu, 0x09FCu},
    {0x0A05u, 0x0A0Au}, {0x0A0Fu, 0x0A10u}, {0x0A13u, 0x0A28u}, {0x0A2Au, 0x0A30u}, {0x0A32u, 0x0A33u}, {0x0A35u, 0x0A36u},
    {0x0A38u, 0x0A39u}, {0x0A59u, 0x0A5Cu}, {0x0A5Eu, 0x0A5Eu}, {0x0A72u, 0x0A74u}, {0x0A85u, 0x0A8Du}, {0x0A8Fu, 0x0A91u},
    {0x0A93u, 0x0AA8u}, {0x0AAAu, 0x0AB0u}, {0x0AB2u, 0x0AB3u}, {0x0AB5u, 0x0AB9u}, {0x0ABDu, 0x0ABDu}, {0x0AD0u, 0x0AD0u},
    {0x0AE0u, 0x0AE1u}, {0x0AF9u, 0x0AF9u}, {0x0B05u, 0x0B0Cu}, {0x0B0Fu, 0x0B10u}, {0x0B13u, 0x0B28u}, {0x0B2Au, 0x0B30u},
    {0x0B32u, 0x0B33u}, {0x0B35u, 0x0B39u}, {0x0B3Du, 0x0B3Du}, {0x0B5Cu, 0x0B5Du}, {0x0B5Fu, 0x0B61u}, {0x0B71u, 0x0B71u},
    {0x0B83u, 0x0B83u}, {0x0B85u, 0x0B8Au}, {0x0B8Eu, 0x0B90u}, {0x0B92u, 0x0B95u}, {0x0B99u, 0x0B9Au}, {0x0B9Cu, 0x0B9Cu},
    {0x0B9Eu, 0x0B9Fu}, {0x0BA3u, 0x0BA4u}, {0x0BA8u, 0x0BAAu}, {0x0BAEu, 0x0BB9u}, {0x0BD0u, 0x0BD0u}, {0x0C05u, 0x0C0Cu},
    {0x0C0Eu, 0x0C10u}, {0x0C12u, 0x0C28u}, {0x0C2Au, 0x0C39u}, {0x0C3Du, 0x0C3Du}, {0x0C58u, 0x0C5Au}, {0x0C60u, 0x0C61u},
    {0x0C80u, 0x0C80u}, {0x0C85u, 0x0C8Cu}, {0x0C8Eu, 0x0C90u}, {0x0C92u, 0x0CA8u}, {0x0CAAu, 0x0CB3u}, {0x0CB5u, 0x0CB9u},
    {0x0CBDu, 0x0CBDu}, {0x0CDEu, 0x0CDEu}, {0x0CE0u, 0x0CE1u}, {0x0CF1u, 0x0CF2u}, {0x0D04u, 0x0D0Cu}, {0x0D0Eu, 0x0D10u},
    {0x0D12u, 0x0D3Au}, {0x0D3Du, 0x0D3Du}, {0x0D4Eu, 0x0D4Eu}, {0x0D54u, 0x0D56u}, {0x0D5Fu, 0x0D61u}, {0x0D7Au, 0x0D7Fu},
    {0x0D85u, 0x0D96u}, {0x0D9Au, 0x0DB1u}, {0x0DB3u, 0x0DBBu}, {0x0DBDu, 0x0DBDu}, {0x0DC0u, 0x0DC6u}, {0x0E01u, 0x0E30u},
    {0x0E32u, 0x0E33u}, {0x0E40u, 0x0E46u}, {0x0E81u, 0x0E82u}, {0x0E84u, 0x0E84u}, {0x0E86u, 0x0E8Au}, {0x0E8Cu, 0x0EA3u},
    {0x0EA5u, 0x0EA5u}, {0x0EA7u, 0x0EB0u}, {0x0EB2u, 0x0EB3u}, {0x0EBDu, 0x0EBDu}, {0x0EC0u, 0x0EC4u}, {0x0EC6u, 0x0EC6u},
    {0x0EDCu, 0x0EDFu}, {0x0F00u, 0x0F00u}, {0x0F40u, 0x0F47u}, {0x0F49u, 0x0F6Cu}, {0x0F88u, 0x0F8Cu}, {0x1000u, 0x102Au},
    {0x103Fu, 0x103Fu}, {0x1050u, 0x1055u}, {0x105Au, 0x105Du}, {0x1061u, 0x1061u}, {0x1065u, 0x1066u}, {0x106Eu, 0x1070u},
    {0x1075u, 0x1081u}, {0x108Eu, 0x108Eu}, {0x10A0u, 0x10C5u}, {0x10C7u, 0x10C7u}, {0x10CDu, 0x10CDu}, {0x10D0u, 0x10FAu},
    {0x10FCu, 0x1248u}, {0x124Au, 0x124Du}, {0x1250u, 0x1256u}, {0x1258u, 0x1258u}, {0x125Au, 0x125Du}, {0x1260u, 0x1288u},
    {0x128Au, 0x128Du}, {0x1290u, 0x12B0u}, {0x12B2u, 0x12B5u}, {0x12B8u, 0x12BEu}, {0x12C0u, 0x12C0u}, {0x12C2u, 0x12C5u},
    {0x12C8u, 0x12D6u}, {0x12D8u, 0x1310u}, {0x1312u, 0x1315u}, {0x1318u, 0x135Au}, {0x1380u, 0x138Fu}, {0x13A0u, 0x13F5u},
    {0x13F8u, 0x13FDu}, {0x1401u, 0x166Cu}, {0x166Fu, 0x167Fu}, {0x1681u, 0x169Au}, {0x16A0u, 0x16EAu}, {0x16F1u, 0x16F8u},
    {0x1700u, 0x170Cu}, {0x170Eu, 0x1711u}, {0x1720u, 0x1731u}, {0x1740u, 0x1751u}, {0x1760u, 0x176Cu}, {0x176Eu, 0x1770u},
    {0x1780u, 0x17B3u}, {0x17D7u, 0x17D7u}, {0x17DCu, 0x17DCu}, {0x1820u, 0x1878u}, {0x1880u, 0x1884u}, {0x1887u, 0x18A8u},
    {0x18AAu, 0x18AAu}, {0x18B0u, 0x18F5u}, {0x1900u, 0x191Eu}, {0x1950u, 0x196Du}, {0x1970u, 0x1974u}, {0x1980u, 0x19ABu},
    {0x19B0u, 0x19C9u}, {0x1A00u, 0x1A16u}, {0x1A20u, 0x1A54u}, {0x1AA7u, 0x1AA7u}, {0x1B05u, 0x1B33u}, {0x1B45u, 0x1B4Bu},
    {0x1B83u, 0x1BA0u}, {0x1BAEu, 0x1BAFu}, {0x1BBAu, 0x1BE5u}, {0x1C00u, 0x1C23u}, {0x1C4Du, 0x1C4Fu}, {0x1C5Au, 0x1C7Du},
    {0x1C80u, 0x1C88u}, {0x1C90u, 0x1CBAu}, {0x1CBDu, 0x1CBFu}, {0x1CE9u, 0x1CECu}, {0x1CEEu, 0x1CF3u}, {0x1CF5u, 0x1CF6u},
    {0x1CFAu, 0x1CFAu}, {0x1D00u, 0x1DBFu}, {0x1E00u, 0x1F15u}, {0x1F18u, 0x1F1Du}, {0x1F20u, 0x1F45u}, {0x1F48u, 0x1F4Du},
    {0x1F50u, 0x1F57u}, {0x1F59u, 0x1F59u}, {0x1F5Bu, 0x1F5Bu}, {0x1F5Du, 0x1F5Du}, {0x1F5Fu, 0x1F7Du}, {0x1F80u, 0x1FB4u},
    {0x1FB6u, 0x1FBCu}, {0x1FBEu, 0x1FBEu}, {0x1FC2u, 0x1FC4u}, {0x1FC6u, 0x1FCCu}, {0x1FD0u, 0x1FD3u}, {0x1FD6u, 0x1FDBu},
    {0x1FE0u, 0x1FECu}, {0x1FF2u, 0x1FF4u}, {0x1FF6u, 0x1FFCu}, {0x2071u, 0x2071u}, {0x207Fu, 0x207Fu}, {0x2090u, 0x209Cu},
    {0x2102u, 0x2102u}, {0x2107u, 0x2107u}, {0x210Au, 0x2113u}, {0x2115u, 0x2115u}, {0x2119u, 0x211Du}, {0x2124u, 0x2124u},
    {0x2126u, 0x2126u}, {0x2128u, 0x2128u}, {0x212Au, 0x212Du}, {0x212Fu, 0x2139u}, {0x213Cu, 0x213Fu}, {0x2145u, 0x2149u},
    {0x214Eu, 0x214Eu}, {0x2183u, 0x2184u}, {0x2C00u, 0x2C2Eu}, {0x2C30u, 0x2C5Eu}, {0x2C60u, 0x2CE4u}, {0x2CEBu, 0x2CEEu},
    {0x2CF2u, 0x2CF3u}, {0x2D00u, 0x2D25u}, {0x2D27u, 0x2D27u}, {0x2D2Du, 0x2D2Du}, {0x2D30u, 0x2D67u}, {0x2D6Fu, 0x2D6Fu},
    {0x2D80u, 0x2D96u}, {0x2DA0u, 0x2DA6u}, {0x2DA8u, 0x2DAEu}, {0x2DB0u, 0x2DB6u}, {0x2DB8u, 0x2DBEu}, {0x2DC0u, 0x2DC6u},
    {0x2DC8u, 0x2DCEu}, {0x2DD0u, 0x2DD6u}, {0x2DD8u, 0x2DDEu}, {0x2E2Fu, 0x2E2Fu}, {0x3005u, 0x3006u}, {0x3031u, 0x3035u},
    {0x303Bu, 0x303Cu}, {0x3041u, 0x3096u}, {0x309Du, 0x309Fu}, {0x30A1u, 0x30FAu}, {0x30FCu, 0x30FFu}, {0x3105u, 0x312Fu},
    {0x3131u, 0x318Eu}, {0x31A0u, 0x31BFu}, {0x31F0u, 0x31FFu}, {0x3400u, 0x4DBFu}, {0x4E00u, 0x9FFCu}, {0xA000u, 0xA48Cu},
    {0xA4D0u, 0xA4FDu}, {0xA500u, 0xA60Cu}, {0xA610u, 0xA61Fu}, {0xA62Au, 0xA62Bu}, {0xA640u, 0xA66Eu}, {0xA67Fu, 0xA69Du},
    {0xA6A0u, 0xA6E5u}, {0xA717u, 0xA71Fu}, {0xA722u, 0xA788u}, {0xA78Bu, 0xA7BFu}, {0xA7C2u, 0xA7CAu}, {0xA7F5u, 0xA801u},
    {0xA803u, 0xA805u}, {0xA807u, 0xA80Au}, {0xA80Cu, 0xA822u}, {0xA840u, 0xA873u}, {0xA882u, 0xA8B3u}, {0xA8F2u, 0xA8F7u},
    {0xA8FBu, 0xA8FBu}, {0xA8FDu, 0xA8FEu}, {0xA90Au, 0xA925u}, {0xA930u, 0xA946u}, {0xA960u, 0xA97Cu}, {0xA984u, 0xA9B2u},
    {0xA9CFu, 0xA9CFu}, {0xA9E0u, 0xA9E4u}, {0xA9E6u, 0xA9EFu}, {0xA9FAu, 0xA9FEu}, {0xAA00u, 0xAA28u}, {0xAA40u, 0xAA42u},
    {0xAA44u, 0xAA4Bu}, {0xAA60u, 0xAA76u}, {0xAA7Au, 0xAA7Au}, {0xAA7Eu, 0xAAAFu}, {0xAAB1u, 0xAAB1u}, {0xAAB5u, 0xAAB6u},
    {0xAAB9u, 0xAABDu}, {0xAAC0u, 0xAAC0u}, {0xAAC2u, 0xAAC2u}, {0xAADBu, 0xAADDu}, {0xAAE0u, 0xAAEAu}, {0xAAF2u, 0xAAF4u},
    {0xAB01u, 0xAB06u}, {0xAB09u, 0xAB0Eu}, {0xAB11u, 0xAB16u}, {0xAB20u, 0xAB26u}, {0xAB28u, 0xAB2Eu}, {0xAB30u, 0xAB5Au},
    {0xAB5Cu, 0xAB69u}, {0xAB70u, 0xABE2u}, {0xAC00u, 0xD7A3u}, {0xD7B0u, 0xD7C6u}, {0xD7CBu, 0xD7FBu}, {0xF900u, 0xFA6Du},
    {0xFA70u, 0xFAD9u}, {0xFB00u, 0xFB06u}, {0xFB13u, 0xFB17u}, {0xFB1Du, 0xFB1Du}, {0xFB1Fu, 0xFB28u}, {0xFB2Au, 0xFB36u},
    {0xFB38u, 0xFB3Cu}, {0xFB3Eu, 0xFB3Eu}, {0xFB40u, 0xFB41u}, {0xFB43u, 0xFB44u}, {0xFB46u, 0xFBB1u}, {0xFBD3u, 0xFD3Du},
    {0xFD50u, 0xFD8Fu}, {0xFD92u, 0xFDC7u}, {0xFDF0u, 0xFDFBu}, {0xFE70u, 0xFE74u}, {0xFE76u, 0xFEFCu}, {0xFF21u, 0xFF3Au},
    {0xFF41u, 0xFF5Au}, {0xFF66u, 0xFFBEu}, {0xFFC2u, 0xFFC7u}, {0xFFCAu, 0xFFCFu}, {0xFFD2u, 0xFFD7u}, {0xFFDAu, 0xFFDCu},
    {0x10000u, 0x1000Bu}, {0x1000Du, 0x10026u}, {0x10028u, 0x1003Au}, {0x1003Cu, 0x1003Du}, {0x1003Fu, 0x1004Du}, {0x10050u, 0x1005Du},
    {0x10080u, 0x100FAu}, {0x10280u, 0x1029Cu}, {0x102A0u, 0x102D0u}, {0x10300u, 0x1031Fu}, {0x1032Du, 0x10340u}, {0x10342u, 0x10349u},
    {0x10350u, 0x10375u}, {0x10380u, 0x1039Du}, {0x103A0u, 0x103C3u}, {0x103C8u, 0x103CFu}, {0x10400u, 0x1049Du}, {0x104B0u, 0x104D3u},
    {0x104D8u, 0x104FBu}, {0x10500u, 0x10527u}, {0x10530u, 0x10563u}, {0x10600u, 0x10736u}, {0x10740u, 0x10755u}, {0x10760u, 0x10767u},
    {0x10800u, 0x10805u}, {0x10808u, 0x10808u}, {0x1080Au, 0x10835u}, {0x10837u, 0x10838u}, {0x1083Cu, 0x1083Cu}, {0x1083Fu, 0x10855u},
    {0x10860u, 0x10876u}, {0x10880u, 0x1089Eu}, {0x108E0u, 0x108F2u}, {0x108F4u, 0x108F5u}, {0x10900u, 0x10915u}, {0x10920u, 0x10939u},
    {0x10980u, 0x109B7u}, {0x109BEu, 0x109BFu}, {0x10A00u, 0x10A00u}, {0x10A10u, 0x10A13u}, {0x10A15u, 0x10A17u}, {0x10A19u, 0x10A35u},
    {0x10A60u, 0x10A7Cu}, {0x10A80u, 0x10A9Cu}, {0x10AC0u, 0x10AC7u}, {0x10AC9u, 0x10AE4u}, {0x10B00u, 0x10B35u}, {0x10B40u, 0x10B55u},
    {0x10B60u, 0x10B72u}, {0x10B80u, 0x10B91u}, {0x10C00u, 0x10C48u}, {0x10C80u, 0x10CB2u}, {0x10CC0u, 0x10CF2u}, {0x10D00u, 0x10D23u},
    {0x10E80u, 0x10EA9u}, {0x10EB0u, 0x10EB1u}, {0x10F00u, 0x10F1Cu}, {0x10F27u, 0x10F27u}, {0x10F30u, 0x10F45u}, {0x10FB0u, 0x10FC4u},
    {0x10FE0u, 0x10FF6u}, {0x11003u, 0x11037u}, {0x11083u, 0x110AFu}, {0x110D0u, 0x110E8u}, {0x11103u, 0x11126u}, {0x11144u, 0x11144u},
    {0x11147u, 0x11147u}, {0x11150u, 0x11172u}, {0x11176u, 0x11176u}, {0x11183u, 0x111B2u}, {0x111C1u, 0x111C4u}, {0x111DAu, 0x111DAu},
    {0x111DCu, 0x111DCu}, {0x11200u, 0x11211u}, {0x11213u, 0x1122Bu}, {0x11280u, 0x11286u}, {0x11288u, 0x11288u}, {0x1128Au, 0x1128Du},
    {0x1128Fu, 0x1129Du}, {0x1129Fu, 0x112A8u}, {0x112B0u, 0x112DEu}, {0x11305u, 0x1130Cu}, {0x1130Fu, 0x11310u}, {0x11313u, 0x11328u},
    {0x1132Au, 0x11330u}, {0x11332u, 0x11333u}, {0x11335u, 0x11339u}, {0x1133Du, 0x1133Du}, {0x11350u, 0x11350u}, {0x1135Du, 0x11361u},
    {0x11400u, 0x11434u}, {0x11447u, 0x1144Au}, {0x1145Fu, 0x11461u}, {0x11480u, 0x114AFu}, {0x114C4u, 0x114C5u}, {0x114C7u, 0x114C7u},
    {0x11580u, 0x115AEu}, {0x115D8u, 0x115DBu}, {0x11600u, 0x1162Fu}, {0x11644u, 0x11644u}, {0x11680u, 0x116AAu}, {0x116B8u, 0x116B8u},
    {0x11700u, 0x1171Au}, {0x11800u, 0x1182Bu}, {0x118A0u, 0x118DFu}, {0x118FFu, 0x11906u}, {0x11909u, 0x11909u}, {0x1190Cu, 0x11913u},
    {0x11915u, 0x11916u}, {0x11918u, 0x1192Fu}, {0x1193Fu, 0x1193Fu}, {0x11941u, 0x11941u}, {0x119A0u, 0x119A7u}, {0x119AAu, 0x119D0u},
    {0x119E1u, 0x119E1u}, {0x119E3u, 0x119E3u}, {0x11A00u, 0x11A00u}, {0x11A0Bu, 0x11A32u}, {0x11A3Au, 0x11A3Au}, {0x11A50u, 0x11A50u},
    {0x11A5Cu, 0x11A89u}, {0x11A9Du, 0x11A9Du}, {0x11AC0u, 0x11AF8u}, {0x11C00u, 0x11C08u}, {0x11C0Au, 0x11C2Eu}, {0x11C40u, 0x11C40u},
    {0x11C72u, 0x11C8Fu}, {0x11D00u, 0x11D06u}, {0x11D08u, 0x11D09u}, {0x11D0Bu, 0x11D30u}, {0x11D46u, 0x11D46u}, {0x11D60u, 0x11D65u},
    {0x11D67u, 0x11D68u}, {0x11D6Au, 0x11D89u}, {0x11D98u, 0x11D98u}, {0x11EE0u, 0x11EF2u}, {0x11FB0u, 0x11FB0u}, {0x12000u, 0x12399u},
    {0x12480u, 0x12543u}, {0x13000u, 0x1342Eu}, {0x14400u, 0x14646u}, {0x16800u, 0x16A38u}, {0x16A40u, 0x16A5Eu}, {0x16AD0u, 0x16AEDu},
    {0x16B00u, 0x16B2Fu}, {0x16B40u, 0x16B43u}, {0x16B63u, 0x16B77u}, {0x16B7Du, 0x16B8Fu}, {0x16E40u, 0x16E7Fu}, {0x16F00u, 0x16F4Au},
    {0x16F50u, 0x16F50u}, {0x16F93u, 0x16F9Fu}, {0x16FE0u, 0x16FE1u}, {0x16FE3u, 0x16FE3u}, {0x17000u, 0x187F7u}, {0x18800u, 0x18CD5u},
    {0x18D00u, 0x18D08u}, {0x1B000u, 0x1B11Eu}, {0x1B150u, 0x1B152u}, {0x1B164u, 0x1B167u}, {0x1B170u, 0x1B2FBu}, {0x1BC00u, 0x1BC6Au},
    {0x1BC70u, 0x1BC7Cu}, {0x1BC80u, 0x1BC88u}, {0x1BC90u, 0x1BC99u}, {0x1D400u, 0x1D454u}, {0x1D456u, 0x1D49Cu}, {0x1D49Eu, 0x1D49Fu},
    {0x1D4A2u, 0x1D4A2u}, {0x1D4A5u, 0x1D4A6u}, {0x1D4A9u, 0x1D4ACu}, {0x1D4AEu, 0x1D4B9u}, {0x1D4BBu, 0x1D4BBu}, {0x1D4BDu, 0x1D4C3u},
    {0x1D4C5u, 0x1D505u}, {0x1D507u, 0x1D50Au}, {0x1D50Du, 0x1D514u}, {0x1D516u, 0x1D51Cu}, {0x1D51Eu, 0x1D539u}, {0x1D53Bu, 0x1D53Eu},
    {0x1D540u, 0x1D544u}, {0x1D546u, 0x1D546u}, {0x1D54Au, 0x1D550u}, {0x1D552u, 0x1D6A5u}, {0x1D6A8u, 0x1D6C0u}, {0x1D6C2u, 0x1D6DAu},
    {0x1D6DCu, 0x1D6FAu}, {0x1D6FCu, 0x1D714u}, {0x1D716u, 0x1D734u}, {0x1D736u, 0x1D74Eu}, {0x1D750u, 0x1D76Eu}, {0x1D770u, 0x1D788u},
    {0x1D78Au, 0x1D7A8u}, {0x1D7AAu, 0x1D7C2u}, {0x1D7C4u, 0x1D7CBu}, {0x1E100u, 0x1E12Cu}, {0x1E137u, 0x1E13Du}, {0x1E14Eu, 0x1E14Eu},
    {0x1E2C0u, 0x1E2EBu}, {0x1E800u, 0x1E8C4u}, {0x1E900u, 0x1E943u}, {0x1E94Bu, 0x1E94Bu}, {0x1EE00u, 0x1EE03u}, {0x1EE05u, 0x1EE1Fu},
    {0x1EE21u, 0x1EE22u}, {0x1EE24u, 0x1EE24u}, {0x1EE27u, 0x1EE27u}, {0x1EE29u, 0x1EE32u}, {0x1EE34u, 0x1EE37u}, {0x1EE39u, 0x1EE39u},
    {0x1EE3Bu, 0x1EE3Bu}, {0x1EE42u, 0x1EE42u}, {0x1EE47u, 0x1EE47u}, {0x1EE49u, 0x1EE49u}, {0x1EE4Bu, 0x1EE4Bu}, {0x1EE4Du, 0x1EE4Fu},
    {0x1EE51u, 0x1EE52u}, {0x1EE54u, 0x1EE54u}, {0x1EE57u, 0x1EE57u}, {0x1EE59u, 0x1EE59u}, {0x1EE5Bu, 0x1EE5Bu}, {0x1EE5Du, 0x1EE5Du},
    {0x1EE5Fu, 0x1EE5Fu}, {0x1EE61u, 0x1EE62u}, {0x1EE64u, 0x1EE64u}, {0x1EE67u, 0x1EE6Au}, {0x1EE6Cu, 0x1EE72u}, {0x1EE74u, 0x1EE77u},
    {0x1EE79u, 0x1EE7Cu}, {0x1EE7Eu, 0x1EE7Eu}, {0x1EE80u, 0x1EE89u}, {0x1EE8Bu, 0x1EE9Bu}, {0x1EEA1u, 0x1EEA3u}, {0x1EEA5u, 0x1EEA9u},
    {0x1EEABu, 0x1EEBBu}, {0x20000u, 0x2A6DDu}, {0x2A700u, 0x2B734u}, {0x2B740u, 0x2B81Du}, {0x2B820u, 0x2CEA1u}, {0x2CEB0u, 0x2EBE0u},
    {0x2F800u, 0x2FA1Du}, {0x30000u, 0x3134Au},
};

/* Non-ASCII Unicode General Category N* (132 exact intervals from Unicode UCD) */
static const uint32_t kUnicodeNumberRanges[][2] = {
    {0x00B2u, 0x00B3u}, {0x00B9u, 0x00B9u}, {0x00BCu, 0x00BEu}, {0x0660u, 0x0669u}, {0x06F0u, 0x06F9u}, {0x07C0u, 0x07C9u},
    {0x0966u, 0x096Fu}, {0x09E6u, 0x09EFu}, {0x09F4u, 0x09F9u}, {0x0A66u, 0x0A6Fu}, {0x0AE6u, 0x0AEFu}, {0x0B66u, 0x0B6Fu},
    {0x0B72u, 0x0B77u}, {0x0BE6u, 0x0BF2u}, {0x0C66u, 0x0C6Fu}, {0x0C78u, 0x0C7Eu}, {0x0CE6u, 0x0CEFu}, {0x0D58u, 0x0D5Eu},
    {0x0D66u, 0x0D78u}, {0x0DE6u, 0x0DEFu}, {0x0E50u, 0x0E59u}, {0x0ED0u, 0x0ED9u}, {0x0F20u, 0x0F33u}, {0x1040u, 0x1049u},
    {0x1090u, 0x1099u}, {0x1369u, 0x137Cu}, {0x16EEu, 0x16F0u}, {0x17E0u, 0x17E9u}, {0x17F0u, 0x17F9u}, {0x1810u, 0x1819u},
    {0x1946u, 0x194Fu}, {0x19D0u, 0x19DAu}, {0x1A80u, 0x1A89u}, {0x1A90u, 0x1A99u}, {0x1B50u, 0x1B59u}, {0x1BB0u, 0x1BB9u},
    {0x1C40u, 0x1C49u}, {0x1C50u, 0x1C59u}, {0x2070u, 0x2070u}, {0x2074u, 0x2079u}, {0x2080u, 0x2089u}, {0x2150u, 0x2182u},
    {0x2185u, 0x2189u}, {0x2460u, 0x249Bu}, {0x24EAu, 0x24FFu}, {0x2776u, 0x2793u}, {0x2CFDu, 0x2CFDu}, {0x3007u, 0x3007u},
    {0x3021u, 0x3029u}, {0x3038u, 0x303Au}, {0x3192u, 0x3195u}, {0x3220u, 0x3229u}, {0x3248u, 0x324Fu}, {0x3251u, 0x325Fu},
    {0x3280u, 0x3289u}, {0x32B1u, 0x32BFu}, {0xA620u, 0xA629u}, {0xA6E6u, 0xA6EFu}, {0xA830u, 0xA835u}, {0xA8D0u, 0xA8D9u},
    {0xA900u, 0xA909u}, {0xA9D0u, 0xA9D9u}, {0xA9F0u, 0xA9F9u}, {0xAA50u, 0xAA59u}, {0xABF0u, 0xABF9u}, {0xFF10u, 0xFF19u},
    {0x10107u, 0x10133u}, {0x10140u, 0x10178u}, {0x1018Au, 0x1018Bu}, {0x102E1u, 0x102FBu}, {0x10320u, 0x10323u}, {0x10341u, 0x10341u},
    {0x1034Au, 0x1034Au}, {0x103D1u, 0x103D5u}, {0x104A0u, 0x104A9u}, {0x10858u, 0x1085Fu}, {0x10879u, 0x1087Fu}, {0x108A7u, 0x108AFu},
    {0x108FBu, 0x108FFu}, {0x10916u, 0x1091Bu}, {0x109BCu, 0x109BDu}, {0x109C0u, 0x109CFu}, {0x109D2u, 0x109FFu}, {0x10A40u, 0x10A48u},
    {0x10A7Du, 0x10A7Eu}, {0x10A9Du, 0x10A9Fu}, {0x10AEBu, 0x10AEFu}, {0x10B58u, 0x10B5Fu}, {0x10B78u, 0x10B7Fu}, {0x10BA9u, 0x10BAFu},
    {0x10CFAu, 0x10CFFu}, {0x10D30u, 0x10D39u}, {0x10E60u, 0x10E7Eu}, {0x10F1Du, 0x10F26u}, {0x10F51u, 0x10F54u}, {0x10FC5u, 0x10FCBu},
    {0x11052u, 0x1106Fu}, {0x110F0u, 0x110F9u}, {0x11136u, 0x1113Fu}, {0x111D0u, 0x111D9u}, {0x111E1u, 0x111F4u}, {0x112F0u, 0x112F9u},
    {0x11450u, 0x11459u}, {0x114D0u, 0x114D9u}, {0x11650u, 0x11659u}, {0x116C0u, 0x116C9u}, {0x11730u, 0x1173Bu}, {0x118E0u, 0x118F2u},
    {0x11950u, 0x11959u}, {0x11C50u, 0x11C6Cu}, {0x11D50u, 0x11D59u}, {0x11DA0u, 0x11DA9u}, {0x11FC0u, 0x11FD4u}, {0x12400u, 0x1246Eu},
    {0x16A60u, 0x16A69u}, {0x16B50u, 0x16B59u}, {0x16B5Bu, 0x16B61u}, {0x16E80u, 0x16E96u}, {0x1D2E0u, 0x1D2F3u}, {0x1D360u, 0x1D378u},
    {0x1D7CEu, 0x1D7FFu}, {0x1E140u, 0x1E149u}, {0x1E2F0u, 0x1E2F9u}, {0x1E8C7u, 0x1E8CFu}, {0x1E950u, 0x1E959u}, {0x1EC71u, 0x1ECABu},
    {0x1ECADu, 0x1ECAFu}, {0x1ECB1u, 0x1ECB4u}, {0x1ED01u, 0x1ED2Du}, {0x1ED2Fu, 0x1ED3Du}, {0x1F100u, 0x1F10Cu}, {0x1FBF0u, 0x1FBF9u},
};

static inline bool in_range_table(uint32_t cp, const uint32_t table[][2], size_t count) {
    size_t lo = 0;
    size_t hi = count;
    while (lo < hi) {
        size_t mid = lo + ((hi - lo) >> 1);
        if (cp < table[mid][0]) {
            hi = mid;
        } else if (cp > table[mid][1]) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

static inline bool is_unicode_whitespace(uint32_t cp) {
    if (cp <= 0x20u) {
        return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f';
    }
    return cp == 0x0085u || cp == 0x00A0u || cp == 0x1680u ||
           (cp >= 0x2000u && cp <= 0x200Au) ||
           cp == 0x2028u || cp == 0x2029u || cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u;
}

static inline bool is_unicode_letter(uint32_t cp) {
    if (cp < 0x80u) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    return in_range_table(cp, kUnicodeLetterRanges,
                          sizeof(kUnicodeLetterRanges) / sizeof(kUnicodeLetterRanges[0]));
}

static inline bool is_unicode_number(uint32_t cp) {
    if (cp < 0x80u) return cp >= '0' && cp <= '9';
    return in_range_table(cp, kUnicodeNumberRanges,
                          sizeof(kUnicodeNumberRanges) / sizeof(kUnicodeNumberRanges[0]));
}

static inline char ascii_lower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return static_cast<char>(cp + ('a' - 'A'));
    return (cp < 0x80u) ? static_cast<char>(cp) : '\0';
}

struct CodepointSpan {
    uint32_t cp;
    uint32_t byte_off;
    uint32_t byte_len;
};

static void glm53_pretokenize_span(
    const char* text,
    size_t len,
    std::vector<std::pair<uint32_t, uint32_t>>& pieces_out
) {
    pieces_out.clear();
    if (len == 0) return;

    std::vector<CodepointSpan> cps;
    cps.reserve(len);
    size_t off = 0;
    const uint8_t* u = reinterpret_cast<const uint8_t*>(text);
    while (off < len) {
        size_t adv = 0;
        uint32_t cp = decode_utf8_one(u + off, len - off, &adv);
        cps.push_back({cp, static_cast<uint32_t>(off), static_cast<uint32_t>(adv)});
        off += adv;
    }

    const size_t N = cps.size();
    size_t i = 0;
    while (i < N) {
        const uint32_t c0 = cps[i].cp;

        /* 1. `(?i:'s|'t|'re|'ve|'m|'ll|'d)` */
        if (c0 == '\'' && i + 1 < N) {
            char c1 = ascii_lower(cps[i + 1].cp);
            if (i + 2 < N) {
                char c2 = ascii_lower(cps[i + 2].cp);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    uint32_t b0 = cps[i].byte_off;
                    uint32_t b1 = cps[i + 2].byte_off + cps[i + 2].byte_len;
                    pieces_out.emplace_back(b0, b1 - b0);
                    i += 3;
                    continue;
                }
            }
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                uint32_t b0 = cps[i].byte_off;
                uint32_t b1 = cps[i + 1].byte_off + cps[i + 1].byte_len;
                pieces_out.emplace_back(b0, b1 - b0);
                i += 2;
                continue;
            }
        }

        /* 2. `[^\r\n\p{L}\p{N}]?\p{L}+` */
        if (is_unicode_letter(c0)) {
            size_t j = i + 1;
            while (j < N && is_unicode_letter(cps[j].cp)) ++j;
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }
        if (c0 != '\r' && c0 != '\n' && !is_unicode_number(c0) &&
            i + 1 < N && is_unicode_letter(cps[i + 1].cp)) {
            size_t j = i + 2;
            while (j < N && is_unicode_letter(cps[j].cp)) ++j;
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }

        /* 3. `\p{N}{1,3}` (up to 3 consecutive digits per pre-token!) */
        if (is_unicode_number(c0)) {
            size_t j = i + 1;
            size_t cnt = 1;
            while (cnt < 3 && j < N && is_unicode_number(cps[j].cp)) {
                ++j;
                ++cnt;
            }
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }

        /* 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*` */
        auto is_punct_sym = [](uint32_t cp) {
            return !is_unicode_whitespace(cp) && !is_unicode_letter(cp) && !is_unicode_number(cp);
        };
        if (is_punct_sym(c0) || (c0 == ' ' && i + 1 < N && is_punct_sym(cps[i + 1].cp))) {
            size_t j = (c0 == ' ') ? (i + 2) : (i + 1);
            while (j < N && is_punct_sym(cps[j].cp)) ++j;
            while (j < N && (cps[j].cp == '\r' || cps[j].cp == '\n')) ++j;
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }

        /* 5/6/7. `\s*[\r\n]+ | \s+(?!\S) | \s+` */
        if (is_unicode_whitespace(c0)) {
            size_t ws_end = i + 1;
            ssize_t last_crlf = (c0 == '\r' || c0 == '\n') ? static_cast<ssize_t>(i) : -1;
            while (ws_end < N && is_unicode_whitespace(cps[ws_end].cp)) {
                if (cps[ws_end].cp == '\r' || cps[ws_end].cp == '\n') {
                    last_crlf = static_cast<ssize_t>(ws_end);
                }
                ++ws_end;
            }
            size_t match_end = ws_end;
            if (last_crlf >= 0) {
                match_end = static_cast<size_t>(last_crlf) + 1;
            } else if (ws_end < N && (ws_end - i) >= 2) {
                match_end = ws_end - 1;
            } else {
                match_end = ws_end;
            }
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[match_end - 1].byte_off + cps[match_end - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = match_end;
            continue;
        }

        pieces_out.emplace_back(cps[i].byte_off, cps[i].byte_len);
        ++i;
    }
}

struct MergeSlot {
    uint64_t pair_key;
    uint32_t rank;
    uint32_t merged_id;
};

struct AddedTokenEntry {
    std::string content;
    uint32_t    id;
    bool        special;
};

struct TransparentSvHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }
    size_t operator()(const std::string& s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
};

} /* anonymous namespace */

struct StrataGlm53Tokenizer {
    uint32_t                     vocab_size    = 0;
    bool                         ignore_merges = true;
    uint32_t                     byte_token_ids[256] = {};
    std::vector<std::string>     id_to_raw_bytes;
    std::vector<uint8_t>         id_is_special;
    std::unordered_map<std::string, uint32_t, TransparentSvHash, std::equal_to<>> raw_piece_to_id;
    std::vector<MergeSlot>       merge_ht;
    uint64_t                     merge_mask = 0;
    std::vector<AddedTokenEntry> added_sorted;
    std::vector<uint32_t>        added_by_b0[256];

    inline bool lookup_merge(uint32_t left_id, uint32_t right_id, uint32_t* rank_out, uint32_t* merged_out) const {
        uint64_t key = (static_cast<uint64_t>(left_id) << 32) | static_cast<uint64_t>(right_id);
        uint64_t h   = (key * 11400714819323198485ULL) >> 44;
        uint64_t idx = h & merge_mask;
        while (true) {
            const MergeSlot& s = merge_ht[idx];
            if (s.pair_key == key) {
                *rank_out   = s.rank;
                *merged_out = s.merged_id;
                return true;
            }
            if (s.pair_key == std::numeric_limits<uint64_t>::max()) {
                return false;
            }
            idx = (idx + 1ULL) & merge_mask;
        }
    }

    inline void insert_merge(uint32_t left_id, uint32_t right_id, uint32_t rank, uint32_t merged_id) {
        uint64_t key = (static_cast<uint64_t>(left_id) << 32) | static_cast<uint64_t>(right_id);
        uint64_t h   = (key * 11400714819323198485ULL) >> 44;
        uint64_t idx = h & merge_mask;
        while (true) {
            MergeSlot& s = merge_ht[idx];
            if (s.pair_key == std::numeric_limits<uint64_t>::max()) {
                s.pair_key  = key;
                s.rank      = rank;
                s.merged_id = merged_id;
                return;
            }
            if (s.pair_key == key) {
                if (rank < s.rank) {
                    s.rank      = rank;
                    s.merged_id = merged_id;
                }
                return;
            }
            idx = (idx + 1ULL) & merge_mask;
        }
    }
};

namespace {

static void build_gpt2_byte_unicode_tables(
    std::string b2u_utf8[256],
    int16_t u2b_cp[512]
) {
    for (int i = 0; i < 512; ++i) u2b_cp[i] = -1;
    bool direct[256] = {};
    for (int b = 33; b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;

    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra++);
        b2u_utf8[b].clear();
        append_utf8_codepoint(b2u_utf8[b], cp);
        u2b_cp[cp] = static_cast<int16_t>(b);
    }
}

static bool decode_bpe_vocab_key_to_bytes(
    const std::string& vocab_key,
    const int16_t u2b_cp[512],
    std::string& raw_out
) {
    raw_out.clear();
    raw_out.reserve(vocab_key.size());
    const uint8_t* u = reinterpret_cast<const uint8_t*>(vocab_key.data());
    size_t len = vocab_key.size();
    size_t off = 0;
    while (off < len) {
        size_t adv = 0;
        uint32_t cp = decode_utf8_one(u + off, len - off, &adv);
        if (cp >= 512u || u2b_cp[cp] < 0) return false;
        raw_out.push_back(static_cast<char>(static_cast<uint8_t>(u2b_cp[cp])));
        off += adv;
    }
    return true;
}

static void bpe_encode_piece(
    const StrataGlm53Tokenizer* tok,
    const uint8_t* bytes,
    uint32_t len,
    std::vector<uint32_t>& out_ids
) {
    if (len == 0) return;
    if (len == 1) {
        out_ids.push_back(tok->byte_token_ids[bytes[0]]);
        return;
    }

    /* Official HF `tokenizers` `ignore_merges = true` fast-path */
    if (tok->ignore_merges) {
        std::string_view sv(reinterpret_cast<const char*>(bytes), len);
        auto it = tok->raw_piece_to_id.find(sv);
        if (it != tok->raw_piece_to_id.end()) {
            out_ids.push_back(it->second);
            return;
        }
    }

    if (len <= 64) {
        uint32_t ids[64];
        uint32_t n = len;
        for (uint32_t i = 0; i < n; ++i) ids[i] = tok->byte_token_ids[bytes[i]];

        while (n >= 2) {
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();
            uint32_t best_merged = 0;
            uint32_t best_pos = 0;
            for (uint32_t i = 0; i + 1 < n; ++i) {
                uint32_t r = 0, m = 0;
                if (tok->lookup_merge(ids[i], ids[i + 1], &r, &m) && r < best_rank) {
                    best_rank   = r;
                    best_merged = m;
                    best_pos    = i;
                }
            }
            if (best_rank == std::numeric_limits<uint32_t>::max()) break;
            ids[best_pos] = best_merged;
            for (uint32_t j = best_pos + 1; j + 1 < n; ++j) {
                ids[j] = ids[j + 1];
            }
            --n;
        }
        for (uint32_t i = 0; i < n; ++i) out_ids.push_back(ids[i]);
        return;
    }

    struct Node {
        uint32_t id;
        int32_t  prev;
        int32_t  next;
        uint32_t ver;
        bool     alive;
    };
    struct HeapItem {
        uint32_t rank;
        int32_t  left;
        uint32_t left_ver;
        uint32_t right_ver;
        uint32_t merged_id;
        bool operator>(const HeapItem& o) const {
            return (rank != o.rank) ? (rank > o.rank) : (left > o.left);
        }
    };

    std::vector<Node> nodes(len);
    for (uint32_t i = 0; i < len; ++i) {
        nodes[i] = {
            tok->byte_token_ids[bytes[i]],
            static_cast<int32_t>(i) - 1,
            (i + 1 < len) ? static_cast<int32_t>(i + 1) : -1,
            0u,
            true
        };
    }

    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> pq;
    auto try_push = [&](int32_t l) {
        if (l < 0) return;
        int32_t r = nodes[l].next;
        if (r < 0) return;
        uint32_t rank = 0, merged = 0;
        if (tok->lookup_merge(nodes[l].id, nodes[r].id, &rank, &merged)) {
            pq.push({rank, l, nodes[l].ver, nodes[r].ver, merged});
        }
    };
    for (uint32_t i = 0; i + 1 < len; ++i) {
        try_push(static_cast<int32_t>(i));
    }

    while (!pq.empty()) {
        HeapItem top = pq.top();
        pq.pop();
        int32_t l = top.left;
        if (!nodes[l].alive || nodes[l].ver != top.left_ver) continue;
        int32_t r = nodes[l].next;
        if (r < 0 || !nodes[r].alive || nodes[r].ver != top.right_ver) continue;

        nodes[l].id = top.merged_id;
        ++nodes[l].ver;
        nodes[r].alive = false;
        int32_t rn = nodes[r].next;
        nodes[l].next = rn;
        if (rn >= 0) nodes[rn].prev = l;

        try_push(nodes[l].prev);
        try_push(l);
    }

    for (int32_t cur = 0; cur >= 0; cur = nodes[cur].next) {
        if (nodes[cur].alive) out_ids.push_back(nodes[cur].id);
    }
}

static bool parse_config_into(
    const std::string& cfg_path,
    const std::string& gen_cfg_path,
    StrataGlm53Config& cfg
) {
    strata_glm53_default_config(&cfg);
    std::string raw;
    if (!read_whole_file(cfg_path, raw)) return false;
    SmallJson root;
    if (!parse_small_json(raw, root) || root.type != SmallJson::J_OBJ) return false;

    const SmallJson* tc = root.get("text_config");
    if (!tc || tc->type != SmallJson::J_OBJ) tc = &root;

    cfg.num_hidden_layers        = static_cast<uint32_t>(tc->get_i64("num_hidden_layers", cfg.num_hidden_layers));
    cfg.num_nextn_predict_layers = static_cast<uint32_t>(tc->get_i64("num_nextn_predict_layers", cfg.num_nextn_predict_layers));
    cfg.first_k_dense_replace    = static_cast<uint32_t>(tc->get_i64("first_k_dense_replace", cfg.first_k_dense_replace));
    cfg.hidden_size              = static_cast<uint32_t>(tc->get_i64("hidden_size", cfg.hidden_size));
    cfg.vocab_size               = static_cast<uint32_t>(tc->get_i64("vocab_size", cfg.vocab_size));
    cfg.intermediate_size        = static_cast<uint32_t>(tc->get_i64("intermediate_size", cfg.intermediate_size));

    cfg.mhc_enabled              = tc->get_bool("mhc", cfg.mhc_enabled != 0u) ? 1u : 0u;
    cfg.hc_mult                  = static_cast<uint32_t>(tc->get_i64("hc_mult", cfg.hc_mult));
    cfg.hc_sinkhorn_iters        = static_cast<uint32_t>(tc->get_i64("hc_sinkhorn_iters", cfg.hc_sinkhorn_iters));
    cfg.hc_eps                   = static_cast<float>(tc->get_f64("hc_eps", cfg.hc_eps));

    if (const SmallJson* lac = tc->get("linear_attn_config")) {
        cfg.linear_num_heads        = static_cast<uint32_t>(lac->get_i64("num_heads", cfg.linear_num_heads));
        cfg.linear_head_dim         = static_cast<uint32_t>(lac->get_i64("head_dim", cfg.linear_head_dim));
        cfg.linear_conv_kernel_dim  = static_cast<uint32_t>(lac->get_i64("short_conv_kernel_size", cfg.linear_conv_kernel_dim));
        cfg.linear_gate_lower_bound = static_cast<float>(lac->get_f64("gate_lower_bound", cfg.linear_gate_lower_bound));
    }

    cfg.num_attention_heads      = static_cast<uint32_t>(tc->get_i64("num_attention_heads", cfg.num_attention_heads));
    cfg.num_key_value_heads      = static_cast<uint32_t>(tc->get_i64("num_key_value_heads", cfg.num_key_value_heads));
    cfg.q_lora_rank              = static_cast<uint32_t>(tc->get_i64("q_lora_rank", cfg.q_lora_rank));
    cfg.kv_lora_rank             = static_cast<uint32_t>(tc->get_i64("kv_lora_rank", cfg.kv_lora_rank));
    cfg.qk_nope_head_dim         = static_cast<uint32_t>(tc->get_i64("qk_nope_head_dim", cfg.qk_nope_head_dim));
    cfg.qk_rope_head_dim         = static_cast<uint32_t>(tc->get_i64("qk_rope_head_dim", cfg.qk_rope_head_dim));
    cfg.v_head_dim               = static_cast<uint32_t>(tc->get_i64("v_head_dim", cfg.v_head_dim));
    cfg.mla_use_nope             = tc->get_bool("mla_use_nope", cfg.mla_use_nope != 0u) ? 1u : 0u;
    cfg.index_n_heads            = static_cast<uint32_t>(tc->get_i64("index_n_heads", cfg.index_n_heads));
    cfg.index_head_dim           = static_cast<uint32_t>(tc->get_i64("index_head_dim", cfg.index_head_dim));
    cfg.index_kpool              = static_cast<uint32_t>(tc->get_i64("index_kpool", cfg.index_kpool));
    cfg.index_topk               = static_cast<uint32_t>(tc->get_i64("index_topk", cfg.index_topk));
    cfg.index_always_select_tail = tc->get_bool("index_kpool_always_select_tail", cfg.index_always_select_tail != 0u) ? 1u : 0u;

    cfg.n_routed_experts         = static_cast<uint32_t>(tc->get_i64("n_routed_experts", cfg.n_routed_experts));
    cfg.num_experts_per_tok      = static_cast<uint32_t>(tc->get_i64("num_experts_per_tok", cfg.num_experts_per_tok));
    cfg.n_shared_experts         = static_cast<uint32_t>(tc->get_i64("n_shared_experts", cfg.n_shared_experts));
    cfg.moe_intermediate_size    = static_cast<uint32_t>(tc->get_i64("moe_intermediate_size", cfg.moe_intermediate_size));
    cfg.n_group                  = static_cast<uint32_t>(tc->get_i64("n_group", cfg.n_group));
    cfg.topk_group               = static_cast<uint32_t>(tc->get_i64("topk_group", cfg.topk_group));
    cfg.norm_topk_prob           = tc->get_bool("norm_topk_prob", cfg.norm_topk_prob != 0u) ? 1u : 0u;
    cfg.routed_scaling_factor    = static_cast<float>(tc->get_f64("routed_scaling_factor", cfg.routed_scaling_factor));
    cfg.swiglu_limit             = static_cast<float>(tc->get_f64("swiglu_limit", cfg.swiglu_limit));

    cfg.rms_norm_eps             = static_cast<float>(tc->get_f64("rms_norm_eps", cfg.rms_norm_eps));
    const SmallJson* qc = root.get("quantization");
    if (!qc) qc = root.get("quantization_config");
    if (qc && qc->type == SmallJson::J_OBJ) {
        cfg.quant_group_size = static_cast<uint32_t>(qc->get_i64("group_size", cfg.quant_group_size));
        cfg.quant_bits       = static_cast<uint32_t>(qc->get_i64("bits", cfg.quant_bits));
    }

    cfg.tie_word_embeddings     = root.get_bool("tie_word_embeddings", tc->get_bool("tie_word_embeddings", false)) ? 1u : 0u;
    cfg.max_position_embeddings = static_cast<uint32_t>(tc->get_i64("max_position_embeddings", cfg.max_position_embeddings));
    cfg.pad_token_id            = static_cast<uint32_t>(tc->get_i64("pad_token_id", cfg.pad_token_id));

    auto parse_eos_field = [&](const SmallJson* eos) {
        if (!eos) return;
        if (eos->type == SmallJson::J_ARR && !eos->arr.empty()) {
            cfg.num_eos_token_ids = 0u;
            for (size_t i = 0; i < eos->arr.size() && cfg.num_eos_token_ids < 4u; ++i) {
                if (eos->arr[i].type == SmallJson::J_NUM) {
                    cfg.eos_token_ids[cfg.num_eos_token_ids++] = static_cast<uint32_t>(eos->arr[i].i_val);
                }
            }
        } else if (eos->type == SmallJson::J_NUM) {
            cfg.eos_token_ids[0]  = static_cast<uint32_t>(eos->i_val);
            cfg.num_eos_token_ids = 1u;
        }
    };
    parse_eos_field(tc->get("eos_token_id"));
    std::string gen_raw;
    if (read_whole_file(gen_cfg_path, gen_raw)) {
        SmallJson gen_root;
        if (parse_small_json(gen_raw, gen_root) && gen_root.type == SmallJson::J_OBJ) {
            parse_eos_field(gen_root.get("eos_token_id"));
        }
    }

    if (cfg.num_hidden_layers == 0 || cfg.num_hidden_layers > STRATA_GLM53_MAX_LAYERS) {
        return false;
    }

    const SmallJson* lt = tc->get("layer_types");
    const SmallJson* mt = tc->get("mlp_layer_types");
    cfg.num_kda_layers = 0u;
    cfg.num_dsa_layers = 0u;
    for (uint32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        if (lt && lt->type == SmallJson::J_ARR && l < lt->arr.size() && lt->arr[l].type == SmallJson::J_STR) {
            cfg.layer_attn_types[l] = (lt->arr[l].s_val == "deepseek_sparse_attention")
                                          ? STRATA_GLM53_ATTN_DSA_MLA
                                          : STRATA_GLM53_ATTN_KDA_LINEAR;
        } else {
            cfg.layer_attn_types[l] = ((l % 4u) == 3u) ? STRATA_GLM53_ATTN_DSA_MLA : STRATA_GLM53_ATTN_KDA_LINEAR;
        }
        if (mt && mt->type == SmallJson::J_ARR && l < mt->arr.size() && mt->arr[l].type == SmallJson::J_STR) {
            cfg.layer_mlp_types[l] = (mt->arr[l].s_val == "dense")
                                         ? STRATA_GLM53_MLP_DENSE
                                         : STRATA_GLM53_MLP_SPARSE_MOE;
        } else {
            cfg.layer_mlp_types[l] = (l < cfg.first_k_dense_replace) ? STRATA_GLM53_MLP_DENSE : STRATA_GLM53_MLP_SPARSE_MOE;
        }
        if (cfg.layer_attn_types[l] == STRATA_GLM53_ATTN_KDA_LINEAR) ++cfg.num_kda_layers;
        else                                                         ++cfg.num_dsa_layers;
    }

    return true;
}

static bool check_dir_has_no_files(const std::string& dir_path) {
    DIR* d = opendir(dir_path.c_str());
    if (!d) return true;
    bool empty = true;
    while (struct dirent* de = readdir(d)) {
        if (std::strcmp(de->d_name, ".") == 0 || std::strcmp(de->d_name, "..") == 0) continue;
        empty = false;
        break;
    }
    closedir(d);
    return empty;
}

static void clean_staging_part_files(const std::string& staging_dir) {
    DIR* d = opendir(staging_dir.c_str());
    if (!d) return;
    std::vector<std::string> to_rm;
    while (struct dirent* de = readdir(d)) {
        std::string name(de->d_name);
        if (name.size() >= 5 && name.substr(name.size() - 5) == ".part") {
            to_rm.push_back(staging_dir + "/" + name);
        }
    }
    closedir(d);
    for (const auto& p : to_rm) unlink(p.c_str());
}

/* ============================================================================
 * 6b. v2 RAM-Resident `RHT-64 + E8S-18-V4` Target Registry & FP8/BF16 Converter
 * ============================================================================ */

enum FastTargetKind {
    FAST_SPINE_F32      = 0,
    FAST_SPINE_F16      = 1,
    FAST_SPINE_Q4       = 2,
    FAST_MLA_KVB_BF16   = 3,
    FAST_MAIN_EXPERT_E8 = 4,
    FAST_MTP_EXPERT_Q4  = 5
};

struct FastTargetSpec {
    uint32_t               id              = 0;
    FastTargetKind         kind            = FAST_SPINE_F32;
    bool                   is_mtp          = false;
    int32_t                global_exp_idx  = -1;   /* 0..12383 for routed experts */
    uint8_t                exp_proj_slot   = 0u;   /* 0=gate, 1=up, 2=down */
    uint8_t                part_idx        = 0u;   /* 0=part0, 1=part1 for FAST_MAIN_EXPERT_E8 */
    uint32_t               mla_layer_slot  = 0u;   /* 0..44 or 45 (MTP) */
    uint64_t               e8_proj_off     = 0ULL; /* Byte offset inside part0/part1 for E8 sub-proj */
    uint32_t               e8_rows         = 0u;
    uint32_t               e8_cols         = 0u;
    uint64_t               tail_zero_off   = 0ULL;
    uint64_t               tail_zero_bytes = 0ULL;
    StrataGlm53QuantMatrix qm{};
    StrataGlm53TensorSlice slice{};
};

class FastTargetRegistry {
public:
    std::vector<FastTargetSpec>               specs;
    std::unordered_map<std::string, uint32_t> name_to_id;

    uint32_t add_spine_f32(const std::string& name, const StrataGlm53TensorSlice& sl, bool is_mtp) {
        FastTargetSpec s{};
        s.id     = static_cast<uint32_t>(specs.size());
        s.kind   = FAST_SPINE_F32;
        s.is_mtp = is_mtp;
        s.slice  = sl;
        specs.push_back(s);
        name_to_id[name] = s.id;
        return s.id;
    }

    uint32_t add_spine_f16(const std::string& name, const StrataGlm53TensorSlice& sl, bool is_mtp) {
        FastTargetSpec s{};
        s.id     = static_cast<uint32_t>(specs.size());
        s.kind   = FAST_SPINE_F16;
        s.is_mtp = is_mtp;
        s.slice  = sl;
        specs.push_back(s);
        name_to_id[name] = s.id;
        return s.id;
    }

    uint32_t add_spine_q4(const std::string& prefix, const StrataGlm53QuantMatrix& qm, bool is_mtp) {
        FastTargetSpec s{};
        s.id     = static_cast<uint32_t>(specs.size());
        s.kind   = FAST_SPINE_Q4;
        s.is_mtp = is_mtp;
        s.qm     = qm;
        specs.push_back(s);
        name_to_id[prefix + ".weight"] = s.id;
        return s.id;
    }

    uint32_t add_mla_kv_b(const std::string& prefix, uint32_t mla_slot, bool is_mtp) {
        FastTargetSpec s{};
        s.id             = static_cast<uint32_t>(specs.size());
        s.kind           = FAST_MLA_KVB_BF16;
        s.is_mtp         = is_mtp;
        s.mla_layer_slot = mla_slot;
        specs.push_back(s);
        name_to_id[prefix + ".weight"] = s.id;
        return s.id;
    }

    uint32_t add_main_expert_e8(
        const std::string& prefix,
        int32_t            global_exp_idx,
        uint8_t            proj_slot,
        uint8_t            part_idx,
        uint64_t           e8_proj_off,
        uint32_t           rows,
        uint32_t           cols,
        uint64_t           rec_tail_off = 0ULL,
        uint64_t           rec_tail_len = 0ULL
    ) {
        FastTargetSpec s{};
        s.id              = static_cast<uint32_t>(specs.size());
        s.kind            = FAST_MAIN_EXPERT_E8;
        s.is_mtp          = false;
        s.global_exp_idx  = global_exp_idx;
        s.exp_proj_slot   = proj_slot;
        s.part_idx        = part_idx;
        s.e8_proj_off     = e8_proj_off;
        s.e8_rows         = rows;
        s.e8_cols         = cols;
        s.tail_zero_off   = rec_tail_off;
        s.tail_zero_bytes = rec_tail_len;
        specs.push_back(s);
        name_to_id[prefix + ".weight"] = s.id;
        return s.id;
    }

    uint32_t add_mtp_expert_q4(
        const std::string&            prefix,
        int32_t                       global_exp_idx,
        uint8_t                       proj_slot,
        const StrataGlm53QuantMatrix& qm,
        uint64_t                      rec_tail_off = 0ULL,
        uint64_t                      rec_tail_len = 0ULL
    ) {
        FastTargetSpec s{};
        s.id              = static_cast<uint32_t>(specs.size());
        s.kind            = FAST_MTP_EXPERT_Q4;
        s.is_mtp          = true;
        s.global_exp_idx  = global_exp_idx;
        s.exp_proj_slot   = proj_slot;
        s.qm              = qm;
        s.tail_zero_off   = rec_tail_off;
        s.tail_zero_bytes = rec_tail_len;
        specs.push_back(s);
        name_to_id[prefix + ".weight"] = s.id;
        return s.id;
    }
};

static void register_fast_decoder_layer_targets(
    FastTargetRegistry&           reg,
    const std::string&            lp,
    const StrataGlm53SpineHeader& hdr,
    const StrataGlm53LayerDesc&   ld,
    uint32_t                      mla_slot,
    bool                          is_mtp
) {
    const StrataGlm53Config& cfg = hdr.cfg;

    if (ld.has_mhc) {
        reg.add_spine_f32(lp + "hc_attn_fn",    ld.hc_attn.fn,    is_mtp);
        reg.add_spine_f32(lp + "hc_attn_base",  ld.hc_attn.base,  is_mtp);
        reg.add_spine_f32(lp + "hc_attn_scale", ld.hc_attn.scale, is_mtp);
        reg.add_spine_f32(lp + "hc_ffn_fn",     ld.hc_ffn.fn,     is_mtp);
        reg.add_spine_f32(lp + "hc_ffn_base",   ld.hc_ffn.base,   is_mtp);
        reg.add_spine_f32(lp + "hc_ffn_scale",  ld.hc_ffn.scale,  is_mtp);
    }

    /* CRITICAL: Glm5NextTextRMSNorm is direct T5LayerNorm (NO +1.0f!) */
    reg.add_spine_f32(lp + "input_layernorm.weight",          ld.input_layernorm,          is_mtp);
    reg.add_spine_f32(lp + "post_attention_layernorm.weight", ld.post_attention_layernorm, is_mtp);

    const std::string ap = lp + "self_attn.";
    if (ld.attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) {
        const uint32_t nh      = cfg.linear_num_heads;
        const uint32_t hd      = cfg.linear_head_dim;
        const uint32_t qkv_dim = nh * hd;
        const uint32_t k_dim   = cfg.linear_conv_kernel_dim;

        reg.add_spine_q4(ap + "q_proj", strata_glm53_q4_row_subslice(&ld.lin_qkv_proj, 0u,           qkv_dim), is_mtp);
        reg.add_spine_q4(ap + "k_proj", strata_glm53_q4_row_subslice(&ld.lin_qkv_proj, qkv_dim,      qkv_dim), is_mtp);
        reg.add_spine_q4(ap + "v_proj", strata_glm53_q4_row_subslice(&ld.lin_qkv_proj, 2u * qkv_dim, qkv_dim), is_mtp);

        reg.add_spine_q4(ap + "f_a_proj", strata_glm53_q4_row_subslice(&ld.lin_fbg_a_proj, 0u,      hd), is_mtp);
        reg.add_spine_q4(ap + "b_proj",   strata_glm53_q4_row_subslice(&ld.lin_fbg_a_proj, hd,      nh), is_mtp);
        reg.add_spine_q4(ap + "g_a_proj", strata_glm53_q4_row_subslice(&ld.lin_fbg_a_proj, hd + nh, hd), is_mtp);

        reg.add_spine_q4(ap + "f_b_proj", ld.lin_f_b_proj, is_mtp);
        reg.add_spine_q4(ap + "g_b_proj", ld.lin_g_b_proj, is_mtp);
        reg.add_spine_q4(ap + "o_proj",   ld.lin_o_proj,   is_mtp);

        reg.add_spine_f32(ap + "q_conv1d.weight", make_conv1d_subslice(ld.lin_qkv_conv1d, 0u,           qkv_dim, k_dim), is_mtp);
        reg.add_spine_f32(ap + "k_conv1d.weight", make_conv1d_subslice(ld.lin_qkv_conv1d, qkv_dim,      qkv_dim, k_dim), is_mtp);
        reg.add_spine_f32(ap + "v_conv1d.weight", make_conv1d_subslice(ld.lin_qkv_conv1d, 2u * qkv_dim, qkv_dim, k_dim), is_mtp);

        reg.add_spine_f32(ap + "A_log",         ld.lin_A_log,   is_mtp);
        reg.add_spine_f32(ap + "dt_bias",       ld.lin_dt_bias, is_mtp);
        reg.add_spine_f32(ap + "o_norm.weight", ld.lin_o_norm,  is_mtp);
    } else {
        const uint32_t qr  = cfg.q_lora_rank;
        const uint32_t kvr = cfg.kv_lora_rank;

        reg.add_spine_q4(ap + "q_a_proj",           strata_glm53_q4_row_subslice(&ld.mla_qkv_a_proj, 0u, qr),  is_mtp);
        reg.add_spine_q4(ap + "kv_a_proj_with_mqa", strata_glm53_q4_row_subslice(&ld.mla_qkv_a_proj, qr, kvr), is_mtp);

        reg.add_spine_f32(ap + "q_a_layernorm.weight",  ld.mla_q_a_layernorm,  is_mtp);
        reg.add_spine_q4(ap + "q_b_proj",               ld.mla_q_b_proj,       is_mtp);
        reg.add_spine_f32(ap + "kv_a_layernorm.weight", ld.mla_kv_a_layernorm, is_mtp);
        reg.add_mla_kv_b(ap + "kv_b_proj",              mla_slot,              is_mtp);
        reg.add_spine_q4(ap + "o_proj",                 ld.mla_o_proj,         is_mtp);

        const std::string ip = ap + "indexer.";
        reg.add_spine_q4(ip + "wq_b",         ld.indexer_wq_b,         is_mtp);
        reg.add_spine_q4(ip + "wk",           ld.indexer_wk,           is_mtp);
        reg.add_spine_q4(ip + "weights_proj", ld.indexer_weights_proj, is_mtp);
        reg.add_spine_f32(ip + "k_norm.weight",             ld.indexer_k_norm_weight, is_mtp);
        reg.add_spine_f32(ip + "k_norm.bias",               ld.indexer_k_norm_bias,   is_mtp);
        reg.add_spine_f32(ip + "index_kpool_compress_ape",  ld.indexer_compress_ape,  is_mtp);
        reg.add_spine_f32(ip + "index_kpool_compress_gate", ld.indexer_compress_gate, is_mtp);
    }

    const std::string mp = lp + "mlp.";
    if (ld.mlp_type == STRATA_GLM53_MLP_DENSE) {
        const uint32_t d_ff = cfg.intermediate_size;
        reg.add_spine_q4(mp + "gate_proj", strata_glm53_q4_row_subslice(&ld.dense_gate_up_proj, 0u,   d_ff), is_mtp);
        reg.add_spine_q4(mp + "up_proj",   strata_glm53_q4_row_subslice(&ld.dense_gate_up_proj, d_ff, d_ff), is_mtp);
        reg.add_spine_q4(mp + "down_proj", ld.dense_down_proj, is_mtp);
    } else {
        const uint32_t d    = cfg.hidden_size;
        const uint32_t d_mo = cfg.moe_intermediate_size;
        const uint32_t d_sh = cfg.moe_intermediate_size * cfg.n_shared_experts;
        const uint32_t gs   = cfg.quant_group_size;

        reg.add_spine_f16(mp + "gate.weight",                  ld.moe_router_gate,  is_mtp);
        reg.add_spine_f32(mp + "gate.e_score_correction_bias", ld.moe_e_score_bias, is_mtp);

        reg.add_spine_q4(mp + "shared_experts.gate_proj", strata_glm53_q4_row_subslice(&ld.shared_gate_up_proj, 0u,   d_sh), is_mtp);
        reg.add_spine_q4(mp + "shared_experts.up_proj",   strata_glm53_q4_row_subslice(&ld.shared_gate_up_proj, d_sh, d_sh), is_mtp);
        reg.add_spine_q4(mp + "shared_experts.down_proj", ld.shared_down_proj, is_mtp);

        const uint32_t moe_l = static_cast<uint32_t>(ld.moe_layer_idx);
        if (is_mtp) {
            /* Layer-45 MTP experts -> Q4_A in `mtp_experts.strata` (index `e` from 0..287) */
            const uint64_t prj_b = hdr.expert_proj_bytes;
            const uint64_t w_b   = hdr.expert_weight_bytes;
            const uint64_t s_b   = hdr.expert_scales_bytes;
            const uint64_t rec_b = hdr.expert_record_bytes;
            const uint64_t used  = 3ULL * prj_b;
            const uint64_t tail  = (rec_b > used) ? (rec_b - used) : 0ULL;

            for (uint32_t e = 0; e < cfg.n_routed_experts; ++e) {
                const int32_t  g_exp = static_cast<int32_t>(moe_l * cfg.n_routed_experts + e);
                const uint64_t e_off = static_cast<uint64_t>(e) * rec_b;
                const std::string ep = mp + "experts." + std::to_string(e) + ".";

                reg.add_mtp_expert_q4(ep + "gate_proj", g_exp, 0u,
                                      make_expert_proj_qm(e_off + 0ULL * prj_b, w_b, s_b, d_mo, d, gs));
                reg.add_mtp_expert_q4(ep + "up_proj",   g_exp, 1u,
                                      make_expert_proj_qm(e_off + 1ULL * prj_b, w_b, s_b, d_mo, d, gs));
                reg.add_mtp_expert_q4(ep + "down_proj", g_exp, 2u,
                                      make_expert_proj_qm(e_off + 2ULL * prj_b, w_b, s_b, d, d_mo, gs),
                                      e_off + used, tail);
            }
        } else {
            /* Main MoE layers 0..41 -> RHT-64 + E8S-18-V4 in `part0` (0..20) / `part1` (21..41) */
            const uint64_t e8_prj_b = hdr.e8_proj_bytes;
            const uint64_t e8_rec_b = hdr.e8_record_bytes;
            const uint64_t used     = 3ULL * e8_prj_b;
            const uint64_t tail     = (e8_rec_b > used) ? (e8_rec_b - used) : 0ULL;

            for (uint32_t e = 0; e < cfg.n_routed_experts; ++e) {
                const int32_t g_exp = static_cast<int32_t>(moe_l * cfg.n_routed_experts + e);
                uint32_t part_idx = 0u;
                const uint64_t e_off = strata_glm53_v2_expert_offset(&hdr, moe_l, e, &part_idx);
                const std::string ep = mp + "experts." + std::to_string(e) + ".";

                reg.add_main_expert_e8(ep + "gate_proj", g_exp, 0u, static_cast<uint8_t>(part_idx),
                                       e_off + 0ULL * e8_prj_b, d_mo, d);
                reg.add_main_expert_e8(ep + "up_proj",   g_exp, 1u, static_cast<uint8_t>(part_idx),
                                       e_off + 1ULL * e8_prj_b, d_mo, d);
                reg.add_main_expert_e8(ep + "down_proj", g_exp, 2u, static_cast<uint8_t>(part_idx),
                                       e_off + 2ULL * e8_prj_b, d, d_mo, e_off + used, tail);
            }
        }
    }
}

static void build_fast_target_registry(
    const StrataGlm53SpineHeader& hdr,
    FastTargetRegistry&           reg
) {
    const StrataGlm53Config& cfg = hdr.cfg;
    reg.specs.reserve(40000);
    reg.name_to_id.reserve(65536);

    reg.add_spine_q4("model.embed_tokens", hdr.embed_tokens, false);
    for (uint32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        register_fast_decoder_layer_targets(
            reg, "model.layers." + std::to_string(l) + ".", hdr, hdr.layers[l], l, false);
    }
    reg.add_spine_f32("model.norm.weight", hdr.final_norm, false);
    if (!cfg.tie_word_embeddings) {
        reg.add_spine_q4("lm_head", hdr.lm_head, false);
    }

    if (hdr.mtp.enabled) {
        const uint32_t mtp_l = hdr.mtp.global_layer_idx;
        const std::string mp = "model.layers." + std::to_string(mtp_l) + ".";
        reg.add_spine_f32(mp + "enorm.weight",            hdr.mtp.enorm,            true);
        reg.add_spine_f32(mp + "hnorm.weight",            hdr.mtp.hnorm,            true);
        reg.add_spine_q4(mp + "eh_proj",                  hdr.mtp.eh_proj,          true);
        register_fast_decoder_layer_targets(reg, mp, hdr, hdr.mtp.layer, mtp_l, true);
        reg.add_spine_f32(mp + "shared_head.norm.weight", hdr.mtp.shared_head_norm, true);
    }
}

static bool is_scale_inv_name(const std::string& name, std::string* base_weight_out) {
    constexpr const char* kSuffix = ".weight_scale_inv";
    constexpr size_t      kSufLen = 17u;
    if (name.size() > kSufLen &&
        name.compare(name.size() - kSufLen, kSufLen, kSuffix) == 0) {
        if (base_weight_out) {
            *base_weight_out = name.substr(0, name.size() - kSufLen) + ".weight";
        }
        return true;
    }
    return false;
}

static uint64_t fast_st_dtype_bpe(SafetensorDtype dt) {
    switch (dt) {
        case ST_F8_E4M3: return 1ULL;
        case ST_BF16:
        case ST_F16:     return 2ULL;
        case ST_F32:
        case ST_U32:     return 4ULL;
        case ST_I64:     return 8ULL;
        default:         return 0ULL;
    }
}

struct FastShardConverter {
    int                             spine_fd   = -1;
    int                             part0_fd   = -1;
    int                             part1_fd   = -1;
    int                             mtp_fd     = -1;
    uint32_t                        ols_trials = 3u;
    const StrataGlm53SpineHeader*   spine_hdr  = nullptr;
    NeonWorkerPool*                 pool       = nullptr;
    std::atomic<bool>               nan_inf_detected{false};

    /* Precomputed L1/L2 lookup tables (`217 KiB` total) */
    float                           fp8_lut[256];
    std::unique_ptr<StrataGlm53E8QuantTables> e8_tbl;

    /* Reusable scratch arena (`~76 MiB`, zero per-tensor heap churn) */
    std::vector<uint8_t>            in_buf;     /* 32 MiB */
    std::vector<uint8_t>            out_w_buf;  /* 32 MiB */
    std::vector<uint8_t>            out_s_buf;  /* 4 MiB  */
    std::vector<uint8_t>            out_b_buf;  /* 4 MiB  */
    std::vector<uint8_t>            e8_out_buf; /* e8_proj_bytes (~2.375 MiB) */
    std::vector<float>              scale_inv_buf;

    /* Running E8S-18-V4 quality & speed telemetry */
    std::mutex                      stat_mu;
    double                          total_e8_sig_energy = 0.0;
    double                          total_e8_err_energy = 0.0;
    double                          total_e8_quant_sec  = 0.0;
    uint64_t                        total_e8_projs_done = 0ULL;

    FastShardConverter(
        int sfd, int p0fd, int p1fd, int mtpfd,
        uint32_t trials,
        const StrataGlm53SpineHeader* hdr,
        NeonWorkerPool* wp
    ) : spine_fd(sfd), part0_fd(p0fd), part1_fd(p1fd), mtp_fd(mtpfd),
        ols_trials(trials == 0u ? 3u : std::min<uint32_t>(trials, 3u)),
        spine_hdr(hdr), pool(wp),
        e8_tbl(std::make_unique<StrataGlm53E8QuantTables>()) {
        for (uint32_t u = 0; u < 256u; ++u) {
            fp8_lut[u] = sg53_fp8_e4m3fn_to_f32(static_cast<uint8_t>(u));
        }
        sg53_e8s18_build_quant_tables(e8_tbl.get());
        in_buf.resize(32ULL * 1024ULL * 1024ULL);
        out_w_buf.resize(32ULL * 1024ULL * 1024ULL);
        out_s_buf.resize(4ULL * 1024ULL * 1024ULL);
        out_b_buf.resize(4ULL * 1024ULL * 1024ULL);
        e8_out_buf.resize(std::max<uint64_t>(hdr->e8_proj_bytes, 4096ULL));
        scale_inv_buf.reserve(8192);
    }

    static uint64_t shape_elems(const std::vector<uint64_t>& shape) {
        if (shape.empty()) return 0;
        uint64_t n = 1;
        for (uint64_t d : shape) n *= d;
        return n;
    }

    bool load_scale_inv(
        int shard_fd,
        const SafetensorEntry* sc_ent,
        uint64_t rows,
        uint64_t cols,
        uint64_t* sc_cols_out
    ) {
        if (!sc_ent) return false;
        const uint64_t exp_sr = (rows + 127ULL) / 128ULL;
        const uint64_t exp_sc = (cols + 127ULL) / 128ULL;
        const uint64_t exp_n  = exp_sr * exp_sc;
        const uint64_t bpe    = fast_st_dtype_bpe(sc_ent->dtype);
        if (sc_ent->shape.size() != 2 || sc_ent->shape[0] != exp_sr || sc_ent->shape[1] != exp_sc ||
            bpe == 0 || sc_ent->byte_len != exp_n * bpe) {
            std::fprintf(stderr, "[glm53-fast] ERROR: scale_inv shape mismatch on '%s' (got [%llu,%llu], exp [%llu,%llu])\n",
                         sc_ent->canon_name.c_str(),
                         sc_ent->shape.size() > 0 ? static_cast<unsigned long long>(sc_ent->shape[0]) : 0ULL,
                         sc_ent->shape.size() > 1 ? static_cast<unsigned long long>(sc_ent->shape[1]) : 0ULL,
                         static_cast<unsigned long long>(exp_sr),
                         static_cast<unsigned long long>(exp_sc));
            return false;
        }
        scale_inv_buf.resize(static_cast<size_t>(exp_n));
        if (sc_ent->dtype == ST_F32) {
            if (!pread_exact(shard_fd, scale_inv_buf.data(), static_cast<size_t>(exp_n * sizeof(float)), sc_ent->abs_offset)) {
                return false;
            }
            if (!check_f32_span_finite(scale_inv_buf.data(), static_cast<size_t>(exp_n))) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                return false;
            }
        } else if (sc_ent->dtype == ST_BF16) {
            std::vector<uint16_t> tmp(static_cast<size_t>(exp_n));
            if (!pread_exact(shard_fd, tmp.data(), static_cast<size_t>(exp_n * 2ULL), sc_ent->abs_offset)) return false;
            if (!convert_bf16_to_f32_span_checked(tmp.data(), scale_inv_buf.data(), static_cast<size_t>(exp_n))) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                return false;
            }
        } else if (sc_ent->dtype == ST_F16) {
            std::vector<uint16_t> tmp(static_cast<size_t>(exp_n));
            if (!pread_exact(shard_fd, tmp.data(), static_cast<size_t>(exp_n * 2ULL), sc_ent->abs_offset)) return false;
            if (!check_f16_span_finite(tmp.data(), static_cast<size_t>(exp_n))) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                return false;
            }
            for (size_t i = 0; i < exp_n; ++i) scale_inv_buf[i] = fp16_to_f32_fast(tmp[i]);
        } else {
            return false;
        }
        *sc_cols_out = exp_sc;
        return true;
    }

    bool convert_spine_f32(int shard_fd, const SafetensorEntry& ent, const FastTargetSpec& spec) {
        const uint64_t elems = shape_elems(ent.shape);
        const uint64_t bpe   = fast_st_dtype_bpe(ent.dtype);
        if (elems != spec.slice.elem_count || bpe == 0 || ent.byte_len != elems * bpe) {
            std::fprintf(stderr, "[glm53-fast] ERROR: F32 shape mismatch on '%s' (got %llu, expected %u)\n",
                         ent.canon_name.c_str(), static_cast<unsigned long long>(elems), spec.slice.elem_count);
            return false;
        }
        uint64_t rem_elems = elems;
        uint64_t r_off     = ent.abs_offset;
        uint64_t w_off     = spec.slice.offset;
        const uint64_t max_batch = std::min<uint64_t>(in_buf.size() / bpe, out_w_buf.size() / sizeof(float));
        while (rem_elems > 0) {
            const uint64_t n = std::min<uint64_t>(rem_elems, max_batch);
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(n * bpe), r_off)) return false;
            float* dst = reinterpret_cast<float*>(out_w_buf.data());
            bool finite_ok = true;
            if (ent.dtype == ST_BF16) {
                finite_ok = convert_bf16_to_f32_span_checked(
                    reinterpret_cast<const uint16_t*>(in_buf.data()), dst, static_cast<size_t>(n));
            } else if (ent.dtype == ST_F16) {
                const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
                finite_ok = check_f16_span_finite(s16, static_cast<size_t>(n));
                for (uint64_t i = 0; i < n; ++i) dst[i] = fp16_to_f32_fast(s16[i]);
            } else if (ent.dtype == ST_F32) {
                const float* sf = reinterpret_cast<const float*>(in_buf.data());
                finite_ok = check_f32_span_finite(sf, static_cast<size_t>(n));
                std::memcpy(dst, sf, static_cast<size_t>(n * sizeof(float)));
            } else {
                return false;
            }
            if (!finite_ok) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                std::fprintf(stderr, "[glm53-fast] ERROR: NaN/Inf in '%s'\n", ent.canon_name.c_str());
                return false;
            }
            const uint64_t chunk_out = n * sizeof(float);
            if (!pwrite_exact(spine_fd, dst, static_cast<size_t>(chunk_out), w_off)) return false;
            r_off += n * bpe;
            w_off += chunk_out;
            rem_elems -= n;
        }
        const uint64_t written = elems * sizeof(float);
        if (spec.slice.bytes > written) {
            if (!pwrite_zeros(spine_fd, spec.slice.offset + written, spec.slice.bytes - written)) return false;
        }
        return true;
    }

    bool convert_spine_f16(int shard_fd, const SafetensorEntry& ent, const FastTargetSpec& spec) {
        const uint64_t elems = shape_elems(ent.shape);
        const uint64_t bpe   = fast_st_dtype_bpe(ent.dtype);
        if (elems != spec.slice.elem_count || bpe == 0 || ent.byte_len != elems * bpe) {
            std::fprintf(stderr, "[glm53-fast] ERROR: F16 shape mismatch on '%s' (got %llu, expected %u)\n",
                         ent.canon_name.c_str(), static_cast<unsigned long long>(elems), spec.slice.elem_count);
            return false;
        }
        uint64_t rem_elems = elems;
        uint64_t r_off     = ent.abs_offset;
        uint64_t w_off     = spec.slice.offset;
        const uint64_t max_batch = std::min<uint64_t>(in_buf.size() / bpe, out_w_buf.size() / sizeof(uint16_t));
        while (rem_elems > 0) {
            const uint64_t n = std::min<uint64_t>(rem_elems, max_batch);
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(n * bpe), r_off)) return false;
            uint16_t* dst16 = reinterpret_cast<uint16_t*>(out_w_buf.data());
            std::atomic<bool> batch_finite{true};
            if (ent.dtype == ST_BF16) {
                const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
                pool->parallel_for_rows(n, 32768ULL, [&](uint64_t i0, uint64_t i1) {
                    if (!convert_bf16_to_fp16_span_checked(s16 + i0, dst16 + i0, static_cast<size_t>(i1 - i0))) {
                        batch_finite.store(false, std::memory_order_relaxed);
                    }
                });
            } else if (ent.dtype == ST_F16) {
                const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
                if (!check_f16_span_finite(s16, static_cast<size_t>(n))) batch_finite.store(false);
                std::memcpy(dst16, s16, static_cast<size_t>(n * sizeof(uint16_t)));
            } else if (ent.dtype == ST_F32) {
                const float* sf = reinterpret_cast<const float*>(in_buf.data());
                pool->parallel_for_rows(n, 32768ULL, [&](uint64_t i0, uint64_t i1) {
                    if (!convert_f32_to_fp16_span_checked(sf + i0, dst16 + i0, static_cast<size_t>(i1 - i0))) {
                        batch_finite.store(false, std::memory_order_relaxed);
                    }
                });
            } else {
                return false;
            }
            if (!batch_finite.load(std::memory_order_relaxed)) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                return false;
            }
            const uint64_t out_chunk = n * sizeof(uint16_t);
            if (!pwrite_exact(spine_fd, dst16, static_cast<size_t>(out_chunk), w_off)) return false;
            r_off += n * bpe;
            w_off += out_chunk;
            rem_elems -= n;
        }
        const uint64_t written = elems * sizeof(uint16_t);
        if (spec.slice.bytes > written) {
            if (!pwrite_zeros(spine_fd, spec.slice.offset + written, spec.slice.bytes - written)) return false;
        }
        return true;
    }

    /* Lossless `BF16 -> FP16` unpack directly into `mla_w_uk` (transposed) & `mla_w_uv` (ZERO Q4 loss!) */
    bool convert_mla_kv_b_bf16(
        int shard_fd,
        const SafetensorEntry& ent,
        const StrataGlm53LayerDesc& ld
    ) {
        const StrataGlm53Config& cfg = spine_hdr->cfg;
        const uint32_t nh            = cfg.num_attention_heads; /* 64  */
        const uint32_t q_nope        = cfg.qk_nope_head_dim;    /* 256 */
        const uint32_t v_dim         = cfg.v_head_dim;          /* 256 */
        const uint32_t kvr           = cfg.kv_lora_rank;        /* 512 */
        const uint32_t rows_per_head = q_nope + v_dim;          /* 512 */
        const uint32_t total_rows    = nh * rows_per_head;      /* 32768 */
        const uint64_t bpe           = fast_st_dtype_bpe(ent.dtype);
        const uint64_t elems         = shape_elems(ent.shape);

        if ((ent.dtype != ST_BF16 && ent.dtype != ST_F16 && ent.dtype != ST_F32) ||
            elems != static_cast<uint64_t>(total_rows) * kvr || ent.byte_len != elems * bpe) {
            std::fprintf(stderr, "[glm53-fast] ERROR: kv_b_proj shape/dtype mismatch on '%s'\n",
                         ent.canon_name.c_str());
            return false;
        }

        const uint64_t in_head_b = static_cast<uint64_t>(rows_per_head) * kvr * bpe;
        const uint64_t uk_head_b = static_cast<uint64_t>(kvr) * q_nope * sizeof(uint16_t);
        const uint64_t uv_head_b = static_cast<uint64_t>(v_dim) * kvr * sizeof(uint16_t);
        const uint32_t max_bh = static_cast<uint32_t>(std::min<uint64_t>(
            nh, std::min<uint64_t>(in_buf.size() / in_head_b, out_w_buf.size() / (uk_head_b + uv_head_b))));
        if (max_bh == 0) return false;

        uint32_t h_done = 0;
        while (h_done < nh) {
            const uint32_t bh = std::min<uint32_t>(max_bh, nh - h_done);
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(bh * in_head_b),
                             ent.abs_offset + static_cast<uint64_t>(h_done) * in_head_b)) {
                return false;
            }
            uint8_t* uk_base = out_w_buf.data();
            uint8_t* uv_base = uk_base + bh * uk_head_b;
            std::atomic<bool> batch_finite{true};

            pool->parallel_for_rows(bh, 1ULL, [&](uint64_t i0, uint64_t i1) {
                for (uint64_t i = i0; i < i1; ++i) {
                    uint16_t* dst_uk = reinterpret_cast<uint16_t*>(uk_base + i * uk_head_b);
                    uint16_t* dst_uv = reinterpret_cast<uint16_t*>(uv_base + i * uv_head_b);
                    const uint8_t* src_h = in_buf.data() + i * in_head_b;
                    if (ent.dtype == ST_BF16) {
                        const uint16_t* sb = reinterpret_cast<const uint16_t*>(src_h);
                        const size_t n_h = static_cast<size_t>(rows_per_head) * kvr;
                        for (size_t j = 0; j < n_h; ++j) {
                            if (!bf16_is_finite(sb[j])) { batch_finite.store(false, std::memory_order_relaxed); break; }
                        }
                        sg53_unpack_kv_b_proj_head_bf16(sb, q_nope, v_dim, kvr, dst_uk, dst_uv);
                    } else if (ent.dtype == ST_F16) {
                        const uint16_t* sh = reinterpret_cast<const uint16_t*>(src_h);
                        if (!check_f16_span_finite(sh, static_cast<size_t>(rows_per_head) * kvr)) {
                            batch_finite.store(false, std::memory_order_relaxed);
                        }
                        for (uint32_t r = 0; r < q_nope; ++r) {
                            for (uint32_t c = 0; c < kvr; ++c) {
                                dst_uk[static_cast<size_t>(c) * q_nope + r] = sh[static_cast<size_t>(r) * kvr + c];
                            }
                        }
                        std::memcpy(dst_uv, sh + static_cast<size_t>(q_nope) * kvr, static_cast<size_t>(uv_head_b));
                    } else {
                        const float* sf = reinterpret_cast<const float*>(src_h);
                        if (!check_f32_span_finite(sf, static_cast<size_t>(rows_per_head) * kvr)) {
                            batch_finite.store(false, std::memory_order_relaxed);
                        }
                        for (uint32_t r = 0; r < q_nope; ++r) {
                            for (uint32_t c = 0; c < kvr; ++c) {
                                dst_uk[static_cast<size_t>(c) * q_nope + r] = f32_to_fp16_fast(sf[static_cast<size_t>(r) * kvr + c]);
                            }
                        }
                        const float* sv = sf + static_cast<size_t>(q_nope) * kvr;
                        for (size_t j = 0; j < static_cast<size_t>(v_dim) * kvr; ++j) {
                            dst_uv[j] = f32_to_fp16_fast(sv[j]);
                        }
                    }
                }
            });

            if (!batch_finite.load(std::memory_order_relaxed) ||
                !check_f16_span_finite(reinterpret_cast<const uint16_t*>(uk_base), static_cast<size_t>(bh * uk_head_b / 2ULL)) ||
                !check_f16_span_finite(reinterpret_cast<const uint16_t*>(uv_base), static_cast<size_t>(bh * uv_head_b / 2ULL))) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                std::fprintf(stderr, "[glm53-fast] ERROR: NaN/Inf in BF16->FP16 kv_b_proj '%s'\n", ent.canon_name.c_str());
                return false;
            }

            if (!pwrite_exact(spine_fd, uk_base, static_cast<size_t>(bh * uk_head_b),
                              ld.mla_w_uk.offset + static_cast<uint64_t>(h_done) * uk_head_b) ||
                !pwrite_exact(spine_fd, uv_base, static_cast<size_t>(bh * uv_head_b),
                              ld.mla_w_uv.offset + static_cast<uint64_t>(h_done) * uv_head_b)) {
                return false;
            }
            h_done += bh;
        }

        if (ld.mla_w_uk.bytes > static_cast<uint64_t>(nh) * uk_head_b &&
            !pwrite_zeros(spine_fd, ld.mla_w_uk.offset + static_cast<uint64_t>(nh) * uk_head_b,
                          ld.mla_w_uk.bytes - static_cast<uint64_t>(nh) * uk_head_b)) return false;
        if (ld.mla_w_uv.bytes > static_cast<uint64_t>(nh) * uv_head_b &&
            !pwrite_zeros(spine_fd, ld.mla_w_uv.offset + static_cast<uint64_t>(nh) * uv_head_b,
                          ld.mla_w_uv.bytes - static_cast<uint64_t>(nh) * uv_head_b)) return false;
        return true;
    }

    /* Quantizes FP8+scale_inv OR BF16/F16/F32 matrix into Q4_A gs=64 (`spine_fd` or `mtp_fd`) */
    bool convert_matrix_to_q4(
        int                    shard_fd,
        int                    dst_fd,
        const SafetensorEntry& ent,
        const SafetensorEntry* sc_ent,
        const FastTargetSpec&  spec
    ) {
        const uint64_t rows = spec.qm.rows_logical;
        const uint64_t cols = spec.qm.cols;
        const uint64_t gs   = spec.qm.group_size;
        const uint64_t bpe  = fast_st_dtype_bpe(ent.dtype);
        const uint64_t elems = shape_elems(ent.shape);
        if (bpe == 0 || gs != 64ULL || (cols % 64ULL) != 0ULL || elems != rows * cols ||
            ent.byte_len != elems * bpe) {
            std::fprintf(stderr, "[glm53-fast] ERROR: Q4 matrix mismatch on '%s' (elems=%llu, exp=%llu)\n",
                         ent.canon_name.c_str(),
                         static_cast<unsigned long long>(elems),
                         static_cast<unsigned long long>(rows * cols));
            return false;
        }

        uint64_t sc_cols = 0ULL;
        if (ent.dtype == ST_F8_E4M3) {
            if (!load_scale_inv(shard_fd, sc_ent, rows, cols, &sc_cols)) return false;
        }

        const uint64_t in_row_bytes   = cols * bpe;
        const uint64_t w_row_bytes    = cols / 2ULL;
        const uint64_t groups_per_row = cols / 64ULL;
        const uint64_t sb_row_bytes   = groups_per_row * 2ULL;

        uint64_t max_rows = std::min<uint64_t>(in_buf.size() / in_row_bytes, out_w_buf.size() / w_row_bytes);
        max_rows = std::min<uint64_t>(max_rows, out_s_buf.size() / sb_row_bytes);
        if (max_rows == 0) return false;

        uint64_t row_done = 0;
        while (row_done < rows) {
            const uint64_t batch_rows = std::min<uint64_t>(max_rows, rows - row_done);
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(batch_rows * in_row_bytes),
                             ent.abs_offset + row_done * in_row_bytes)) {
                return false;
            }

            uint8_t*       w_base  = out_w_buf.data();
            uint16_t*      s_base  = reinterpret_cast<uint16_t*>(out_s_buf.data());
            uint16_t*      b_base  = reinterpret_cast<uint16_t*>(out_b_buf.data());
            const uint8_t* src_raw = in_buf.data();
            const float*   sc_ptr  = scale_inv_buf.data();
            std::atomic<bool> batch_finite{true};

            pool->parallel_for_rows(batch_rows, 32ULL, [&](uint64_t r0, uint64_t r1) {
                float tmp[64];
                for (uint64_t r = r0; r < r1; ++r) {
                    const uint64_t r_abs = row_done + r;
                    uint32_t* q_row = reinterpret_cast<uint32_t*>(w_base + r * w_row_bytes);
                    uint16_t* sc_r  = s_base + r * groups_per_row;
                    uint16_t* bi_r  = b_base + r * groups_per_row;

                    for (uint64_t g = 0; g < groups_per_row; ++g) {
                        const uint64_t base_c = r * cols + g * 64ULL;
                        if (ent.dtype == ST_F8_E4M3) {
                            const uint8_t* p8 = src_raw + base_c;
                            const float s_blk = sc_ptr[(r_abs >> 7u) * sc_cols + (g >> 1u)];
                            for (int j = 0; j < 64; ++j) {
                                const uint8_t u = p8[j];
                                if ((u & 0x7Fu) == 0x7Fu) batch_finite.store(false, std::memory_order_relaxed);
                                tmp[j] = fp8_lut[u] * s_blk;
                            }
                        } else if (ent.dtype == ST_BF16) {
                            const uint16_t* p16 = reinterpret_cast<const uint16_t*>(src_raw) + base_c;
                            for (int j = 0; j < 64; ++j) {
                                if (!bf16_is_finite(p16[j])) batch_finite.store(false, std::memory_order_relaxed);
                                tmp[j] = bf16_to_f32_scalar(p16[j]);
                            }
                        } else if (ent.dtype == ST_F16) {
                            const uint16_t* p16 = reinterpret_cast<const uint16_t*>(src_raw) + base_c;
                            for (int j = 0; j < 64; ++j) {
                                if (!fp16_is_finite(p16[j])) batch_finite.store(false, std::memory_order_relaxed);
                                tmp[j] = fp16_to_f32_fast(p16[j]);
                            }
                        } else {
                            const float* pf = reinterpret_cast<const float*>(src_raw) + base_c;
                            for (int j = 0; j < 64; ++j) {
                                if (!std::isfinite(pf[j])) batch_finite.store(false, std::memory_order_relaxed);
                                tmp[j] = pf[j];
                            }
                        }
                        quantize_group_f32(tmp, 64u, q_row + g * 8ULL, &sc_r[g], &bi_r[g]);
                    }
                }
            });

            if (!batch_finite.load(std::memory_order_relaxed)) {
                nan_inf_detected.store(true, std::memory_order_relaxed);
                std::fprintf(stderr, "[glm53-fast] ERROR: NaN/Inf in Q4 input '%s'\n", ent.canon_name.c_str());
                return false;
            }

            if (!pwrite_exact(dst_fd, w_base, static_cast<size_t>(batch_rows * w_row_bytes),
                              spec.qm.weight_off + row_done * w_row_bytes) ||
                !pwrite_exact(dst_fd, s_base, static_cast<size_t>(batch_rows * sb_row_bytes),
                              spec.qm.scales_off + row_done * sb_row_bytes) ||
                !pwrite_exact(dst_fd, b_base, static_cast<size_t>(batch_rows * sb_row_bytes),
                              spec.qm.biases_off + row_done * sb_row_bytes)) {
                return false;
            }
            row_done += batch_rows;
        }

        const uint64_t w_written  = rows * w_row_bytes;
        const uint64_t sb_written = rows * sb_row_bytes;
        if (spec.qm.weight_bytes > w_written &&
            !pwrite_zeros(dst_fd, spec.qm.weight_off + w_written, spec.qm.weight_bytes - w_written)) return false;
        if (spec.qm.scales_bytes > sb_written &&
            !pwrite_zeros(dst_fd, spec.qm.scales_off + sb_written, spec.qm.scales_bytes - sb_written)) return false;
        if (spec.qm.biases_bytes > sb_written &&
            !pwrite_zeros(dst_fd, spec.qm.biases_off + sb_written, spec.qm.biases_bytes - sb_written)) return false;
        if (spec.tail_zero_bytes > 0 &&
            !pwrite_zeros(dst_fd, spec.tail_zero_off, spec.tail_zero_bytes)) return false;
        return true;
    }

    /* Dequant FP8+scale_inv (or BF16/F16/F32) -> RHT-64 along K -> 3-trial OLS E8S-18-V4 + E4M4 -> part0/part1 */
    bool convert_main_expert_e8(
        int                    shard_fd,
        const SafetensorEntry& ent,
        const SafetensorEntry* sc_ent,
        const FastTargetSpec&  spec
    ) {
        auto qt0 = std::chrono::steady_clock::now();
        const uint64_t rows  = spec.e8_rows;
        const uint64_t cols  = spec.e8_cols;
        const uint64_t bpe   = fast_st_dtype_bpe(ent.dtype);
        const uint64_t elems = shape_elems(ent.shape);
        if (bpe == 0 || (cols % 64ULL) != 0ULL || elems != rows * cols || ent.byte_len != elems * bpe) {
            std::fprintf(stderr, "[glm53-fast] ERROR: E8 expert shape mismatch on '%s' (got %llu, exp %llu)\n",
                         ent.canon_name.c_str(),
                         static_cast<unsigned long long>(elems),
                         static_cast<unsigned long long>(rows * cols));
            return false;
        }
        if (ent.byte_len > in_buf.size()) {
            in_buf.resize(static_cast<size_t>(ent.byte_len));
        }

        uint64_t sc_cols = 0ULL;
        if (ent.dtype == ST_F8_E4M3) {
            if (!load_scale_inv(shard_fd, sc_ent, rows, cols, &sc_cols)) return false;
        }
        if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(ent.byte_len), ent.abs_offset)) {
            return false;
        }

        const uint64_t lo16_off   = 0ULL;
        const uint64_t hi2_off    = spine_hdr->e8_lo16_bytes;
        const uint64_t scale_off  = spine_hdr->e8_lo16_bytes + spine_hdr->e8_hi2_bytes;
        const uint64_t total_out  = spine_hdr->e8_proj_bytes;
        if (e8_out_buf.size() < total_out) e8_out_buf.resize(static_cast<size_t>(total_out));
        /* Zero padding bytes if any (canonical 2048x4096 has 0 padding bytes) */
        const uint64_t logical_lo16_b  = (rows * (cols / 8ULL)) * 2ULL;
        const uint64_t logical_hi2_b   = (rows * (cols / 64ULL)) * 2ULL;
        const uint64_t logical_scale_b = (rows * (cols / 64ULL)) * 1ULL;
        if (spine_hdr->e8_lo16_bytes > logical_lo16_b) {
            std::memset(e8_out_buf.data() + lo16_off + logical_lo16_b, 0,
                        static_cast<size_t>(spine_hdr->e8_lo16_bytes - logical_lo16_b));
        }
        if (spine_hdr->e8_hi2_bytes > logical_hi2_b) {
            std::memset(e8_out_buf.data() + hi2_off + logical_hi2_b, 0,
                        static_cast<size_t>(spine_hdr->e8_hi2_bytes - logical_hi2_b));
        }
        if (spine_hdr->e8_scale_bytes > logical_scale_b) {
            std::memset(e8_out_buf.data() + scale_off + logical_scale_b, 0,
                        static_cast<size_t>(spine_hdr->e8_scale_bytes - logical_scale_b));
        }

        uint16_t*      lo16_base  = reinterpret_cast<uint16_t*>(e8_out_buf.data() + lo16_off);
        uint16_t*      hi2_base   = reinterpret_cast<uint16_t*>(e8_out_buf.data() + hi2_off);
        uint8_t*       scale_base = e8_out_buf.data() + scale_off;
        const uint8_t* src_raw    = in_buf.data();
        const float*   sc_ptr     = scale_inv_buf.data();
        const uint64_t grp_cols   = cols / 64ULL;
        const uint64_t vec_cols   = cols / 8ULL;

        std::atomic<bool> finite_ok{true};
        double proj_sig = 0.0;
        double proj_err = 0.0;

        pool->parallel_for_rows(rows, 32ULL, [&](uint64_t r0, uint64_t r1) {
            float w64[64], rot64[64];
            double local_sig = 0.0;
            double local_err = 0.0;

            for (uint64_t r = r0; r < r1; ++r) {
                uint16_t* row_lo16  = lo16_base  + r * vec_cols;
                uint16_t* row_hi2   = hi2_base   + r * grp_cols;
                uint8_t*  row_scale = scale_base + r * grp_cols;

                for (uint64_t g = 0; g < grp_cols; ++g) {
                    const uint64_t base_c = r * cols + g * 64ULL;
                    if (ent.dtype == ST_F8_E4M3) {
                        const uint8_t* p8 = src_raw + base_c;
                        const float s_blk = sc_ptr[(r >> 7u) * sc_cols + (g >> 1u)];
                        for (int j = 0; j < 64; ++j) {
                            const uint8_t u = p8[j];
                            if ((u & 0x7Fu) == 0x7Fu) finite_ok.store(false, std::memory_order_relaxed);
                            w64[j] = fp8_lut[u] * s_blk;
                        }
                    } else if (ent.dtype == ST_BF16) {
                        const uint16_t* p16 = reinterpret_cast<const uint16_t*>(src_raw) + base_c;
                        for (int j = 0; j < 64; ++j) {
                            if (!bf16_is_finite(p16[j])) finite_ok.store(false, std::memory_order_relaxed);
                            w64[j] = bf16_to_f32_scalar(p16[j]);
                        }
                    } else if (ent.dtype == ST_F16) {
                        const uint16_t* p16 = reinterpret_cast<const uint16_t*>(src_raw) + base_c;
                        for (int j = 0; j < 64; ++j) {
                            if (!fp16_is_finite(p16[j])) finite_ok.store(false, std::memory_order_relaxed);
                            w64[j] = fp16_to_f32_fast(p16[j]);
                        }
                    } else {
                        const float* pf = reinterpret_cast<const float*>(src_raw) + base_c;
                        for (int j = 0; j < 64; ++j) {
                            if (!std::isfinite(pf[j])) finite_ok.store(false, std::memory_order_relaxed);
                            w64[j] = pf[j];
                        }
                    }

                    sg53_rht64_f32(w64, rot64);
                    sg53_e8s18_quantize_group64(
                        e8_tbl.get(), rot64, ols_trials,
                        row_lo16 + g * 8ULL, &row_hi2[g], &row_scale[g],
                        &local_sig, &local_err
                    );
                }
            }

            std::lock_guard<std::mutex> lk(stat_mu);
            proj_sig += local_sig;
            proj_err += local_err;
        });

        if (!finite_ok.load(std::memory_order_relaxed)) {
            nan_inf_detected.store(true, std::memory_order_relaxed);
            std::fprintf(stderr, "[glm53-fast] ERROR: NaN/Inf in E8 expert '%s'\n", ent.canon_name.c_str());
            return false;
        }

        const int dst_fd = (spec.part_idx == 0u) ? part0_fd : part1_fd;
        if (!pwrite_exact(dst_fd, e8_out_buf.data(), static_cast<size_t>(total_out), spec.e8_proj_off)) {
            return false;
        }
        if (spec.tail_zero_bytes > 0 &&
            !pwrite_zeros(dst_fd, spec.tail_zero_off, spec.tail_zero_bytes)) {
            return false;
        }

        const double q_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - qt0).count();
        {
            std::lock_guard<std::mutex> lk(stat_mu);
            total_e8_sig_energy += proj_sig;
            total_e8_err_energy += proj_err;
            total_e8_quant_sec  += q_sec;
            ++total_e8_projs_done;
        }
        return true;
    }

    bool verify_spine_slices_on_disk() {
        ShardConverter tmp_chk(spine_fd, -1, spine_hdr, pool);
        return tmp_chk.verify_spine_slices_on_disk() && !nan_inf_detected.load(std::memory_order_relaxed);
    }
};

static std::string json_escape_str(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8U);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20U) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return out;
}

static std::string format_manifest_source_json(const std::string& raw, bool force_local_basename) {
    const bool is_url = (raw.rfind("http://", 0) == 0 || raw.rfind("https://", 0) == 0);
    bool strip_to_basename = force_local_basename;
    if (!strip_to_basename && !is_url) {
        if (!raw.empty() && (raw[0] == '/' || raw[0] == '~' || raw[0] == '.')) {
            strip_to_basename = true;
        } else {
            size_t slashes = 0U;
            for (char c : raw) {
                if (c == '/') ++slashes;
            }
            if (slashes > 1U) strip_to_basename = true;
        }
    }
    std::string label = raw;
    if (strip_to_basename) {
        while (!label.empty() && label.back() == '/') label.pop_back();
        const size_t pos = label.find_last_of('/');
        if (pos != std::string::npos) label = label.substr(pos + 1U);
        if (label.empty() || label == "." || label == "..") label = "local";
    }
    return json_escape_str(label);
}

} /* anonymous namespace */

/* ============================================================================
 * 7. Public C ABI Implementations (`include/strata_glm53.h`)
 * ============================================================================ */

extern "C" {

StrataGlm53Tokenizer* strata_glm53_tokenizer_load(const char* tokenizer_json_path) {
    if (!tokenizer_json_path) return nullptr;
    std::string json_str;
    if (!read_whole_file(tokenizer_json_path, json_str) || json_str.empty()) {
        std::fprintf(stderr, "[glm53-tok] ERROR: failed to read '%s'\n", tokenizer_json_path);
        return nullptr;
    }

    std::string b2u_utf8[256];
    int16_t u2b_cp[512];
    build_gpt2_byte_unicode_tables(b2u_utf8, u2b_cp);

    std::unordered_map<std::string, uint32_t> str_to_id;
    str_to_id.reserve(200000);

    struct RawMerge {
        std::string a;
        std::string b;
    };
    std::vector<RawMerge> raw_merges;
    raw_merges.reserve(330000);
    std::vector<AddedTokenEntry> added_list;
    bool ignore_merges = true;

    JsonCursor cur(json_str.data(), json_str.size());
    if (!cur.consume('{')) return nullptr;

    std::string top_key, sub_key, k_str, v_str;
    uint32_t max_id_seen = 0;

    if (!cur.consume('}')) {
        while (true) {
            if (!cur.parse_string(top_key)) return nullptr;
            if (!cur.consume(':')) return nullptr;

            if (top_key == "added_tokens") {
                if (!cur.consume('[')) return nullptr;
                if (!cur.consume(']')) {
                    while (true) {
                        if (!cur.consume('{')) return nullptr;
                        AddedTokenEntry at{"", 0, false};
                        if (!cur.consume('}')) {
                            while (true) {
                                if (!cur.parse_string(sub_key)) return nullptr;
                                if (!cur.consume(':')) return nullptr;
                                if (sub_key == "id") {
                                    uint64_t id_u64 = 0;
                                    if (!cur.parse_uint64(id_u64) || id_u64 >= (1ULL << 22)) return nullptr;
                                    at.id = static_cast<uint32_t>(id_u64);
                                } else if (sub_key == "content") {
                                    if (!cur.parse_string(at.content)) return nullptr;
                                } else if (sub_key == "special") {
                                    if (!cur.parse_bool(at.special)) return nullptr;
                                } else {
                                    if (!cur.skip_value()) return nullptr;
                                }
                                if (cur.consume('}')) break;
                                if (!cur.consume(',')) return nullptr;
                            }
                        }
                        if (!at.content.empty()) {
                            max_id_seen = std::max(max_id_seen, at.id);
                            added_list.push_back(std::move(at));
                        }
                        if (cur.consume(']')) break;
                        if (!cur.consume(',')) return nullptr;
                    }
                }
            } else if (top_key == "model") {
                if (!cur.consume('{')) return nullptr;
                if (!cur.consume('}')) {
                    while (true) {
                        if (!cur.parse_string(sub_key)) return nullptr;
                        if (!cur.consume(':')) return nullptr;
                        if (sub_key == "ignore_merges") {
                            if (!cur.parse_bool(ignore_merges)) return nullptr;
                        } else if (sub_key == "vocab") {
                            if (!cur.consume('{')) return nullptr;
                            if (!cur.consume('}')) {
                                while (true) {
                                    if (!cur.parse_string(k_str)) return nullptr;
                                    if (!cur.consume(':')) return nullptr;
                                    uint64_t tid = 0;
                                    if (!cur.parse_uint64(tid) || tid >= (1ULL << 22)) return nullptr;
                                    uint32_t id32 = static_cast<uint32_t>(tid);
                                    max_id_seen = std::max(max_id_seen, id32);
                                    str_to_id.emplace(k_str, id32);
                                    if (cur.consume('}')) break;
                                    if (!cur.consume(',')) return nullptr;
                                }
                            }
                        } else if (sub_key == "merges") {
                            if (!cur.consume('[')) return nullptr;
                            if (!cur.consume(']')) {
                                while (true) {
                                    char pk = cur.peek();
                                    if (pk == '[') {
                                        cur.consume('[');
                                        RawMerge rm;
                                        if (!cur.parse_string(rm.a)) return nullptr;
                                        if (!cur.consume(',')) return nullptr;
                                        if (!cur.parse_string(rm.b)) return nullptr;
                                        if (!cur.consume(']')) return nullptr;
                                        raw_merges.push_back(std::move(rm));
                                    } else if (pk == '"') {
                                        if (!cur.parse_string(v_str)) return nullptr;
                                        size_t sp = v_str.find(' ');
                                        if (sp != std::string::npos) {
                                            raw_merges.push_back({v_str.substr(0, sp), v_str.substr(sp + 1)});
                                        }
                                    } else {
                                        return nullptr;
                                    }
                                    if (cur.consume(']')) break;
                                    if (!cur.consume(',')) return nullptr;
                                }
                            }
                        } else {
                            if (!cur.skip_value()) return nullptr;
                        }
                        if (cur.consume('}')) break;
                        if (!cur.consume(',')) return nullptr;
                    }
                }
            } else {
                if (!cur.skip_value()) return nullptr;
            }

            if (cur.consume('}')) break;
            if (!cur.consume(',')) return nullptr;
        }
    }

    if (str_to_id.empty() || max_id_seen >= (1U << 22)) return nullptr;

    auto* tok = new StrataGlm53Tokenizer();
    tok->vocab_size    = max_id_seen + 1u;
    tok->ignore_merges = ignore_merges;
    tok->id_to_raw_bytes.assign(tok->vocab_size, std::string());
    tok->id_is_special.assign(tok->vocab_size, 0u);
    tok->raw_piece_to_id.reserve(str_to_id.size() + 64u);

    std::string raw_tmp;
    for (const auto& kv : str_to_id) {
        if (kv.second >= tok->vocab_size) {
            delete tok;
            return nullptr;
        }
        if (decode_bpe_vocab_key_to_bytes(kv.first, u2b_cp, raw_tmp)) {
            tok->id_to_raw_bytes[kv.second] = raw_tmp;
            tok->raw_piece_to_id.emplace(raw_tmp, kv.second);
        } else {
            tok->id_to_raw_bytes[kv.second] = kv.first;
        }
    }

    for (int b = 0; b < 256; ++b) {
        auto it = str_to_id.find(b2u_utf8[b]);
        if (it == str_to_id.end()) {
            delete tok;
            return nullptr;
        }
        tok->byte_token_ids[b] = it->second;
    }

    for (const auto& at : added_list) {
        if (at.id < tok->vocab_size) {
            tok->id_to_raw_bytes[at.id] = at.content;
            tok->id_is_special[at.id]   = at.special ? 1u : 0u;
        }
    }
    tok->added_sorted = std::move(added_list);
    std::sort(tok->added_sorted.begin(), tok->added_sorted.end(),
              [](const AddedTokenEntry& x, const AddedTokenEntry& y) {
                  if (x.content.size() != y.content.size()) return x.content.size() > y.content.size();
                  return x.id < y.id;
              });
    for (uint32_t idx = 0; idx < tok->added_sorted.size(); ++idx) {
        uint8_t b0 = static_cast<uint8_t>(tok->added_sorted[idx].content[0]);
        tok->added_by_b0[b0].push_back(idx);
    }

    uint64_t ht_cap = 1024;
    while (ht_cap < raw_merges.size() * 2ULL + 64ULL) ht_cap <<= 1;
    tok->merge_ht.assign(static_cast<size_t>(ht_cap),
                         MergeSlot{std::numeric_limits<uint64_t>::max(), 0u, 0u});
    tok->merge_mask = ht_cap - 1ULL;

    std::string combined;
    for (uint32_t rank = 0; rank < raw_merges.size(); ++rank) {
        const auto& rm = raw_merges[rank];
        auto it_a = str_to_id.find(rm.a);
        auto it_b = str_to_id.find(rm.b);
        if (it_a == str_to_id.end() || it_b == str_to_id.end()) continue;
        combined.assign(rm.a);
        combined.append(rm.b);
        auto it_m = str_to_id.find(combined);
        if (it_m == str_to_id.end()) continue;
        tok->insert_merge(it_a->second, it_b->second, rank, it_m->second);
    }

    return tok;
}

void strata_glm53_tokenizer_free(StrataGlm53Tokenizer* tok) {
    delete tok;
}

uint32_t strata_glm53_tokenizer_vocab_size(const StrataGlm53Tokenizer* tok) {
    return tok ? tok->vocab_size : 0u;
}

int strata_glm53_encode(
    const StrataGlm53Tokenizer* tok,
    const char* text,
    size_t      text_len,
    bool        parse_special,
    uint32_t*   out_ids,
    uint32_t    max_ids,
    uint32_t*   out_num_ids
) {
    if (!tok || !text || !out_num_ids) return -1;
    if (text_len == static_cast<size_t>(-1)) {
        text_len = std::strlen(text);
    }
    *out_num_ids = 0;
    if (text_len == 0) return 0;

    std::vector<uint32_t> produced;
    produced.reserve(text_len / 2 + 16);
    std::vector<std::pair<uint32_t, uint32_t>> pieces;

    auto encode_plain_span = [&](const char* span_ptr, size_t span_len) {
        if (span_len == 0) return;
        glm53_pretokenize_span(span_ptr, span_len, pieces);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(span_ptr);
        for (const auto& pc : pieces) {
            bpe_encode_piece(tok, base + pc.first, pc.second, produced);
        }
    };

    if (!parse_special || tok->added_sorted.empty()) {
        encode_plain_span(text, text_len);
    } else {
        size_t span_start = 0;
        size_t pos = 0;
        while (pos < text_len) {
            uint8_t b0 = static_cast<uint8_t>(text[pos]);
            const auto& candidates = tok->added_by_b0[b0];
            int32_t matched_idx = -1;
            for (uint32_t c_idx : candidates) {
                const AddedTokenEntry& at = tok->added_sorted[c_idx];
                size_t mlen = at.content.size();
                if (pos + mlen <= text_len &&
                    std::memcmp(text + pos, at.content.data(), mlen) == 0) {
                    matched_idx = static_cast<int32_t>(c_idx);
                    break;
                }
            }
            if (matched_idx >= 0) {
                if (pos > span_start) {
                    encode_plain_span(text + span_start, pos - span_start);
                }
                const AddedTokenEntry& at = tok->added_sorted[static_cast<size_t>(matched_idx)];
                produced.push_back(at.id);
                pos += at.content.size();
                span_start = pos;
            } else {
                ++pos;
            }
        }
        if (span_start < text_len) {
            encode_plain_span(text + span_start, text_len - span_start);
        }
    }

    *out_num_ids = static_cast<uint32_t>(produced.size());
    if (!out_ids) return 0;
    if (produced.size() > max_ids) return -2;
    if (!produced.empty()) {
        std::memcpy(out_ids, produced.data(), produced.size() * sizeof(uint32_t));
    }
    return 0;
}

int strata_glm53_format_chat_prompt(
    const char* user_prompt,
    const char* system_prompt,
    bool        enable_thinking,
    const char* reasoning_effort,
    char*       out_buf,
    size_t      out_buf_cap,
    size_t*     out_len
) {
    if (!user_prompt) return -1;

    std::string effort_cap = "Low";
    if (reasoning_effort && reasoning_effort[0] != '\0') {
        std::string lower(reasoning_effort);
        for (char& c : lower) {
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
        }
        if (lower == "low") {
            effort_cap = "Low";
        } else if (lower == "high") {
            effort_cap = "High";
        } else if (lower == "max") {
            effort_cap = "Max";
        } else {
            return -1;
        }
    }

    const size_t sys_len = (system_prompt && system_prompt[0] != '\0') ? std::strlen(system_prompt) : 0u;
    std::string res;
    res.reserve(std::strlen(user_prompt) + sys_len + 128u);

    res += "[gMASK]<sop><|system|>Reasoning Effort: ";
    res += effort_cap;
    if (sys_len > 0u) {
        res += "<|system|>";
        res.append(system_prompt, sys_len);
    }
    res += "<|user|>";
    res += user_prompt;
    res += enable_thinking ? "<|assistant|><think>" : "<|assistant|><think></think>";

    if (out_len) *out_len = res.size();
    if (!out_buf) return 0;
    if (res.size() + 1u > out_buf_cap) return -2;
    std::memcpy(out_buf, res.data(), res.size() + 1u);
    return 0;
}

int strata_glm53_stream_decode_step(
    const StrataGlm53Tokenizer* tok,
    StrataGlm53StreamDecoder*   dec,
    uint32_t                    token_id,
    bool                        skip_special,
    char*                       out_utf8,
    size_t                      out_cap,
    size_t*                     out_len
) {
    if (!tok || !dec || !out_utf8 || out_cap == 0) return -1;
    out_utf8[0] = '\0';
    if (out_len) *out_len = 0;

    if (token_id >= tok->vocab_size) return 0;
    if (skip_special && tok->id_is_special[token_id]) return 0;

    const std::string& raw = tok->id_to_raw_bytes[token_id];
    if (raw.empty() && dec->pending_len == 0) return 0;

    std::string combined;
    combined.reserve(dec->pending_len + raw.size());
    if (dec->pending_len > 0) {
        combined.append(reinterpret_cast<const char*>(dec->pending_bytes), dec->pending_len);
        dec->pending_len = 0;
    }
    combined.append(raw);

    const uint8_t* u = reinterpret_cast<const uint8_t*>(combined.data());
    size_t n = combined.size();
    size_t hold = 0;
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        uint8_t b = u[n - back];
        if ((b & 0xC0u) == 0x80u) continue;
        size_t need = 1;
        if ((b & 0xE0u) == 0xC0u)      need = 2;
        else if ((b & 0xF0u) == 0xE0u) need = 3;
        else if ((b & 0xF8u) == 0xF0u) need = 4;
        if (need > back) hold = back;
        break;
    }

    size_t emit_len = n - hold;
    if (hold > 0) {
        std::memcpy(dec->pending_bytes, u + emit_len, hold);
        dec->pending_len = static_cast<uint8_t>(hold);
    }
    if (emit_len + 1 > out_cap) return -2;
    if (emit_len > 0) {
        std::memcpy(out_utf8, combined.data(), emit_len);
    }
    out_utf8[emit_len] = '\0';
    if (out_len) *out_len = emit_len;
    return 0;
}

int strata_glm53_decode(
    const StrataGlm53Tokenizer* tok,
    const uint32_t* ids,
    uint32_t        num_ids,
    bool            skip_special,
    char*           out_utf8,
    size_t          out_cap,
    size_t*         out_len
) {
    if (!tok || (!ids && num_ids > 0)) return -1;
    std::string acc;
    acc.reserve(static_cast<size_t>(num_ids) * 4 + 16);
    for (uint32_t i = 0; i < num_ids; ++i) {
        uint32_t tid = ids[i];
        if (tid >= tok->vocab_size) continue;
        if (skip_special && tok->id_is_special[tid]) continue;
        acc.append(tok->id_to_raw_bytes[tid]);
    }
    if (out_len) *out_len = acc.size();
    if (!out_utf8) return 0;
    if (acc.size() + 1 > out_cap) return -2;
    std::memcpy(out_utf8, acc.data(), acc.size() + 1);
    return 0;
}

int strata_glm53_convert_hf(
    const char* hf_repo_or_dir,
    const char* out_bundle_dir,
    uint32_t    download_concurrency,
    uint32_t    convert_threads,
    StrataGlm53ConvertStats* stats_out
) {
    auto wall_t0 = std::chrono::steady_clock::now();
    if (stats_out) std::memset(stats_out, 0, sizeof(*stats_out));

    const std::string src_spec = (hf_repo_or_dir && hf_repo_or_dir[0] != '\0')
                                     ? std::string(hf_repo_or_dir)
                                     : std::string(STRATA_GLM53_DEFAULT_HF_REPO);
    const std::string out_dir  = (out_bundle_dir && out_bundle_dir[0] != '\0')
                                     ? std::string(out_bundle_dir)
                                     : std::string(STRATA_GLM53_DEFAULT_BUNDLE_DIR);

    if (!mkdirs(out_dir)) {
        std::fprintf(stderr, "[glm53-stream] ERROR: cannot create bundle dir '%s'\n", out_dir.c_str());
        return -1;
    }

    bool is_local_dir = false;
    if (is_directory_path(src_spec) && file_exists_nonempty(src_spec + "/config.json")) {
        if (file_exists_nonempty(src_spec + "/model.safetensors")) {
            is_local_dir = true;
        } else {
            std::string tmp_idx;
            if (read_whole_file(src_spec + "/model.safetensors.index.json", tmp_idx)) {
                SmallJson ij;
                if (parse_small_json(tmp_idx, ij)) {
                    if (const SmallJson* wm = ij.get("weight_map")) {
                        if (!wm->obj.empty() &&
                            is_safe_shard_basename(wm->obj.begin()->second.s_val) &&
                            file_exists_nonempty(src_spec + "/" + wm->obj.begin()->second.s_val)) {
                            is_local_dir = true;
                        }
                    }
                }
            }
        }
    }

    std::string base_url;
    if (!is_local_dir) {
        if (src_spec.rfind("http://", 0) == 0 || src_spec.rfind("https://", 0) == 0) {
            base_url = src_spec;
            while (!base_url.empty() && base_url.back() == '/') base_url.pop_back();
            if (base_url.find("/resolve/") == std::string::npos) {
                base_url += "/resolve/main";
            }
        } else {
            const char* ep = std::getenv("HF_ENDPOINT");
            std::string host = (ep && ep[0] != '\0') ? std::string(ep) : std::string("https://huggingface.co");
            while (!host.empty() && host.back() == '/') host.pop_back();
            base_url = host + "/" + src_spec + "/resolve/main";
        }
    }

    /* 1. Fetch or copy metadata JSONs + `tokenizer.json` */
    if (is_local_dir) {
        if (!copy_file_if_exists(src_spec + "/config.json", out_dir + "/config.json", true)) {
            std::fprintf(stderr, "[glm53-stream] ERROR: missing config.json in '%s'\n", src_spec.c_str());
            return -1;
        }
        (void)copy_file_if_exists(src_spec + "/tokenizer.json", out_dir + "/tokenizer.json", false);
        (void)copy_file_if_exists(src_spec + "/tokenizer_config.json", out_dir + "/tokenizer_config.json", false);
        (void)copy_file_if_exists(src_spec + "/chat_template.jinja", out_dir + "/chat_template.jinja", false);
        (void)copy_file_if_exists(src_spec + "/generation_config.json", out_dir + "/generation_config.json", false);
        (void)copy_file_if_exists(src_spec + "/model.safetensors.index.json", out_dir + "/model.safetensors.index.json", false);
    } else {
        const char* req_meta[] = {"config.json", "tokenizer.json", "model.safetensors.index.json"};
        for (const char* mf : req_meta) {
            std::string dst = out_dir + "/" + mf;
            if (!file_exists_nonempty(dst)) {
                std::fprintf(stdout, "[glm53-stream] Downloading %s ...\n", mf);
                std::fflush(stdout);
                if (!curl_download_single(base_url + "/" + mf, dst, nullptr, nullptr)) {
                    std::fprintf(stderr, "[glm53-stream] ERROR: failed to download '%s'\n", mf);
                    return -1;
                }
            }
        }
        const char* opt_meta[] = {"tokenizer_config.json", "chat_template.jinja", "generation_config.json"};
        for (const char* mf : opt_meta) {
            std::string dst = out_dir + "/" + mf;
            if (!file_exists_nonempty(dst)) {
                (void)curl_download_single(base_url + "/" + mf, dst, nullptr, nullptr);
            }
        }
    }

    /* 2. Parse `config.json` & build deterministic `spine.strata` + `experts_nvme.strata` layout */
    StrataGlm53Config cfg;
    if (!parse_config_into(out_dir + "/config.json", out_dir + "/generation_config.json", cfg)) {
        std::fprintf(stderr, "[glm53-stream] ERROR: failed to parse '%s/config.json'\n", out_dir.c_str());
        return -1;
    }

    auto spine_hdr_ptr = std::make_unique<StrataGlm53SpineHeader>();
    StrataGlm53SpineHeader& spine_hdr = *spine_hdr_ptr;
    strata_glm53_init_layout(&cfg, &spine_hdr);

    TargetRegistry reg;
    build_target_registry(spine_hdr, reg);

    /* 3. Parse `model.safetensors.index.json` inside inner scope so temporary DOM is freed immediately */
    std::vector<std::string> shard_names;
    std::unordered_map<std::string, std::vector<std::string>> shard_raw_keys;
    uint64_t index_total_bytes = 0ULL;
    {
        std::string idx_raw;
        if (read_whole_file(out_dir + "/model.safetensors.index.json", idx_raw)) {
            SmallJson idx_json;
            if (!parse_small_json(idx_raw, idx_json)) return -1;
            if (const SmallJson* md = idx_json.get("metadata")) {
                index_total_bytes = static_cast<uint64_t>(md->get_f64("total_size", 0.0));
            }
            const SmallJson* wm = idx_json.get("weight_map");
            if (!wm || wm->type != SmallJson::J_OBJ) return -1;
            std::set<std::string> uniq;
            for (const auto& kv : wm->obj) {
                if (kv.second.type == SmallJson::J_STR) {
                    if (!is_safe_shard_basename(kv.second.s_val)) {
                        std::fprintf(stderr, "[glm53-stream] ERROR: unsafe shard filename '%s' in model.safetensors.index.json\n",
                                     kv.second.s_val.c_str());
                        return -1;
                    }
                    uniq.insert(kv.second.s_val);
                    shard_raw_keys[kv.second.s_val].push_back(kv.first);
                }
            }
            shard_names.assign(uniq.begin(), uniq.end());
        } else if (is_local_dir && file_exists_nonempty(src_spec + "/model.safetensors")) {
            shard_names.push_back("model.safetensors");
        } else {
            std::fprintf(stderr, "[glm53-stream] ERROR: missing model.safetensors.index.json\n");
            return -1;
        }
    }

    /* 4. Open or resume `spine.strata` (`5.3611 GiB`) & `experts_nvme.strata` (`163.2656 GiB`) */
    const std::string spine_path   = out_dir + "/" + STRATA_GLM53_SPINE_FILENAME;
    const std::string experts_path = out_dir + "/" + STRATA_GLM53_EXPERTS_NVME_FILENAME;
    const std::string done_path    = out_dir + "/.convert_done_shards";
    const std::string stage_dir    = out_dir + "/.staging";

    const bool has_experts = (spine_hdr.experts_nvme_bytes > 0ULL);
    std::set<std::string> done_shards;
    bool can_resume = false;
    {
        uint64_t sz_s = 0, sz_e = 0;
        if (file_exists_nonempty(done_path) &&
            file_exists_nonempty(spine_path, &sz_s) && sz_s == spine_hdr.total_spine_bytes &&
            (!has_experts || (file_exists_nonempty(experts_path, &sz_e) && sz_e == spine_hdr.experts_nvme_bytes))) {
            std::string done_txt;
            if (read_whole_file(done_path, done_txt)) {
                can_resume = true;
                size_t p = 0;
                while (p < done_txt.size()) {
                    size_t nl = done_txt.find('\n', p);
                    std::string line = (nl == std::string::npos) ? done_txt.substr(p) : done_txt.substr(p, nl - p);
                    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
                    if (!line.empty()) done_shards.insert(line);
                    if (nl == std::string::npos) break;
                    p = nl + 1;
                }
            }
        }
    }

    const int open_flags = can_resume ? O_RDWR : (O_CREAT | O_TRUNC | O_RDWR);
    int sfd_out = open(spine_path.c_str(), open_flags, 0644);
    int efd_out = has_experts ? open(experts_path.c_str(), open_flags, 0644) : -1;
    if (sfd_out < 0 || (has_experts && efd_out < 0)) {
        if (sfd_out >= 0) close(sfd_out);
        if (efd_out >= 0) close(efd_out);
        std::fprintf(stderr, "[glm53-stream] ERROR: failed to open output .strata files\n");
        return -1;
    }

#if defined(F_NOCACHE)
    (void)fcntl(sfd_out, F_NOCACHE, 1);
    if (efd_out >= 0) (void)fcntl(efd_out, F_NOCACHE, 1);
#endif

    if (!can_resume) {
        done_shards.clear();
        try_preallocate_apfs(sfd_out, spine_hdr.total_spine_bytes);
        if (efd_out >= 0) try_preallocate_apfs(efd_out, spine_hdr.experts_nvme_bytes);

        if (ftruncate(sfd_out, static_cast<off_t>(spine_hdr.total_spine_bytes)) != 0 ||
            (efd_out >= 0 && ftruncate(efd_out, static_cast<off_t>(spine_hdr.experts_nvme_bytes)) != 0)) {
            close(sfd_out);
            if (efd_out >= 0) close(efd_out);
            std::fprintf(stderr, "[glm53-stream] ERROR: ftruncate failed on .strata files\n");
            return -1;
        }
        /* Zero header region (`128 KiB`) and write initial `StrataGlm53SpineHeader` */
        if (!pwrite_zeros(sfd_out, 0, STRATA_GLM53_HEADER_PADDED_BYTES) ||
            !pwrite_exact(sfd_out, &spine_hdr, sizeof(spine_hdr), 0)) {
            close(sfd_out);
            if (efd_out >= 0) close(efd_out);
            return -1;
        }
        int dfd = open(done_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (dfd >= 0) close(dfd);
    }

    /* Track target coverage, expert completion masks, and resumed statistics */
    std::vector<uint8_t>  target_seen(reg.specs.size(), 0u);
    std::vector<uint16_t> expert_mask(spine_hdr.total_routed_experts, 0u);
    uint32_t num_shards_resumed       = 0;
    uint32_t num_shards_converted     = 0;
    uint32_t text_tensors_converted   = 0;
    uint32_t mtp_tensors_converted    = 0;
    uint32_t routed_experts_packed    = 0;
    uint32_t vision_tensors_skipped   = 0;
    uint64_t total_hf_bytes_processed = 0;

    auto mark_spec_completed = [&](const TargetSpec& ts, bool quantized_unpacked) {
        if (!target_seen[ts.id]) {
            target_seen[ts.id] = 1u;
            if (ts.is_mtp) ++mtp_tensors_converted;
            else           ++text_tensors_converted;
        }
        if (quantized_unpacked) {
            if (ts.companion_s_id != UINT32_MAX) target_seen[ts.companion_s_id] = 1u;
            if (ts.companion_b_id != UINT32_MAX) target_seen[ts.companion_b_id] = 1u;
        }
        if (ts.global_exp_idx >= 0 && static_cast<uint32_t>(ts.global_exp_idx) < expert_mask.size()) {
            uint16_t prev = expert_mask[ts.global_exp_idx];
            uint16_t add  = ts.exp_part_mask;
            if (quantized_unpacked) {
                if (ts.companion_s_id != UINT32_MAX) add |= reg.specs[ts.companion_s_id].exp_part_mask;
                if (ts.companion_b_id != UINT32_MAX) add |= reg.specs[ts.companion_b_id].exp_part_mask;
            }
            uint16_t next = static_cast<uint16_t>(prev | add);
            expert_mask[ts.global_exp_idx] = next;
            if (prev != 0x01FFu && next == 0x01FFu) {
                ++routed_experts_packed;
            }
        }
    };

    std::vector<uint32_t> pending_shard_indices;
    for (uint32_t s = 0; s < shard_names.size(); ++s) {
        const std::string& sname = shard_names[s];
        if (done_shards.count(sname) > 0) {
            ++num_shards_resumed;
            auto it_k = shard_raw_keys.find(sname);
            if (it_k != shard_raw_keys.end()) {
                for (const std::string& rk : it_k->second) {
                    if (is_vision_tensor_name(rk)) {
                        ++vision_tensors_skipped;
                    } else {
                        std::string cn = canonicalize_tensor_name(rk);
                        auto it_t = reg.name_to_id.find(cn);
                        if (it_t != reg.name_to_id.end()) {
                            mark_spec_completed(reg.specs[it_t->second], false);
                        }
                    }
                }
            }
        } else {
            pending_shard_indices.push_back(s);
        }
    }

    std::fprintf(stdout,
                 "[glm53-stream] Layout initialized: spine=%.4f GiB (%llu B), experts_nvme=%.4f GiB (%llu B = %u x 13.5 MiB) | "
                 "Shards=%zu (%u resumed, %zu pending)\n",
                 static_cast<double>(spine_hdr.total_spine_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<unsigned long long>(spine_hdr.total_spine_bytes),
                 static_cast<double>(spine_hdr.experts_nvme_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<unsigned long long>(spine_hdr.experts_nvme_bytes),
                 spine_hdr.total_routed_experts,
                 shard_names.size(), num_shards_resumed, pending_shard_indices.size());
    std::fflush(stdout);

    /* 5. Execute Bounded Producer-Consumer Pipeline (`< 8 GiB` `.staging/`) + NEON Converter */
    NeonWorkerPool pool(convert_threads);
    ShardConverter converter(sfd_out, efd_out, &spine_hdr, &pool);

    int done_fd = open(done_path.c_str(), O_CREAT | O_APPEND | O_WRONLY, 0644);
    uint64_t total_downloaded_bytes = 0ULL;
    uint64_t peak_staging_bytes     = 0ULL;
    auto dl_phase_t0 = std::chrono::steady_clock::now();
    bool ok = true;

    auto process_one_shard = [&](uint32_t shard_idx, const std::string& spath, bool delete_after) -> bool {
        auto st0 = std::chrono::steady_clock::now();
        if (delete_after) {
            peak_staging_bytes = std::max(peak_staging_bytes, measure_dir_bytes(stage_dir));
        }

        int sfd = open(spath.c_str(), O_RDONLY);
        if (sfd < 0) {
            std::fprintf(stderr, "[glm53-stream] ERROR: cannot open shard '%s'\n", spath.c_str());
            return false;
        }
#if defined(F_NOCACHE)
        (void)fcntl(sfd, F_NOCACHE, 1);
#endif
        struct stat st;
        if (fstat(sfd, &st) != 0 || st.st_size <= 8) {
            close(sfd);
            return false;
        }
        const uint64_t sbytes = static_cast<uint64_t>(st.st_size);
        total_hf_bytes_processed += sbytes;

        std::vector<SafetensorEntry> entries;
        uint32_t vis_skip = 0;
        if (!parse_safetensors_shard_entries(sfd, sbytes, entries, vis_skip)) {
            close(sfd);
            std::fprintf(stderr, "[glm53-stream] ERROR: corrupt safetensors header in '%s'\n", spath.c_str());
            return false;
        }
        vision_tensors_skipped += vis_skip;

        /* Sort tensor entries by byte offset for strictly sequential NVMe shard reads */
        std::sort(entries.begin(), entries.end(),
                  [](const SafetensorEntry& a, const SafetensorEntry& b) {
                      return a.abs_offset < b.abs_offset;
                  });

        /* Pre-index any `kv_b_proj.{weight,scales,biases}` triplets present in this shard */
        struct MlaKvBGroup {
            const SafetensorEntry* w = nullptr;
            const SafetensorEntry* s = nullptr;
            const SafetensorEntry* b = nullptr;
            uint32_t w_spec_id = UINT32_MAX;
            uint32_t s_spec_id = UINT32_MAX;
            uint32_t b_spec_id = UINT32_MAX;
            bool     done      = false;
        };
        std::map<uint32_t, MlaKvBGroup> kvb_groups;
        for (const SafetensorEntry& ent : entries) {
            auto it = reg.name_to_id.find(ent.canon_name);
            if (it == reg.name_to_id.end()) {
                close(sfd);
                std::fprintf(stderr, "[glm53-stream] ERROR: unexpected tensor '%s' in '%s'\n",
                             ent.canon_name.c_str(), shard_names[shard_idx].c_str());
                return false;
            }
            const TargetSpec& ts = reg.specs[it->second];
            if (ts.kind == TGT_MLA_KVB_W) {
                kvb_groups[ts.mla_layer_slot].w         = &ent;
                kvb_groups[ts.mla_layer_slot].w_spec_id = ts.id;
            } else if (ts.kind == TGT_MLA_KVB_S) {
                kvb_groups[ts.mla_layer_slot].s         = &ent;
                kvb_groups[ts.mla_layer_slot].s_spec_id = ts.id;
            } else if (ts.kind == TGT_MLA_KVB_B) {
                kvb_groups[ts.mla_layer_slot].b         = &ent;
                kvb_groups[ts.mla_layer_slot].b_spec_id = ts.id;
            }
        }

        for (const SafetensorEntry& ent : entries) {
            const TargetSpec& ts = reg.specs[reg.name_to_id[ent.canon_name]];
            bool conv_ok = false;
            bool q_unpacked = false;

            switch (ts.kind) {
                case TGT_SPINE_F32:
                    conv_ok = converter.convert_spine_f32(sfd, ent, ts);
                    if (conv_ok) mark_spec_completed(ts, false);
                    break;

                case TGT_SPINE_F16:
                    conv_ok = converter.convert_spine_f16(sfd, ent, ts);
                    if (conv_ok) mark_spec_completed(ts, false);
                    break;

                case TGT_Q4_WEIGHT:
                    conv_ok = converter.convert_q4_weight(sfd, ent, ts, &q_unpacked);
                    if (conv_ok) mark_spec_completed(ts, q_unpacked);
                    break;

                case TGT_Q4_SCALES:
                case TGT_Q4_BIASES:
                    conv_ok = converter.convert_q4_scales_or_biases(sfd, ent, ts);
                    if (conv_ok) mark_spec_completed(ts, false);
                    break;

                case TGT_MLA_KVB_W:
                case TGT_MLA_KVB_S:
                case TGT_MLA_KVB_B: {
                    MlaKvBGroup& grp = kvb_groups[ts.mla_layer_slot];
                    if (grp.done) {
                        conv_ok = true;
                        break;
                    }
                    if (!grp.w || !grp.s || !grp.b) {
                        std::fprintf(stderr, "[glm53-stream] ERROR: incomplete kv_b_proj triplet at layer %u\n",
                                     ts.mla_layer_slot);
                        conv_ok = false;
                        break;
                    }
                    const StrataGlm53LayerDesc& ld =
                        (ts.mla_layer_slot == spine_hdr.mtp.global_layer_idx)
                            ? spine_hdr.mtp.layer
                            : spine_hdr.layers[ts.mla_layer_slot];
                    conv_ok = converter.convert_mla_kv_b_triplet(sfd, *grp.w, grp.s, grp.b, ld);
                    if (conv_ok) {
                        grp.done = true;
                        mark_spec_completed(reg.specs[grp.w_spec_id], false);
                        mark_spec_completed(reg.specs[grp.s_spec_id], false);
                        mark_spec_completed(reg.specs[grp.b_spec_id], false);
                    }
                    break;
                }
            }

            if (!conv_ok) {
                close(sfd);
                return false;
            }
        }
        close(sfd);

        if (delete_after) {
            peak_staging_bytes = std::max(peak_staging_bytes, measure_dir_bytes(stage_dir));
            unlink(spath.c_str());
        }

        if (done_fd >= 0) {
            std::string rec = shard_names[shard_idx] + "\n";
            (void)write(done_fd, rec.data(), rec.size());
            (void)fsync(done_fd);
        }
        ++num_shards_converted;

        double conv_sec    = std::chrono::duration<double>(std::chrono::steady_clock::now() - st0).count();
        double wall_so_far = std::chrono::duration<double>(std::chrono::steady_clock::now() - dl_phase_t0).count();
        double net_mbps    = (wall_so_far > 1e-6 && total_downloaded_bytes > 0)
                                 ? (static_cast<double>(total_downloaded_bytes) / (1e6 * wall_so_far))
                                 : 0.0;
        uint32_t done_all  = num_shards_resumed + num_shards_converted;
        std::fprintf(stdout,
                     "[glm53-stream] [%2u/%zu] %-34s (%5.2f GiB, %4zu keys) pack %.2fs | "
                     "net %6.1f MB/s | stage_pk %.2f GiB | RSS %.3f GiB\n",
                     done_all, shard_names.size(), shard_names[shard_idx].c_str(),
                     static_cast<double>(sbytes) / (1024.0 * 1024.0 * 1024.0),
                     entries.size(), conv_sec, net_mbps,
                     static_cast<double>(peak_staging_bytes) / (1024.0 * 1024.0 * 1024.0),
                     current_peak_rss_gib());
        std::fflush(stdout);
        return true;
    };

    if (!pending_shard_indices.empty()) {
        if (is_local_dir) {
            for (uint32_t s_idx : pending_shard_indices) {
                if (!process_one_shard(s_idx, src_spec + "/" + shard_names[s_idx], false)) {
                    ok = false;
                    break;
                }
            }
        } else {
            if (!mkdirs(stage_dir)) {
                if (done_fd >= 0) close(done_fd);
                close(sfd_out);
                if (efd_out >= 0) close(efd_out);
                return -1;
            }
            clean_staging_part_files(stage_dir);

            /*
             * Default (`download_concurrency == 0`): 1 background downloader thread + `max_staged = 2`
             * permits (`1` shard `i` being converted + `1` shard `i + 1` prefetched in `.staging/`),
             * strictly bounding `.staging/` disk usage to `<= 2 x 3.999 GiB = 7.998 GiB < 8.000 GiB`!
             */
            const uint32_t n_dl       = (download_concurrency == 0u) ? 1u : std::min<uint32_t>(download_concurrency, 2u);
            const uint32_t max_staged = n_dl + 1u;

            struct ReadyShard {
                uint32_t    shard_idx;
                std::string path;
                uint64_t    downloaded_bytes;
                bool        ok;
            };

            std::mutex              q_mu;
            std::condition_variable cv_stage;
            std::condition_variable cv_ready;
            std::deque<ReadyShard>  ready_q;
            uint32_t                staged_permits = max_staged;
            std::atomic<size_t>     next_job{0};
            std::atomic<bool>       abort_dl{false};
            uint32_t                active_dl = n_dl;

            auto dl_worker = [&]() {
                while (!abort_dl.load(std::memory_order_relaxed)) {
                    size_t j = next_job.fetch_add(1, std::memory_order_relaxed);
                    if (j >= pending_shard_indices.size()) break;
                    const uint32_t s_idx = pending_shard_indices[j];
                    const std::string& sname = shard_names[s_idx];
                    const std::string dst = stage_dir + "/" + sname;

                    {
                        std::unique_lock<std::mutex> lk(q_mu);
                        cv_stage.wait(lk, [&]() {
                            return abort_dl.load(std::memory_order_relaxed) || staged_permits > 0;
                        });
                        if (abort_dl.load(std::memory_order_relaxed)) break;
                        --staged_permits;
                    }

                    uint64_t dl_bytes = 0;
                    bool shard_ok = false;
                    uint64_t existing_sz = 0;
                    if (file_exists_nonempty(dst, &existing_sz)) {
                        int tfd = open(dst.c_str(), O_RDONLY);
                        if (tfd >= 0) {
                            std::vector<SafetensorEntry> tmp_e;
                            uint32_t tmp_v = 0;
                            if (parse_safetensors_shard_entries(tfd, existing_sz, tmp_e, tmp_v)) {
                                shard_ok = true;
                            }
                            close(tfd);
                        }
                        if (!shard_ok) unlink(dst.c_str());
                    }

                    if (!shard_ok) {
                        double dl_sec = 0.0;
                        shard_ok = curl_download_shard_fast(base_url + "/" + sname, dst, &dl_bytes, &dl_sec);
                    }

                    {
                        std::lock_guard<std::mutex> lk(q_mu);
                        if (!shard_ok) {
                            abort_dl.store(true, std::memory_order_relaxed);
                            ++staged_permits;
                            cv_stage.notify_all();
                        }
                        ready_q.push_back({s_idx, dst, dl_bytes, shard_ok});
                    }
                    cv_ready.notify_one();
                }

                std::lock_guard<std::mutex> lk(q_mu);
                if (--active_dl == 0) {
                    cv_ready.notify_all();
                }
            };

            std::vector<std::thread> dl_threads;
            dl_threads.reserve(n_dl);
            for (uint32_t i = 0; i < n_dl; ++i) {
                dl_threads.emplace_back(dl_worker);
            }

            size_t consumed = 0;
            while (consumed < pending_shard_indices.size()) {
                ReadyShard item;
                {
                    std::unique_lock<std::mutex> lk(q_mu);
                    cv_ready.wait(lk, [&]() { return !ready_q.empty() || active_dl == 0; });
                    if (ready_q.empty()) {
                        ok = false;
                        break;
                    }
                    item = std::move(ready_q.front());
                    ready_q.pop_front();
                }

                if (!item.ok) {
                    std::fprintf(stderr, "[glm53-stream] ERROR: download failed for '%s'\n",
                                 shard_names[item.shard_idx].c_str());
                    ok = false;
                    abort_dl.store(true, std::memory_order_relaxed);
                    cv_stage.notify_all();
                    break;
                }

                total_downloaded_bytes += item.downloaded_bytes;
                bool conv_ok = process_one_shard(item.shard_idx, item.path, true);

                {
                    std::lock_guard<std::mutex> lk(q_mu);
                    ++staged_permits;
                }
                cv_stage.notify_one();

                if (!conv_ok) {
                    ok = false;
                    abort_dl.store(true, std::memory_order_relaxed);
                    cv_stage.notify_all();
                    break;
                }
                ++consumed;
            }

            abort_dl.store(true, std::memory_order_relaxed);
            cv_stage.notify_all();
            for (auto& th : dl_threads) {
                if (th.joinable()) th.join();
            }
        }
    }

    if (done_fd >= 0) close(done_fd);

    /* 6. Verify 100% tensor coverage across all expected Spine + NVMe Expert + MTP targets */
    if (ok) {
        for (size_t i = 0; i < reg.specs.size(); ++i) {
            if (!target_seen[i]) {
                std::string missing_name = "<unknown>";
                for (const auto& kv : reg.name_to_id) {
                    if (kv.second == i) {
                        missing_name = kv.first;
                        break;
                    }
                }
                std::fprintf(stderr, "[glm53-stream] ERROR: missing expected tensor '%s' after processing all shards\n",
                             missing_name.c_str());
                ok = false;
                break;
            }
        }
    }

    /* 7. Re-stamp `StrataGlm53SpineHeader`, run full on-disk NaN/Inf check, and fsync */
    bool nan_inf_ok = false;
    if (ok) {
        if (!pwrite_exact(sfd_out, &spine_hdr, sizeof(spine_hdr), 0)) {
            ok = false;
        } else {
            nan_inf_ok = converter.verify_spine_slices_on_disk();
            if (!nan_inf_ok) {
                std::fprintf(stderr, "[glm53-stream] ERROR: on-disk NaN/Inf verification failed\n");
                ok = false;
            }
        }
    }

    if (!ok) {
        close(sfd_out);
        if (efd_out >= 0) close(efd_out);
        return -1;
    }

    (void)fsync(sfd_out);
    if (efd_out >= 0) (void)fsync(efd_out);
    close(sfd_out);
    if (efd_out >= 0) close(efd_out);

    /* 8. Clean `.staging/` directory & verify 16 KiB / 256 KiB alignments */
    clean_staging_part_files(stage_dir);
    (void)rmdir(stage_dir.c_str());
    bool staging_clean = check_dir_has_no_files(stage_dir);

    uint64_t actual_spine   = 0ULL;
    uint64_t actual_experts = 0ULL;
    bool aligned_ok =
        file_exists_nonempty(spine_path, &actual_spine) &&
        actual_spine == spine_hdr.total_spine_bytes &&
        (actual_spine % STRATA_GLM53_PAGE_BYTES) == 0ULL &&
        (!has_experts || (file_exists_nonempty(experts_path, &actual_experts) &&
                          actual_experts == spine_hdr.experts_nvme_bytes &&
                          (actual_experts % STRATA_GLM53_NVME_STRIPE_BYTES) == 0ULL &&
                          (actual_experts % STRATA_GLM53_PAGE_BYTES) == 0ULL));

    double elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_t0).count();
    double dl_elapsed  = std::chrono::duration<double>(std::chrono::steady_clock::now() - dl_phase_t0).count();
    double net_mbps    = (dl_elapsed > 1e-6 && total_downloaded_bytes > 0)
                             ? (static_cast<double>(total_downloaded_bytes) / (1e6 * dl_elapsed))
                             : 0.0;
    double peak_rss    = current_peak_rss_gib();
    uint64_t hf_bytes  = (index_total_bytes > 0) ? index_total_bytes : total_hf_bytes_processed;

    /* 9. Write human-readable `manifest.json` */
    {
        std::string mf_path = out_dir + "/manifest.json";
        FILE* mf = std::fopen(mf_path.c_str(), "wb");
        if (mf) {
            const std::string safe_src = format_manifest_source_json(src_spec, is_local_dir);
            std::fprintf(mf,
                "{\n"
                "  \"format\": \"STRATA_GLM53_V1\",\n"
                "  \"architecture\": \"Glm5NextForConditionalGeneration\",\n"
                "  \"source\": \"%s\",\n"
                "  \"num_hidden_layers\": %u,\n"
                "  \"num_nextn_predict_layers\": %u,\n"
                "  \"first_k_dense_replace\": %u,\n"
                "  \"num_kda_layers\": %u,\n"
                "  \"num_dsa_layers\": %u,\n"
                "  \"hidden_size\": %u,\n"
                "  \"intermediate_size\": %u,\n"
                "  \"n_routed_experts\": %u,\n"
                "  \"num_experts_per_tok\": %u,\n"
                "  \"moe_intermediate_size\": %u,\n"
                "  \"total_routed_experts\": %u,\n"
                "  \"expert_record_bytes\": %llu,\n"
                "  \"vocab_size\": %u,\n"
                "  \"page_align_bytes\": %llu,\n"
                "  \"nvme_stripe_bytes\": %llu,\n"
                "  \"hf_safetensors_bytes\": %llu,\n"
                "  \"spine_bytes\": %llu,\n"
                "  \"experts_nvme_bytes\": %llu,\n"
                "  \"total_bundle_bytes\": %llu,\n"
                "  \"active_spine_bytes_per_tok\": %llu,\n"
                "  \"active_routed_bytes_per_tok\": %llu,\n"
                "  \"active_bytes_per_tok\": %llu,\n"
                "  \"mtp_active_bytes_per_draft\": %llu,\n"
                "  \"num_shards_total\": %zu,\n"
                "  \"num_text_tensors_converted\": %u,\n"
                "  \"num_mtp_tensors_converted\": %u,\n"
                "  \"num_routed_experts_packed\": %u,\n"
                "  \"num_vision_tensors_skipped\": %u,\n"
                "  \"peak_staging_gib\": %.4f,\n"
                "  \"elapsed_sec\": %.3f,\n"
                "  \"network_mbps\": %.2f,\n"
                "  \"peak_rss_gib\": %.4f,\n"
                "  \"nan_inf_checked\": %s\n"
                "}\n",
                safe_src.c_str(),
                cfg.num_hidden_layers, cfg.num_nextn_predict_layers, cfg.first_k_dense_replace,
                cfg.num_kda_layers, cfg.num_dsa_layers,
                cfg.hidden_size, cfg.intermediate_size,
                cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size,
                spine_hdr.total_routed_experts,
                static_cast<unsigned long long>(spine_hdr.expert_record_bytes),
                cfg.vocab_size,
                static_cast<unsigned long long>(STRATA_GLM53_PAGE_BYTES),
                static_cast<unsigned long long>(STRATA_GLM53_NVME_STRIPE_BYTES),
                static_cast<unsigned long long>(hf_bytes),
                static_cast<unsigned long long>(actual_spine),
                static_cast<unsigned long long>(actual_experts),
                static_cast<unsigned long long>(actual_spine + actual_experts),
                static_cast<unsigned long long>(spine_hdr.active_spine_bytes_per_tok),
                static_cast<unsigned long long>(spine_hdr.active_routed_bytes_per_tok),
                static_cast<unsigned long long>(spine_hdr.active_bytes_per_tok),
                static_cast<unsigned long long>(spine_hdr.mtp_active_bytes_per_draft),
                shard_names.size(),
                text_tensors_converted, mtp_tensors_converted,
                routed_experts_packed, vision_tensors_skipped,
                static_cast<double>(peak_staging_bytes) / (1024.0 * 1024.0 * 1024.0),
                elapsed_sec, net_mbps, peak_rss,
                nan_inf_ok ? "true" : "false");
            std::fclose(mf);
        }
    }

    if (stats_out) {
        stats_out->num_shards_total           = static_cast<uint32_t>(shard_names.size());
        stats_out->num_shards_converted       = num_shards_converted;
        stats_out->num_shards_resumed         = num_shards_resumed;
        stats_out->num_text_tensors_converted = text_tensors_converted;
        stats_out->num_mtp_tensors_converted  = mtp_tensors_converted;
        stats_out->num_routed_experts_packed  = routed_experts_packed;
        stats_out->num_vision_tensors_skipped = vision_tensors_skipped;
        stats_out->hf_safetensors_bytes       = hf_bytes;
        stats_out->spine_bytes                = actual_spine;
        stats_out->experts_nvme_bytes         = actual_experts;
        stats_out->total_bundle_bytes         = actual_spine + actual_experts;
        stats_out->active_bytes_per_tok       = spine_hdr.active_bytes_per_tok;
        stats_out->elapsed_sec                = elapsed_sec;
        stats_out->network_mbps               = net_mbps;
        stats_out->peak_rss_gib               = peak_rss;
        stats_out->page_aligned_verified      = aligned_ok;
        stats_out->staging_clean_verified     = staging_clean;
        stats_out->nan_inf_checked            = nan_inf_ok;
    }

    std::fprintf(stdout,
                 "[glm53-stream] COMPLETE in %.2f s (net %.1f MB/s): spine=%.4f GiB + experts_nvme=%.4f GiB "
                 "= %.4f GiB | aligned=%d | staging_clean=%d | nan_inf_ok=%d | Peak RSS=%.4f GiB\n",
                 elapsed_sec, net_mbps,
                 static_cast<double>(actual_spine) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(actual_experts) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(actual_spine + actual_experts) / (1024.0 * 1024.0 * 1024.0),
                 aligned_ok ? 1 : 0, staging_clean ? 1 : 0, nan_inf_ok ? 1 : 0, peak_rss);
    std::fflush(stdout);
    return (aligned_ok && staging_clean && nan_inf_ok) ? 0 : -1;
}

int strata_glm53_convert_hf_directory(
    const char* hf_repo_or_dir,
    const char* out_bundle_dir,
    StrataGlm53ConvertStats* stats_out
) {
    return strata_glm53_convert_hf(hf_repo_or_dir, out_bundle_dir, 0u, 0u, stats_out);
}

int strata_glm53_convert_fast(
    const char*              hf_repo_or_dir,
    const char*              out_bundle_dir,
    uint32_t                 num_streams,
    uint32_t                 ols_trials,
    uint32_t                 keep_temp_shards,
    const char*              hf_token,
    StrataGlm53ConvertStats* stats_out
) {
    auto wall_t0 = std::chrono::steady_clock::now();
    if (stats_out) std::memset(stats_out, 0, sizeof(*stats_out));

    const std::string src_spec = (hf_repo_or_dir && hf_repo_or_dir[0] != '\0')
                                     ? std::string(hf_repo_or_dir)
                                     : std::string(STRATA_GLM53_DEFAULT_FAST_HF_REPO);
    const std::string out_dir  = (out_bundle_dir && out_bundle_dir[0] != '\0')
                                     ? std::string(out_bundle_dir)
                                     : std::string(STRATA_GLM53_DEFAULT_FAST_BUNDLE_DIR);
    const uint32_t eff_streams = (num_streams == 0u) ? 4u : std::min<uint32_t>(num_streams, 8u);
    const uint32_t eff_trials  = (ols_trials  == 0u) ? 3u : std::min<uint32_t>(ols_trials,  3u);

    if (!mkdirs(out_dir)) {
        std::fprintf(stderr, "[glm53-fast] ERROR: cannot create bundle dir '%s'\n", out_dir.c_str());
        return -1;
    }

    bool is_local_dir = false;
    if (is_directory_path(src_spec) && file_exists_nonempty(src_spec + "/config.json")) {
        if (file_exists_nonempty(src_spec + "/model.safetensors")) {
            is_local_dir = true;
        } else {
            std::string tmp_idx;
            if (read_whole_file(src_spec + "/model.safetensors.index.json", tmp_idx)) {
                SmallJson ij;
                if (parse_small_json(tmp_idx, ij)) {
                    if (const SmallJson* wm = ij.get("weight_map")) {
                        if (!wm->obj.empty() &&
                            is_safe_shard_basename(wm->obj.begin()->second.s_val) &&
                            file_exists_nonempty(src_spec + "/" + wm->obj.begin()->second.s_val)) {
                            is_local_dir = true;
                        }
                    }
                }
            }
        }
    }

    std::string base_url;
    if (!is_local_dir) {
        if (src_spec.rfind("http://", 0) == 0 || src_spec.rfind("https://", 0) == 0) {
            base_url = src_spec;
            while (!base_url.empty() && base_url.back() == '/') base_url.pop_back();
            if (base_url.find("/resolve/") == std::string::npos) {
                base_url += "/resolve/main";
            }
        } else {
            const char* ep = std::getenv("HF_ENDPOINT");
            std::string host = (ep && ep[0] != '\0') ? std::string(ep) : std::string("https://huggingface.co");
            while (!host.empty() && host.back() == '/') host.pop_back();
            base_url = host + "/" + src_spec + "/resolve/main";
        }
    }

    /* 1. Fetch or copy metadata JSONs + `tokenizer.json` + `chat_template.jinja` */
    if (is_local_dir) {
        if (!copy_file_if_exists(src_spec + "/config.json", out_dir + "/config.json", true)) {
            std::fprintf(stderr, "[glm53-fast] ERROR: missing config.json in '%s'\n", src_spec.c_str());
            return -1;
        }
        (void)copy_file_if_exists(src_spec + "/tokenizer.json", out_dir + "/tokenizer.json", false);
        (void)copy_file_if_exists(src_spec + "/tokenizer_config.json", out_dir + "/tokenizer_config.json", false);
        (void)copy_file_if_exists(src_spec + "/chat_template.jinja", out_dir + "/chat_template.jinja", false);
        (void)copy_file_if_exists(src_spec + "/generation_config.json", out_dir + "/generation_config.json", false);
        (void)copy_file_if_exists(src_spec + "/model.safetensors.index.json", out_dir + "/model.safetensors.index.json", false);
    } else {
        const char* req_meta[] = {"config.json", "tokenizer.json", "model.safetensors.index.json"};
        for (const char* mf : req_meta) {
            std::string dst = out_dir + "/" + mf;
            if (!file_exists_nonempty(dst)) {
                std::fprintf(stdout, "[glm53-fast] Downloading %s ...\n", mf);
                std::fflush(stdout);
                if (!curl_download_single(base_url + "/" + mf, dst, nullptr, nullptr, hf_token)) {
                    std::fprintf(stderr, "[glm53-fast] ERROR: failed to download '%s'\n", mf);
                    return -1;
                }
            }
        }
        const char* opt_meta[] = {"tokenizer_config.json", "chat_template.jinja", "generation_config.json"};
        for (const char* mf : opt_meta) {
            std::string dst = out_dir + "/" + mf;
            if (!file_exists_nonempty(dst)) {
                (void)curl_download_single(base_url + "/" + mf, dst, nullptr, nullptr, hf_token);
            }
        }
    }

    /* 2. Parse `config.json` & build deterministic v2 RAM-Resident layout */
    StrataGlm53Config cfg;
    if (!parse_config_into(out_dir + "/config.json", out_dir + "/generation_config.json", cfg)) {
        std::fprintf(stderr, "[glm53-fast] ERROR: failed to parse '%s/config.json'\n", out_dir.c_str());
        return -1;
    }
    /* Spine projections & MTP experts always use Q4_A gs=64 regardless of source FP8 quantization_config */
    cfg.quant_group_size = STRATA_GLM53_QUANT_GROUP_SIZE;
    cfg.quant_bits       = STRATA_GLM53_QUANT_BITS;

    auto spine_hdr_ptr = std::make_unique<StrataGlm53SpineHeader>();
    StrataGlm53SpineHeader& spine_hdr = *spine_hdr_ptr;
    strata_glm53_init_layout_v2(&cfg, &spine_hdr);

    FastTargetRegistry reg;
    build_fast_target_registry(spine_hdr, reg);

    /* 3. Parse `model.safetensors.index.json` inside inner scope so temporary DOM frees immediately */
    std::vector<std::string> shard_names;
    std::unordered_map<std::string, std::vector<std::string>> shard_raw_keys;
    uint64_t index_total_bytes = 0ULL;
    {
        std::string idx_raw;
        if (read_whole_file(out_dir + "/model.safetensors.index.json", idx_raw)) {
            SmallJson idx_json;
            if (!parse_small_json(idx_raw, idx_json)) return -1;
            if (const SmallJson* md = idx_json.get("metadata")) {
                index_total_bytes = static_cast<uint64_t>(md->get_f64("total_size", 0.0));
            }
            const SmallJson* wm = idx_json.get("weight_map");
            if (!wm || wm->type != SmallJson::J_OBJ) return -1;
            std::set<std::string> uniq;
            for (const auto& kv : wm->obj) {
                if (kv.second.type == SmallJson::J_STR) {
                    if (!is_safe_shard_basename(kv.second.s_val)) {
                        std::fprintf(stderr, "[glm53-stream] ERROR: unsafe shard filename '%s' in model.safetensors.index.json\n",
                                     kv.second.s_val.c_str());
                        return -1;
                    }
                    uniq.insert(kv.second.s_val);
                    shard_raw_keys[kv.second.s_val].push_back(kv.first);
                }
            }
            shard_names.assign(uniq.begin(), uniq.end());
        } else if (is_local_dir && file_exists_nonempty(src_spec + "/model.safetensors")) {
            shard_names.push_back("model.safetensors");
        } else {
            std::fprintf(stderr, "[glm53-fast] ERROR: missing model.safetensors.index.json\n");
            return -1;
        }
    }

    /* 4. Open or resume `spine.strata`, `experts_uma_part0.strata`, `experts_uma_part1.strata`, `mtp_experts.strata` */
    const std::string spine_path = out_dir + "/" + STRATA_GLM53_SPINE_FILENAME;
    const std::string part0_path = out_dir + "/" + STRATA_GLM53_EXPERTS_PART0_FILENAME;
    const std::string part1_path = out_dir + "/" + STRATA_GLM53_EXPERTS_PART1_FILENAME;
    const std::string mtp_path   = out_dir + "/" + STRATA_GLM53_MTP_EXPERTS_FILENAME;
    const std::string done_path  = out_dir + "/.convert_done_shards";
    const std::string stage_dir  = out_dir + "/.tmp_shards";

    const bool has_part0 = (spine_hdr.part0_bytes > 0ULL);
    const bool has_part1 = (spine_hdr.part1_bytes > 0ULL);
    const bool has_mtp   = (spine_hdr.mtp_experts_bytes > 0ULL);

    std::set<std::string> done_shards;
    bool can_resume = false;
    {
        uint64_t sz_s = 0, sz_p0 = 0, sz_p1 = 0, sz_m = 0;
        if (file_exists_nonempty(done_path) &&
            file_exists_nonempty(spine_path, &sz_s) && sz_s == spine_hdr.total_spine_bytes &&
            (!has_part0 || (file_exists_nonempty(part0_path, &sz_p0) && sz_p0 == spine_hdr.part0_bytes)) &&
            (!has_part1 || (file_exists_nonempty(part1_path, &sz_p1) && sz_p1 == spine_hdr.part1_bytes)) &&
            (!has_mtp   || (file_exists_nonempty(mtp_path,   &sz_m)  && sz_m  == spine_hdr.mtp_experts_bytes))) {
            std::string done_txt;
            if (read_whole_file(done_path, done_txt)) {
                can_resume = true;
                size_t p = 0;
                while (p < done_txt.size()) {
                    size_t nl = done_txt.find('\n', p);
                    std::string line = (nl == std::string::npos) ? done_txt.substr(p) : done_txt.substr(p, nl - p);
                    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
                    if (!line.empty()) done_shards.insert(line);
                    if (nl == std::string::npos) break;
                    p = nl + 1;
                }
            }
        }
    }

    const int open_flags = can_resume ? O_RDWR : (O_CREAT | O_TRUNC | O_RDWR);
    int sfd_out = open(spine_path.c_str(), open_flags, 0644);
    int p0_out  = has_part0 ? open(part0_path.c_str(), open_flags, 0644) : -1;
    int p1_out  = has_part1 ? open(part1_path.c_str(), open_flags, 0644) : -1;
    int mtp_out = has_mtp   ? open(mtp_path.c_str(),   open_flags, 0644) : -1;

    auto close_outputs = [&]() {
        if (sfd_out >= 0) { close(sfd_out); sfd_out = -1; }
        if (p0_out  >= 0) { close(p0_out);  p0_out  = -1; }
        if (p1_out  >= 0) { close(p1_out);  p1_out  = -1; }
        if (mtp_out >= 0) { close(mtp_out); mtp_out = -1; }
    };

    if (sfd_out < 0 || (has_part0 && p0_out < 0) || (has_part1 && p1_out < 0) || (has_mtp && mtp_out < 0)) {
        close_outputs();
        std::fprintf(stderr, "[glm53-fast] ERROR: failed to open v2 output .strata files\n");
        return -1;
    }

#if defined(F_NOCACHE)
    (void)fcntl(sfd_out, F_NOCACHE, 1);
    if (p0_out  >= 0) (void)fcntl(p0_out,  F_NOCACHE, 1);
    if (p1_out  >= 0) (void)fcntl(p1_out,  F_NOCACHE, 1);
    if (mtp_out >= 0) (void)fcntl(mtp_out, F_NOCACHE, 1);
#endif

    if (!can_resume) {
        done_shards.clear();
        try_preallocate_apfs(sfd_out, spine_hdr.total_spine_bytes);
        if (p0_out  >= 0) try_preallocate_apfs(p0_out,  spine_hdr.part0_bytes);
        if (p1_out  >= 0) try_preallocate_apfs(p1_out,  spine_hdr.part1_bytes);
        if (mtp_out >= 0) try_preallocate_apfs(mtp_out, spine_hdr.mtp_experts_bytes);

        if (ftruncate(sfd_out, static_cast<off_t>(spine_hdr.total_spine_bytes)) != 0 ||
            (p0_out  >= 0 && ftruncate(p0_out,  static_cast<off_t>(spine_hdr.part0_bytes))       != 0) ||
            (p1_out  >= 0 && ftruncate(p1_out,  static_cast<off_t>(spine_hdr.part1_bytes))       != 0) ||
            (mtp_out >= 0 && ftruncate(mtp_out, static_cast<off_t>(spine_hdr.mtp_experts_bytes)) != 0)) {
            close_outputs();
            std::fprintf(stderr, "[glm53-fast] ERROR: ftruncate failed on v2 .strata files\n");
            return -1;
        }
        if (!pwrite_zeros(sfd_out, 0, STRATA_GLM53_HEADER_PADDED_BYTES) ||
            !pwrite_exact(sfd_out, &spine_hdr, sizeof(spine_hdr), 0)) {
            close_outputs();
            return -1;
        }
        int dfd = open(done_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (dfd >= 0) close(dfd);
    }

    std::vector<uint8_t> target_seen(reg.specs.size(), 0u);
    std::vector<uint8_t> expert_mask(spine_hdr.total_routed_experts, 0u);
    uint32_t num_shards_resumed       = 0;
    uint32_t num_shards_converted     = 0;
    uint32_t text_tensors_converted   = 0;
    uint32_t mtp_tensors_converted    = 0;
    uint32_t routed_experts_packed    = 0;
    uint32_t vision_tensors_skipped   = 0;
    uint64_t total_hf_bytes_processed = 0;

    auto mark_fast_completed = [&](const FastTargetSpec& ts) {
        if (!target_seen[ts.id]) {
            target_seen[ts.id] = 1u;
            if (ts.is_mtp) ++mtp_tensors_converted;
            else           ++text_tensors_converted;
        }
        if (ts.global_exp_idx >= 0 && static_cast<uint32_t>(ts.global_exp_idx) < expert_mask.size()) {
            const uint8_t prev = expert_mask[ts.global_exp_idx];
            const uint8_t next = static_cast<uint8_t>(prev | (1u << ts.exp_proj_slot));
            expert_mask[ts.global_exp_idx] = next;
            if (prev != 0x07u && next == 0x07u) {
                ++routed_experts_packed;
            }
        }
    };

    std::vector<uint32_t> pending_shard_indices;
    for (uint32_t s = 0; s < shard_names.size(); ++s) {
        const std::string& sname = shard_names[s];
        if (done_shards.count(sname) > 0) {
            ++num_shards_resumed;
            auto it_k = shard_raw_keys.find(sname);
            if (it_k != shard_raw_keys.end()) {
                for (const std::string& rk : it_k->second) {
                    if (is_vision_tensor_name(rk)) {
                        ++vision_tensors_skipped;
                    } else {
                        std::string cn = canonicalize_tensor_name(rk);
                        if (is_scale_inv_name(cn, nullptr)) continue;
                        auto it_t = reg.name_to_id.find(cn);
                        if (it_t != reg.name_to_id.end()) {
                            mark_fast_completed(reg.specs[it_t->second]);
                        }
                    }
                }
            }
        } else {
            pending_shard_indices.push_back(s);
        }
    }

    const uint64_t planned_total =
        spine_hdr.total_spine_bytes + spine_hdr.part0_bytes +
        spine_hdr.part1_bytes + spine_hdr.mtp_experts_bytes;

    NeonWorkerPool pool(0u);
    FastShardConverter converter(sfd_out, p0_out, p1_out, mtp_out, eff_trials, &spine_hdr, &pool);

    std::fprintf(stdout,
                 "[glm53-fast] v2 RAM-Resident E8S-18-V4 Layout: spine=%.3f GiB + part0=%.3f GiB + part1=%.3f GiB + "
                 "mtp=%.3f GiB = %.3f GiB | streams=%u trials=%u threads=%u | Shards=%zu (%u resumed, %zu pending)\n",
                 static_cast<double>(spine_hdr.total_spine_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(spine_hdr.part0_bytes)       / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(spine_hdr.part1_bytes)       / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(spine_hdr.mtp_experts_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(planned_total)               / (1024.0 * 1024.0 * 1024.0),
                 eff_streams, eff_trials, pool.num_threads(),
                 shard_names.size(), num_shards_resumed, pending_shard_indices.size());
    std::fflush(stdout);

    int done_fd = open(done_path.c_str(), O_CREAT | O_APPEND | O_WRONLY, 0644);
    uint64_t total_downloaded_bytes = 0ULL;
    uint64_t peak_staging_bytes     = 0ULL;
    auto dl_phase_t0 = std::chrono::steady_clock::now();
    bool ok = true;

    auto process_one_shard = [&](uint32_t shard_idx, const std::string& spath, bool delete_after) -> bool {
        auto st0 = std::chrono::steady_clock::now();
        if (delete_after) {
            peak_staging_bytes = std::max(peak_staging_bytes, measure_dir_bytes(stage_dir));
        }

        int sfd = open(spath.c_str(), O_RDONLY);
        if (sfd < 0) {
            std::fprintf(stderr, "[glm53-fast] ERROR: cannot open shard '%s'\n", spath.c_str());
            return false;
        }
#if defined(F_NOCACHE)
        (void)fcntl(sfd, F_NOCACHE, 1);
#endif
        struct stat st;
        if (fstat(sfd, &st) != 0 || st.st_size <= 8) {
            close(sfd);
            return false;
        }
        const uint64_t sbytes = static_cast<uint64_t>(st.st_size);
        total_hf_bytes_processed += sbytes;

        std::vector<SafetensorEntry> entries;
        uint32_t vis_skip = 0;
        if (!parse_safetensors_shard_entries(sfd, sbytes, entries, vis_skip)) {
            close(sfd);
            std::fprintf(stderr, "[glm53-fast] ERROR: corrupt safetensors header in '%s'\n", spath.c_str());
            return false;
        }
        vision_tensors_skipped += vis_skip;

        std::sort(entries.begin(), entries.end(),
                  [](const SafetensorEntry& a, const SafetensorEntry& b) {
                      return a.abs_offset < b.abs_offset;
                  });

        /* Index all `.weight_scale_inv` entries in this shard and verify 100% key parity */
        std::unordered_map<std::string, const SafetensorEntry*> scale_map;
        scale_map.reserve(entries.size());
        for (const SafetensorEntry& ent : entries) {
            std::string base_w;
            if (is_scale_inv_name(ent.canon_name, &base_w)) {
                if (reg.name_to_id.find(base_w) == reg.name_to_id.end()) {
                    close(sfd);
                    std::fprintf(stderr, "[glm53-fast] ERROR: unexpected scale_inv '%s' in '%s'\n",
                                 ent.canon_name.c_str(), shard_names[shard_idx].c_str());
                    return false;
                }
                scale_map[base_w] = &ent;
            } else {
                if (reg.name_to_id.find(ent.canon_name) == reg.name_to_id.end()) {
                    close(sfd);
                    std::fprintf(stderr, "[glm53-fast] ERROR: unexpected tensor '%s' in '%s'\n",
                                 ent.canon_name.c_str(), shard_names[shard_idx].c_str());
                    return false;
                }
            }
        }

        for (const SafetensorEntry& ent : entries) {
            if (is_scale_inv_name(ent.canon_name, nullptr)) continue;
            const FastTargetSpec& ts = reg.specs[reg.name_to_id[ent.canon_name]];
            const SafetensorEntry* sc_ent = nullptr;
            auto it_sc = scale_map.find(ent.canon_name);
            if (it_sc != scale_map.end()) sc_ent = it_sc->second;

            bool conv_ok = false;
            switch (ts.kind) {
                case FAST_SPINE_F32:
                    conv_ok = converter.convert_spine_f32(sfd, ent, ts);
                    break;
                case FAST_SPINE_F16:
                    conv_ok = converter.convert_spine_f16(sfd, ent, ts);
                    break;
                case FAST_SPINE_Q4:
                    conv_ok = converter.convert_matrix_to_q4(sfd, sfd_out, ent, sc_ent, ts);
                    break;
                case FAST_MLA_KVB_BF16: {
                    const StrataGlm53LayerDesc& ld =
                        (ts.mla_layer_slot == spine_hdr.mtp.global_layer_idx)
                            ? spine_hdr.mtp.layer
                            : spine_hdr.layers[ts.mla_layer_slot];
                    conv_ok = converter.convert_mla_kv_b_bf16(sfd, ent, ld);
                    break;
                }
                case FAST_MAIN_EXPERT_E8:
                    conv_ok = converter.convert_main_expert_e8(sfd, ent, sc_ent, ts);
                    break;
                case FAST_MTP_EXPERT_Q4:
                    conv_ok = converter.convert_matrix_to_q4(sfd, mtp_out, ent, sc_ent, ts);
                    break;
            }

            if (!conv_ok) {
                close(sfd);
                return false;
            }
            mark_fast_completed(ts);
        }
        close(sfd);

        if (delete_after) {
            peak_staging_bytes = std::max(peak_staging_bytes, measure_dir_bytes(stage_dir));
            unlink(spath.c_str());
        }

        if (done_fd >= 0) {
            std::string rec = shard_names[shard_idx] + "\n";
            (void)write(done_fd, rec.data(), rec.size());
            (void)fsync(done_fd);
        }
        ++num_shards_converted;

        double conv_sec    = std::chrono::duration<double>(std::chrono::steady_clock::now() - st0).count();
        double wall_so_far = std::chrono::duration<double>(std::chrono::steady_clock::now() - dl_phase_t0).count();
        double net_mbps    = (wall_so_far > 1e-6 && total_downloaded_bytes > 0)
                                 ? (static_cast<double>(total_downloaded_bytes) / (1e6 * wall_so_far))
                                 : 0.0;
        double cur_snr_db  = (converter.total_e8_err_energy > 0.0)
                                 ? (10.0 * std::log10(converter.total_e8_sig_energy / converter.total_e8_err_energy))
                                 : 0.0;
        double cur_exp_ms  = (converter.total_e8_projs_done > 0)
                                 ? (3000.0 * converter.total_e8_quant_sec / static_cast<double>(converter.total_e8_projs_done))
                                 : 0.0;
        uint32_t done_all  = num_shards_resumed + num_shards_converted;
        std::fprintf(stdout,
                     "[glm53-fast] [%2u/%zu] %-34s (%5.2f GiB, %4zu keys) conv %.2fs | "
                     "E8 %.2f dB (%.1f ms/exp) | net %5.1f MB/s | tmp_pk %.2f GiB | RSS %.3f GiB\n",
                     done_all, shard_names.size(), shard_names[shard_idx].c_str(),
                     static_cast<double>(sbytes) / (1024.0 * 1024.0 * 1024.0),
                     entries.size(), conv_sec, cur_snr_db, cur_exp_ms, net_mbps,
                     static_cast<double>(peak_staging_bytes) / (1024.0 * 1024.0 * 1024.0),
                     current_peak_rss_gib());
        std::fflush(stdout);
        return true;
    };

    if (!pending_shard_indices.empty()) {
        if (is_local_dir) {
            for (uint32_t s_idx : pending_shard_indices) {
                if (!process_one_shard(s_idx, src_spec + "/" + shard_names[s_idx], false)) {
                    ok = false;
                    break;
                }
            }
        } else {
            if (!mkdirs(stage_dir)) {
                if (done_fd >= 0) close(done_fd);
                close_outputs();
                return -1;
            }
            clean_staging_part_files(stage_dir);

            /* 1 background downloader using `eff_streams` parallel HTTP range streams + `max_staged = 2`
             * (1 converting + 1 prefetching => peak temp SSD ~10 GiB << 20 GiB!) */
            constexpr uint32_t n_dl       = 1u;
            constexpr uint32_t max_staged = 2u;

            struct ReadyShard {
                uint32_t    shard_idx;
                std::string path;
                uint64_t    downloaded_bytes;
                bool        ok;
            };

            std::mutex              q_mu;
            std::condition_variable cv_stage;
            std::condition_variable cv_ready;
            std::deque<ReadyShard>  ready_q;
            uint32_t                staged_permits = max_staged;
            std::atomic<size_t>     next_job{0};
            std::atomic<bool>       abort_dl{false};
            uint32_t                active_dl = n_dl;

            auto dl_worker = [&]() {
                while (!abort_dl.load(std::memory_order_relaxed)) {
                    size_t j = next_job.fetch_add(1, std::memory_order_relaxed);
                    if (j >= pending_shard_indices.size()) break;
                    const uint32_t s_idx = pending_shard_indices[j];
                    const std::string& sname = shard_names[s_idx];
                    const std::string dst = stage_dir + "/" + sname;

                    {
                        std::unique_lock<std::mutex> lk(q_mu);
                        cv_stage.wait(lk, [&]() {
                            return abort_dl.load(std::memory_order_relaxed) || staged_permits > 0;
                        });
                        if (abort_dl.load(std::memory_order_relaxed)) break;
                        --staged_permits;
                    }

                    uint64_t dl_bytes = 0;
                    bool shard_ok = false;
                    uint64_t existing_sz = 0;
                    if (file_exists_nonempty(dst, &existing_sz)) {
                        int tfd = open(dst.c_str(), O_RDONLY);
                        if (tfd >= 0) {
                            std::vector<SafetensorEntry> tmp_e;
                            uint32_t tmp_v = 0;
                            if (parse_safetensors_shard_entries(tfd, existing_sz, tmp_e, tmp_v)) {
                                shard_ok = true;
                            }
                            close(tfd);
                        }
                        if (!shard_ok) unlink(dst.c_str());
                    }

                    if (!shard_ok) {
                        double dl_sec = 0.0;
                        shard_ok = curl_download_shard_fast(
                            base_url + "/" + sname, dst, &dl_bytes, &dl_sec, eff_streams, hf_token);
                    }

                    {
                        std::lock_guard<std::mutex> lk(q_mu);
                        if (!shard_ok) {
                            abort_dl.store(true, std::memory_order_relaxed);
                            ++staged_permits;
                            cv_stage.notify_all();
                        }
                        ready_q.push_back({s_idx, dst, dl_bytes, shard_ok});
                    }
                    cv_ready.notify_one();
                }

                std::lock_guard<std::mutex> lk(q_mu);
                if (--active_dl == 0) cv_ready.notify_all();
            };

            std::thread dl_thread(dl_worker);

            size_t consumed = 0;
            while (consumed < pending_shard_indices.size()) {
                ReadyShard item;
                {
                    std::unique_lock<std::mutex> lk(q_mu);
                    cv_ready.wait(lk, [&]() { return !ready_q.empty() || active_dl == 0; });
                    if (ready_q.empty()) {
                        ok = false;
                        break;
                    }
                    item = std::move(ready_q.front());
                    ready_q.pop_front();
                }

                if (!item.ok) {
                    std::fprintf(stderr, "[glm53-fast] ERROR: download failed for '%s'\n",
                                 shard_names[item.shard_idx].c_str());
                    ok = false;
                    abort_dl.store(true, std::memory_order_relaxed);
                    cv_stage.notify_all();
                    break;
                }

                total_downloaded_bytes += item.downloaded_bytes;
                bool conv_ok = process_one_shard(item.shard_idx, item.path, keep_temp_shards == 0u);

                {
                    std::lock_guard<std::mutex> lk(q_mu);
                    ++staged_permits;
                }
                cv_stage.notify_one();

                if (!conv_ok) {
                    ok = false;
                    abort_dl.store(true, std::memory_order_relaxed);
                    cv_stage.notify_all();
                    break;
                }
                ++consumed;
            }

            abort_dl.store(true, std::memory_order_relaxed);
            cv_stage.notify_all();
            if (dl_thread.joinable()) dl_thread.join();
        }
    }

    if (done_fd >= 0) close(done_fd);

    /* 5. Verify 100% tensor coverage across all expected Spine + Main E8 + MTP Q4 targets */
    if (ok) {
        for (size_t i = 0; i < reg.specs.size(); ++i) {
            if (!target_seen[i]) {
                std::string missing_name = "<unknown>";
                for (const auto& kv : reg.name_to_id) {
                    if (kv.second == i) { missing_name = kv.first; break; }
                }
                std::fprintf(stderr, "[glm53-fast] ERROR: missing expected tensor '%s' after all shards\n",
                             missing_name.c_str());
                ok = false;
                break;
            }
        }
    }

    /* 6. Re-stamp `StrataGlm53SpineHeader`, run full on-disk NaN/Inf sweep, and fsync all 4 files */
    bool nan_inf_ok = false;
    if (ok) {
        if (!pwrite_exact(sfd_out, &spine_hdr, sizeof(spine_hdr), 0)) {
            ok = false;
        } else {
            nan_inf_ok = converter.verify_spine_slices_on_disk();
            if (!nan_inf_ok) {
                std::fprintf(stderr, "[glm53-fast] ERROR: on-disk NaN/Inf verification failed\n");
                ok = false;
            }
        }
    }

    if (!ok) {
        close_outputs();
        return -1;
    }

    (void)fsync(sfd_out);
    if (p0_out  >= 0) (void)fsync(p0_out);
    if (p1_out  >= 0) (void)fsync(p1_out);
    if (mtp_out >= 0) (void)fsync(mtp_out);
    close_outputs();

    /* 7. Clean `.tmp_shards/` & verify exact 16 KiB alignments and file sizes */
    bool staging_clean = true;
    if (keep_temp_shards == 0u) {
        clean_staging_part_files(stage_dir);
        (void)rmdir(stage_dir.c_str());
        staging_clean = check_dir_has_no_files(stage_dir);
    }

    uint64_t actual_spine = 0ULL, actual_part0 = 0ULL, actual_part1 = 0ULL, actual_mtp = 0ULL;
    bool aligned_ok =
        file_exists_nonempty(spine_path, &actual_spine) &&
        actual_spine == spine_hdr.total_spine_bytes &&
        (actual_spine % STRATA_GLM53_PAGE_BYTES) == 0ULL &&
        (!has_part0 || (file_exists_nonempty(part0_path, &actual_part0) &&
                        actual_part0 == spine_hdr.part0_bytes &&
                        (actual_part0 % STRATA_GLM53_PAGE_BYTES) == 0ULL)) &&
        (!has_part1 || (file_exists_nonempty(part1_path, &actual_part1) &&
                        actual_part1 == spine_hdr.part1_bytes &&
                        (actual_part1 % STRATA_GLM53_PAGE_BYTES) == 0ULL)) &&
        (!has_mtp   || (file_exists_nonempty(mtp_path,   &actual_mtp)   &&
                        actual_mtp == spine_hdr.mtp_experts_bytes &&
                        (actual_mtp % STRATA_GLM53_PAGE_BYTES) == 0ULL));

    const uint64_t total_bundle_b = actual_spine + actual_part0 + actual_part1 + actual_mtp;
    const double elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_t0).count();
    const double dl_elapsed  = std::chrono::duration<double>(std::chrono::steady_clock::now() - dl_phase_t0).count();
    const double net_mbps    = (dl_elapsed > 1e-6 && total_downloaded_bytes > 0)
                                   ? (static_cast<double>(total_downloaded_bytes) / (1e6 * dl_elapsed))
                                   : 0.0;
    const double peak_rss    = current_peak_rss_gib();
    const uint64_t hf_bytes  = (index_total_bytes > 0) ? index_total_bytes : total_hf_bytes_processed;
    const double avg_snr_db  = (converter.total_e8_err_energy > 0.0)
                                   ? (10.0 * std::log10(converter.total_e8_sig_energy / converter.total_e8_err_energy))
                                   : 0.0;
    const double avg_exp_ms  = (converter.total_e8_projs_done > 0)
                                   ? (3000.0 * converter.total_e8_quant_sec / static_cast<double>(converter.total_e8_projs_done))
                                   : 0.0;

    /* 8. Write human-readable v2 `manifest.json` */
    {
        std::string mf_path = out_dir + "/manifest.json";
        FILE* mf = std::fopen(mf_path.c_str(), "wb");
        if (mf) {
            const std::string safe_src = format_manifest_source_json(src_spec, is_local_dir);
            std::fprintf(mf,
                "{\n"
                "  \"format\": \"STRATA_GLM53_V2_E8S18_V4\",\n"
                "  \"format_version\": 2,\n"
                "  \"is_ram_resident\": true,\n"
                "  \"main_expert_dtype\": \"E8S-18-V4\",\n"
                "  \"mtp_expert_dtype\": \"Q4_A\",\n"
                "  \"rht_block\": 64,\n"
                "  \"rht_seed\": \"0x5354524154414538\",\n"
                "  \"architecture\": \"Glm5NextForConditionalGeneration\",\n"
                "  \"source\": \"%s\",\n"
                "  \"num_hidden_layers\": %u,\n"
                "  \"num_nextn_predict_layers\": %u,\n"
                "  \"first_k_dense_replace\": %u,\n"
                "  \"num_kda_layers\": %u,\n"
                "  \"num_dsa_layers\": %u,\n"
                "  \"hidden_size\": %u,\n"
                "  \"intermediate_size\": %u,\n"
                "  \"n_routed_experts\": %u,\n"
                "  \"num_experts_per_tok\": %u,\n"
                "  \"moe_intermediate_size\": %u,\n"
                "  \"total_routed_experts\": %u,\n"
                "  \"e8_record_bytes\": %llu,\n"
                "  \"e8_proj_bytes\": %llu,\n"
                "  \"mtp_expert_record_bytes\": %llu,\n"
                "  \"vocab_size\": %u,\n"
                "  \"page_align_bytes\": %llu,\n"
                "  \"hf_safetensors_bytes\": %llu,\n"
                "  \"spine_bytes\": %llu,\n"
                "  \"part0_bytes\": %llu,\n"
                "  \"part1_bytes\": %llu,\n"
                "  \"mtp_experts_bytes\": %llu,\n"
                "  \"total_bundle_bytes\": %llu,\n"
                "  \"active_spine_bytes_per_tok\": %llu,\n"
                "  \"active_routed_bytes_per_tok\": %llu,\n"
                "  \"active_bytes_per_tok\": %llu,\n"
                "  \"mtp_active_bytes_per_draft\": %llu,\n"
                "  \"num_shards_total\": %zu,\n"
                "  \"num_text_tensors_converted\": %u,\n"
                "  \"num_mtp_tensors_converted\": %u,\n"
                "  \"num_routed_experts_packed\": %u,\n"
                "  \"num_vision_tensors_skipped\": %u,\n"
                "  \"ols_trials\": %u,\n"
                "  \"avg_e8_snr_db\": %.4f,\n"
                "  \"avg_expert_quant_ms\": %.2f,\n"
                "  \"peak_staging_gib\": %.4f,\n"
                "  \"elapsed_sec\": %.3f,\n"
                "  \"network_mbps\": %.2f,\n"
                "  \"peak_rss_gib\": %.4f,\n"
                "  \"nan_inf_checked\": %s\n"
                "}\n",
                safe_src.c_str(),
                cfg.num_hidden_layers, cfg.num_nextn_predict_layers, cfg.first_k_dense_replace,
                cfg.num_kda_layers, cfg.num_dsa_layers,
                cfg.hidden_size, cfg.intermediate_size,
                cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size,
                spine_hdr.total_routed_experts,
                static_cast<unsigned long long>(spine_hdr.e8_record_bytes),
                static_cast<unsigned long long>(spine_hdr.e8_proj_bytes),
                static_cast<unsigned long long>(spine_hdr.expert_record_bytes),
                cfg.vocab_size,
                static_cast<unsigned long long>(STRATA_GLM53_PAGE_BYTES),
                static_cast<unsigned long long>(hf_bytes),
                static_cast<unsigned long long>(actual_spine),
                static_cast<unsigned long long>(actual_part0),
                static_cast<unsigned long long>(actual_part1),
                static_cast<unsigned long long>(actual_mtp),
                static_cast<unsigned long long>(total_bundle_b),
                static_cast<unsigned long long>(spine_hdr.active_spine_bytes_per_tok),
                static_cast<unsigned long long>(spine_hdr.active_routed_bytes_per_tok),
                static_cast<unsigned long long>(spine_hdr.active_bytes_per_tok),
                static_cast<unsigned long long>(spine_hdr.mtp_active_bytes_per_draft),
                shard_names.size(),
                text_tensors_converted, mtp_tensors_converted,
                routed_experts_packed, vision_tensors_skipped,
                eff_trials, avg_snr_db, avg_exp_ms,
                static_cast<double>(peak_staging_bytes) / (1024.0 * 1024.0 * 1024.0),
                elapsed_sec, net_mbps, peak_rss,
                nan_inf_ok ? "true" : "false");
            std::fclose(mf);
        }
    }

    if (stats_out) {
        stats_out->num_shards_total           = static_cast<uint32_t>(shard_names.size());
        stats_out->num_shards_converted       = num_shards_converted;
        stats_out->num_shards_resumed         = num_shards_resumed;
        stats_out->num_text_tensors_converted = text_tensors_converted;
        stats_out->num_mtp_tensors_converted  = mtp_tensors_converted;
        stats_out->num_routed_experts_packed  = routed_experts_packed;
        stats_out->num_vision_tensors_skipped = vision_tensors_skipped;
        stats_out->hf_safetensors_bytes       = hf_bytes;
        stats_out->spine_bytes                = actual_spine;
        stats_out->experts_nvme_bytes         = 0ULL;
        stats_out->part0_bytes                = actual_part0;
        stats_out->part1_bytes                = actual_part1;
        stats_out->mtp_experts_bytes          = actual_mtp;
        stats_out->total_bundle_bytes         = total_bundle_b;
        stats_out->active_bytes_per_tok       = spine_hdr.active_bytes_per_tok;
        stats_out->elapsed_sec                = elapsed_sec;
        stats_out->network_mbps               = net_mbps;
        stats_out->peak_rss_gib               = peak_rss;
        stats_out->page_aligned_verified      = aligned_ok;
        stats_out->staging_clean_verified     = staging_clean;
        stats_out->nan_inf_checked            = nan_inf_ok;
        stats_out->avg_e8_snr_db              = avg_snr_db;
        stats_out->avg_expert_quant_ms        = avg_exp_ms;
    }

    std::fprintf(stdout,
                 "[glm53-fast] COMPLETE in %.2f s (net %.1f MB/s): spine=%.3f + part0=%.3f + part1=%.3f + mtp=%.3f "
                 "= %.3f GiB | E8 avg SNR=%.3f dB (%.1f ms/exp) | aligned=%d | clean=%d | nan_ok=%d | Peak RSS=%.4f GiB\n",
                 elapsed_sec, net_mbps,
                 static_cast<double>(actual_spine)   / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(actual_part0)   / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(actual_part1)   / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(actual_mtp)     / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(total_bundle_b) / (1024.0 * 1024.0 * 1024.0),
                 avg_snr_db, avg_exp_ms,
                 aligned_ok ? 1 : 0, staging_clean ? 1 : 0, nan_inf_ok ? 1 : 0, peak_rss);
    std::fflush(stdout);
    return (aligned_ok && staging_clean && nan_inf_ok) ? 0 : -1;
}

int strata_glm53_load_bundle(
    const char* bundle_dir,
    StrataGlm53Bundle* bundle_out
) {
    if (!bundle_out) return -1;
    std::memset(bundle_out, 0, sizeof(*bundle_out));
    bundle_out->spine_fd       = -1;
    bundle_out->part0_fd       = -1;
    bundle_out->part1_fd       = -1;
    bundle_out->mtp_experts_fd = -1;
    for (uint32_t i = 0; i < STRATA_GLM53_NUM_NVME_FDS; ++i) {
        bundle_out->experts_fds[i] = -1;
    }

    const char* dir = (bundle_dir && bundle_dir[0] != '\0')
                          ? bundle_dir
                          : STRATA_GLM53_DEFAULT_BUNDLE_DIR;
    std::snprintf(bundle_out->bundle_dir, sizeof(bundle_out->bundle_dir), "%s", dir);
    std::snprintf(bundle_out->spine_path, sizeof(bundle_out->spine_path), "%s/%s",
                  dir, STRATA_GLM53_SPINE_FILENAME);
    std::snprintf(bundle_out->experts_nvme_path, sizeof(bundle_out->experts_nvme_path), "%s/%s",
                  dir, STRATA_GLM53_EXPERTS_NVME_FILENAME);
    std::snprintf(bundle_out->experts_part0_path, sizeof(bundle_out->experts_part0_path), "%s/%s",
                  dir, STRATA_GLM53_EXPERTS_PART0_FILENAME);
    std::snprintf(bundle_out->experts_part1_path, sizeof(bundle_out->experts_part1_path), "%s/%s",
                  dir, STRATA_GLM53_EXPERTS_PART1_FILENAME);
    std::snprintf(bundle_out->mtp_experts_path, sizeof(bundle_out->mtp_experts_path), "%s/%s",
                  dir, STRATA_GLM53_MTP_EXPERTS_FILENAME);

    auto fail_cleanup = [&]() -> int {
        strata_glm53_free_bundle(bundle_out);
        return -1;
    };

    auto mmap_ro_aligned = [](const char* path, uint64_t expected_bytes, int* fd_out,
                              const uint8_t** base_out, size_t* size_out) -> bool {
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "[glm53-load] ERROR: cannot open '%s'\n", path);
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 ||
            static_cast<uint64_t>(st.st_size) != expected_bytes ||
            (static_cast<uint64_t>(st.st_size) % STRATA_GLM53_PAGE_BYTES) != 0ULL) {
            std::fprintf(stderr, "[glm53-load] ERROR: size/alignment mismatch on '%s' (got %lld, expected %llu)\n",
                         path, static_cast<long long>(st.st_size), static_cast<unsigned long long>(expected_bytes));
            close(fd);
            return false;
        }
        void* m = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) {
            close(fd);
            return false;
        }
        if ((reinterpret_cast<uintptr_t>(m) % STRATA_GLM53_PAGE_BYTES) != 0u) {
            munmap(m, static_cast<size_t>(st.st_size));
            close(fd);
            return false;
        }
        *fd_out   = fd;
        *base_out = static_cast<const uint8_t*>(m);
        *size_out = static_cast<size_t>(st.st_size);
        return true;
    };

    /* 1. Zero-copy 16 KiB page-aligned read-only `mmap` of `spine.strata` */
    bundle_out->spine_fd = open(bundle_out->spine_path, O_RDONLY);
    if (bundle_out->spine_fd < 0) {
        std::fprintf(stderr, "[glm53-load] ERROR: cannot open '%s'\n", bundle_out->spine_path);
        return fail_cleanup();
    }
    struct stat st_spine;
    if (fstat(bundle_out->spine_fd, &st_spine) != 0 ||
        static_cast<uint64_t>(st_spine.st_size) < STRATA_GLM53_HEADER_PADDED_BYTES ||
        (static_cast<uint64_t>(st_spine.st_size) % STRATA_GLM53_PAGE_BYTES) != 0ULL) {
        std::fprintf(stderr, "[glm53-load] ERROR: invalid/unaligned size on '%s'\n", bundle_out->spine_path);
        return fail_cleanup();
    }
    bundle_out->spine_bytes = static_cast<size_t>(st_spine.st_size);
    void* ms = mmap(nullptr, bundle_out->spine_bytes, PROT_READ, MAP_PRIVATE, bundle_out->spine_fd, 0);
    if (ms == MAP_FAILED) return fail_cleanup();
    bundle_out->spine_base = static_cast<const uint8_t*>(ms);
    if ((reinterpret_cast<uintptr_t>(bundle_out->spine_base) % STRATA_GLM53_PAGE_BYTES) != 0u) {
        return fail_cleanup();
    }
    bundle_out->header = reinterpret_cast<const StrataGlm53SpineHeader*>(bundle_out->spine_base);

    const StrataGlm53SpineHeader* hdr = bundle_out->header;
    auto validate_glm53_spine = [&]() -> bool {
        if (hdr->magic != STRATA_GLM53_SPINE_MAGIC_U64 ||
            hdr->header_padded_bytes != STRATA_GLM53_HEADER_PADDED_BYTES ||
            hdr->total_spine_bytes != bundle_out->spine_bytes ||
            (hdr->version != STRATA_GLM53_FORMAT_VERSION_V1 && hdr->version != STRATA_GLM53_FORMAT_VERSION_V2)) {
            return false;
        }
        const auto& c = hdr->cfg;
        if (c.num_hidden_layers == 0U || c.num_hidden_layers > STRATA_GLM53_MAX_LAYERS ||
            c.hidden_size == 0U || c.hidden_size > 32768U ||
            c.vocab_size == 0U || c.vocab_size > (1U << 22) ||
            c.num_attention_heads == 0U || c.num_attention_heads > 256U ||
            c.quant_group_size != STRATA_GLM53_QUANT_GROUP_SIZE ||
            c.hc_mult == 0U || c.hc_mult > 16U ||
            /* Round 7: `hc_sinkhorn_iters` is a per-thread GPU loop count (stock 20); an unbounded value hangs the GPU */
            c.hc_sinkhorn_iters > 64U ||
            c.index_kpool != STRATA_GLM53_KV_BLOCK_TOKENS ||
            c.n_routed_experts > 512U ||
            c.num_experts_per_tok > 16U ||
            c.num_experts_per_tok > c.n_routed_experts ||
            /* Round 7: routed experts with a zero top-k would launch empty MoE dispatches */
            (c.n_routed_experts > 0U && c.num_experts_per_tok == 0U) ||
            c.num_eos_token_ids > 4U) {
            return false;
        }
        const uint64_t fsz = bundle_out->spine_bytes;
        auto in_rng = [&](uint64_t off, uint64_t len) -> bool {
            return off >= STRATA_GLM53_HEADER_PADDED_BYTES && off <= fsz && len <= fsz - off;
        };
        auto chk_sl = [&](const StrataGlm53TensorSlice& s, uint32_t exp_dtype, uint64_t exp_rows, uint64_t exp_cols) -> bool {
            if (s.dtype != exp_dtype) return false;
            if (s.shape0 == 0U || static_cast<uint64_t>(s.shape0) != exp_rows) return false;
            const uint64_t s1 = (s.shape1 > 0U) ? static_cast<uint64_t>(s.shape1) : 1ULL;
            if (s1 != exp_cols) return false;
            const uint64_t bpe = (s.dtype == STRATA_GLM53_DTYPE_F32 || s.dtype == STRATA_GLM53_DTYPE_U32) ? 4ULL : 2ULL;
            if (static_cast<uint64_t>(s.shape0) > (fsz / bpe) / s1) return false;
            const uint64_t need = static_cast<uint64_t>(s.shape0) * s1 * bpe;
            return s.bytes >= need && need > 0ULL && in_rng(s.offset, s.bytes);
        };
        auto chk_q4 = [&](const StrataGlm53QuantMatrix& m, bool required, uint64_t exp_rows, uint64_t exp_cols) -> bool {
            if (!required && m.weight_bytes == 0ULL && m.scales_bytes == 0ULL && m.biases_bytes == 0ULL) return true;
            if (exp_rows == 0ULL || exp_cols == 0ULL || exp_rows > 0xFFFFFFFFULL || exp_cols > 0xFFFFFFFFULL) return false;
            if (m.group_size != STRATA_GLM53_QUANT_GROUP_SIZE) return false;
            if (m.cols == 0U || (m.cols % m.group_size) != 0U || (m.cols & 7U) != 0U) return false;
            if (static_cast<uint64_t>(m.rows_logical) != exp_rows || static_cast<uint64_t>(m.cols) != exp_cols) return false;
            const uint64_t exp_rp = (exp_rows + 63ULL) & ~63ULL;
            if (static_cast<uint64_t>(m.rows_padded) != exp_rp) return false;
            const uint64_t min_w = (static_cast<uint64_t>(m.rows_padded) * m.cols) / 2ULL;
            const uint64_t min_s = static_cast<uint64_t>(m.rows_padded) * (m.cols / m.group_size) * 2ULL;
            if (m.weight_bytes < min_w || m.scales_bytes < min_s || m.biases_bytes < min_s) return false;
            return in_rng(m.weight_off, m.weight_bytes) &&
                   in_rng(m.scales_off, m.scales_bytes) &&
                   in_rng(m.biases_off, m.biases_bytes);
        };
        auto chk_mhc = [&](const StrataGlm53MhcDesc& m) -> bool {
            const uint64_t hc     = c.hc_mult;
            const uint64_t mix_sz = (2ULL + hc) * hc;
            const uint64_t hc_dim = hc * c.hidden_size;
            return hc > 0ULL && hc <= 16ULL &&
                   chk_sl(m.fn, STRATA_GLM53_DTYPE_F32, mix_sz, hc_dim) &&
                   chk_sl(m.base, STRATA_GLM53_DTYPE_F32, mix_sz, 1ULL) &&
                   chk_sl(m.scale, STRATA_GLM53_DTYPE_F32, 3ULL, 1ULL);
        };
        const uint64_t d = c.hidden_size;
        auto chk_layer = [&](const StrataGlm53LayerDesc& ld, uint32_t exp_global_idx) -> bool {
            if (ld.global_layer_idx != exp_global_idx) return false;
            if (!in_rng(ld.layer_page_off, ld.layer_page_bytes)) return false;
            if (ld.has_mhc && (!chk_mhc(ld.hc_attn) || !chk_mhc(ld.hc_ffn))) return false;
            if (!chk_sl(ld.input_layernorm, STRATA_GLM53_DTYPE_F32, d, 1ULL) ||
                !chk_sl(ld.post_attention_layernorm, STRATA_GLM53_DTYPE_F32, d, 1ULL)) return false;
            if (ld.attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) {
                const uint64_t nh = c.linear_num_heads;
                const uint64_t hd = c.linear_head_dim;
                if (nh == 0ULL || hd == 0ULL || nh > 256ULL || hd > 1024ULL ||
                    c.linear_conv_kernel_dim == 0U || c.linear_conv_kernel_dim > 64U) return false;
                const uint64_t qkv_dim  = nh * hd;
                const uint64_t conv_dim = 3ULL * qkv_dim;
                const uint64_t fbg_dim  = 2ULL * hd + nh;
                if (!chk_q4(ld.lin_qkv_proj, true, conv_dim, d) || !chk_q4(ld.lin_fbg_a_proj, true, fbg_dim, d) ||
                    !chk_q4(ld.lin_f_b_proj, true, qkv_dim, hd) || !chk_q4(ld.lin_g_b_proj, true, qkv_dim, hd) ||
                    !chk_q4(ld.lin_o_proj, true, d, qkv_dim) ||
                    !chk_sl(ld.lin_qkv_conv1d, STRATA_GLM53_DTYPE_F32, conv_dim, c.linear_conv_kernel_dim) ||
                    !chk_sl(ld.lin_A_log, STRATA_GLM53_DTYPE_F32, nh, 1ULL) ||
                    !chk_sl(ld.lin_dt_bias, STRATA_GLM53_DTYPE_F32, qkv_dim, 1ULL) ||
                    !chk_sl(ld.lin_o_norm, STRATA_GLM53_DTYPE_F32, hd, 1ULL)) {
                    return false;
                }
            } else if (ld.attn_type == STRATA_GLM53_ATTN_DSA_MLA) {
                const uint64_t nh     = c.num_attention_heads;
                const uint64_t qr     = c.q_lora_rank;
                const uint64_t kvr    = c.kv_lora_rank;
                const uint64_t q_nope = c.qk_nope_head_dim;
                const uint64_t v_dim  = c.v_head_dim;
                const uint64_t idx_nh = c.index_n_heads;
                const uint64_t idx_hd = c.index_head_dim;
                if (qr == 0ULL || qr > 16384ULL || kvr == 0ULL || kvr > 16384ULL ||
                    q_nope == 0ULL || q_nope > 1024ULL || v_dim == 0ULL || v_dim > 1024ULL ||
                    idx_nh == 0ULL || idx_nh > 256ULL || idx_hd == 0ULL || idx_hd > 1024ULL ||
                    c.index_kpool != STRATA_GLM53_KV_BLOCK_TOKENS) {
                    return false;
                }
                const uint64_t q_tot = nh * q_nope;
                const uint64_t v_tot = nh * v_dim;
                if (!chk_q4(ld.mla_qkv_a_proj, true, qr + kvr, d) ||
                    !chk_sl(ld.mla_q_a_layernorm, STRATA_GLM53_DTYPE_F32, qr, 1ULL) ||
                    !chk_q4(ld.mla_q_b_proj, true, q_tot, qr) ||
                    !chk_sl(ld.mla_kv_a_layernorm, STRATA_GLM53_DTYPE_F32, kvr, 1ULL) ||
                    !chk_q4(ld.mla_kv_b_k, false, q_tot, kvr) || !chk_q4(ld.mla_kv_b_v, false, v_tot, kvr) ||
                    !chk_sl(ld.mla_w_uk, STRATA_GLM53_DTYPE_F16, nh * kvr, q_nope) ||
                    !chk_sl(ld.mla_w_uv, STRATA_GLM53_DTYPE_F16, v_tot, kvr) ||
                    !chk_q4(ld.mla_o_proj, true, d, v_tot) ||
                    !chk_q4(ld.indexer_wq_b, true, idx_nh * idx_hd, qr) || !chk_q4(ld.indexer_wk, true, idx_hd, d) ||
                    !chk_q4(ld.indexer_weights_proj, true, idx_nh, d) ||
                    !chk_sl(ld.indexer_k_norm_weight, STRATA_GLM53_DTYPE_F32, idx_hd, 1ULL) ||
                    !chk_sl(ld.indexer_k_norm_bias, STRATA_GLM53_DTYPE_F32, idx_hd, 1ULL) ||
                    !chk_sl(ld.indexer_compress_ape, STRATA_GLM53_DTYPE_F32, c.index_kpool, idx_hd) ||
                    !chk_sl(ld.indexer_compress_gate, STRATA_GLM53_DTYPE_F32, idx_hd, d)) {
                    return false;
                }
            } else {
                return false;
            }
            if (ld.mlp_type == STRATA_GLM53_MLP_DENSE) {
                if (c.intermediate_size == 0U || c.intermediate_size > 131072U) return false;
                if (!chk_q4(ld.dense_gate_up_proj, true, 2ULL * c.intermediate_size, d) ||
                    !chk_q4(ld.dense_down_proj, true, d, c.intermediate_size)) {
                    return false;
                }
            } else if (ld.mlp_type == STRATA_GLM53_MLP_SPARSE_MOE) {
                if (c.n_routed_experts == 0U || c.moe_intermediate_size == 0U || c.moe_intermediate_size > 65536U ||
                    c.n_shared_experts == 0U || c.n_shared_experts > 64U) return false;
                const uint64_t d_sh = static_cast<uint64_t>(c.moe_intermediate_size) * c.n_shared_experts;
                if (!chk_sl(ld.moe_router_gate, STRATA_GLM53_DTYPE_F16, c.n_routed_experts, d) ||
                    !chk_sl(ld.moe_e_score_bias, STRATA_GLM53_DTYPE_F32, c.n_routed_experts, 1ULL) ||
                    !chk_q4(ld.shared_gate_up_proj, true, 2ULL * d_sh, d) ||
                    !chk_q4(ld.shared_down_proj, true, d, d_sh)) {
                    return false;
                }
            } else {
                return false;
            }
            return true;
        };
        if (!in_rng(hdr->embed_page_off, hdr->embed_page_bytes) || !chk_q4(hdr->embed_tokens, true, c.vocab_size, d) ||
            !in_rng(hdr->head_page_off, hdr->head_page_bytes) ||
            !chk_sl(hdr->final_norm, STRATA_GLM53_DTYPE_F32, d, 1ULL) ||
            !chk_q4(hdr->lm_head, true, c.vocab_size, d)) {
            return false;
        }
        uint32_t n_kda = 0U, n_dsa = 0U;
        for (uint32_t l = 0U; l < c.num_hidden_layers; ++l) {
            if (hdr->layers[l].attn_type != c.layer_attn_types[l] ||
                hdr->layers[l].mlp_type != c.layer_mlp_types[l] ||
                !chk_layer(hdr->layers[l], l)) {
                return false;
            }
            if (hdr->layers[l].attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) ++n_kda;
            else ++n_dsa;
        }
        if (c.num_kda_layers != n_kda || c.num_dsa_layers != n_dsa) return false;
        if (hdr->mtp.enabled) {
            if (hdr->mtp.global_layer_idx != c.num_hidden_layers ||
                hdr->mtp.layer.has_mhc != 0U ||
                hdr->mtp.layer.attn_type != STRATA_GLM53_ATTN_DSA_MLA ||
                hdr->mtp.layer.mlp_type != STRATA_GLM53_MLP_SPARSE_MOE ||
                !in_rng(hdr->mtp.mtp_page_off, hdr->mtp.mtp_page_bytes) ||
                !chk_sl(hdr->mtp.enorm, STRATA_GLM53_DTYPE_F32, d, 1ULL) ||
                !chk_sl(hdr->mtp.hnorm, STRATA_GLM53_DTYPE_F32, d, 1ULL) ||
                !chk_q4(hdr->mtp.eh_proj, true, d, 2ULL * d) ||
                !chk_sl(hdr->mtp.shared_head_norm, STRATA_GLM53_DTYPE_F32, d, 1ULL) ||
                !chk_layer(hdr->mtp.layer, c.num_hidden_layers)) {
                return false;
            }
        }
        return true;
    };
    if (!validate_glm53_spine()) {
        std::fprintf(stderr, "[glm53-load] ERROR: spine header validation failed on '%s' (version=%u)\n",
                     bundle_out->spine_path, hdr->version);
        return fail_cleanup();
    }

    bundle_out->cfg                = hdr->cfg;
    bundle_out->experts_nvme_bytes = hdr->experts_nvme_bytes;

    uint32_t counted_main_moe = 0U;
    for (uint32_t l = 0U; l < hdr->cfg.num_hidden_layers; ++l) {
        if (hdr->layers[l].mlp_type == STRATA_GLM53_MLP_SPARSE_MOE) {
            if (hdr->layers[l].moe_layer_idx != static_cast<int32_t>(counted_main_moe)) {
                std::fprintf(stderr, "[glm53-load] ERROR: moe_layer_idx mismatch on '%s'\n", bundle_out->spine_path);
                return fail_cleanup();
            }
            ++counted_main_moe;
        } else if (hdr->layers[l].moe_layer_idx != -1) {
            std::fprintf(stderr, "[glm53-load] ERROR: dense layer has non-negative moe_layer_idx on '%s'\n",
                         bundle_out->spine_path);
            return fail_cleanup();
        }
    }
    const uint32_t counted_total_moe = counted_main_moe + (hdr->mtp.enabled ? 1U : 0U);
    if (hdr->num_main_moe_layers != counted_main_moe ||
        hdr->num_total_moe_layers != counted_total_moe ||
        (hdr->mtp.enabled && (hdr->mtp.moe_layer_idx != static_cast<int32_t>(counted_main_moe) ||
                              hdr->mtp.layer.moe_layer_idx != static_cast<int32_t>(counted_main_moe)))) {
        std::fprintf(stderr, "[glm53-load] ERROR: MoE layer count mismatch on '%s'\n", bundle_out->spine_path);
        return fail_cleanup();
    }
    const uint64_t exp_d     = hdr->cfg.hidden_size;
    const uint64_t exp_d_moe = hdr->cfg.moe_intermediate_size;
    const uint64_t exp_gs    = STRATA_GLM53_QUANT_GROUP_SIZE;
    const uint64_t exp_w_b   = sg53_align_up_u64((exp_d_moe * exp_d) / 2ULL, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t exp_s_b   = sg53_align_up_u64((exp_d_moe * (exp_d / exp_gs)) * 2ULL, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t exp_prj_b = exp_w_b + 2ULL * exp_s_b;
    const uint64_t exp_rec_b = sg53_align_up_u64(3ULL * exp_prj_b, STRATA_GLM53_NVME_STRIPE_BYTES);

    if (hdr->version == STRATA_GLM53_FORMAT_VERSION_V1) {
        /* v1 Regime II: validate NVMe expert geometry and open `N = 8` parallel `F_NOCACHE` FDs */
        const uint64_t exp_tot_experts = static_cast<uint64_t>(counted_total_moe) * hdr->cfg.n_routed_experts;
        if (hdr->total_routed_experts != exp_tot_experts ||
            (counted_total_moe > 0U &&
             (exp_d_moe == 0ULL || exp_d_moe > 65536ULL || (exp_d % exp_gs) != 0ULL ||
              hdr->expert_weight_bytes != exp_w_b ||
              hdr->expert_scales_bytes != exp_s_b ||
              hdr->expert_proj_bytes != exp_prj_b ||
              hdr->expert_record_bytes != exp_rec_b)) ||
            (hdr->expert_record_bytes % STRATA_GLM53_NVME_STRIPE_BYTES) != 0ULL ||
            (hdr->expert_record_bytes % STRATA_GLM53_PAGE_BYTES) != 0ULL ||
            hdr->experts_nvme_bytes != exp_tot_experts * hdr->expert_record_bytes) {
            std::fprintf(stderr, "[glm53-load] ERROR: v1 expert header validation failed on '%s'\n",
                         bundle_out->spine_path);
            return fail_cleanup();
        }
        bundle_out->is_ram_resident = false;
        if (bundle_out->experts_nvme_bytes > 0ULL) {
            for (uint32_t i = 0; i < STRATA_GLM53_NUM_NVME_FDS; ++i) {
                int efd = open(bundle_out->experts_nvme_path, O_RDONLY);
                if (efd < 0) {
                    std::fprintf(stderr, "[glm53-load] ERROR: cannot open '%s' (FD %u)\n",
                                 bundle_out->experts_nvme_path, i);
                    return fail_cleanup();
                }
#if defined(F_NOCACHE)
                if (fcntl(efd, F_NOCACHE, 1) != 0) {
                    close(efd);
                    std::fprintf(stderr, "[glm53-load] ERROR: fcntl(F_NOCACHE) failed on '%s'\n",
                                 bundle_out->experts_nvme_path);
                    return fail_cleanup();
                }
#endif
                if (i == 0u) {
                    struct stat st_exp;
                    if (fstat(efd, &st_exp) != 0 ||
                        static_cast<uint64_t>(st_exp.st_size) != bundle_out->experts_nvme_bytes) {
                        close(efd);
                        std::fprintf(stderr, "[glm53-load] ERROR: size mismatch on '%s'\n",
                                     bundle_out->experts_nvme_path);
                        return fail_cleanup();
                    }
                }
                bundle_out->experts_fds[i]  = efd;
                bundle_out->num_experts_fds = i + 1u;
            }
        }
    } else {
        /* v2 RAM-Resident: zero-copy `16 KiB`-aligned read-only `mmap` of `part0`, `part1`, `mtp_experts` */
        const uint64_t n_w          = exp_d_moe * exp_d;
        const uint64_t exp_e8_lo16  = sg53_align_up_u64((n_w / 8ULL) * 2ULL, STRATA_GLM53_TENSOR_ALIGN);
        const uint64_t exp_e8_hi2   = sg53_align_up_u64((n_w / 64ULL) * 2ULL, STRATA_GLM53_TENSOR_ALIGN);
        const uint64_t exp_e8_scale = sg53_align_up_u64((n_w / 64ULL) * 1ULL, STRATA_GLM53_TENSOR_ALIGN);
        const uint64_t exp_e8_proj  = exp_e8_lo16 + exp_e8_hi2 + exp_e8_scale;
        const uint64_t exp_e8_rec   = sg53_align_up_u64(3ULL * exp_e8_proj, STRATA_GLM53_PAGE_BYTES);
        const uint64_t exp_p0_bytes = sg53_align_up_u64(
            static_cast<uint64_t>(hdr->part0_moe_layers) * hdr->cfg.n_routed_experts * exp_e8_rec,
            STRATA_GLM53_PAGE_BYTES);
        const uint64_t exp_p1_bytes = sg53_align_up_u64(
            static_cast<uint64_t>(hdr->part1_moe_layers) * hdr->cfg.n_routed_experts * exp_e8_rec,
            STRATA_GLM53_PAGE_BYTES);
        const uint64_t exp_mtp_bytes = hdr->mtp.enabled
            ? sg53_align_up_u64(static_cast<uint64_t>(hdr->cfg.n_routed_experts) * exp_rec_b, STRATA_GLM53_PAGE_BYTES)
            : 0ULL;
        if (hdr->is_ram_resident != 1U ||
            hdr->experts_nvme_bytes != 0ULL ||
            hdr->part0_moe_layers > counted_main_moe ||
            hdr->part1_moe_layers != counted_main_moe - hdr->part0_moe_layers ||
            (counted_main_moe > 0U &&
             (exp_d_moe == 0ULL || exp_d_moe > 65536ULL || (n_w % 64ULL) != 0ULL ||
              hdr->e8_lo16_bytes != exp_e8_lo16 ||
              hdr->e8_hi2_bytes != exp_e8_hi2 ||
              hdr->e8_scale_bytes != exp_e8_scale ||
              hdr->e8_proj_bytes != exp_e8_proj ||
              hdr->e8_record_bytes != exp_e8_rec)) ||
            (hdr->mtp.enabled &&
             (exp_d_moe == 0ULL || exp_d_moe > 65536ULL || (exp_d % exp_gs) != 0ULL ||
              hdr->expert_weight_bytes != exp_w_b ||
              hdr->expert_scales_bytes != exp_s_b ||
              hdr->expert_proj_bytes != exp_prj_b ||
              hdr->expert_record_bytes != exp_rec_b)) ||
            hdr->part0_bytes != exp_p0_bytes ||
            hdr->part1_bytes != exp_p1_bytes ||
            hdr->mtp_experts_bytes != exp_mtp_bytes ||
            (hdr->e8_record_bytes % STRATA_GLM53_PAGE_BYTES) != 0ULL ||
            (hdr->part0_bytes % STRATA_GLM53_PAGE_BYTES) != 0ULL ||
            (hdr->part1_bytes % STRATA_GLM53_PAGE_BYTES) != 0ULL ||
            (hdr->mtp_experts_bytes % STRATA_GLM53_PAGE_BYTES) != 0ULL) {
            std::fprintf(stderr, "[glm53-load] ERROR: v2 RAM-resident header validation failed on '%s'\n",
                         bundle_out->spine_path);
            return fail_cleanup();
        }
        bundle_out->is_ram_resident = true;
        if (hdr->part0_bytes > 0ULL &&
            !mmap_ro_aligned(bundle_out->experts_part0_path, hdr->part0_bytes,
                             &bundle_out->part0_fd, &bundle_out->part0_base, &bundle_out->part0_bytes)) {
            return fail_cleanup();
        }
        if (hdr->part1_bytes > 0ULL &&
            !mmap_ro_aligned(bundle_out->experts_part1_path, hdr->part1_bytes,
                             &bundle_out->part1_fd, &bundle_out->part1_base, &bundle_out->part1_bytes)) {
            return fail_cleanup();
        }
        if (hdr->mtp_experts_bytes > 0ULL &&
            !mmap_ro_aligned(bundle_out->mtp_experts_path, hdr->mtp_experts_bytes,
                             &bundle_out->mtp_experts_fd, &bundle_out->mtp_experts_base, &bundle_out->mtp_experts_bytes)) {
            return fail_cleanup();
        }
    }

    /* 3. Load `<bundle_dir>/tokenizer.json` if present */
    std::string tok_path = std::string(dir) + "/tokenizer.json";
    if (file_exists_nonempty(tok_path)) {
        bundle_out->tokenizer = strata_glm53_tokenizer_load(tok_path.c_str());
        if (!bundle_out->tokenizer) {
            std::fprintf(stderr, "[glm53-load] ERROR: failed to parse '%s'\n", tok_path.c_str());
            return fail_cleanup();
        }
    }

    return 0;
}

void strata_glm53_free_bundle(StrataGlm53Bundle* bundle) {
    if (!bundle) return;
    if (bundle->tokenizer) {
        strata_glm53_tokenizer_free(bundle->tokenizer);
        bundle->tokenizer = nullptr;
    }
    for (uint32_t i = 0; i < STRATA_GLM53_NUM_NVME_FDS; ++i) {
        if (bundle->experts_fds[i] >= 0) {
            close(bundle->experts_fds[i]);
            bundle->experts_fds[i] = -1;
        }
    }
    if (bundle->mtp_experts_base && bundle->mtp_experts_bytes > 0) {
        munmap(const_cast<uint8_t*>(bundle->mtp_experts_base), bundle->mtp_experts_bytes);
    }
    if (bundle->mtp_experts_fd >= 0) {
        close(bundle->mtp_experts_fd);
    }
    if (bundle->part1_base && bundle->part1_bytes > 0) {
        munmap(const_cast<uint8_t*>(bundle->part1_base), bundle->part1_bytes);
    }
    if (bundle->part1_fd >= 0) {
        close(bundle->part1_fd);
    }
    if (bundle->part0_base && bundle->part0_bytes > 0) {
        munmap(const_cast<uint8_t*>(bundle->part0_base), bundle->part0_bytes);
    }
    if (bundle->part0_fd >= 0) {
        close(bundle->part0_fd);
    }
    if (bundle->spine_base && bundle->spine_bytes > 0) {
        munmap(const_cast<uint8_t*>(bundle->spine_base), bundle->spine_bytes);
    }
    if (bundle->spine_fd >= 0) {
        close(bundle->spine_fd);
    }
    std::memset(bundle, 0, sizeof(*bundle));
    bundle->spine_fd       = -1;
    bundle->part0_fd       = -1;
    bundle->part1_fd       = -1;
    bundle->mtp_experts_fd = -1;
    for (uint32_t i = 0; i < STRATA_GLM53_NUM_NVME_FDS; ++i) {
        bundle->experts_fds[i] = -1;
    }
}

} /* extern "C" */
