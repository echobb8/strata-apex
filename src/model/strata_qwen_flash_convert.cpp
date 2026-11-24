/*
 * STRATA-M5 v2.0 — Qwen/Qwen3.8-Flash-Next (180B: 127B 512-Expert MoE + 51.2B PLE + Native MTP)
 * High-Speed Pipelined Multi-Shard Downloader, Multi-Threaded ARM NEON Streaming Converter,
 * Immediate Shard Deleter, Two-Part Zero-Copy UMA + F_NOCACHE NVMe PLE Loader & BPE Tokenizer
 *
 * File: src/model/strata_qwen_flash_convert.cpp
 *
 * Implements:
 *   1. Bounded producer-consumer Hugging Face downloader (`curl -sSL --fail --retry 8` into `.staging/`
 *      with strict semaphore `<= 4` raw `.safetensors` shards on disk at once) + crash-safe resume
 *      via `.convert_done_shards` + immediate `unlink` after per-shard conversion.
 *   2. Multi-threaded ARM NEON `BF16 -> Q4_affine (gs=64) / IEEE-FP16 / IEEE-FP32` streaming converter
 *      (`F_NOCACHE` chunked `pread`/`pwrite`, `< 250 MiB` Peak RSS across 335.28 GiB BF16 master shards,
 *      exact `+1.0f` RMSNorm folding on all `Qwen4ExpTextRMSNorm` / `GemmaRMSNorm` weights while
 *      preserving `linear_attn.norm.weight` `RMSNormGated` without `+1.0f`).
 *   3. Deterministic `pwrite` into 16 KiB page-aligned `uma_part0.strata`, `uma_part1.strata`, and
 *      `ple_ngram.strata` (`92 B/row` compact Q4 N-gram store across 128 shards in any arrival order).
 *   4. Zero-copy `mmap` two-part UMA + `F_NOCACHE` NVMe PLE + `mlock` 16-bank 4-way LFU cache loader.
 *   5. Exact `248,320`-vocab Qwen Byte-Level BPE tokenizer & `Qwen3.8-Flash-Next` ChatML formatter.
 */

#include "../../include/strata_qwen_flash.h"

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
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

extern char** environ;

namespace {

/* ============================================================================
 * 1. Low-Level Conversion, NEON Quantization & Filesystem Utilities
 * ============================================================================ */

static inline float bf16_to_f32_scalar(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

static inline float fp16_to_f32_fast(uint16_t h) {
#if defined(__aarch64__)
    float16x4_t v = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(v), 0);
#else
    return sqf_f16_to_f32(h);
#endif
}

static inline uint16_t f32_to_fp16_fast(float f) {
    if (f > 65504.0f) f = 65504.0f;
    else if (f < -65504.0f) f = -65504.0f;
#if defined(__aarch64__)
    float16x4_t h = vcvt_f16_f32(vdupq_n_f32(f));
    return vget_lane_u16(vreinterpret_u16_f16(h), 0);
#else
    return sqf_f32_to_f16(f);
#endif
}

static void convert_bf16_to_fp16_span(const uint16_t* src_bf16, uint16_t* dst_fp16, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    const float32x4_t v_max = vdupq_n_f32(65504.0f);
    const float32x4_t v_min = vdupq_n_f32(-65504.0f);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t bf = vld1q_u16(src_bf16 + i);
        float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
        float32x4_t hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
        lo = vminq_f32(v_max, vmaxq_f32(v_min, lo));
        hi = vminq_f32(v_max, vmaxq_f32(v_min, hi));
        float16x8_t h8 = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
        vst1q_u16(dst_fp16 + i, vreinterpretq_u16_f16(h8));
    }
#endif
    for (; i < n; ++i) {
        dst_fp16[i] = f32_to_fp16_fast(bf16_to_f32_scalar(src_bf16[i]));
    }
}

static void convert_f32_to_fp16_span(const float* src_f32, uint16_t* dst_fp16, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    const float32x4_t v_max = vdupq_n_f32(65504.0f);
    const float32x4_t v_min = vdupq_n_f32(-65504.0f);
    for (; i + 8 <= n; i += 8) {
        float32x4_t lo = vminq_f32(v_max, vmaxq_f32(v_min, vld1q_f32(src_f32 + i)));
        float32x4_t hi = vminq_f32(v_max, vmaxq_f32(v_min, vld1q_f32(src_f32 + i + 4)));
        float16x8_t h8 = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
        vst1q_u16(dst_fp16 + i, vreinterpretq_u16_f16(h8));
    }
#endif
    for (; i < n; ++i) {
        dst_fp16[i] = f32_to_fp16_fast(src_f32[i]);
    }
}

/*
 * ARM NEON 4-bit Affine Group-64 Quantizer (`BF16 -> packed U32 nibbles + FP16 scale + FP16 bias`):
 * Rounds `scale` & `bias` to IEEE FP16 BEFORE computing `round((w - f16_to_f32(bias_h)) / f16_to_f32(scale_h))`
 * so runtime dequantization `nibble * float(scale_h) + float(bias_h)` achieves optimal SQNR.
 */
static inline void quantize_group64_bf16(
    const uint16_t* src_bf16,
    uint32_t*       dst_u32,    /* 8 x uint32_t (32 bytes) */
    uint16_t*       dst_scale,  /* 1 x FP16 */
    uint16_t*       dst_bias    /* 1 x FP16 */
) {
#if defined(__aarch64__)
    float32x4_t v[16];
    for (int k = 0; k < 8; ++k) {
        uint16x8_t bf = vld1q_u16(src_bf16 + k * 8);
        v[2 * k]     = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
        v[2 * k + 1] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
    }

    float32x4_t vmin4 = vminq_f32(vminq_f32(v[0], v[1]), vminq_f32(v[2], v[3]));
    float32x4_t vmax4 = vmaxq_f32(vmaxq_f32(v[0], v[1]), vmaxq_f32(v[2], v[3]));
    for (int k = 4; k < 16; k += 4) {
        float32x4_t mn = vminq_f32(vminq_f32(v[k], v[k + 1]), vminq_f32(v[k + 2], v[k + 3]));
        float32x4_t mx = vmaxq_f32(vmaxq_f32(v[k], v[k + 1]), vmaxq_f32(v[k + 2], v[k + 3]));
        vmin4 = vminq_f32(vmin4, mn);
        vmax4 = vmaxq_f32(vmax4, mx);
    }
    const float w_min = vminvq_f32(vmin4);
    const float w_max = vmaxvq_f32(vmax4);

    const uint16_t bias_h  = f32_to_fp16_fast(w_min);
    const uint16_t scale_h = f32_to_fp16_fast((w_max - w_min) * (1.0f / 15.0f));
    *dst_scale = scale_h;
    *dst_bias  = bias_h;

    const float scale_eff = fp16_to_f32_fast(scale_h);
    const float bias_eff  = fp16_to_f32_fast(bias_h);
    if (!(scale_eff > 0.0f)) {
        std::memset(dst_u32, 0, 8 * sizeof(uint32_t));
        return;
    }

    const float32x4_t v_bias = vdupq_n_f32(bias_eff);
    const float32x4_t v_invs = vdupq_n_f32(1.0f / scale_eff);
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_15   = vdupq_n_f32(15.0f);
    const int32x4_t   sh_lo  = {0, 4, 8, 12};
    const int32x4_t   sh_hi  = {16, 20, 24, 28};

    for (int u = 0; u < 8; ++u) {
        float32x4_t f0 = vrndnq_f32(vmulq_f32(vsubq_f32(v[2 * u],     v_bias), v_invs));
        float32x4_t f1 = vrndnq_f32(vmulq_f32(vsubq_f32(v[2 * u + 1], v_bias), v_invs));
        f0 = vminq_f32(v_15, vmaxq_f32(v_zero, f0));
        f1 = vminq_f32(v_15, vmaxq_f32(v_zero, f1));
        uint32x4_t q0 = vcvtq_u32_f32(f0);
        uint32x4_t q1 = vcvtq_u32_f32(f1);
        dst_u32[u] = vaddvq_u32(vshlq_u32(q0, sh_lo)) | vaddvq_u32(vshlq_u32(q1, sh_hi));
    }
#else
    float vals[64];
    float w_min = bf16_to_f32_scalar(src_bf16[0]);
    float w_max = w_min;
    vals[0] = w_min;
    for (int i = 1; i < 64; ++i) {
        float x = bf16_to_f32_scalar(src_bf16[i]);
        vals[i] = x;
        if (x < w_min) w_min = x;
        if (x > w_max) w_max = x;
    }
    const uint16_t bias_h  = f32_to_fp16_fast(w_min);
    const uint16_t scale_h = f32_to_fp16_fast((w_max - w_min) * (1.0f / 15.0f));
    *dst_scale = scale_h;
    *dst_bias  = bias_h;
    const float scale_eff = fp16_to_f32_fast(scale_h);
    const float bias_eff  = fp16_to_f32_fast(bias_h);
    if (!(scale_eff > 0.0f)) {
        std::memset(dst_u32, 0, 8 * sizeof(uint32_t));
        return;
    }
    const float inv_s = 1.0f / scale_eff;
    for (int u = 0; u < 8; ++u) {
        uint32_t packed = 0;
        for (int j = 0; j < 8; ++j) {
            float qf = std::nearbyintf((vals[u * 8 + j] - bias_eff) * inv_s);
            if (qf < 0.0f) qf = 0.0f;
            if (qf > 15.0f) qf = 15.0f;
            packed |= (static_cast<uint32_t>(qf) & 0xFu) << (4 * j);
        }
        dst_u32[u] = packed;
    }
#endif
}

/* General Q4 quantizer for variable group length `g_len` (multiple of 8, `<= 64`, e.g. PLE tail group of 32)
 * or non-BF16 float arrays. */
static inline void quantize_group_f32(
    const float* src_f32,
    uint32_t     g_len,      /* multiple of 8, <= 64 */
    uint32_t*    dst_u32,    /* g_len / 8 words */
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
    bool ok = (out.empty()) ? true : pread_exact(fd, out.data(), out.size(), 0);
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

static bool curl_download_atomic(
    const std::string& url,
    const std::string& dst_path,
    uint64_t*          bytes_out,
    double*            elapsed_out
) {
    auto t0 = std::chrono::steady_clock::now();
    std::string part_path = dst_path + ".part";
    const std::string auth_cfg = make_curl_auth_config(url, std::getenv("HF_TOKEN"));

    for (int outer_try = 0; outer_try < 6; ++outer_try) {
        if (outer_try == 0 || outer_try >= 3) {
            unlink(part_path.c_str());
        }
        std::vector<const char*> argv;
        argv.push_back("curl");
        argv.push_back("--globoff");
        argv.push_back("-sSL");
        argv.push_back("-C");
        argv.push_back("-");
        argv.push_back("--fail");
        argv.push_back("--retry");
        argv.push_back("8");
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
            if (rc == 0 && pid > 0) {
                size_t woff = 0U;
                while (woff < auth_cfg.size()) {
                    ssize_t wn = write(cfg_pipe[1], auth_cfg.data() + woff, auth_cfg.size() - woff);
                    if (wn < 0 && errno == EINTR) continue;
                    if (wn <= 0) break;
                    woff += static_cast<size_t>(wn);
                }
            }
            close(cfg_pipe[1]);
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
            std::this_thread::sleep_for(std::chrono::seconds(2 * (outer_try + 1)));
        }
    }
    unlink(part_path.c_str());
    return false;
}

/* ============================================================================
 * 2. Persistent Thread Pool for Multi-GB/s Parallel NEON Row Quantization
 * ============================================================================ */

class NeonWorkerPool {
public:
    explicit NeonWorkerPool(uint32_t num_threads) {
        if (num_threads == 0) {
            unsigned hc = std::thread::hardware_concurrency();
            num_threads = (hc > 0) ? std::min(hc, 18u) : 8u;
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
            fn_          = &fn;
            total_rows_  = total_rows;
            grain_       = std::max<uint64_t>(1ULL, grain);
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
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++p;
            } else {
                break;
            }
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
 * 4. Safetensors Header Parser & Deterministic Target Layout Mapper
 * ============================================================================ */

enum SafetensorDtype {
    ST_UNKNOWN = 0,
    ST_BF16    = 1,
    ST_F16     = 2,
    ST_F32     = 3,
    ST_I64     = 4,
    ST_U32     = 5
};

struct SafetensorEntry {
    std::string           canon_name;
    SafetensorDtype       dtype = ST_UNKNOWN;
    std::vector<uint64_t> shape;
    uint64_t              abs_offset = 0;
    uint64_t              byte_len   = 0;
};

static SafetensorDtype parse_st_dtype(const std::string& s) {
    if (s == "BF16") return ST_BF16;
    if (s == "F16")  return ST_F16;
    if (s == "F32")  return ST_F32;
    if (s == "I64")  return ST_I64;
    if (s == "U32")  return ST_U32;
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
    if (key.rfind("language_model.", 0) == 0) {
        std::string sub = key.substr(std::strlen("language_model."));
        if (sub.rfind("layers.", 0) == 0 || sub.rfind("embed_tokens.", 0) == 0 ||
            sub.rfind("hyper_connection_mixer.", 0) == 0) {
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

    /* Detect truncated `.safetensors` files where trailing tensor data is missing */
    if (max_end_seen > shard_file_bytes) return false;
    return true;
}

enum TargetKind {
    TGT_UMA_Q4              = 0,
    TGT_UMA_F16             = 1,
    TGT_UMA_F32             = 2,
    TGT_PLE_SHARD           = 3,
    TGT_PLE_CHK_MULTIPLIERS = 4,
    TGT_PLE_CHK_OFFSETS     = 5,
    TGT_PLE_CHK_VOCABS      = 6
};

struct TargetSpec {
    uint32_t                   id = 0;
    TargetKind                 kind = TGT_UMA_F32;
    uint32_t                   part_id = 0;      /* 0 -> uma_part0, 1 -> uma_part1 */
    bool                       add_one = false;  /* +1.0f RMSNorm folding rule */
    bool                       is_mtp  = false;
    StrataQwenFlashQuantMatrix qm{};
    StrataQwenFlashTensorSlice slice{};
    uint32_t                   ple_shard_idx = 0;
};

class TargetRegistry {
public:
    std::vector<TargetSpec>                   specs;
    std::unordered_map<std::string, uint32_t> name_to_id;

    void add_q4(const std::string& name, uint32_t part_id, const StrataQwenFlashQuantMatrix& qm, bool is_mtp) {
        TargetSpec s{};
        s.id      = static_cast<uint32_t>(specs.size());
        s.kind    = TGT_UMA_Q4;
        s.part_id = part_id;
        s.is_mtp  = is_mtp;
        s.qm      = qm;
        specs.push_back(s);
        name_to_id[name] = s.id;
    }

    void add_f16(const std::string& name, uint32_t part_id, const StrataQwenFlashTensorSlice& sl, bool is_mtp) {
        TargetSpec s{};
        s.id      = static_cast<uint32_t>(specs.size());
        s.kind    = TGT_UMA_F16;
        s.part_id = part_id;
        s.is_mtp  = is_mtp;
        s.slice   = sl;
        specs.push_back(s);
        name_to_id[name] = s.id;
    }

    void add_f32(const std::string& name, uint32_t part_id, const StrataQwenFlashTensorSlice& sl, bool add_one, bool is_mtp) {
        TargetSpec s{};
        s.id      = static_cast<uint32_t>(specs.size());
        s.kind    = TGT_UMA_F32;
        s.part_id = part_id;
        s.add_one = add_one;
        s.is_mtp  = is_mtp;
        s.slice   = sl;
        specs.push_back(s);
        name_to_id[name] = s.id;
    }

    void add_ple_shard(const std::string& name, uint32_t shard_idx) {
        TargetSpec s{};
        s.id            = static_cast<uint32_t>(specs.size());
        s.kind          = TGT_PLE_SHARD;
        s.ple_shard_idx = shard_idx;
        specs.push_back(s);
        name_to_id[name] = s.id;
    }

    void add_ple_check(const std::string& name, TargetKind chk_kind) {
        TargetSpec s{};
        s.id   = static_cast<uint32_t>(specs.size());
        s.kind = chk_kind;
        specs.push_back(s);
        name_to_id[name] = s.id;
    }
};

static void register_hc_targets(
    TargetRegistry& reg,
    const std::string& prefix,
    uint32_t part_id,
    const StrataQwenFlashHCDesc& hc,
    bool is_mtp
) {
    /* CRITICAL: `hc_norm.weight` is `Qwen4ExpTextRMSNorm` -> `add_one = true`! */
    reg.add_f32(prefix + "hc_norm.weight",               part_id, hc.hc_norm, true, is_mtp);
    reg.add_f16(prefix + "input_mix_weight_down.weight", part_id, hc.input_mix_weight_down, is_mtp);
    reg.add_f16(prefix + "input_mix_weight_up.weight",   part_id, hc.input_mix_weight_up, is_mtp);
    reg.add_f16(prefix + "block_inject_weight.weight",   part_id, hc.block_inject_weight, is_mtp);
}

static void register_mixer_targets(
    TargetRegistry& reg,
    const std::string& prefix,
    uint32_t part_id,
    const StrataQwenFlashMixerDesc& mx,
    bool is_mtp
) {
    /* CRITICAL: `hc_norm.weight` is `Qwen4ExpTextRMSNorm` -> `add_one = true`! */
    reg.add_f32(prefix + "hc_norm.weight",               part_id, mx.hc_norm, true, is_mtp);
    reg.add_f16(prefix + "input_mix_weight_down.weight", part_id, mx.input_mix_weight_down, is_mtp);
    reg.add_f16(prefix + "input_mix_weight_up.weight",   part_id, mx.input_mix_weight_up, is_mtp);
}

static void register_decoder_layer_targets(
    TargetRegistry& reg,
    const std::string& lp,
    const StrataQwenFlashLayerDesc& ld,
    const StrataQwenFlashPleHeader& ple_hdr,
    bool is_mtp
) {
    const uint32_t p = ld.part_id;
    if (ld.has_ple) {
        const std::string pp = lp + "ple.";
        reg.add_f16(pp + "key_proj.weight",   p, ld.ple.key_proj, is_mtp);
        reg.add_f16(pp + "value_proj.weight", p, ld.ple.value_proj, is_mtp);
        /* CRITICAL: `ple.norm_{key,query,conv}.weight` all fold `+1.0f`! */
        reg.add_f32(pp + "norm_key.weight",   p, ld.ple.norm_key,   true,  is_mtp);
        reg.add_f32(pp + "norm_query.weight", p, ld.ple.norm_query, true,  is_mtp);
        reg.add_f32(pp + "norm_conv.weight",  p, ld.ple.norm_conv,  true,  is_mtp);
        reg.add_f32(pp + "conv1d.weight",     p, ld.ple.conv1d_weight, false, is_mtp);

        const std::string ep = pp + "ple_embedding.";
        reg.add_ple_check(ep + "layer_multipliers",       TGT_PLE_CHK_MULTIPLIERS);
        reg.add_ple_check(ep + "ngram_heads_offsets",     TGT_PLE_CHK_OFFSETS);
        reg.add_ple_check(ep + "ngram_heads_vocab_sizes", TGT_PLE_CHK_VOCABS);
        for (uint32_t s = 0; s < ple_hdr.num_shards; ++s) {
            reg.add_ple_shard(ep + "ngram_embedding.shard_" + std::to_string(s) + ".weight", s);
        }
    }

    register_hc_targets(reg, lp + "attn_hyper_connection.", p, ld.attn_hc, is_mtp);
    register_hc_targets(reg, lp + "mlp_hyper_connection.",  p, ld.mlp_hc,  is_mtp);

    if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) {
        const std::string ap = lp + "linear_attn.";
        reg.add_q4(ap + "in_proj_qkv.weight", p, ld.lin_in_proj_qkv, is_mtp);
        reg.add_q4(ap + "in_proj_z.weight",   p, ld.lin_in_proj_z,   is_mtp);
        reg.add_q4(ap + "out_proj.weight",    p, ld.lin_out_proj,    is_mtp);
        reg.add_f16(ap + "in_proj_a.weight",  p, ld.lin_in_proj_a,   is_mtp);
        reg.add_f16(ap + "in_proj_b.weight",  p, ld.lin_in_proj_b,   is_mtp);
        reg.add_f32(ap + "conv1d.weight",     p, ld.lin_conv1d_weight, false, is_mtp);
        reg.add_f32(ap + "A_log",             p, ld.lin_A_log,         false, is_mtp);
        reg.add_f32(ap + "dt_bias",           p, ld.lin_dt_bias,       false, is_mtp);
        /* CRITICAL EXCEPTION: `linear_attn.norm.weight` is `RMSNormGated` -> NO `+1.0f`! */
        reg.add_f32(ap + "norm.weight",       p, ld.lin_norm,          false, is_mtp);
    } else {
        const std::string ap = lp + "self_attn.";
        reg.add_q4(ap + "q_proj.weight", p, ld.attn_q_proj, is_mtp);
        reg.add_q4(ap + "k_proj.weight", p, ld.attn_k_proj, is_mtp);
        reg.add_q4(ap + "v_proj.weight", p, ld.attn_v_proj, is_mtp);
        reg.add_q4(ap + "o_proj.weight", p, ld.attn_o_proj, is_mtp);
        /* CRITICAL: `q_norm`, `k_norm`, `indexer.{q,k}_layernorm` all fold `+1.0f`! */
        reg.add_f32(ap + "q_norm.weight",                  p, ld.attn_q_norm,     true, is_mtp);
        reg.add_f32(ap + "k_norm.weight",                  p, ld.attn_k_norm,     true, is_mtp);
        reg.add_f16(ap + "indexer.index_qk_proj.weight",   p, ld.indexer_qk_proj,       is_mtp);
        reg.add_f32(ap + "indexer.q_layernorm.weight",     p, ld.indexer_q_norm,  true, is_mtp);
        reg.add_f32(ap + "indexer.k_layernorm.weight",     p, ld.indexer_k_norm,  true, is_mtp);
    }

    const std::string mp = lp + "mlp.";
    reg.add_f16(mp + "gate.weight",                    p, ld.moe_router_gate,    is_mtp);
    reg.add_q4(mp + "experts.gate_up_proj",            p, ld.moe_experts_gate_up, is_mtp);
    reg.add_q4(mp + "experts.down_proj",               p, ld.moe_experts_down,    is_mtp);
    /* Alias fallback in case a synthetic checkpoint appends `.weight` to 3D expert tensors */
    reg.name_to_id[mp + "experts.gate_up_proj.weight"] = reg.name_to_id[mp + "experts.gate_up_proj"];
    reg.name_to_id[mp + "experts.down_proj.weight"]    = reg.name_to_id[mp + "experts.down_proj"];

    reg.add_f16(mp + "shared_expert.gate_proj.weight", p, ld.shared_gate_proj,   is_mtp);
    reg.add_f16(mp + "shared_expert.up_proj.weight",   p, ld.shared_up_proj,     is_mtp);
    reg.add_f16(mp + "shared_expert.down_proj.weight", p, ld.shared_down_proj,   is_mtp);
    reg.add_f32(mp + "shared_expert_gate.weight",      p, ld.shared_expert_gate, false, is_mtp);
}

static void build_target_registry(
    const StrataQwenFlashConfig&      cfg,
    const StrataQwenFlashPart0Header& p0,
    const StrataQwenFlashPart1Header& p1,
    const StrataQwenFlashPleHeader&   ple,
    TargetRegistry&                   reg
) {
    reg.specs.reserve(1400);
    reg.add_q4("model.embed_tokens.weight", 0u, p0.embed_tokens, false);

    for (uint32_t l = 0; l < p0.num_part0_layers; ++l) {
        register_decoder_layer_targets(
            reg, "model.layers." + std::to_string(l) + ".", p0.layers[l], ple, false);
    }
    for (uint32_t i = 0; i < p1.num_part1_layers; ++i) {
        uint32_t gl = p1.first_global_layer_idx + i;
        register_decoder_layer_targets(
            reg, "model.layers." + std::to_string(gl) + ".", p1.layers[i], ple, false);
    }

    register_mixer_targets(reg, "model.hyper_connection_mixer.", 1u, p1.final_mixer, false);
    if (!cfg.tie_word_embeddings) {
        reg.add_q4("lm_head.weight", 1u, p1.lm_head, false);
    }

    if (cfg.has_mtp && p1.mtp.enabled) {
        /* CRITICAL: `mtp.pre_fc_norm_embedding` & `mtp.pre_fc_norm_hidden` fold `+1.0f`! */
        reg.add_f32("mtp.pre_fc_norm_embedding.weight", 1u, p1.mtp.pre_fc_norm_embedding, true, true);
        reg.add_f32("mtp.pre_fc_norm_hidden.weight",    1u, p1.mtp.pre_fc_norm_hidden,    true, true);
        reg.add_f16("mtp.fc_embedding.weight",          1u, p1.mtp.fc_embedding,                true);
        reg.add_f16("mtp.fc_hidden.weight",             1u, p1.mtp.fc_hidden,                   true);
        register_decoder_layer_targets(reg, "mtp.layers.0.", p1.mtp.layer0, ple, true);
        register_mixer_targets(reg, "mtp.hyper_connection_mixer.", 1u, p1.mtp.mixer, true);
    }
}

/* ============================================================================
 * 5. Low-RSS Chunked NEON Shard Converter (`< 250 MiB` Peak RSS)
 * ============================================================================ */

struct ShardConverter {
    int                             part_fds[2] = {-1, -1};
    int                             ple_fd      = -1;
    const StrataQwenFlashPleHeader* ple_hdr     = nullptr;
    NeonWorkerPool*                 pool        = nullptr;

    /* Single reusable 104 MiB scratch arena across all 131 shards */
    std::vector<uint8_t>            in_buf;     /* 64 MiB */
    std::vector<uint8_t>            out_w_buf;  /* 32 MiB */
    std::vector<uint8_t>            out_s_buf;  /* 4 MiB  */
    std::vector<uint8_t>            out_b_buf;  /* 4 MiB  */

    ShardConverter(int fd0, int fd1, int pfd, const StrataQwenFlashPleHeader* ph, NeonWorkerPool* wp)
        : ple_fd(pfd), ple_hdr(ph), pool(wp) {
        part_fds[0] = fd0;
        part_fds[1] = fd1;
        in_buf.resize(64ULL * 1024ULL * 1024ULL);
        out_w_buf.resize(32ULL * 1024ULL * 1024ULL);
        out_s_buf.resize(4ULL * 1024ULL * 1024ULL);
        out_b_buf.resize(4ULL * 1024ULL * 1024ULL);
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

    bool convert_f32_target(int shard_fd, const SafetensorEntry& ent, const TargetSpec& spec) {
        const uint64_t elems = shape_elems(ent.shape);
        const uint64_t bpe   = dtype_bpe(ent.dtype);
        if (elems != spec.slice.elem_count || bpe == 0 || ent.byte_len != elems * bpe) {
            std::fprintf(stderr, "[flash-stream] ERROR: F32 shape/size mismatch on '%s' (got %llu, expected %u)\n",
                         ent.canon_name.c_str(), static_cast<unsigned long long>(elems), spec.slice.elem_count);
            return false;
        }
        if (ent.byte_len > in_buf.size() || spec.slice.bytes > out_w_buf.size()) {
            std::fprintf(stderr, "[flash-stream] ERROR: F32 slice '%s' exceeds scratch arena\n",
                         ent.canon_name.c_str());
            return false;
        }

        if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(ent.byte_len), ent.abs_offset)) {
            return false;
        }
        std::memset(out_w_buf.data(), 0, static_cast<size_t>(spec.slice.bytes));
        float* dst = reinterpret_cast<float*>(out_w_buf.data());
        const float bias = spec.add_one ? 1.0f : 0.0f;

        if (ent.dtype == ST_BF16) {
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
            for (uint64_t i = 0; i < elems; ++i) dst[i] = bf16_to_f32_scalar(s16[i]) + bias;
        } else if (ent.dtype == ST_F16) {
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
            for (uint64_t i = 0; i < elems; ++i) dst[i] = fp16_to_f32_fast(s16[i]) + bias;
        } else if (ent.dtype == ST_F32) {
            const float* sf = reinterpret_cast<const float*>(in_buf.data());
            for (uint64_t i = 0; i < elems; ++i) dst[i] = sf[i] + bias;
        } else {
            return false;
        }

        return pwrite_exact(part_fds[spec.part_id], out_w_buf.data(),
                            static_cast<size_t>(spec.slice.bytes), spec.slice.offset);
    }

    bool convert_f16_target(int shard_fd, const SafetensorEntry& ent, const TargetSpec& spec) {
        const uint64_t elems = shape_elems(ent.shape);
        const uint64_t bpe   = dtype_bpe(ent.dtype);
        if (elems != spec.slice.elem_count || bpe == 0 || ent.byte_len != elems * bpe) {
            std::fprintf(stderr, "[flash-stream] ERROR: F16 shape/size mismatch on '%s' (got %llu, expected %u)\n",
                         ent.canon_name.c_str(), static_cast<unsigned long long>(elems), spec.slice.elem_count);
            return false;
        }

        const int out_fd = part_fds[spec.part_id];
        uint64_t rem_elems = elems;
        uint64_t r_off     = ent.abs_offset;
        uint64_t w_off     = spec.slice.offset;
        const uint64_t max_batch = std::min<uint64_t>(in_buf.size() / bpe, out_w_buf.size() / sizeof(uint16_t));

        while (rem_elems > 0) {
            const uint64_t n = std::min<uint64_t>(rem_elems, max_batch);
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(n * bpe), r_off)) return false;

            uint16_t* dst16 = reinterpret_cast<uint16_t*>(out_w_buf.data());
            if (ent.dtype == ST_BF16) {
                const uint16_t* s16 = reinterpret_cast<const uint16_t*>(in_buf.data());
                pool->parallel_for_rows(n, 32768ULL, [&](uint64_t i0, uint64_t i1) {
                    convert_bf16_to_fp16_span(s16 + i0, dst16 + i0, static_cast<size_t>(i1 - i0));
                });
            } else if (ent.dtype == ST_F16) {
                std::memcpy(dst16, in_buf.data(), static_cast<size_t>(n * sizeof(uint16_t)));
            } else if (ent.dtype == ST_F32) {
                const float* sf = reinterpret_cast<const float*>(in_buf.data());
                pool->parallel_for_rows(n, 32768ULL, [&](uint64_t i0, uint64_t i1) {
                    convert_f32_to_fp16_span(sf + i0, dst16 + i0, static_cast<size_t>(i1 - i0));
                });
            } else {
                return false;
            }

            const uint64_t out_chunk_bytes = n * sizeof(uint16_t);
            if (!pwrite_exact(out_fd, out_w_buf.data(), static_cast<size_t>(out_chunk_bytes), w_off)) {
                return false;
            }
            r_off     += n * bpe;
            w_off     += out_chunk_bytes;
            rem_elems -= n;
        }

        /* Zero-fill any 256B alignment tail */
        const uint64_t written = elems * sizeof(uint16_t);
        if (spec.slice.bytes > written) {
            const size_t pad = static_cast<size_t>(spec.slice.bytes - written);
            std::memset(out_w_buf.data(), 0, pad);
            if (!pwrite_exact(out_fd, out_w_buf.data(), pad, spec.slice.offset + written)) return false;
        }
        return true;
    }

    bool convert_q4_target(int shard_fd, const SafetensorEntry& ent, const TargetSpec& spec) {
        const uint64_t rows = spec.qm.rows_logical;
        const uint64_t cols = spec.qm.cols;
        const uint64_t gs   = spec.qm.group_size;
        const uint64_t elems = shape_elems(ent.shape);
        const uint64_t bpe   = dtype_bpe(ent.dtype);
        if (gs != 64ULL || (cols % 64ULL) != 0ULL || elems != rows * cols ||
            bpe == 0 || ent.byte_len != elems * bpe || ent.shape.empty() || ent.shape.back() != cols) {
            std::fprintf(stderr, "[flash-stream] ERROR: Q4 shape/size mismatch on '%s' (got %llu elems, expected [%llu, %llu])\n",
                         ent.canon_name.c_str(), static_cast<unsigned long long>(elems),
                         static_cast<unsigned long long>(rows), static_cast<unsigned long long>(cols));
            return false;
        }

        const int      out_fd         = part_fds[spec.part_id];
        const uint64_t in_row_bytes   = cols * bpe;
        const uint64_t w_row_bytes    = cols / 2ULL;              /* 8 nibbles -> 4B = 0.5B/elem */
        const uint64_t groups_per_row = cols / 64ULL;
        const uint64_t sb_row_bytes   = groups_per_row * 2ULL;

        uint64_t max_rows = in_buf.size() / in_row_bytes;
        max_rows = std::min<uint64_t>(max_rows, out_w_buf.size() / w_row_bytes);
        max_rows = std::min<uint64_t>(max_rows, out_s_buf.size() / sb_row_bytes);
        if (max_rows == 0) return false;

        uint64_t row_done = 0;
        while (row_done < rows) {
            const uint64_t batch_rows = std::min<uint64_t>(max_rows, rows - row_done);
            const uint64_t rd_bytes   = batch_rows * in_row_bytes;
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(rd_bytes),
                             ent.abs_offset + row_done * in_row_bytes)) {
                return false;
            }

            uint8_t*  w_base = out_w_buf.data();
            uint16_t* s_base = reinterpret_cast<uint16_t*>(out_s_buf.data());
            uint16_t* b_base = reinterpret_cast<uint16_t*>(out_b_buf.data());

            const uint64_t grain = std::max<uint64_t>(1ULL, std::min<uint64_t>(64ULL, batch_rows / (pool->num_threads() * 4ULL + 1ULL)));
            if (ent.dtype == ST_BF16) {
                const uint16_t* src_bf = reinterpret_cast<const uint16_t*>(in_buf.data());
                pool->parallel_for_rows(batch_rows, grain, [&](uint64_t r0, uint64_t r1) {
                    for (uint64_t r = r0; r < r1; ++r) {
                        const uint16_t* s_row = src_bf + r * cols;
                        uint32_t*       q_row = reinterpret_cast<uint32_t*>(w_base + r * w_row_bytes);
                        uint16_t*       sc_r  = s_base + r * groups_per_row;
                        uint16_t*       bi_r  = b_base + r * groups_per_row;
                        for (uint64_t g = 0; g < groups_per_row; ++g) {
                            quantize_group64_bf16(s_row + g * 64ULL, q_row + g * 8ULL, &sc_r[g], &bi_r[g]);
                        }
                    }
                });
            } else {
                const uint8_t* src_raw = in_buf.data();
                pool->parallel_for_rows(batch_rows, grain, [&](uint64_t r0, uint64_t r1) {
                    float tmp[64];
                    for (uint64_t r = r0; r < r1; ++r) {
                        uint32_t* q_row = reinterpret_cast<uint32_t*>(w_base + r * w_row_bytes);
                        uint16_t* sc_r  = s_base + r * groups_per_row;
                        uint16_t* bi_r  = b_base + r * groups_per_row;
                        for (uint64_t g = 0; g < groups_per_row; ++g) {
                            const uint64_t base_c = r * cols + g * 64ULL;
                            if (ent.dtype == ST_F16) {
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
            }

            if (!pwrite_exact(out_fd, w_base, static_cast<size_t>(batch_rows * w_row_bytes),
                              spec.qm.weight_off + row_done * w_row_bytes) ||
                !pwrite_exact(out_fd, s_base, static_cast<size_t>(batch_rows * sb_row_bytes),
                              spec.qm.scales_off + row_done * sb_row_bytes) ||
                !pwrite_exact(out_fd, b_base, static_cast<size_t>(batch_rows * sb_row_bytes),
                              spec.qm.biases_off + row_done * sb_row_bytes)) {
                return false;
            }
            row_done += batch_rows;
        }
        return true;
    }

    bool convert_ple_shard(int shard_fd, const SafetensorEntry& ent, uint32_t ple_shard_idx) {
        if (!ple_hdr || ple_fd < 0 || ple_shard_idx >= ple_hdr->num_shards ||
            ple_shard_idx >= STRATA_QWEN_FLASH_MAX_PLE_SHARDS) return false;
        const uint64_t rows       = ple_hdr->rows_per_shard;
        const uint32_t head_dim   = ple_hdr->head_dim_per_ngram;
        const uint32_t gs         = ple_hdr->quant_group_size;
        const uint32_t num_groups = ple_hdr->num_groups_per_row;
        const uint32_t row_bytes  = ple_hdr->row_bytes;
        const uint64_t elems      = shape_elems(ent.shape);
        const uint64_t bpe        = dtype_bpe(ent.dtype);

        if (head_dim == 0u || head_dim > STRATA_QWEN_FLASH_MAX_PLE_HEAD_DIM ||
            (head_dim % 8u) != 0u || gs != STRATA_QWEN_FLASH_QUANT_GROUP_SIZE ||
            num_groups == 0u || num_groups > STRATA_QWEN_FLASH_MAX_PLE_GROUPS ||
            elems != rows * head_dim || bpe == 0 || ent.byte_len != elems * bpe) {
            std::fprintf(stderr, "[flash-stream] ERROR: PLE shard %u shape mismatch on '%s' (got %llu, expected [%llu, %u])\n",
                         ple_shard_idx, ent.canon_name.c_str(),
                         static_cast<unsigned long long>(elems),
                         static_cast<unsigned long long>(rows), head_dim);
            return false;
        }

        const uint64_t in_row_bytes = static_cast<uint64_t>(head_dim) * bpe;
        const uint64_t q_bytes      = static_cast<uint64_t>(head_dim / 2u);
        const uint64_t s_bytes      = static_cast<uint64_t>(num_groups * 2u);
        const uint64_t max_rows     = std::min<uint64_t>(in_buf.size() / in_row_bytes,
                                                         out_w_buf.size() / row_bytes);
        if (max_rows == 0) return false;

        const uint64_t dst_shard_base = ple_hdr->shard_byte_offsets[ple_shard_idx];
        uint64_t row_done = 0;

        while (row_done < rows) {
            const uint64_t batch_rows = std::min<uint64_t>(max_rows, rows - row_done);
            if (!pread_exact(shard_fd, in_buf.data(), static_cast<size_t>(batch_rows * in_row_bytes),
                             ent.abs_offset + row_done * in_row_bytes)) {
                return false;
            }

            uint8_t* dst_base = out_w_buf.data();
            const uint8_t* src_raw = in_buf.data();

            pool->parallel_for_rows(batch_rows, 512ULL, [&](uint64_t r0, uint64_t r1) {
                float tail_f32[64];
                for (uint64_t r = r0; r < r1; ++r) {
                    uint8_t*  out_row = dst_base + r * row_bytes;
                    uint32_t* q_ptr   = reinterpret_cast<uint32_t*>(out_row);
                    uint16_t  sc_arr[STRATA_QWEN_FLASH_MAX_PLE_GROUPS];
                    uint16_t  bi_arr[STRATA_QWEN_FLASH_MAX_PLE_GROUPS];

                    for (uint32_t g = 0; g < num_groups; ++g) {
                        const uint32_t c0    = g * gs;
                        const uint32_t g_len = std::min<uint32_t>(gs, head_dim - c0);
                        if (ent.dtype == ST_BF16 && g_len == 64u) {
                            const uint16_t* s_bf = reinterpret_cast<const uint16_t*>(src_raw + r * in_row_bytes) + c0;
                            quantize_group64_bf16(s_bf, q_ptr + (c0 >> 3u), &sc_arr[g], &bi_arr[g]);
                        } else if (ent.dtype == ST_BF16) {
                            const uint16_t* s_bf = reinterpret_cast<const uint16_t*>(src_raw + r * in_row_bytes) + c0;
                            for (uint32_t j = 0; j < g_len; ++j) tail_f32[j] = bf16_to_f32_scalar(s_bf[j]);
                            quantize_group_f32(tail_f32, g_len, q_ptr + (c0 >> 3u), &sc_arr[g], &bi_arr[g]);
                        } else if (ent.dtype == ST_F16) {
                            const uint16_t* s_h = reinterpret_cast<const uint16_t*>(src_raw + r * in_row_bytes) + c0;
                            for (uint32_t j = 0; j < g_len; ++j) tail_f32[j] = fp16_to_f32_fast(s_h[j]);
                            quantize_group_f32(tail_f32, g_len, q_ptr + (c0 >> 3u), &sc_arr[g], &bi_arr[g]);
                        } else {
                            const float* s_f = reinterpret_cast<const float*>(src_raw + r * in_row_bytes) + c0;
                            quantize_group_f32(s_f, g_len, q_ptr + (c0 >> 3u), &sc_arr[g], &bi_arr[g]);
                        }
                    }
                    std::memcpy(out_row + q_bytes, sc_arr, static_cast<size_t>(s_bytes));
                    std::memcpy(out_row + q_bytes + s_bytes, bi_arr, static_cast<size_t>(s_bytes));
                }
            });

            if (!pwrite_exact(ple_fd, dst_base, static_cast<size_t>(batch_rows * row_bytes),
                              dst_shard_base + row_done * row_bytes)) {
                return false;
            }
            row_done += batch_rows;
        }
        return true;
    }

    bool verify_ple_i64(int shard_fd, const SafetensorEntry& ent, TargetKind kind) {
        if (!ple_hdr) return true;
        const uint64_t elems = shape_elems(ent.shape);
        if (ent.dtype != ST_I64 || ent.byte_len != elems * sizeof(int64_t) || elems > 64) {
            std::fprintf(stderr, "[flash-stream] ERROR: invalid PLE I64 metadata tensor '%s'\n",
                         ent.canon_name.c_str());
            return false;
        }
        int64_t vals[64] = {};
        if (!pread_exact(shard_fd, vals, static_cast<size_t>(ent.byte_len), ent.abs_offset)) return false;

        if (kind == TGT_PLE_CHK_MULTIPLIERS) {
            if (elems != ple_hdr->ngram_size) return false;
            for (uint64_t i = 0; i < elems; ++i) {
                if (static_cast<uint64_t>(vals[i]) != ple_hdr->layer_multipliers[i]) {
                    std::fprintf(stderr, "[flash-stream] ERROR: PLE multiplier[%llu] mismatch (%lld vs %llu)\n",
                                 static_cast<unsigned long long>(i), static_cast<long long>(vals[i]),
                                 static_cast<unsigned long long>(ple_hdr->layer_multipliers[i]));
                    return false;
                }
            }
        } else if (kind == TGT_PLE_CHK_OFFSETS) {
            if (elems != ple_hdr->ngram_heads) return false;
            for (uint64_t i = 0; i < elems; ++i) {
                if (static_cast<uint64_t>(vals[i]) != ple_hdr->head_offsets[i]) return false;
            }
        } else if (kind == TGT_PLE_CHK_VOCABS) {
            if (elems != ple_hdr->ngram_heads) return false;
            for (uint64_t i = 0; i < elems; ++i) {
                if (static_cast<uint64_t>(vals[i]) != ple_hdr->head_vocab_sizes[i]) return false;
            }
        }
        return true;
    }
};

/* ============================================================================
 * 6. Qwen2 / Qwen3.8 / Qwen4-Exp Byte-Level BPE Tokenizer (`tokenizer.json`)
 * ============================================================================ */

static inline bool is_unicode_whitespace(uint32_t cp) {
    if (cp <= 0x20u) {
        return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f';
    }
    return cp == 0x0085u || cp == 0x00A0u || cp == 0x1680u ||
           (cp >= 0x2000u && cp <= 0x200Au) ||
           cp == 0x2028u || cp == 0x2029u || cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u;
}

static inline bool is_unicode_number(uint32_t cp) {
    if (cp < 0x80u) return cp >= '0' && cp <= '9';
    if (cp == 0x00B2u || cp == 0x00B3u || cp == 0x00B9u || (cp >= 0x00BCu && cp <= 0x00BEu)) return true;
    if ((cp >= 0x0660u && cp <= 0x0669u) || (cp >= 0x06F0u && cp <= 0x06F9u)) return true;
    if ((cp >= 0x07C0u && cp <= 0x07C9u) || (cp >= 0x0966u && cp <= 0x096Fu)) return true;
    if ((cp >= 0x09E6u && cp <= 0x09EFu) || (cp >= 0x0A66u && cp <= 0x0A6Fu)) return true;
    if ((cp >= 0x0AE6u && cp <= 0x0AEFu) || (cp >= 0x0B66u && cp <= 0x0B6Fu)) return true;
    if ((cp >= 0x0BE6u && cp <= 0x0BEFu) || (cp >= 0x0C66u && cp <= 0x0C6Fu)) return true;
    if ((cp >= 0x0CE6u && cp <= 0x0CEFu) || (cp >= 0x0D66u && cp <= 0x0D6Fu)) return true;
    if ((cp >= 0x0E50u && cp <= 0x0E59u) || (cp >= 0x0ED0u && cp <= 0x0ED9u)) return true;
    if ((cp >= 0x0F20u && cp <= 0x0F33u) || (cp >= 0x1040u && cp <= 0x1049u)) return true;
    if ((cp >= 0x17E0u && cp <= 0x17E9u) || (cp >= 0x1810u && cp <= 0x1819u)) return true;
    if (cp == 0x2070u || (cp >= 0x2074u && cp <= 0x2079u) || (cp >= 0x2080u && cp <= 0x2089u)) return true;
    if (cp >= 0x2150u && cp <= 0x218Bu) return true;
    if ((cp >= 0x2460u && cp <= 0x249Bu) || cp == 0x24EAu || (cp >= 0x24EBu && cp <= 0x24FFu)) return true;
    if ((cp >= 0x2776u && cp <= 0x2793u) || cp == 0x3007u || (cp >= 0x3021u && cp <= 0x3029u)) return true;
    if ((cp >= 0x3038u && cp <= 0x303Au) || (cp >= 0x3220u && cp <= 0x3229u)) return true;
    if ((cp >= 0xFF10u && cp <= 0xFF19u) || (cp >= 0x1D7CEu && cp <= 0x1D7FFu)) return true;
    return false;
}

static inline bool is_unicode_letter_or_mark(uint32_t cp) {
    if (cp < 0x80u) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    if (is_unicode_whitespace(cp) || is_unicode_number(cp)) return false;
    /* Combining diacritical marks (`\p{M}`) */
    if ((cp >= 0x0300u && cp <= 0x036Fu) || (cp >= 0x1AB0u && cp <= 0x1AFFu) ||
        (cp >= 0x1DC0u && cp <= 0x1DFFu) || (cp >= 0x20D0u && cp <= 0x20FFu) ||
        (cp >= 0xFE20u && cp <= 0xFE2Fu)) return true;
    /* Latin-1 & Extended */
    if (cp == 0x00AAu || cp == 0x00B5u || cp == 0x00BAu) return true;
    if (cp >= 0x00C0u && cp <= 0x00D6u) return true;
    if (cp >= 0x00D8u && cp <= 0x00F6u) return true;
    if (cp >= 0x00F8u && cp <= 0x02C1u) return true;
    if ((cp >= 0x02C6u && cp <= 0x02D1u) || (cp >= 0x02E0u && cp <= 0x02E4u) ||
        cp == 0x02ECu || cp == 0x02EEu) return true;
    /* World scripts */
    if (cp >= 0x0370u && cp <= 0x0374u) return true;
    if (cp >= 0x0376u && cp <= 0x0377u) return true;
    if (cp >= 0x037Au && cp <= 0x037Du) return true;
    if (cp == 0x037Fu || cp == 0x0386u || (cp >= 0x0388u && cp <= 0x038Au) ||
        cp == 0x038Cu || (cp >= 0x038Eu && cp <= 0x03A1u) || (cp >= 0x03A3u && cp <= 0x03F5u) ||
        (cp >= 0x03F7u && cp <= 0x0481u) || (cp >= 0x048Au && cp <= 0x052Fu)) return true;
    if ((cp >= 0x0531u && cp <= 0x0556u) || cp == 0x0559u || (cp >= 0x0560u && cp <= 0x0588u)) return true;
    if ((cp >= 0x05D0u && cp <= 0x05EAu) || (cp >= 0x05EFu && cp <= 0x05F2u)) return true;
    if ((cp >= 0x0620u && cp <= 0x064Au) || (cp >= 0x066Eu && cp <= 0x066Fu) ||
        (cp >= 0x0671u && cp <= 0x06D3u) || cp == 0x06D5u || (cp >= 0x06E5u && cp <= 0x06E6u) ||
        (cp >= 0x06EEu && cp <= 0x06EFu) || (cp >= 0x06FAu && cp <= 0x06FCu) || cp == 0x06FFu) return true;
    if (cp >= 0x0900u && cp <= 0x0963u) return true;
    if (cp >= 0x0E01u && cp <= 0x0E4Eu) return true;
    if (cp >= 0x10A0u && cp <= 0x1248u) return true;
    if (cp >= 0x1D00u && cp <= 0x1DBFu) return true;
    if (cp >= 0x1E00u && cp <= 0x1FBCu) return true;
    if (cp == 0x2071u || cp == 0x207Fu || (cp >= 0x2090u && cp <= 0x209Cu)) return true;
    if (cp == 0x2102u || cp == 0x2107u || (cp >= 0x210Au && cp <= 0x2113u) || cp == 0x2115u ||
        (cp >= 0x2119u && cp <= 0x211Du) || cp == 0x2124u || cp == 0x2126u || cp == 0x2128u ||
        (cp >= 0x212Au && cp <= 0x212Du) || (cp >= 0x212Fu && cp <= 0x2139u) ||
        (cp >= 0x213Cu && cp <= 0x213Fu) || (cp >= 0x2145u && cp <= 0x2149u)) return true;
    if ((cp >= 0x3005u && cp <= 0x3006u) || (cp >= 0x3031u && cp <= 0x3035u) ||
        cp == 0x303Bu || cp == 0x303Cu) return true;
    if ((cp >= 0x3041u && cp <= 0x3096u) || (cp >= 0x3099u && cp <= 0x309Fu)) return true;
    if ((cp >= 0x30A1u && cp <= 0x30FAu) || (cp >= 0x30FCu && cp <= 0x30FFu)) return true;
    if (cp >= 0x3105u && cp <= 0x318Eu) return true;
    if (cp >= 0x3400u && cp <= 0x9FFFu) return true;
    if (cp >= 0xAC00u && cp <= 0xD7A3u) return true;
    if (cp >= 0xF900u && cp <= 0xFAFFu) return true;
    if ((cp >= 0xFF21u && cp <= 0xFF3Au) || (cp >= 0xFF41u && cp <= 0xFF5Au) ||
        (cp >= 0xFF66u && cp <= 0xFFBEu)) return true;
    if ((cp >= 0x1D400u && cp <= 0x1D7CBu) || (cp >= 0x20000u && cp <= 0x2FA1Fu)) return true;
    return false;
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

static void qwen_pretokenize_span(
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

        /* 2. `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` */
        if (is_unicode_letter_or_mark(c0)) {
            size_t j = i + 1;
            while (j < N && is_unicode_letter_or_mark(cps[j].cp)) ++j;
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }
        if (c0 != '\r' && c0 != '\n' && !is_unicode_number(c0) &&
            i + 1 < N && is_unicode_letter_or_mark(cps[i + 1].cp)) {
            size_t j = i + 2;
            while (j < N && is_unicode_letter_or_mark(cps[j].cp)) ++j;
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }

        /* 3. `\p{N}` (single digit per token) */
        if (is_unicode_number(c0)) {
            pieces_out.emplace_back(cps[i].byte_off, cps[i].byte_len);
            ++i;
            continue;
        }

        /* 4. ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` */
        auto is_punct_sym = [](uint32_t cp) {
            return !is_unicode_whitespace(cp) && !is_unicode_letter_or_mark(cp) && !is_unicode_number(cp);
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

} /* anonymous namespace */

struct StrataQwenFlashTokenizer {
    uint32_t                     vocab_size = 0;
    uint32_t                     byte_token_ids[256] = {};
    std::vector<std::string>     id_to_raw_bytes;
    std::vector<uint8_t>         id_is_special;
    std::vector<MergeSlot>       merge_ht;
    uint64_t                     merge_mask = 0;
    std::vector<AddedTokenEntry> added_sorted;
    std::vector<uint32_t>        added_by_b0[256];

    inline bool lookup_merge(uint32_t left_id, uint32_t right_id, uint32_t* rank_out, uint32_t* merged_out) const {
        uint64_t key = (static_cast<uint64_t>(left_id) << 32) | static_cast<uint64_t>(right_id);
        uint64_t h = (key * 11400714819323198485ULL) >> 45;
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
        uint64_t h = (key * 11400714819323198485ULL) >> 45;
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
    const StrataQwenFlashTokenizer* tok,
    const uint8_t* bytes,
    uint32_t len,
    std::vector<uint32_t>& out_ids
) {
    if (len == 0) return;
    if (len == 1) {
        out_ids.push_back(tok->byte_token_ids[bytes[0]]);
        return;
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
    StrataQwenFlashConfig& cfg
) {
    strata_qwen_flash_default_config(&cfg);
    std::string raw;
    if (!read_whole_file(cfg_path, raw)) return false;
    SmallJson root;
    if (!parse_small_json(raw, root) || root.type != SmallJson::J_OBJ) return false;

    const SmallJson* tc = root.get("text_config");
    if (!tc || tc->type != SmallJson::J_OBJ) tc = &root;

    cfg.num_hidden_layers       = static_cast<uint32_t>(tc->get_i64("num_hidden_layers", cfg.num_hidden_layers));
    cfg.split_layer_idx         = cfg.num_hidden_layers / 2u;
    cfg.hidden_size             = static_cast<uint32_t>(tc->get_i64("hidden_size", cfg.hidden_size));
    cfg.vocab_size              = static_cast<uint32_t>(tc->get_i64("vocab_size", cfg.vocab_size));
    cfg.full_attention_interval = static_cast<uint32_t>(tc->get_i64("full_attention_interval", cfg.full_attention_interval));

    cfg.hc_count                = static_cast<uint32_t>(tc->get_i64("hc_count", cfg.hc_count));
    cfg.hc_lowrank              = static_cast<uint32_t>(tc->get_i64("hc_lowrank", cfg.hc_lowrank));

    cfg.num_attention_heads     = static_cast<uint32_t>(tc->get_i64("num_attention_heads", cfg.num_attention_heads));
    cfg.num_key_value_heads     = static_cast<uint32_t>(tc->get_i64("num_key_value_heads", cfg.num_key_value_heads));
    cfg.head_dim                = static_cast<uint32_t>(tc->get_i64("head_dim", cfg.head_dim));
    double prf                  = tc->get_f64("partial_rotary_factor", 0.25);
    cfg.rope_theta              = 10000000.0f;
    if (const SmallJson* rp = tc->get("rope_parameters")) {
        prf            = rp->get_f64("partial_rotary_factor", prf);
        cfg.rope_theta = static_cast<float>(rp->get_f64("rope_theta", 10000000.0));
    } else {
        cfg.rope_theta = static_cast<float>(tc->get_f64("rope_theta", 10000000.0));
    }
    cfg.rotary_dim              = static_cast<uint32_t>(std::llround(cfg.head_dim * prf));
    cfg.indexer_n_heads         = static_cast<uint32_t>(tc->get_i64("indexer_n_heads", cfg.indexer_n_heads));
    cfg.indexer_kv_heads        = static_cast<uint32_t>(tc->get_i64("indexer_kv_heads", cfg.indexer_kv_heads));
    cfg.indexer_head_dim        = static_cast<uint32_t>(tc->get_i64("indexer_head_dim", cfg.indexer_head_dim));
    cfg.indexer_budget          = static_cast<uint32_t>(tc->get_i64("indexer_budget", cfg.indexer_budget));
    cfg.indexer_compress_ratio  = static_cast<uint32_t>(tc->get_i64("indexer_compress_ratio", cfg.indexer_compress_ratio));

    cfg.linear_num_key_heads       = static_cast<uint32_t>(tc->get_i64("linear_num_key_heads", cfg.linear_num_key_heads));
    cfg.linear_key_head_dim        = static_cast<uint32_t>(tc->get_i64("linear_key_head_dim", cfg.linear_key_head_dim));
    cfg.linear_num_value_heads     = static_cast<uint32_t>(tc->get_i64("linear_num_value_heads", cfg.linear_num_value_heads));
    cfg.linear_value_head_dim      = static_cast<uint32_t>(tc->get_i64("linear_value_head_dim", cfg.linear_value_head_dim));
    cfg.linear_conv_kernel_dim     = static_cast<uint32_t>(tc->get_i64("linear_conv_kernel_dim", cfg.linear_conv_kernel_dim));
    std::string gate_type          = tc->get_str("output_gate_type", "sigmoid");
    cfg.linear_output_gate_sigmoid = (gate_type == "sigmoid") ? 1u : 0u;

    cfg.num_experts              = static_cast<uint32_t>(tc->get_i64("num_experts", cfg.num_experts));
    cfg.num_experts_per_tok      = static_cast<uint32_t>(tc->get_i64("num_experts_per_tok", cfg.num_experts_per_tok));
    cfg.moe_intermediate_size    = static_cast<uint32_t>(tc->get_i64("moe_intermediate_size", cfg.moe_intermediate_size));
    cfg.shared_intermediate_size = static_cast<uint32_t>(tc->get_i64("shared_expert_intermediate_size",
                                                         tc->get_i64("shared_intermediate_size", cfg.shared_intermediate_size)));
    cfg.norm_topk_prob           = tc->get_bool("norm_topk_prob", true) ? 1u : 0u;

    if (const SmallJson* ple_ids = tc->get("ple_layer_ids")) {
        if (ple_ids->type == SmallJson::J_ARR) {
            if (ple_ids->arr.empty()) {
                cfg.has_ple = 0u;
            } else if (ple_ids->arr[0].type == SmallJson::J_NUM && ple_ids->arr[0].i_val >= 1) {
                cfg.has_ple       = 1u;
                cfg.ple_layer_idx = static_cast<uint32_t>(ple_ids->arr[0].i_val - 1);
            }
        }
    }
    const int64_t raw_fai             = tc->get_i64("full_attention_interval", cfg.full_attention_interval);
    const int64_t raw_ple_embed_dim   = tc->get_i64("ple_embed_dim", cfg.ple_embed_dim);
    const int64_t raw_ngram_size      = tc->get_i64("ngram_size", cfg.ngram_size);
    const int64_t raw_heads_per_ngram = tc->get_i64("heads_per_ngram", cfg.heads_per_ngram);
    const int64_t raw_ngram_base      = tc->get_i64("ngram_vocab_size_base", cfg.ngram_vocab_size_base);
    const int64_t raw_split_parts     = tc->get_i64("split_ngram_parts", cfg.split_ngram_parts);
    if (raw_fai <= 0 || raw_fai > 0xFFFFFFFFLL) return false;
    cfg.full_attention_interval = static_cast<uint32_t>(raw_fai);

    cfg.ple_embed_dim         = static_cast<uint32_t>(raw_ple_embed_dim);
    cfg.ple_conv_kernel_size  = static_cast<uint32_t>(tc->get_i64("ple_conv_kernel_size", cfg.ple_conv_kernel_size));
    cfg.ngram_size            = static_cast<uint32_t>(raw_ngram_size);
    cfg.heads_per_ngram       = static_cast<uint32_t>(raw_heads_per_ngram);
    cfg.ngram_vocab_size_base = static_cast<uint32_t>(raw_ngram_base);
    cfg.ngram_vocab_pad_div   = static_cast<uint32_t>(tc->get_i64("make_ngram_vocab_size_divisible_by", cfg.ngram_vocab_pad_div));
    cfg.split_ngram_parts     = static_cast<uint32_t>(raw_split_parts);
    cfg.ple_eos_token_id      = static_cast<uint32_t>(tc->get_i64("eos_token_id", cfg.ple_eos_token_id));
    cfg.has_mtp               = (tc->get_i64("mtp_num_hidden_layers", 1) > 0) ? 1u : 0u;

    cfg.rms_norm_eps            = static_cast<float>(tc->get_f64("rms_norm_eps", 1e-6));
    cfg.tie_word_embeddings     = root.get_bool("tie_word_embeddings", tc->get_bool("tie_word_embeddings", false)) ? 1u : 0u;
    cfg.max_position_embeddings = static_cast<uint32_t>(tc->get_i64("max_position_embeddings", cfg.max_position_embeddings));
    cfg.bos_token_id            = static_cast<uint32_t>(tc->get_i64("bos_token_id", cfg.bos_token_id));

    if (cfg.num_hidden_layers == 0u || cfg.num_hidden_layers > STRATA_QWEN_FLASH_MAX_LAYERS ||
        cfg.split_layer_idx > cfg.num_hidden_layers ||
        cfg.split_layer_idx > STRATA_QWEN_FLASH_MAX_PART_LAYERS ||
        (cfg.num_hidden_layers - cfg.split_layer_idx) > STRATA_QWEN_FLASH_MAX_PART_LAYERS ||
        cfg.full_attention_interval == 0u) {
        return false;
    }
    if (cfg.has_ple != 0u) {
        if (raw_ngram_size < 2 || raw_ngram_size > static_cast<int64_t>(STRATA_QWEN_FLASH_MAX_NGRAM_SIZE) ||
            raw_heads_per_ngram < 1 || raw_heads_per_ngram > static_cast<int64_t>(STRATA_QWEN_FLASH_MAX_NGRAM_HEADS) ||
            raw_ple_embed_dim < 1 || raw_ple_embed_dim > 32768 ||
            raw_split_parts < 1 || raw_split_parts > static_cast<int64_t>(STRATA_QWEN_FLASH_MAX_PLE_SHARDS) ||
            raw_ngram_base < 2 || raw_ngram_base > 0xFFFFFFFFLL) {
            return false;
        }
        const uint64_t ngram_heads =
            static_cast<uint64_t>(cfg.ngram_size - 1u) * static_cast<uint64_t>(cfg.heads_per_ngram);
        if (ngram_heads == 0ULL || ngram_heads > STRATA_QWEN_FLASH_MAX_NGRAM_HEADS ||
            (static_cast<uint64_t>(cfg.ple_embed_dim) % ngram_heads) != 0ULL) {
            return false;
        }
        const uint64_t head_dim = static_cast<uint64_t>(cfg.ple_embed_dim) / ngram_heads;
        if (head_dim < 8ULL || head_dim > STRATA_QWEN_FLASH_MAX_PLE_HEAD_DIM || (head_dim % 8ULL) != 0ULL) {
            return false;
        }
    }

    cfg.num_linear_layers    = 0u;
    cfg.num_full_attn_layers = 0u;
    for (uint32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        if (((l + 1u) % cfg.full_attention_interval) == 0u) ++cfg.num_full_attn_layers;
        else                                                 ++cfg.num_linear_layers;
    }

    /* Parse `generation_config.json` if present so `<|im_end|>` (`248046`) + `<|endoftext|>` (`248044`) stay preserved */
    std::string gen_raw;
    if (read_whole_file(gen_cfg_path, gen_raw)) {
        SmallJson gen_root;
        if (parse_small_json(gen_raw, gen_root) && gen_root.type == SmallJson::J_OBJ) {
            if (const SmallJson* eos = gen_root.get("eos_token_id")) {
                if (eos->type == SmallJson::J_ARR && !eos->arr.empty()) {
                    cfg.num_eos_token_ids = 0u;
                    for (size_t i = 0; i < eos->arr.size() && cfg.num_eos_token_ids < 4u; ++i) {
                        if (eos->arr[i].type == SmallJson::J_NUM) {
                            cfg.eos_token_ids[cfg.num_eos_token_ids++] =
                                static_cast<uint32_t>(eos->arr[i].i_val);
                        }
                    }
                }
            }
        }
    }

    return true;
}

static bool check_dir_has_no_files(const std::string& dir_path) {
    DIR* d = opendir(dir_path.c_str());
    if (!d) return true; /* Directory removed */
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

/* Shared header validation used by both `strata_qwen_flash_load_bundle` and `strata_qwen_flash_repack_bundle` */
static constexpr uint32_t kSqfDtF32   = 1U << STRATA_QWEN_FLASH_DTYPE_F32;
static constexpr uint32_t kSqfDtF16   = 1U << STRATA_QWEN_FLASH_DTYPE_F16;
static constexpr uint32_t kSqfDtF16Q8 = kSqfDtF16 | (1U << STRATA_QWEN_FLASH_DTYPE_Q8_0);
static constexpr uint32_t kSqfDtF16Q4 = kSqfDtF16 | (1U << STRATA_QWEN_FLASH_DTYPE_Q4_A);

static bool validate_sqf_config(const StrataQwenFlashConfig& c) {
    return c.num_hidden_layers > 0U && c.num_hidden_layers <= STRATA_QWEN_FLASH_MAX_LAYERS &&
           c.split_layer_idx <= c.num_hidden_layers &&
           c.split_layer_idx <= STRATA_QWEN_FLASH_MAX_PART_LAYERS &&
           (c.num_hidden_layers - c.split_layer_idx) <= STRATA_QWEN_FLASH_MAX_PART_LAYERS &&
           c.full_attention_interval > 0U &&
           c.hidden_size > 0U && c.hidden_size <= 32768U &&
           c.vocab_size > 0U && c.vocab_size <= (1U << 22) &&
           c.hc_count > 0U && c.hc_count <= 16U &&
           c.hc_lowrank > 0U && c.hc_lowrank <= 16384U &&
           c.num_attention_heads > 0U && c.num_attention_heads <= 256U &&
           c.num_key_value_heads > 0U && c.num_key_value_heads <= c.num_attention_heads &&
           c.head_dim > 0U && c.head_dim <= 1024U &&
           c.num_experts > 0U && c.num_experts <= 4096U &&
           c.num_experts_per_tok > 0U && c.num_experts_per_tok <= c.num_experts &&
           c.num_experts_per_tok <= 16U && /* router kernels keep `sel_ids[16]` and clamp `top_k` to 16 */
           c.moe_intermediate_size > 0U && c.moe_intermediate_size <= 65536U &&
           c.shared_intermediate_size > 0U && c.shared_intermediate_size <= 65536U &&
           c.quant_group_size == STRATA_QWEN_FLASH_QUANT_GROUP_SIZE &&
           c.indexer_compress_ratio == STRATA_QWEN_FLASH_KV_BLOCK_TOKENS &&
           /* Round 7 (F10): whole KV blocks, and no larger than the budget the paged KV pool is sized for */
           c.indexer_budget >= STRATA_QWEN_FLASH_KV_BLOCK_TOKENS &&
           (c.indexer_budget % STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) == 0U &&
           c.indexer_budget <= STRATA_QWEN_FLASH_MAX_INDEXER_BUDGET &&
           (c.has_ple == 0U || (c.ngram_size >= 2U && c.ngram_size <= STRATA_QWEN_FLASH_MAX_NGRAM_SIZE)) &&
           c.num_eos_token_ids <= 4U;
}

static bool validate_sqf_slice(
    const StrataQwenFlashTensorSlice& s,
    uint32_t dt_mask,
    uint64_t exp_rows,
    uint64_t exp_cols,
    uint64_t fsz
) {
    if (s.dtype >= 16U || (((1U << s.dtype) & dt_mask) == 0U)) return false;
    if (s.offset < STRATA_QWEN_FLASH_HEADER_PADDED_BYTES || s.offset > fsz || s.bytes > fsz - s.offset) return false;
    if (s.shape0 == 0U || static_cast<uint64_t>(s.shape0) != exp_rows) return false;
    const uint64_t s1 = (s.shape1 > 0U) ? static_cast<uint64_t>(s.shape1) : 1ULL;
    if (s1 != exp_cols) return false;
    uint64_t need = 0ULL;
    if (s.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) {
        if ((s1 % STRATA_QWEN_FLASH_QUANT_GROUP_SIZE) != 0ULL) return false;
        if (static_cast<uint64_t>(s.shape0) > fsz / s1) return false;
        uint64_t wb = 0ULL, sb = 0ULL;
        sqf_q8_slice_layout(s.shape0, static_cast<uint32_t>(s1), &wb, &sb);
        need = wb + sb;
    } else if (s.dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A) {
        if ((s1 % STRATA_QWEN_FLASH_QUANT_GROUP_SIZE) != 0ULL) return false;
        const uint64_t rp = (static_cast<uint64_t>(s.shape0) + 63ULL) & ~63ULL;
        if (rp > (fsz * 2ULL) / s1) return false;
        const uint64_t wb = sqf_align_up_u64((rp * s1) / 2ULL, STRATA_QWEN_FLASH_TENSOR_ALIGN);
        const uint64_t sb = sqf_align_up_u64(rp * (s1 / STRATA_QWEN_FLASH_QUANT_GROUP_SIZE) * 2ULL,
                                             STRATA_QWEN_FLASH_TENSOR_ALIGN);
        need = wb + 2ULL * sb;
    } else {
        const uint64_t bpe = (s.dtype == STRATA_QWEN_FLASH_DTYPE_F32 || s.dtype == STRATA_QWEN_FLASH_DTYPE_U32) ? 4ULL : 2ULL;
        if (static_cast<uint64_t>(s.shape0) > (fsz / bpe) / s1) return false;
        need = static_cast<uint64_t>(s.shape0) * s1 * bpe;
    }
    return need > 0ULL && s.bytes >= need;
}

static bool validate_sqf_q4(
    const StrataQwenFlashQuantMatrix& m,
    uint32_t gs_cfg,
    uint64_t exp_rows,
    uint64_t exp_cols,
    uint64_t fsz
) {
    if (exp_rows == 0ULL || exp_cols == 0ULL || exp_rows > 0xFFFFFFFFULL || exp_cols > 0xFFFFFFFFULL) return false;
    if (m.group_size != STRATA_QWEN_FLASH_QUANT_GROUP_SIZE || m.group_size != gs_cfg) return false;
    if (m.cols == 0U || (m.cols % m.group_size) != 0U || (m.cols & 7U) != 0U) return false;
    if (static_cast<uint64_t>(m.rows_logical) != exp_rows || static_cast<uint64_t>(m.cols) != exp_cols) return false;
    const uint64_t exp_rp = (exp_rows + 63ULL) & ~63ULL;
    if (static_cast<uint64_t>(m.rows_padded) != exp_rp) return false;
    const uint64_t min_w = (static_cast<uint64_t>(m.rows_padded) * m.cols) / 2ULL;
    const uint64_t min_s = static_cast<uint64_t>(m.rows_padded) * (m.cols / m.group_size) * 2ULL;
    if (m.weight_bytes < min_w || m.scales_bytes < min_s || m.biases_bytes < min_s) return false;
    auto in_rng = [&](uint64_t off, uint64_t len) -> bool {
        return off >= STRATA_QWEN_FLASH_HEADER_PADDED_BYTES && off <= fsz && len <= fsz - off;
    };
    return in_rng(m.weight_off, m.weight_bytes) &&
           in_rng(m.scales_off, m.scales_bytes) &&
           in_rng(m.biases_off, m.biases_bytes);
}

static bool validate_sqf_layer(
    const StrataQwenFlashLayerDesc& ld,
    const StrataQwenFlashConfig& c,
    uint64_t fsz,
    uint32_t exp_part_id,
    uint32_t exp_global_idx
) {
    if (ld.part_id != exp_part_id || ld.global_layer_idx != exp_global_idx) return false;
    auto in_rng = [&](uint64_t off, uint64_t len) -> bool {
        return off >= STRATA_QWEN_FLASH_HEADER_PADDED_BYTES && off <= fsz && len <= fsz - off;
    };
    auto chk_sl = [&](const StrataQwenFlashTensorSlice& s, uint32_t dm, uint64_t er, uint64_t ec) -> bool {
        return validate_sqf_slice(s, dm, er, ec, fsz);
    };
    auto chk_q4 = [&](const StrataQwenFlashQuantMatrix& m, uint64_t er, uint64_t ec) -> bool {
        return validate_sqf_q4(m, c.quant_group_size, er, ec, fsz);
    };
    const uint64_t d      = c.hidden_size;
    const uint64_t hc_dim = static_cast<uint64_t>(c.hc_count) * d;
    auto chk_hc = [&](const StrataQwenFlashHCDesc& h) -> bool {
        return chk_sl(h.hc_norm, kSqfDtF32, hc_dim, 1ULL) &&
               chk_sl(h.input_mix_weight_down, kSqfDtF16Q8, c.hc_lowrank, hc_dim) &&
               chk_sl(h.input_mix_weight_up, kSqfDtF16Q8, hc_dim, c.hc_lowrank) &&
               chk_sl(h.block_inject_weight, kSqfDtF16, c.hc_count, hc_dim);
    };
    if (!in_rng(ld.layer_page_off, ld.layer_page_bytes)) return false;
    if (ld.has_ple) {
        if (!c.has_ple) return false;
        if (c.ple_embed_dim == 0U || c.ple_embed_dim > 32768U ||
            c.ple_conv_kernel_size == 0U || c.ple_conv_kernel_size > 64U) return false;
        if (!chk_sl(ld.ple.key_proj, kSqfDtF16Q8, hc_dim, c.ple_embed_dim) ||
            !chk_sl(ld.ple.value_proj, kSqfDtF16Q8, d, c.ple_embed_dim) ||
            !chk_sl(ld.ple.norm_key, kSqfDtF32, hc_dim, 1ULL) || !chk_sl(ld.ple.norm_query, kSqfDtF32, hc_dim, 1ULL) ||
            !chk_sl(ld.ple.norm_conv, kSqfDtF32, hc_dim, 1ULL) ||
            !chk_sl(ld.ple.conv1d_weight, kSqfDtF32, hc_dim, c.ple_conv_kernel_size)) {
            return false;
        }
    }
    if (!chk_hc(ld.attn_hc) || !chk_hc(ld.mlp_hc)) return false;
    if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) {
        if (c.linear_num_key_heads == 0U || c.linear_num_key_heads > 256U ||
            c.linear_key_head_dim == 0U || c.linear_key_head_dim > 1024U ||
            c.linear_num_value_heads == 0U || c.linear_num_value_heads > 256U ||
            c.linear_value_head_dim == 0U || c.linear_value_head_dim > 1024U ||
            c.linear_conv_kernel_dim == 0U || c.linear_conv_kernel_dim > 64U) return false;
        const uint64_t k_dim    = static_cast<uint64_t>(c.linear_num_key_heads) * c.linear_key_head_dim;
        const uint64_t v_dim    = static_cast<uint64_t>(c.linear_num_value_heads) * c.linear_value_head_dim;
        const uint64_t conv_dim = 2ULL * k_dim + v_dim;
        const uint64_t nv       = c.linear_num_value_heads;
        if (!chk_q4(ld.lin_in_proj_qkv, conv_dim, d) || !chk_q4(ld.lin_in_proj_z, v_dim, d) ||
            !chk_q4(ld.lin_out_proj, d, v_dim) ||
            !chk_sl(ld.lin_in_proj_a, kSqfDtF16, nv, d) || !chk_sl(ld.lin_in_proj_b, kSqfDtF16, nv, d) ||
            !chk_sl(ld.lin_conv1d_weight, kSqfDtF32, conv_dim, c.linear_conv_kernel_dim) ||
            !chk_sl(ld.lin_A_log, kSqfDtF32, nv, 1ULL) || !chk_sl(ld.lin_dt_bias, kSqfDtF32, nv, 1ULL) ||
            !chk_sl(ld.lin_norm, kSqfDtF32, c.linear_value_head_dim, 1ULL)) {
            return false;
        }
    } else if (ld.layer_type == STRATA_QWEN_FLASH_LAYER_FULL_ATTN) {
        if (c.indexer_n_heads > 256U || c.indexer_kv_heads > 256U ||
            c.indexer_head_dim == 0U || c.indexer_head_dim > 1024U ||
            c.indexer_compress_ratio != STRATA_QWEN_FLASH_KV_BLOCK_TOKENS) return false;
        const uint64_t q_out   = static_cast<uint64_t>(c.num_attention_heads) * c.head_dim * 2ULL;
        const uint64_t kv_out  = static_cast<uint64_t>(c.num_key_value_heads) * c.head_dim;
        const uint64_t o_in    = static_cast<uint64_t>(c.num_attention_heads) * c.head_dim;
        const uint64_t idx_out = (static_cast<uint64_t>(c.indexer_n_heads) + c.indexer_kv_heads) * c.indexer_head_dim;
        if (idx_out == 0ULL) return false;
        if (!chk_q4(ld.attn_q_proj, q_out, d) || !chk_q4(ld.attn_k_proj, kv_out, d) ||
            !chk_q4(ld.attn_v_proj, kv_out, d) || !chk_q4(ld.attn_o_proj, d, o_in) ||
            !chk_sl(ld.attn_q_norm, kSqfDtF32, c.head_dim, 1ULL) || !chk_sl(ld.attn_k_norm, kSqfDtF32, c.head_dim, 1ULL) ||
            !chk_sl(ld.indexer_qk_proj, kSqfDtF16, idx_out, d) ||
            !chk_sl(ld.indexer_q_norm, kSqfDtF32, c.indexer_head_dim, 1ULL) ||
            !chk_sl(ld.indexer_k_norm, kSqfDtF32, c.indexer_head_dim, 1ULL)) {
            return false;
        }
    } else {
        return false;
    }
    return chk_sl(ld.moe_router_gate, kSqfDtF16, c.num_experts, d) &&
           chk_q4(ld.moe_experts_gate_up, static_cast<uint64_t>(c.num_experts) * (2ULL * c.moe_intermediate_size), d) &&
           chk_q4(ld.moe_experts_down, static_cast<uint64_t>(c.num_experts) * d, c.moe_intermediate_size) &&
           chk_sl(ld.shared_gate_proj, kSqfDtF16Q4, c.shared_intermediate_size, d) &&
           chk_sl(ld.shared_up_proj, kSqfDtF16Q4, c.shared_intermediate_size, d) &&
           chk_sl(ld.shared_down_proj, kSqfDtF16Q4, d, c.shared_intermediate_size) &&
           chk_sl(ld.shared_expert_gate, kSqfDtF32, d, 1ULL);
}

static bool validate_sqf_part0_header(const StrataQwenFlashPart0Header* h0, uint64_t fsz) {
    if (!h0 ||
        h0->magic != STRATA_QWEN_FLASH_PART0_MAGIC_U64 ||
        h0->version != STRATA_QWEN_FLASH_FORMAT_VERSION ||
        h0->header_padded_bytes != STRATA_QWEN_FLASH_HEADER_PADDED_BYTES ||
        h0->total_file_bytes != fsz ||
        !validate_sqf_config(h0->cfg) ||
        h0->num_part0_layers > STRATA_QWEN_FLASH_MAX_PART_LAYERS ||
        h0->num_part0_layers != h0->cfg.split_layer_idx) {
        return false;
    }
    auto in_rng = [&](uint64_t off, uint64_t len) -> bool {
        return off >= STRATA_QWEN_FLASH_HEADER_PADDED_BYTES && off <= fsz && len <= fsz - off;
    };
    if (!in_rng(h0->embed_page_off, h0->embed_page_bytes) ||
        !validate_sqf_q4(h0->embed_tokens, h0->cfg.quant_group_size, h0->cfg.vocab_size, h0->cfg.hidden_size, fsz)) {
        return false;
    }
    for (uint32_t l = 0U; l < h0->num_part0_layers; ++l) {
        if (!validate_sqf_layer(h0->layers[l], h0->cfg, fsz, 0U, l)) return false;
    }
    return true;
}

static bool validate_sqf_part1_header(
    const StrataQwenFlashPart0Header* h0,
    const StrataQwenFlashPart1Header* h1,
    uint64_t fsz
) {
    if (!h0 || !h1 ||
        h1->magic != STRATA_QWEN_FLASH_PART1_MAGIC_U64 ||
        h1->version != STRATA_QWEN_FLASH_FORMAT_VERSION ||
        h1->header_padded_bytes != STRATA_QWEN_FLASH_HEADER_PADDED_BYTES ||
        h1->total_file_bytes != fsz ||
        !validate_sqf_config(h1->cfg) ||
        std::memcmp(&h1->cfg, &h0->cfg, sizeof(h1->cfg)) != 0 ||
        h1->num_part1_layers > STRATA_QWEN_FLASH_MAX_PART_LAYERS ||
        h1->first_global_layer_idx != h1->cfg.split_layer_idx ||
        h1->num_part1_layers != h1->cfg.num_hidden_layers - h1->cfg.split_layer_idx ||
        (h1->cfg.has_mtp != 0U) != (h1->mtp.enabled != 0U)) {
        return false;
    }
    const auto& c = h1->cfg;
    const uint64_t d = c.hidden_size;
    const uint64_t hc_dim = static_cast<uint64_t>(c.hc_count) * d;
    auto in_rng = [&](uint64_t off, uint64_t len) -> bool {
        return off >= STRATA_QWEN_FLASH_HEADER_PADDED_BYTES && off <= fsz && len <= fsz - off;
    };
    auto chk_sl = [&](const StrataQwenFlashTensorSlice& s, uint32_t dm, uint64_t er, uint64_t ec) -> bool {
        return validate_sqf_slice(s, dm, er, ec, fsz);
    };
    auto chk_mx = [&](const StrataQwenFlashMixerDesc& mx) -> bool {
        return c.hc_lowrank > 0U &&
               chk_sl(mx.hc_norm, kSqfDtF32, hc_dim, 1ULL) &&
               chk_sl(mx.input_mix_weight_down, kSqfDtF16Q8, c.hc_lowrank, hc_dim) &&
               chk_sl(mx.input_mix_weight_up, kSqfDtF16Q8, hc_dim, c.hc_lowrank);
    };
    uint32_t n_lin = 0U, n_full = 0U;
    for (uint32_t l = 0U; l < h0->num_part0_layers; ++l) {
        if (h0->layers[l].layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++n_lin;
        else ++n_full;
    }
    for (uint32_t l = 0U; l < h1->num_part1_layers; ++l) {
        if (!validate_sqf_layer(h1->layers[l], c, fsz, 1U, h1->first_global_layer_idx + l)) return false;
        if (h1->layers[l].layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) ++n_lin;
        else ++n_full;
    }
    if (c.num_linear_layers != n_lin || c.num_full_attn_layers != n_full) return false;
    if (!in_rng(h1->head_page_off, h1->head_page_bytes) ||
        !chk_mx(h1->final_mixer) ||
        !validate_sqf_q4(h1->lm_head, c.quant_group_size, c.vocab_size, d, fsz)) {
        return false;
    }
    if (h1->mtp.enabled) {
        if (h1->mtp.layer0.layer_type != STRATA_QWEN_FLASH_LAYER_FULL_ATTN ||
            !in_rng(h1->mtp.mtp_page_off, h1->mtp.mtp_page_bytes) ||
            !chk_sl(h1->mtp.pre_fc_norm_embedding, kSqfDtF32, d, 1ULL) ||
            !chk_sl(h1->mtp.pre_fc_norm_hidden, kSqfDtF32, hc_dim, 1ULL) ||
            !chk_sl(h1->mtp.fc_embedding, kSqfDtF16Q8, d, d) ||
            !chk_sl(h1->mtp.fc_hidden, kSqfDtF16Q8, d, d) ||
            !validate_sqf_layer(h1->mtp.layer0, c, fsz, 1U, c.num_hidden_layers) ||
            !chk_mx(h1->mtp.mixer)) {
            return false;
        }
    }
    return true;
}

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
 * 7. Public C ABI Implementations (`include/strata_qwen_flash.h`)
 * ============================================================================ */

extern "C" {

StrataQwenFlashTokenizer* strata_qwen_flash_tokenizer_load(const char* tokenizer_json_path) {
    if (!tokenizer_json_path) return nullptr;
    std::string json_str;
    if (!read_whole_file(tokenizer_json_path, json_str) || json_str.empty()) {
        std::fprintf(stderr, "[flash-tok] ERROR: failed to read '%s'\n", tokenizer_json_path);
        return nullptr;
    }

    std::string b2u_utf8[256];
    int16_t u2b_cp[512];
    build_gpt2_byte_unicode_tables(b2u_utf8, u2b_cp);

    std::unordered_map<std::string, uint32_t> str_to_id;
    str_to_id.reserve(300000);

    struct RawMerge {
        std::string a;
        std::string b;
    };
    std::vector<RawMerge> raw_merges;
    raw_merges.reserve(260000);
    std::vector<AddedTokenEntry> added_list;

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
                        if (sub_key == "vocab") {
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

    auto* tok = new StrataQwenFlashTokenizer();
    tok->vocab_size = max_id_seen + 1u;
    tok->id_to_raw_bytes.assign(tok->vocab_size, std::string());
    tok->id_is_special.assign(tok->vocab_size, 0u);

    std::string raw_tmp;
    for (const auto& kv : str_to_id) {
        if (kv.second >= tok->vocab_size) {
            delete tok;
            return nullptr;
        }
        if (decode_bpe_vocab_key_to_bytes(kv.first, u2b_cp, raw_tmp)) {
            tok->id_to_raw_bytes[kv.second] = raw_tmp;
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

void strata_qwen_flash_tokenizer_free(StrataQwenFlashTokenizer* tok) {
    delete tok;
}

uint32_t strata_qwen_flash_tokenizer_vocab_size(const StrataQwenFlashTokenizer* tok) {
    return tok ? tok->vocab_size : 0u;
}

int strata_qwen_flash_encode(
    const StrataQwenFlashTokenizer* tok,
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
        qwen_pretokenize_span(span_ptr, span_len, pieces);
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

int strata_qwen_flash_format_chat_prompt(
    const char* user_prompt,
    const char* system_prompt,
    bool        enable_thinking,
    const char* reasoning_effort,
    char*       out_buf,
    size_t      out_buf_cap,
    size_t*     out_len
) {
    if (!user_prompt) return -1;

    std::string effort = (reasoning_effort && reasoning_effort[0] != '\0')
                             ? std::string(reasoning_effort)
                             : std::string("low");
    std::string reasoning_instructions;
    if (enable_thinking) {
        if (effort == "low") {
            reasoning_instructions =
                "Reasoning effort is set to low. Keep your thinking brief and focused, "
                "moving directly to the conclusion without unnecessary elaboration.";
        } else if (effort == "medium") {
            reasoning_instructions.clear();
        } else if (effort == "xhigh") {
            reasoning_instructions =
                "Reasoning effort is set to xhigh. Please think carefully through the task, "
                "validate key assumptions, consider plausible alternatives, and prioritize "
                "correctness, consistency, and clarity in the final answer.";
        } else {
            return -1;
        }
    }

    std::string sys_str = (system_prompt && system_prompt[0] != '\0') ? std::string(system_prompt) : std::string();
    std::string res;
    res.reserve(std::strlen(user_prompt) + sys_str.size() + reasoning_instructions.size() + 256);

    if (!sys_str.empty() && !reasoning_instructions.empty()) {
        res += "<|im_start|>system\n";
        res += reasoning_instructions;
        res += "\n\n";
        res += sys_str;
        res += "<|im_end|>\n";
    } else if (!sys_str.empty()) {
        res += "<|im_start|>system\n";
        res += sys_str;
        res += "<|im_end|>\n";
    } else if (!reasoning_instructions.empty()) {
        res += "<|im_start|>system\n";
        res += reasoning_instructions;
        res += "<|im_end|>\n";
    }

    res += "<|im_start|>user\n";
    res += user_prompt;
    res += "<|im_end|>\n<|im_start|>assistant\n";
    if (enable_thinking) {
        res += "<think>\n";
    } else {
        res += "<think>\n\n</think>\n\n";
    }

    if (out_len) *out_len = res.size();
    if (!out_buf) return 0;
    if (res.size() + 1 > out_buf_cap) return -2;
    std::memcpy(out_buf, res.data(), res.size() + 1);
    return 0;
}

int strata_qwen_flash_stream_decode_step(
    const StrataQwenFlashTokenizer* tok,
    StrataQwenFlashStreamDecoder*   dec,
    uint32_t                        token_id,
    bool                            skip_special,
    char*                           out_utf8,
    size_t                          out_cap,
    size_t*                         out_len
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

int strata_qwen_flash_decode(
    const StrataQwenFlashTokenizer* tok,
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

int strata_qwen_flash_convert_hf(
    const char* hf_repo_or_dir,
    const char* out_bundle_dir,
    uint32_t    download_concurrency,
    uint32_t    convert_threads,
    StrataQwenFlashConvertStats* stats_out
) {
    auto wall_t0 = std::chrono::steady_clock::now();
    if (stats_out) std::memset(stats_out, 0, sizeof(*stats_out));

    const std::string src_spec = (hf_repo_or_dir && hf_repo_or_dir[0] != '\0')
                                     ? std::string(hf_repo_or_dir)
                                     : std::string(STRATA_QWEN_FLASH_DEFAULT_HF_REPO);
    const std::string out_dir  = (out_bundle_dir && out_bundle_dir[0] != '\0')
                                     ? std::string(out_bundle_dir)
                                     : std::string(STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR);

    /* Pre-validate local config.json before creating output dir or writing any files */
    if (is_directory_path(src_spec) && file_exists_nonempty(src_spec + "/config.json")) {
        StrataQwenFlashConfig pre_cfg;
        if (!parse_config_into(src_spec + "/config.json", src_spec + "/generation_config.json", pre_cfg)) {
            std::fprintf(stderr, "[flash-stream] ERROR: invalid configuration in '%s/config.json'\n", src_spec.c_str());
            return -1;
        }
    }

    if (!mkdirs(out_dir)) {
        std::fprintf(stderr, "[flash-stream] ERROR: cannot create bundle dir '%s'\n", out_dir.c_str());
        return -1;
    }

    /* Determine whether `src_spec` is an existing local directory with shards on disk or a remote HF repo */
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
        if (!copy_file_if_exists(src_spec + "/config.json", out_dir + "/config.json", true) ||
            !copy_file_if_exists(src_spec + "/tokenizer.json", out_dir + "/tokenizer.json", true)) {
            std::fprintf(stderr, "[flash-stream] ERROR: missing config.json or tokenizer.json in '%s'\n",
                         src_spec.c_str());
            return -1;
        }
        (void)copy_file_if_exists(src_spec + "/tokenizer_config.json", out_dir + "/tokenizer_config.json", false);
        (void)copy_file_if_exists(src_spec + "/chat_template.jinja", out_dir + "/chat_template.jinja", false);
        (void)copy_file_if_exists(src_spec + "/generation_config.json", out_dir + "/generation_config.json", false);
        (void)copy_file_if_exists(src_spec + "/model.safetensors.index.json", out_dir + "/model.safetensors.index.json", false);
    } else {
        const char* req_meta[] = {"config.json", "tokenizer.json", "model.safetensors.index.json"};
        for (const char* mf : req_meta) {
            std::string dst = out_dir + "/" + mf;
            if (!file_exists_nonempty(dst)) {
                std::fprintf(stdout, "[flash-stream] Downloading %s ...\n", mf);
                std::fflush(stdout);
                if (!curl_download_atomic(base_url + "/" + mf, dst, nullptr, nullptr)) {
                    std::fprintf(stderr, "[flash-stream] ERROR: failed to download '%s'\n", mf);
                    return -1;
                }
            }
        }
        const char* opt_meta[] = {"tokenizer_config.json", "chat_template.jinja", "generation_config.json"};
        for (const char* mf : opt_meta) {
            std::string dst = out_dir + "/" + mf;
            if (!file_exists_nonempty(dst)) {
                (void)curl_download_atomic(base_url + "/" + mf, dst, nullptr, nullptr);
            }
        }
    }

    /* 2. Parse `config.json` (+ `generation_config.json`) & build deterministic 3-file layout */
    StrataQwenFlashConfig cfg;
    if (!parse_config_into(out_dir + "/config.json", out_dir + "/generation_config.json", cfg)) {
        std::fprintf(stderr, "[flash-stream] ERROR: failed to parse '%s/config.json'\n", out_dir.c_str());
        return -1;
    }

    StrataQwenFlashPart0Header part0_hdr;
    StrataQwenFlashPart1Header part1_hdr;
    StrataQwenFlashPleHeader   ple_hdr;
    strata_qwen_flash_init_layout(&cfg, &part0_hdr, &part1_hdr, &ple_hdr);

    TargetRegistry reg;
    build_target_registry(cfg, part0_hdr, part1_hdr, ple_hdr, reg);

    /* 3. Parse `model.safetensors.index.json` (or single `model.safetensors`) */
    std::vector<std::string> shard_names;
    std::unordered_map<std::string, std::vector<std::string>> shard_raw_keys;
    uint64_t index_total_bytes = 0ULL;

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
                    std::fprintf(stderr, "[flash-stream] ERROR: unsafe shard filename '%s' in model.safetensors.index.json\n",
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
        std::fprintf(stderr, "[flash-stream] ERROR: missing model.safetensors.index.json\n");
        return -1;
    }

    /* 4. Open or resume `uma_part0.strata`, `uma_part1.strata`, and `ple_ngram.strata` */
    const std::string part0_path = out_dir + "/" + STRATA_QWEN_FLASH_PART0_FILENAME;
    const std::string part1_path = out_dir + "/" + STRATA_QWEN_FLASH_PART1_FILENAME;
    const std::string ple_path   = out_dir + "/" + STRATA_QWEN_FLASH_PLE_FILENAME;
    const std::string done_path  = out_dir + "/.convert_done_shards";
    const std::string stage_dir  = out_dir + "/.staging";

    std::set<std::string> done_shards;
    bool can_resume = false;
    {
        uint64_t sz0 = 0, sz1 = 0, szp = 0;
        if (file_exists_nonempty(done_path) &&
            file_exists_nonempty(part0_path, &sz0) && sz0 == part0_hdr.total_file_bytes &&
            file_exists_nonempty(part1_path, &sz1) && sz1 == part1_hdr.total_file_bytes &&
            (!cfg.has_ple || (file_exists_nonempty(ple_path, &szp) && szp == ple_hdr.total_file_bytes))) {
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
    int fd0 = open(part0_path.c_str(), open_flags, 0644);
    int fd1 = open(part1_path.c_str(), open_flags, 0644);
    int fdp = cfg.has_ple ? open(ple_path.c_str(), open_flags, 0644) : -1;
    if (fd0 < 0 || fd1 < 0 || (cfg.has_ple && fdp < 0)) {
        if (fd0 >= 0) close(fd0);
        if (fd1 >= 0) close(fd1);
        if (fdp >= 0) close(fdp);
        std::fprintf(stderr, "[flash-stream] ERROR: failed to open output .strata files\n");
        return -1;
    }

#if defined(F_NOCACHE)
    (void)fcntl(fd0, F_NOCACHE, 1);
    (void)fcntl(fd1, F_NOCACHE, 1);
    if (fdp >= 0) (void)fcntl(fdp, F_NOCACHE, 1);
#endif

    if (!can_resume) {
        done_shards.clear();
        if (ftruncate(fd0, static_cast<off_t>(part0_hdr.total_file_bytes)) != 0 ||
            ftruncate(fd1, static_cast<off_t>(part1_hdr.total_file_bytes)) != 0 ||
            (cfg.has_ple && ftruncate(fdp, static_cast<off_t>(ple_hdr.total_file_bytes)) != 0)) {
            close(fd0);
            close(fd1);
            if (fdp >= 0) close(fdp);
            std::fprintf(stderr, "[flash-stream] ERROR: ftruncate failed on .strata files\n");
            return -1;
        }
        if (!pwrite_exact(fd0, &part0_hdr, sizeof(part0_hdr), 0) ||
            !pwrite_exact(fd1, &part1_hdr, sizeof(part1_hdr), 0) ||
            (cfg.has_ple && !pwrite_exact(fdp, &ple_hdr, sizeof(ple_hdr), 0))) {
            close(fd0);
            close(fd1);
            if (fdp >= 0) close(fdp);
            return -1;
        }
        int dfd = open(done_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (dfd >= 0) close(dfd);
    }

    /* Track target coverage and resumed statistics */
    std::vector<uint8_t> target_seen(reg.specs.size(), 0u);
    uint32_t num_shards_resumed       = 0;
    uint32_t num_shards_converted     = 0;
    uint32_t text_tensors_converted   = 0;
    uint32_t ple_shards_converted     = 0;
    uint32_t mtp_tensors_converted    = 0;
    uint32_t vision_tensors_skipped   = 0;
    uint64_t total_hf_bytes_processed = 0;

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
                            const TargetSpec& ts = reg.specs[it_t->second];
                            target_seen[ts.id] = 1u;
                            if (ts.kind == TGT_PLE_SHARD) {
                                ++ple_shards_converted;
                            } else if (ts.kind == TGT_UMA_Q4 || ts.kind == TGT_UMA_F16 || ts.kind == TGT_UMA_F32) {
                                ++text_tensors_converted;
                                if (ts.is_mtp) ++mtp_tensors_converted;
                            }
                        }
                    }
                }
            }
        } else {
            pending_shard_indices.push_back(s);
        }
    }

    std::fprintf(stdout,
                 "[flash-stream] Layout initialized: Part0=%.3f GiB, Part1=%.3f GiB, PLE=%.3f GiB | "
                 "Total shards=%zu (%u resumed, %zu pending)\n",
                 static_cast<double>(part0_hdr.total_file_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(part1_hdr.total_file_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(cfg.has_ple ? ple_hdr.total_file_bytes : 0ULL) / (1024.0 * 1024.0 * 1024.0),
                 shard_names.size(), num_shards_resumed, pending_shard_indices.size());
    std::fflush(stdout);

    /* 5. Execute Bounded Producer-Consumer Pipeline + Multi-Threaded NEON Converter */
    NeonWorkerPool pool(convert_threads);
    ShardConverter converter(fd0, fd1, fdp, cfg.has_ple ? &ple_hdr : nullptr, &pool);

    int done_fd = open(done_path.c_str(), O_CREAT | O_APPEND | O_WRONLY, 0644);
    uint64_t total_downloaded_bytes = 0ULL;
    auto dl_phase_t0 = std::chrono::steady_clock::now();
    bool ok = true;

    auto process_one_shard = [&](uint32_t shard_idx, const std::string& spath, bool delete_after) -> bool {
        auto st0 = std::chrono::steady_clock::now();
        int sfd = open(spath.c_str(), O_RDONLY);
        if (sfd < 0) {
            std::fprintf(stderr, "[flash-stream] ERROR: cannot open shard '%s'\n", spath.c_str());
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
            std::fprintf(stderr, "[flash-stream] ERROR: corrupt safetensors header in '%s'\n", spath.c_str());
            return false;
        }
        vision_tensors_skipped += vis_skip;

        for (const SafetensorEntry& ent : entries) {
            auto it = reg.name_to_id.find(ent.canon_name);
            if (it == reg.name_to_id.end()) {
                close(sfd);
                std::fprintf(stderr, "[flash-stream] ERROR: unexpected tensor '%s' in '%s'\n",
                             ent.canon_name.c_str(), shard_names[shard_idx].c_str());
                return false;
            }
            const TargetSpec& ts = reg.specs[it->second];
            bool conv_ok = false;
            switch (ts.kind) {
                case TGT_UMA_Q4:
                    conv_ok = converter.convert_q4_target(sfd, ent, ts);
                    if (conv_ok) {
                        ++text_tensors_converted;
                        if (ts.is_mtp) ++mtp_tensors_converted;
                    }
                    break;
                case TGT_UMA_F16:
                    conv_ok = converter.convert_f16_target(sfd, ent, ts);
                    if (conv_ok) {
                        ++text_tensors_converted;
                        if (ts.is_mtp) ++mtp_tensors_converted;
                    }
                    break;
                case TGT_UMA_F32:
                    conv_ok = converter.convert_f32_target(sfd, ent, ts);
                    if (conv_ok) {
                        ++text_tensors_converted;
                        if (ts.is_mtp) ++mtp_tensors_converted;
                    }
                    break;
                case TGT_PLE_SHARD:
                    conv_ok = converter.convert_ple_shard(sfd, ent, ts.ple_shard_idx);
                    if (conv_ok) ++ple_shards_converted;
                    break;
                case TGT_PLE_CHK_MULTIPLIERS:
                case TGT_PLE_CHK_OFFSETS:
                case TGT_PLE_CHK_VOCABS:
                    conv_ok = converter.verify_ple_i64(sfd, ent, ts.kind);
                    break;
            }
            if (!conv_ok) {
                close(sfd);
                return false;
            }
            target_seen[ts.id] = 1u;
        }
        close(sfd);

        /* Immediately unlink staged `.safetensors` shard so disk usage stays bounded <= 4 shards! */
        if (delete_after) {
            unlink(spath.c_str());
        }

        /* Persist crash-safe completion record */
        if (done_fd >= 0) {
            std::string rec = shard_names[shard_idx] + "\n";
            (void)write(done_fd, rec.data(), rec.size());
            (void)fsync(done_fd);
        }
        ++num_shards_converted;

        double conv_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - st0).count();
        double wall_so_far = std::chrono::duration<double>(std::chrono::steady_clock::now() - dl_phase_t0).count();
        double net_mbps = (wall_so_far > 1e-6 && total_downloaded_bytes > 0)
                              ? (static_cast<double>(total_downloaded_bytes) / (1e6 * wall_so_far))
                              : 0.0;
        uint32_t done_all = num_shards_resumed + num_shards_converted;
        std::fprintf(stdout,
                     "[flash-stream] [%3u/%zu] %-34s (%5.2f GiB, %3zu tensors) conv %.2fs | "
                     "net %6.1f MB/s | RSS %.3f GiB\n",
                     done_all, shard_names.size(), shard_names[shard_idx].c_str(),
                     static_cast<double>(sbytes) / (1024.0 * 1024.0 * 1024.0),
                     entries.size(), conv_sec, net_mbps, current_peak_rss_gib());
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
                close(fd0);
                close(fd1);
                if (fdp >= 0) close(fdp);
                return -1;
            }
            clean_staging_part_files(stage_dir);

            const uint32_t max_staged = 4u;
            const uint32_t n_dl = (download_concurrency == 0u)
                                      ? 4u
                                      : std::min<uint32_t>(download_concurrency, max_staged);

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

                    /* Acquire bounded staging permit (guarantees <= 4 shards in `.staging/` on disk!) */
                    {
                        std::unique_lock<std::mutex> lk(q_mu);
                        cv_stage.wait(lk, [&]() {
                            return abort_dl.load(std::memory_order_relaxed) || staged_permits > 0;
                        });
                        if (abort_dl.load(std::memory_order_relaxed)) break;
                        --staged_permits;
                    }

                    /* Reuse complete pre-staged shard if valid; otherwise download via `curl` */
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
                        shard_ok = curl_download_atomic(base_url + "/" + sname, dst, &dl_bytes, &dl_sec);
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
                    std::fprintf(stderr, "[flash-stream] ERROR: download failed for '%s'\n",
                                 shard_names[item.shard_idx].c_str());
                    ok = false;
                    abort_dl.store(true, std::memory_order_relaxed);
                    cv_stage.notify_all();
                    break;
                }

                total_downloaded_bytes += item.downloaded_bytes;
                bool conv_ok = process_one_shard(item.shard_idx, item.path, true);

                /* Release staging permit AFTER `process_one_shard` unlinked `item.path`! */
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

    /* Verify 100% tensor coverage across all expected model + PLE + MTP targets */
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
                std::fprintf(stderr, "[flash-stream] ERROR: missing expected tensor '%s' after processing all shards\n",
                             missing_name.c_str());
                ok = false;
                break;
            }
        }
    }

    if (!ok) {
        close(fd0);
        close(fd1);
        if (fdp >= 0) close(fdp);
        return -1;
    }

    /* 6. Re-stamp headers at offset 0 & fsync all `.strata` files */
    if (!pwrite_exact(fd0, &part0_hdr, sizeof(part0_hdr), 0) ||
        !pwrite_exact(fd1, &part1_hdr, sizeof(part1_hdr), 0) ||
        (cfg.has_ple && !pwrite_exact(fdp, &ple_hdr, sizeof(ple_hdr), 0))) {
        close(fd0);
        close(fd1);
        if (fdp >= 0) close(fdp);
        return -1;
    }
    (void)fsync(fd0);
    (void)fsync(fd1);
    if (fdp >= 0) (void)fsync(fdp);
    close(fd0);
    close(fd1);
    if (fdp >= 0) close(fdp);

    /* 7. Clean `.staging/` directory & verify 16 KiB alignment */
    clean_staging_part_files(stage_dir);
    (void)rmdir(stage_dir.c_str());
    bool staging_clean = check_dir_has_no_files(stage_dir);

    uint64_t actual_p0 = 0, actual_p1 = 0, actual_ple = 0;
    bool aligned_ok =
        file_exists_nonempty(part0_path, &actual_p0) &&
        file_exists_nonempty(part1_path, &actual_p1) &&
        actual_p0 == part0_hdr.total_file_bytes &&
        actual_p1 == part1_hdr.total_file_bytes &&
        (actual_p0 % STRATA_QWEN_FLASH_PAGE_BYTES) == 0ULL &&
        (actual_p1 % STRATA_QWEN_FLASH_PAGE_BYTES) == 0ULL &&
        (!cfg.has_ple || (file_exists_nonempty(ple_path, &actual_ple) &&
                          actual_ple == ple_hdr.total_file_bytes &&
                          (actual_ple % STRATA_QWEN_FLASH_PAGE_BYTES) == 0ULL));

    double elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_t0).count();
    double dl_elapsed  = std::chrono::duration<double>(std::chrono::steady_clock::now() - dl_phase_t0).count();
    double net_mbps    = (dl_elapsed > 1e-6 && total_downloaded_bytes > 0)
                             ? (static_cast<double>(total_downloaded_bytes) / (1e6 * dl_elapsed))
                             : 0.0;
    double peak_rss    = current_peak_rss_gib();
    uint64_t hf_bytes  = (index_total_bytes > 0) ? index_total_bytes : total_hf_bytes_processed;

    /* 8. Write human-readable `manifest.json` */
    {
        std::string mf_path = out_dir + "/manifest.json";
        FILE* mf = std::fopen(mf_path.c_str(), "wb");
        if (mf) {
            const std::string safe_src = format_manifest_source_json(src_spec, is_local_dir);
            std::fprintf(mf,
                "{\n"
                "  \"format\": \"STRATA_QWEN_FLASH_V1\",\n"
                "  \"architecture\": \"Qwen4ExpForConditionalGeneration\",\n"
                "  \"source\": \"%s\",\n"
                "  \"num_hidden_layers\": %u,\n"
                "  \"split_layer_idx\": %u,\n"
                "  \"hidden_size\": %u,\n"
                "  \"num_experts\": %u,\n"
                "  \"num_experts_per_tok\": %u,\n"
                "  \"moe_intermediate_size\": %u,\n"
                "  \"vocab_size\": %u,\n"
                "  \"has_ple\": %u,\n"
                "  \"has_mtp\": %u,\n"
                "  \"page_align_bytes\": %llu,\n"
                "  \"hf_safetensors_bytes\": %llu,\n"
                "  \"uma_part0_bytes\": %llu,\n"
                "  \"uma_part1_bytes\": %llu,\n"
                "  \"ple_ngram_bytes\": %llu,\n"
                "  \"total_bundle_bytes\": %llu,\n"
                "  \"active_bytes_per_tok\": %llu,\n"
                "  \"mtp_active_bytes_per_tok\": %llu,\n"
                "  \"num_shards_total\": %zu,\n"
                "  \"num_text_tensors_converted\": %u,\n"
                "  \"num_ple_shards_converted\": %u,\n"
                "  \"num_mtp_tensors_converted\": %u,\n"
                "  \"num_vision_tensors_skipped\": %u,\n"
                "  \"elapsed_sec\": %.3f,\n"
                "  \"network_mbps\": %.2f,\n"
                "  \"peak_rss_gib\": %.4f\n"
                "}\n",
                safe_src.c_str(),
                cfg.num_hidden_layers, cfg.split_layer_idx, cfg.hidden_size,
                cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size,
                cfg.vocab_size, cfg.has_ple, cfg.has_mtp,
                static_cast<unsigned long long>(STRATA_QWEN_FLASH_PAGE_BYTES),
                static_cast<unsigned long long>(hf_bytes),
                static_cast<unsigned long long>(actual_p0),
                static_cast<unsigned long long>(actual_p1),
                static_cast<unsigned long long>(actual_ple),
                static_cast<unsigned long long>(actual_p0 + actual_p1 + actual_ple),
                static_cast<unsigned long long>(part0_hdr.active_bytes_per_tok),
                static_cast<unsigned long long>(part0_hdr.mtp_active_bytes_per_tok),
                shard_names.size(),
                text_tensors_converted, ple_shards_converted,
                mtp_tensors_converted, vision_tensors_skipped,
                elapsed_sec, net_mbps, peak_rss);
            std::fclose(mf);
        }
    }

    if (stats_out) {
        stats_out->num_shards_total           = static_cast<uint32_t>(shard_names.size());
        stats_out->num_shards_converted       = num_shards_converted;
        stats_out->num_shards_resumed         = num_shards_resumed;
        stats_out->num_text_tensors_converted = text_tensors_converted;
        stats_out->num_ple_shards_converted   = ple_shards_converted;
        stats_out->num_mtp_tensors_converted  = mtp_tensors_converted;
        stats_out->num_vision_tensors_skipped = vision_tensors_skipped;
        stats_out->hf_safetensors_bytes       = hf_bytes;
        stats_out->uma_part0_bytes            = actual_p0;
        stats_out->uma_part1_bytes            = actual_p1;
        stats_out->ple_ngram_bytes            = actual_ple;
        stats_out->total_bundle_bytes         = actual_p0 + actual_p1 + actual_ple;
        stats_out->active_bytes_per_tok       = part0_hdr.active_bytes_per_tok;
        stats_out->elapsed_sec                = elapsed_sec;
        stats_out->network_mbps               = net_mbps;
        stats_out->peak_rss_gib               = peak_rss;
        stats_out->page_aligned_verified      = aligned_ok;
        stats_out->staging_clean_verified     = staging_clean;
    }

    std::fprintf(stdout,
                 "[flash-stream] COMPLETE in %.2f s (net %.1f MB/s): Part0=%.3f GiB + Part1=%.3f GiB + PLE=%.3f GiB "
                 "= %.3f GiB | aligned=%d | staging_clean=%d | Peak RSS=%.3f GiB\n",
                 elapsed_sec, net_mbps,
                 static_cast<double>(actual_p0) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(actual_p1) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(actual_ple) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(actual_p0 + actual_p1 + actual_ple) / (1024.0 * 1024.0 * 1024.0),
                 aligned_ok ? 1 : 0, staging_clean ? 1 : 0, peak_rss);
    std::fflush(stdout);
    return (aligned_ok && staging_clean) ? 0 : -1;
}

int strata_qwen_flash_convert_hf_directory(
    const char* hf_repo_or_dir,
    const char* out_bundle_dir,
    StrataQwenFlashConvertStats* stats_out
) {
    return strata_qwen_flash_convert_hf(hf_repo_or_dir, out_bundle_dir, 0u, 0u, stats_out);
}

int strata_qwen_flash_load_bundle(
    const char* bundle_dir,
    size_t      ple_cache_bytes,
    StrataQwenFlashBundle* bundle_out
) {
    if (!bundle_out) return -1;
    std::memset(bundle_out, 0, sizeof(*bundle_out));
    bundle_out->part0_fd = -1;
    bundle_out->part1_fd = -1;
    bundle_out->ple_fd   = -1;

    const char* dir = (bundle_dir && bundle_dir[0] != '\0')
                          ? bundle_dir
                          : STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR;
    std::snprintf(bundle_out->bundle_dir, sizeof(bundle_out->bundle_dir), "%s", dir);
    std::snprintf(bundle_out->part0_path, sizeof(bundle_out->part0_path), "%s/%s",
                  dir, STRATA_QWEN_FLASH_PART0_FILENAME);
    std::snprintf(bundle_out->part1_path, sizeof(bundle_out->part1_path), "%s/%s",
                  dir, STRATA_QWEN_FLASH_PART1_FILENAME);
    std::snprintf(bundle_out->ple_path, sizeof(bundle_out->ple_path), "%s/%s",
                  dir, STRATA_QWEN_FLASH_PLE_FILENAME);

    auto fail_cleanup = [&]() -> int {
        strata_qwen_flash_free_bundle(bundle_out);
        return -1;
    };

    /* 1. Zero-copy 16 KiB-aligned read-only mmap of `uma_part0.strata` */
    bundle_out->part0_fd = open(bundle_out->part0_path, O_RDONLY);
    if (bundle_out->part0_fd < 0) {
        std::fprintf(stderr, "[flash-load] ERROR: cannot open '%s'\n", bundle_out->part0_path);
        return fail_cleanup();
    }
    struct stat st0;
    if (fstat(bundle_out->part0_fd, &st0) != 0 ||
        static_cast<uint64_t>(st0.st_size) < STRATA_QWEN_FLASH_HEADER_PADDED_BYTES ||
        (static_cast<uint64_t>(st0.st_size) % STRATA_QWEN_FLASH_PAGE_BYTES) != 0ULL) {
        std::fprintf(stderr, "[flash-load] ERROR: invalid/unaligned size on '%s'\n", bundle_out->part0_path);
        return fail_cleanup();
    }
    bundle_out->part0_bytes = static_cast<size_t>(st0.st_size);
    void* m0 = mmap(nullptr, bundle_out->part0_bytes, PROT_READ, MAP_PRIVATE, bundle_out->part0_fd, 0);
    if (m0 == MAP_FAILED) return fail_cleanup();
    bundle_out->part0_base = static_cast<const uint8_t*>(m0);
    if ((reinterpret_cast<uintptr_t>(bundle_out->part0_base) % STRATA_QWEN_FLASH_PAGE_BYTES) != 0u) {
        return fail_cleanup();
    }
    bundle_out->header0 = reinterpret_cast<const StrataQwenFlashPart0Header*>(bundle_out->part0_base);
    if (!validate_sqf_part0_header(bundle_out->header0, bundle_out->part0_bytes)) {
        std::fprintf(stderr, "[flash-load] ERROR: header0 validation failed on '%s'\n", bundle_out->part0_path);
        return fail_cleanup();
    }

    /* 2. Zero-copy 16 KiB-aligned read-only mmap of `uma_part1.strata` */
    bundle_out->part1_fd = open(bundle_out->part1_path, O_RDONLY);
    if (bundle_out->part1_fd < 0) {
        std::fprintf(stderr, "[flash-load] ERROR: cannot open '%s'\n", bundle_out->part1_path);
        return fail_cleanup();
    }
    struct stat st1;
    if (fstat(bundle_out->part1_fd, &st1) != 0 ||
        static_cast<uint64_t>(st1.st_size) < STRATA_QWEN_FLASH_HEADER_PADDED_BYTES ||
        (static_cast<uint64_t>(st1.st_size) % STRATA_QWEN_FLASH_PAGE_BYTES) != 0ULL) {
        std::fprintf(stderr, "[flash-load] ERROR: invalid/unaligned size on '%s'\n", bundle_out->part1_path);
        return fail_cleanup();
    }
    bundle_out->part1_bytes = static_cast<size_t>(st1.st_size);
    void* m1 = mmap(nullptr, bundle_out->part1_bytes, PROT_READ, MAP_PRIVATE, bundle_out->part1_fd, 0);
    if (m1 == MAP_FAILED) return fail_cleanup();
    bundle_out->part1_base = static_cast<const uint8_t*>(m1);
    if ((reinterpret_cast<uintptr_t>(bundle_out->part1_base) % STRATA_QWEN_FLASH_PAGE_BYTES) != 0u) {
        return fail_cleanup();
    }
    bundle_out->header1 = reinterpret_cast<const StrataQwenFlashPart1Header*>(bundle_out->part1_base);
    if (!validate_sqf_part1_header(bundle_out->header0, bundle_out->header1, bundle_out->part1_bytes)) {
        std::fprintf(stderr, "[flash-load] ERROR: header1 validation failed on '%s'\n", bundle_out->part1_path);
        return fail_cleanup();
    }

    bundle_out->cfg = bundle_out->header0->cfg;

    /* 3. `F_NOCACHE` NVMe PLE N-gram store + `mlock` 16-bank 4-way LFU cache */
    if (bundle_out->cfg.has_ple) {
        bundle_out->ple_fd = open(bundle_out->ple_path, O_RDONLY);
        if (bundle_out->ple_fd < 0) {
            std::fprintf(stderr, "[flash-load] ERROR: cannot open '%s'\n", bundle_out->ple_path);
            return fail_cleanup();
        }
#if defined(F_NOCACHE)
        (void)fcntl(bundle_out->ple_fd, F_NOCACHE, 1);
#endif
        struct stat stp;
        if (fstat(bundle_out->ple_fd, &stp) != 0 ||
            static_cast<uint64_t>(stp.st_size) < STRATA_QWEN_FLASH_PLE_HEADER_BYTES ||
            (static_cast<uint64_t>(stp.st_size) % STRATA_QWEN_FLASH_PAGE_BYTES) != 0ULL) {
            std::fprintf(stderr, "[flash-load] ERROR: invalid/unaligned size on '%s'\n", bundle_out->ple_path);
            return fail_cleanup();
        }
        bundle_out->ple_file_bytes = static_cast<uint64_t>(stp.st_size);
        auto validate_ple_hdr = [&]() -> bool {
            const auto& ph = bundle_out->ple_header;
            if (ph.magic != STRATA_QWEN_FLASH_PLE_MAGIC_U64 ||
                ph.version != STRATA_QWEN_FLASH_FORMAT_VERSION ||
                ph.total_file_bytes != bundle_out->ple_file_bytes ||
                ph.data_page_off < STRATA_QWEN_FLASH_PLE_HEADER_BYTES ||
                ph.data_page_off > bundle_out->ple_file_bytes ||
                ph.ngram_heads == 0U ||
                ph.ngram_heads > STRATA_QWEN_FLASH_MAX_NGRAM_HEADS ||
                ph.ngram_size < 2U ||
                ph.ngram_size > 4U ||
                ph.ngram_size > STRATA_QWEN_FLASH_MAX_NGRAM_SIZE ||
                bundle_out->cfg.ngram_size != ph.ngram_size ||
                ph.heads_per_ngram == 0U ||
                static_cast<uint64_t>(ph.ngram_size - 1U) * ph.heads_per_ngram != ph.ngram_heads ||
                ph.quant_group_size != STRATA_QWEN_FLASH_QUANT_GROUP_SIZE ||
                bundle_out->cfg.ple_embed_dim == 0U ||
                bundle_out->cfg.ple_embed_dim > 32768U ||
                (bundle_out->cfg.ple_embed_dim % ph.ngram_heads) != 0U ||
                ph.head_dim_per_ngram != bundle_out->cfg.ple_embed_dim / ph.ngram_heads ||
                ph.head_dim_per_ngram == 0U ||
                ph.head_dim_per_ngram > STRATA_QWEN_FLASH_MAX_PLE_HEAD_DIM ||
                (ph.head_dim_per_ngram & 7U) != 0U) {
                return false;
            }
            const uint32_t exp_grps = (ph.head_dim_per_ngram + ph.quant_group_size - 1U) / ph.quant_group_size;
            const uint32_t exp_rb   = (ph.head_dim_per_ngram / 2U) + exp_grps * 4U;
            if (ph.row_bytes < exp_rb || ph.row_bytes > 512U ||
                ph.padded_vocab_rows > (bundle_out->ple_file_bytes - ph.data_page_off) / ph.row_bytes) {
                return false;
            }
            for (uint32_t h = 0U; h < ph.ngram_heads; ++h) {
                if (ph.head_vocab_sizes[h] == 0ULL ||
                    ph.head_offsets[h] > ph.padded_vocab_rows ||
                    ph.head_vocab_sizes[h] > ph.padded_vocab_rows - ph.head_offsets[h]) {
                    return false;
                }
            }
            return true;
        };
        if (!pread_exact(bundle_out->ple_fd, &bundle_out->ple_header, sizeof(bundle_out->ple_header), 0) ||
            !validate_ple_hdr()) {
            std::fprintf(stderr, "[flash-load] ERROR: ple_header validation failed on '%s'\n", bundle_out->ple_path);
            return fail_cleanup();
        }
        if (strata_qwen_flash_ple_cache_init(
                &bundle_out->ple_cache,
                bundle_out->cfg.ple_embed_dim,
                bundle_out->ple_header.ngram_heads,
                ple_cache_bytes) != 0) {
            std::fprintf(stderr, "[flash-load] ERROR: failed to init PLE mlock LFU cache\n");
            return fail_cleanup();
        }
    }

    /* 4. Load BPE tokenizer (`<bundle_dir>/tokenizer.json` if present) */
    std::string tok_path = std::string(dir) + "/tokenizer.json";
    if (file_exists_nonempty(tok_path)) {
        bundle_out->tokenizer = strata_qwen_flash_tokenizer_load(tok_path.c_str());
        if (!bundle_out->tokenizer) {
            std::fprintf(stderr, "[flash-load] ERROR: failed to parse '%s'\n", tok_path.c_str());
            return fail_cleanup();
        }
    }

    return 0;
}

void strata_qwen_flash_free_bundle(StrataQwenFlashBundle* bundle) {
    if (!bundle) return;
    if (bundle->tokenizer) {
        strata_qwen_flash_tokenizer_free(bundle->tokenizer);
        bundle->tokenizer = nullptr;
    }
    strata_qwen_flash_ple_cache_free(&bundle->ple_cache);
    if (bundle->ple_fd >= 0) {
        close(bundle->ple_fd);
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
    std::memset(bundle, 0, sizeof(*bundle));
    bundle->part0_fd = -1;
    bundle->part1_fd = -1;
    bundle->ple_fd   = -1;
}

static inline void sqf_quantize_group64_fp16_to_q8(
    const uint16_t* src_fp16,
    int8_t*         dst_i8,
    uint16_t*       dst_scale
) {
    float vals[64];
    float max_abs = 0.0f;
#if defined(__aarch64__)
    float32x4_t vmax4 = vdupq_n_f32(0.0f);
    for (int k = 0; k < 16; ++k) {
        float16x4_t h4 = vreinterpret_f16_u16(vld1_u16(src_fp16 + k * 4));
        float32x4_t f4 = vcvt_f32_f16(h4);
        vst1q_f32(vals + k * 4, f4);
        vmax4 = vmaxq_f32(vmax4, vabsq_f32(f4));
    }
    max_abs = vmaxvq_f32(vmax4);
#else
    for (int i = 0; i < 64; ++i) {
        float x = fp16_to_f32_fast(src_fp16[i]);
        vals[i] = x;
        float ax = std::fabs(x);
        if (ax > max_abs) max_abs = ax;
    }
#endif
    const uint16_t scale_h = f32_to_fp16_fast(max_abs * (1.0f / 127.0f));
    *dst_scale = scale_h;
    const float scale_eff = fp16_to_f32_fast(scale_h);
    if (!(scale_eff > 0.0f)) {
        std::memset(dst_i8, 0, 64);
        return;
    }
    const float inv_s = 1.0f / scale_eff;
#if defined(__aarch64__)
    const float32x4_t v_invs = vdupq_n_f32(inv_s);
    const float32x4_t v_lo   = vdupq_n_f32(-127.0f);
    const float32x4_t v_hi   = vdupq_n_f32(127.0f);
    for (int k = 0; k < 16; ++k) {
        float32x4_t qf = vrndnq_f32(vmulq_f32(vld1q_f32(vals + k * 4), v_invs));
        qf = vminq_f32(v_hi, vmaxq_f32(v_lo, qf));
        int32x4_t qi = vcvtq_s32_f32(qf);
        dst_i8[k * 4 + 0] = static_cast<int8_t>(vgetq_lane_s32(qi, 0));
        dst_i8[k * 4 + 1] = static_cast<int8_t>(vgetq_lane_s32(qi, 1));
        dst_i8[k * 4 + 2] = static_cast<int8_t>(vgetq_lane_s32(qi, 2));
        dst_i8[k * 4 + 3] = static_cast<int8_t>(vgetq_lane_s32(qi, 3));
    }
#else
    for (int i = 0; i < 64; ++i) {
        float qf = std::nearbyintf(vals[i] * inv_s);
        if (qf < -127.0f) qf = -127.0f;
        else if (qf > 127.0f) qf = 127.0f;
        dst_i8[i] = static_cast<int8_t>(qf);
    }
#endif
}

static inline void sqf_quantize_group64_fp16_to_q4(
    const uint16_t* src_fp16,
    uint32_t*       dst_u32,
    uint16_t*       dst_scale,
    uint16_t*       dst_bias
) {
    float vals[64];
    for (int i = 0; i < 64; ++i) vals[i] = fp16_to_f32_fast(src_fp16[i]);
    quantize_group_f32(vals, 64u, dst_u32, dst_scale, dst_bias);
}

int strata_qwen_flash_repack_bundle(
    const char* src_bundle_dir,
    const char* dst_bundle_dir,
    uint32_t    convert_threads,
    StrataQwenFlashConvertStats* stats_out
) {
    if (stats_out) std::memset(stats_out, 0, sizeof(*stats_out));
    const std::string src_dir = (src_bundle_dir && src_bundle_dir[0] != '\0')
        ? std::string(src_bundle_dir) : std::string(STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR);
    if (!dst_bundle_dir || dst_bundle_dir[0] == '\0') {
        std::fprintf(stderr, "[flash-repack] ERROR: destination directory is required\n");
        return -1;
    }
    const std::string dst_dir(dst_bundle_dir);
    if (src_dir == dst_dir) {
        std::fprintf(stderr, "[flash-repack] ERROR: src and dst directories must be distinct (repack side-by-side then swap)\n");
        return -2;
    }

    const auto t0 = std::chrono::steady_clock::now();

    const std::string src_p0_path = src_dir + "/" + STRATA_QWEN_FLASH_PART0_FILENAME;
    const std::string src_p1_path = src_dir + "/" + STRATA_QWEN_FLASH_PART1_FILENAME;
    const std::string src_pl_path = src_dir + "/" + STRATA_QWEN_FLASH_PLE_FILENAME;

    int src_fd0 = open(src_p0_path.c_str(), O_RDONLY);
    int src_fd1 = open(src_p1_path.c_str(), O_RDONLY);
    if (src_fd0 < 0 || src_fd1 < 0) {
        std::fprintf(stderr, "[flash-repack] ERROR: failed to open source uma_part{0,1}.strata in '%s'\n", src_dir.c_str());
        if (src_fd0 >= 0) close(src_fd0);
        if (src_fd1 >= 0) close(src_fd1);
        return -3;
    }
#if defined(F_NOCACHE)
    (void)fcntl(src_fd0, F_NOCACHE, 1);
    (void)fcntl(src_fd1, F_NOCACHE, 1);
#endif

    struct stat st0, st1;
    if (fstat(src_fd0, &st0) != 0 || fstat(src_fd1, &st1) != 0 ||
        static_cast<uint64_t>(st0.st_size) < STRATA_QWEN_FLASH_HEADER_PADDED_BYTES ||
        static_cast<uint64_t>(st1.st_size) < STRATA_QWEN_FLASH_HEADER_PADDED_BYTES ||
        (static_cast<uint64_t>(st0.st_size) % STRATA_QWEN_FLASH_PAGE_BYTES) != 0ULL ||
        (static_cast<uint64_t>(st1.st_size) % STRATA_QWEN_FLASH_PAGE_BYTES) != 0ULL) {
        std::fprintf(stderr, "[flash-repack] ERROR: invalid or unaligned source bundle files\n");
        close(src_fd0);
        close(src_fd1);
        return -4;
    }

    auto src_h0 = std::make_unique<StrataQwenFlashPart0Header>();
    auto src_h1 = std::make_unique<StrataQwenFlashPart1Header>();
    if (!pread_exact(src_fd0, src_h0.get(), sizeof(*src_h0), 0) ||
        !pread_exact(src_fd1, src_h1.get(), sizeof(*src_h1), 0) ||
        !validate_sqf_part0_header(src_h0.get(), static_cast<uint64_t>(st0.st_size)) ||
        !validate_sqf_part1_header(src_h0.get(), src_h1.get(), static_cast<uint64_t>(st1.st_size))) {
        std::fprintf(stderr, "[flash-repack] ERROR: invalid source bundle headers\n");
        close(src_fd0);
        close(src_fd1);
        return -4;
    }

    if (!mkdirs(dst_dir)) {
        std::fprintf(stderr, "[flash-repack] ERROR: failed to create dst dir '%s'\n", dst_dir.c_str());
        close(src_fd0);
        close(src_fd1);
        return -5;
    }

    auto dst_h0 = std::make_unique<StrataQwenFlashPart0Header>();
    auto dst_h1 = std::make_unique<StrataQwenFlashPart1Header>();
    strata_qwen_flash_init_layout_quant(&src_h0->cfg, /*quantize_dense=*/true, dst_h0.get(), dst_h1.get(), nullptr);

    const std::string dst_p0_tmp = dst_dir + "/" + STRATA_QWEN_FLASH_PART0_FILENAME + ".part";
    const std::string dst_p1_tmp = dst_dir + "/" + STRATA_QWEN_FLASH_PART1_FILENAME + ".part";
    const std::string dst_p0_fin = dst_dir + "/" + STRATA_QWEN_FLASH_PART0_FILENAME;
    const std::string dst_p1_fin = dst_dir + "/" + STRATA_QWEN_FLASH_PART1_FILENAME;

    int dst_fd0 = open(dst_p0_tmp.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    int dst_fd1 = open(dst_p1_tmp.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (dst_fd0 < 0 || dst_fd1 < 0) {
        std::fprintf(stderr, "[flash-repack] ERROR: failed to create destination .part files\n");
        if (dst_fd0 >= 0) close(dst_fd0);
        if (dst_fd1 >= 0) close(dst_fd1);
        close(src_fd0);
        close(src_fd1);
        return -6;
    }
#if defined(F_NOCACHE)
    (void)fcntl(dst_fd0, F_NOCACHE, 1);
    (void)fcntl(dst_fd1, F_NOCACHE, 1);
#endif

    bool ok = (ftruncate(dst_fd0, static_cast<off_t>(dst_h0->total_file_bytes)) == 0) &&
              (ftruncate(dst_fd1, static_cast<off_t>(dst_h1->total_file_bytes)) == 0);

    std::vector<uint8_t> hdr_pad(STRATA_QWEN_FLASH_HEADER_PADDED_BYTES, 0u);
    if (ok) {
        std::memcpy(hdr_pad.data(), dst_h0.get(), sizeof(*dst_h0));
        ok = pwrite_exact(dst_fd0, hdr_pad.data(), hdr_pad.size(), 0);
    }
    if (ok) {
        std::memset(hdr_pad.data(), 0, hdr_pad.size());
        std::memcpy(hdr_pad.data(), dst_h1.get(), sizeof(*dst_h1));
        ok = pwrite_exact(dst_fd1, hdr_pad.data(), hdr_pad.size(), 0);
    }

    NeonWorkerPool pool(convert_threads);
    constexpr size_t kCopyChunk = 64ULL * 1024ULL * 1024ULL;
    std::vector<uint8_t> buf_in(kCopyChunk);
    std::vector<uint8_t> buf_w(64ULL * 1024ULL * 1024ULL);
    std::vector<uint8_t> buf_s(4ULL * 1024ULL * 1024ULL);
    std::vector<uint8_t> buf_b(4ULL * 1024ULL * 1024ULL);
    uint32_t tensors_processed = 0U;

    auto copy_span = [&](int sfd, uint64_t soff, int dfd, uint64_t doff, uint64_t len) -> bool {
        uint64_t done = 0ULL;
        while (done < len) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(buf_in.size(), len - done));
            if (!pread_exact(sfd, buf_in.data(), n, soff + done) ||
                !pwrite_exact(dfd, buf_in.data(), n, doff + done)) {
                return false;
            }
            done += n;
        }
        return true;
    };

    auto repack_qmat = [&](int sfd, const StrataQwenFlashQuantMatrix& sm,
                           int dfd, const StrataQwenFlashQuantMatrix& dm) -> bool {
        if (sm.weight_bytes != dm.weight_bytes || sm.scales_bytes != dm.scales_bytes ||
            sm.biases_bytes != dm.biases_bytes) {
            return false;
        }
        ++tensors_processed;
        return copy_span(sfd, sm.weight_off, dfd, dm.weight_off, sm.weight_bytes) &&
               copy_span(sfd, sm.scales_off, dfd, dm.scales_off, sm.scales_bytes) &&
               copy_span(sfd, sm.biases_off, dfd, dm.biases_off, sm.biases_bytes);
    };

    auto repack_slice = [&](int sfd, const StrataQwenFlashTensorSlice& ss,
                            int dfd, const StrataQwenFlashTensorSlice& ds) -> bool {
        if (ss.shape0 != ds.shape0 || ss.shape1 != ds.shape1) return false;
        ++tensors_processed;
        if (ss.dtype == ds.dtype) {
            if (ss.bytes != ds.bytes) return false;
            return copy_span(sfd, ss.offset, dfd, ds.offset, ss.bytes);
        }
        if (ss.dtype != STRATA_QWEN_FLASH_DTYPE_F16) return false;

        const uint32_t rows = ds.shape0;
        const uint32_t cols = ds.shape1;
        const uint32_t grp_per_row = cols / STRATA_QWEN_FLASH_QUANT_GROUP_SIZE;
        const uint64_t in_bytes = static_cast<uint64_t>(rows) * cols * sizeof(uint16_t);
        if (in_bytes > buf_in.size()) return false;
        if (!pread_exact(sfd, buf_in.data(), static_cast<size_t>(in_bytes), ss.offset)) return false;
        const uint16_t* src_h = reinterpret_cast<const uint16_t*>(buf_in.data());

        if (ds.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) {
            uint64_t wb = 0ULL, sb = 0ULL;
            sqf_q8_slice_layout(rows, cols, &wb, &sb);
            if (wb > buf_w.size() || sb > buf_s.size()) return false;
            std::memset(buf_w.data(), 0, static_cast<size_t>(wb));
            std::memset(buf_s.data(), 0, static_cast<size_t>(sb));
            int8_t*   w_out = reinterpret_cast<int8_t*>(buf_w.data());
            uint16_t* s_out = reinterpret_cast<uint16_t*>(buf_s.data());

            pool.parallel_for_rows(rows, 16ULL, [&](uint64_t r0, uint64_t r1) {
                for (uint64_t r = r0; r < r1; ++r) {
                    const uint16_t* row_in = src_h + r * cols;
                    int8_t*         row_w  = w_out + r * cols;
                    uint16_t*       row_s  = s_out + r * grp_per_row;
                    for (uint32_t g = 0U; g < grp_per_row; ++g) {
                        sqf_quantize_group64_fp16_to_q8(row_in + g * 64U, row_w + g * 64U, row_s + g);
                    }
                }
            });
            return pwrite_exact(dfd, buf_w.data(), static_cast<size_t>(wb), ds.offset) &&
                   pwrite_exact(dfd, buf_s.data(), static_cast<size_t>(sb), ds.offset + wb);
        }

        if (ds.dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A) {
            const StrataQwenFlashQuantMatrix qm = sqf_q4_slice_as_qmat(&ds);
            if (qm.weight_bytes > buf_w.size() || qm.scales_bytes > buf_s.size() || qm.biases_bytes > buf_b.size()) {
                return false;
            }
            std::memset(buf_w.data(), 0, static_cast<size_t>(qm.weight_bytes));
            std::memset(buf_s.data(), 0, static_cast<size_t>(qm.scales_bytes));
            std::memset(buf_b.data(), 0, static_cast<size_t>(qm.biases_bytes));
            uint32_t* w_out = reinterpret_cast<uint32_t*>(buf_w.data());
            uint16_t* s_out = reinterpret_cast<uint16_t*>(buf_s.data());
            uint16_t* b_out = reinterpret_cast<uint16_t*>(buf_b.data());
            const uint32_t u32_per_row = cols / 8U;

            pool.parallel_for_rows(rows, 16ULL, [&](uint64_t r0, uint64_t r1) {
                for (uint64_t r = r0; r < r1; ++r) {
                    const uint16_t* row_in = src_h + r * cols;
                    uint32_t*       row_w  = w_out + r * u32_per_row;
                    uint16_t*       row_s  = s_out + r * grp_per_row;
                    uint16_t*       row_b  = b_out + r * grp_per_row;
                    for (uint32_t g = 0U; g < grp_per_row; ++g) {
                        sqf_quantize_group64_fp16_to_q4(row_in + g * 64U, row_w + g * 8U, row_s + g, row_b + g);
                    }
                }
            });
            return pwrite_exact(dfd, buf_w.data(), static_cast<size_t>(qm.weight_bytes), qm.weight_off) &&
                   pwrite_exact(dfd, buf_s.data(), static_cast<size_t>(qm.scales_bytes), qm.scales_off) &&
                   pwrite_exact(dfd, buf_b.data(), static_cast<size_t>(qm.biases_bytes), qm.biases_off);
        }

        return false;
    };

    auto repack_hc = [&](int sfd, const StrataQwenFlashHCDesc& sh,
                         int dfd, const StrataQwenFlashHCDesc& dh) -> bool {
        return repack_slice(sfd, sh.hc_norm, dfd, dh.hc_norm) &&
               repack_slice(sfd, sh.input_mix_weight_down, dfd, dh.input_mix_weight_down) &&
               repack_slice(sfd, sh.input_mix_weight_up, dfd, dh.input_mix_weight_up) &&
               repack_slice(sfd, sh.block_inject_weight, dfd, dh.block_inject_weight);
    };

    auto repack_mixer = [&](int sfd, const StrataQwenFlashMixerDesc& sm,
                            int dfd, const StrataQwenFlashMixerDesc& dm) -> bool {
        return repack_slice(sfd, sm.hc_norm, dfd, dm.hc_norm) &&
               repack_slice(sfd, sm.input_mix_weight_down, dfd, dm.input_mix_weight_down) &&
               repack_slice(sfd, sm.input_mix_weight_up, dfd, dm.input_mix_weight_up);
    };

    auto repack_layer = [&](int sfd, const StrataQwenFlashLayerDesc& sl,
                            int dfd, const StrataQwenFlashLayerDesc& dl) -> bool {
        if (sl.has_ple != dl.has_ple || sl.layer_type != dl.layer_type) return false;
        if (sl.has_ple) {
            if (!repack_slice(sfd, sl.ple.key_proj, dfd, dl.ple.key_proj) ||
                !repack_slice(sfd, sl.ple.value_proj, dfd, dl.ple.value_proj) ||
                !repack_slice(sfd, sl.ple.norm_key, dfd, dl.ple.norm_key) ||
                !repack_slice(sfd, sl.ple.norm_query, dfd, dl.ple.norm_query) ||
                !repack_slice(sfd, sl.ple.norm_conv, dfd, dl.ple.norm_conv) ||
                !repack_slice(sfd, sl.ple.conv1d_weight, dfd, dl.ple.conv1d_weight)) {
                return false;
            }
        }
        if (!repack_hc(sfd, sl.attn_hc, dfd, dl.attn_hc) ||
            !repack_hc(sfd, sl.mlp_hc, dfd, dl.mlp_hc)) {
            return false;
        }
        if (sl.layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) {
            if (!repack_qmat(sfd, sl.lin_in_proj_qkv, dfd, dl.lin_in_proj_qkv) ||
                !repack_qmat(sfd, sl.lin_in_proj_z, dfd, dl.lin_in_proj_z) ||
                !repack_qmat(sfd, sl.lin_out_proj, dfd, dl.lin_out_proj) ||
                !repack_slice(sfd, sl.lin_in_proj_a, dfd, dl.lin_in_proj_a) ||
                !repack_slice(sfd, sl.lin_in_proj_b, dfd, dl.lin_in_proj_b) ||
                !repack_slice(sfd, sl.lin_conv1d_weight, dfd, dl.lin_conv1d_weight) ||
                !repack_slice(sfd, sl.lin_A_log, dfd, dl.lin_A_log) ||
                !repack_slice(sfd, sl.lin_dt_bias, dfd, dl.lin_dt_bias) ||
                !repack_slice(sfd, sl.lin_norm, dfd, dl.lin_norm)) {
                return false;
            }
        } else {
            if (!repack_qmat(sfd, sl.attn_q_proj, dfd, dl.attn_q_proj) ||
                !repack_qmat(sfd, sl.attn_k_proj, dfd, dl.attn_k_proj) ||
                !repack_qmat(sfd, sl.attn_v_proj, dfd, dl.attn_v_proj) ||
                !repack_qmat(sfd, sl.attn_o_proj, dfd, dl.attn_o_proj) ||
                !repack_slice(sfd, sl.attn_q_norm, dfd, dl.attn_q_norm) ||
                !repack_slice(sfd, sl.attn_k_norm, dfd, dl.attn_k_norm) ||
                !repack_slice(sfd, sl.indexer_qk_proj, dfd, dl.indexer_qk_proj) ||
                !repack_slice(sfd, sl.indexer_q_norm, dfd, dl.indexer_q_norm) ||
                !repack_slice(sfd, sl.indexer_k_norm, dfd, dl.indexer_k_norm)) {
                return false;
            }
        }
        return repack_slice(sfd, sl.moe_router_gate, dfd, dl.moe_router_gate) &&
               repack_slice(sfd, sl.shared_gate_proj, dfd, dl.shared_gate_proj) &&
               repack_slice(sfd, sl.shared_up_proj, dfd, dl.shared_up_proj) &&
               repack_slice(sfd, sl.shared_down_proj, dfd, dl.shared_down_proj) &&
               repack_slice(sfd, sl.shared_expert_gate, dfd, dl.shared_expert_gate) &&
               repack_qmat(sfd, sl.moe_experts_gate_up, dfd, dl.moe_experts_gate_up) &&
               repack_qmat(sfd, sl.moe_experts_down, dfd, dl.moe_experts_down);
    };

    if (ok) ok = repack_qmat(src_fd0, src_h0->embed_tokens, dst_fd0, dst_h0->embed_tokens);
    for (uint32_t l = 0U; ok && l < src_h0->num_part0_layers; ++l) {
        ok = repack_layer(src_fd0, src_h0->layers[l], dst_fd0, dst_h0->layers[l]);
    }
    for (uint32_t l = 0U; ok && l < src_h1->num_part1_layers; ++l) {
        ok = repack_layer(src_fd1, src_h1->layers[l], dst_fd1, dst_h1->layers[l]);
    }
    if (ok) ok = repack_mixer(src_fd1, src_h1->final_mixer, dst_fd1, dst_h1->final_mixer);
    if (ok && !src_h0->cfg.tie_word_embeddings) {
        ok = repack_qmat(src_fd1, src_h1->lm_head, dst_fd1, dst_h1->lm_head);
    }
    if (ok && src_h1->mtp.enabled) {
        ok = repack_slice(src_fd1, src_h1->mtp.pre_fc_norm_embedding, dst_fd1, dst_h1->mtp.pre_fc_norm_embedding) &&
             repack_slice(src_fd1, src_h1->mtp.pre_fc_norm_hidden,    dst_fd1, dst_h1->mtp.pre_fc_norm_hidden) &&
             repack_slice(src_fd1, src_h1->mtp.fc_embedding,          dst_fd1, dst_h1->mtp.fc_embedding) &&
             repack_slice(src_fd1, src_h1->mtp.fc_hidden,             dst_fd1, dst_h1->mtp.fc_hidden) &&
             repack_layer(src_fd1, src_h1->mtp.layer0,                dst_fd1, dst_h1->mtp.layer0) &&
             repack_mixer(src_fd1, src_h1->mtp.mixer,                 dst_fd1, dst_h1->mtp.mixer);
    }

    close(src_fd0);
    close(src_fd1);
    close(dst_fd0);
    close(dst_fd1);

    if (!ok) {
        unlink(dst_p0_tmp.c_str());
        unlink(dst_p1_tmp.c_str());
        std::fprintf(stderr, "[flash-repack] ERROR: failed during tensor streaming/quantization\n");
        return -7;
    }

    if (rename(dst_p0_tmp.c_str(), dst_p0_fin.c_str()) != 0 ||
        rename(dst_p1_tmp.c_str(), dst_p1_fin.c_str()) != 0) {
        unlink(dst_p0_tmp.c_str());
        unlink(dst_p1_tmp.c_str());
        std::fprintf(stderr, "[flash-repack] ERROR: atomic rename failed\n");
        return -8;
    }

    /* Hardlink `ple_ngram.strata` (0 extra SSD bytes; fallback to symlink if cross-device) */
    uint64_t ple_bytes = 0ULL;
    if (src_h0->cfg.has_ple && file_exists_nonempty(src_pl_path, &ple_bytes)) {
        const std::string dst_pl = dst_dir + "/" + STRATA_QWEN_FLASH_PLE_FILENAME;
        unlink(dst_pl.c_str());
        if (link(src_pl_path.c_str(), dst_pl.c_str()) != 0) {
            if (symlink(src_pl_path.c_str(), dst_pl.c_str()) != 0) {
                std::fprintf(stderr, "[flash-repack] ERROR: failed to link '%s' -> '%s'\n",
                             src_pl_path.c_str(), dst_pl.c_str());
                return -9;
            }
        }
    }

    /* Copy tokenizer & config metadata JSON/Jinja files */
    const char* meta_files[] = {
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "generation_config.json"
    };
    for (const char* mf : meta_files) {
        (void)copy_file_if_exists(src_dir + "/" + mf, dst_dir + "/" + mf, false);
    }

    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (stats_out) {
        stats_out->num_text_tensors_converted = tensors_processed;
        stats_out->uma_part0_bytes            = dst_h0->total_file_bytes;
        stats_out->uma_part1_bytes            = dst_h1->total_file_bytes;
        stats_out->ple_ngram_bytes            = ple_bytes;
        stats_out->total_bundle_bytes         = dst_h0->total_file_bytes + dst_h1->total_file_bytes + ple_bytes;
        stats_out->active_bytes_per_tok       = dst_h0->active_bytes_per_tok;
        stats_out->elapsed_sec                = elapsed;
        stats_out->peak_rss_gib               = current_peak_rss_gib();
        stats_out->page_aligned_verified      =
            ((dst_h0->total_file_bytes % STRATA_QWEN_FLASH_PAGE_BYTES) == 0ULL) &&
            ((dst_h1->total_file_bytes % STRATA_QWEN_FLASH_PAGE_BYTES) == 0ULL);
        stats_out->staging_clean_verified     = true;
    }
    return 0;
}

} /* extern "C" */
