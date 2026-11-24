#ifndef STRATA_EMBEDDED_SHADERS_H
#define STRATA_EMBEDDED_SHADERS_H

/*
 * STRATA-APEX — Embedded Metal 4 Shaders & Shared ABI Headers
 *
 * Generated into `build/strata_embedded_shaders.cpp` by `scripts/embed_shaders.py`
 * so `strata-apex` runs as a self-contained single binary from any directory or
 * Homebrew install path. Set `STRATA_SHADER_DIR=<dir>` to override from disk.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* strata_embedded_asset_lookup(const char* key, size_t* len_out);

#ifdef __cplusplus
}

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

static inline std::string strata_load_shader_asset(const char* key, const std::string& disk_fallback) {
    auto read_file_str = [](const std::string& p) -> std::string {
        if (p.empty()) return {};
        std::ifstream f(p, std::ios::in | std::ios::binary);
        if (!f) return {};
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    };

    /* 1. Explicit developer override via `STRATA_SHADER_DIR` */
    if (const char* env_dir = std::getenv("STRATA_SHADER_DIR"); env_dir && env_dir[0] != '\0' && key) {
        std::string p(env_dir);
        while (p.size() > 1U && p.back() == '/') p.pop_back();
        p += "/";
        p += key;
        std::string s = read_file_str(p);
        if (!s.empty()) return s;
    }

    /* 2. Embedded self-contained binary asset (default for portable single-binary distribution) */
    if (key) {
        size_t len = 0U;
        const char* data = strata_embedded_asset_lookup(key, &len);
        if (data && len > 0U) {
            return std::string(data, len);
        }
    }

    /* 3. Sibling source-tree disk fallback */
    return read_file_str(disk_fallback);
}
#endif

#endif /* STRATA_EMBEDDED_SHADERS_H */
