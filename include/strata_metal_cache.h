#ifndef STRATA_METAL_CACHE_H
#define STRATA_METAL_CACHE_H

/*
 * STRATA-M5 v2.0 — Shared Persistent Metal 4 `MTLBinaryArchive` Shader Cache
 * & `STRATA_METAL_REFLECT=1` Pipeline Register-Pressure / Spill Auditor
 *
 * Used identically across all three STRATA-M5 GPU engines:
 *   - `src/gpu/strata_qwen_flash_gpu.mm` (engine_tag = "qwen_flash")
 *   - `src/gpu/strata_glm53_gpu.mm`      (engine_tag = "glm53")
 *   - `src/gpu/strata_inkling2_gpu.mm`   (engine_tag = "inkling2")
 *
 * Environment controls:
 *   - `STRATA_METAL_CACHE_DIR`     : Override cache directory (default: `~/.cache/strata-apex/metal_cache`).
 *   - `STRATA_METAL_CACHE_DISABLE` : Set to `1` to disable binary archive load/save (reflection still works).
 *   - `STRATA_METAL_REFLECT`       : Set to `1` to print summary + every PSO with `maxTotalThreadsPerThreadgroup < 1024`
 *                                    sorted worst-first; set to `2` or `all` to print every PSO.
 *
 * Thread-safety:
 *   - `strata_metal_cache_make_pso*()` is 100% thread-safe and designed for concurrent `dispatch_apply`
 *     PSO creation across all CPU cores (shared-locked archive reads, lock-free parallel cold compiles,
 *     unique-locked archive insertions).
 *   - `strata_metal_cache_finish()` atomically serializes dirty archives (`<path>.tmp.<pid>.<seq>` -> `rename()`)
 *     so concurrent processes never corrupt the on-disk archive.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "strata_paths.h"

#define STRATA_METAL_CACHE_DEFAULT_DIR strata_apex_metal_cache_dir()

#ifdef __OBJC__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

struct StrataMetalPsoCacheImpl;

typedef struct StrataMetalPsoCache {
    id<MTLDevice>            device;
    id<MTLLibrary>           library;
    id<MTLBinaryArchive>     archive;

    char                     engine_tag[64];
    char                     cache_path[512];
    char                     key_hex16[17];

    bool                     cache_hit;               /* true iff existing disk archive loaded & served PSOs */
    bool                     dirty;                   /* true iff archive has unserialized newly-compiled PSOs */
    bool                     disabled;                /* true iff STRATA_METAL_CACHE_DISABLE=1 */
    bool                     reflect_enabled;         /* true iff STRATA_METAL_REFLECT!=0 */

    /* Reflection & load telemetry (updated thread-safely during make_pso, finalized in finish) */
    uint32_t                 pso_count;               /* Total PSOs successfully created through this cache */
    uint32_t                 archive_hit_count;       /* PSOs loaded directly from MTLBinaryArchive */
    uint32_t                 archive_miss_count;      /* PSOs compiled from AIR (cold miss or archive fallback) */
    uint32_t                 spill_risk_count;        /* PSOs with maxTotalThreadsPerThreadgroup < 1024 */
    uint32_t                 min_max_threads_per_tg;  /* Worst (smallest) maxTotalThreadsPerThreadgroup seen */
    double                   elapsed_ms;              /* Wall-clock ms from open() through finish() */
    double                   serialize_ms;            /* Wall-clock ms spent serializing archive in finish() */

    struct StrataMetalPsoCacheImpl* impl;
} StrataMetalPsoCache;

/*
 * Deterministically packs `MTLCompileOptions` (`languageVersion`, `mathMode`, `mathFloatingPointFunctions`)
 * into a lossless 64-bit salt mixed into the SHA-256 binary archive key.
 */
static inline uint64_t strata_metal_compile_salt(MTLCompileOptions* opts) {
    uint64_t lang = 0ULL;
    uint64_t mm   = 0ULL;
    uint64_t fpf  = 0ULL;
    if (opts != nil) {
        lang = (uint64_t)opts.languageVersion;
        if (@available(macOS 15.0, *)) {
            mm  = (uint64_t)opts.mathMode;
            fpf = (uint64_t)opts.mathFloatingPointFunctions;
        }
    }
    return (lang << 16) | ((mm & 0xFFULL) << 8) | (fpf & 0xFFULL);
}

/*
 * Opens (or creates) a persistent `MTLBinaryArchive` cache keyed by SHA-256 of:
 *   `engine_tag` + `msl_src[0..msl_len]` + `compile_salt` (`languageVersion`, `mathMode`,
 *   `mathFloatingPointFunctions`) + `[device name]` + macOS build string (`kern.osversion`).
 *
 * File path: `<cache_dir>/<engine_tag>_<hex16>.metallib_bin`.
 * Returns a heap-allocated handle (release with `strata_metal_cache_destroy`), or NULL on invalid args.
 */
StrataMetalPsoCache* strata_metal_cache_open(
    id<MTLDevice>  device,
    id<MTLLibrary> library,
    const char*    engine_tag,
    const char*    msl_src,
    size_t         msl_len,
    uint64_t       compile_salt
);

/* Convenience wrapper extracting `compile_salt` directly from `MTLCompileOptions*`. */
static inline StrataMetalPsoCache* strata_metal_cache_open_with_opts(
    id<MTLDevice>      device,
    id<MTLLibrary>     library,
    const char*        engine_tag,
    const char*        msl_src,
    size_t             msl_len,
    MTLCompileOptions* compile_opts
) {
    return strata_metal_cache_open(
        device, library, engine_tag, msl_src, msl_len, strata_metal_compile_salt(compile_opts));
}

/*
 * Creates (or loads from binary archive) a compute PSO for `fn_name` with optional `const_values` (`nil` for
 * unspecialized). Thread-safe for concurrent calls from `dispatch_apply`.
 *   - On warm archive hit: loads directly from `cache->archive` (gracefully falling back to normal compile +
 *     marking `dirty = true` + recording into `cache->archive` if this specific PSO was not in the archive).
 *   - On cold miss: compiles in parallel and records the binary into `cache->archive`.
 *   - Always inspects `maxTotalThreadsPerThreadgroup`, `threadExecutionWidth`, `staticThreadgroupMemoryLength`
 *     for `STRATA_METAL_REFLECT=1` spill auditing.
 */
id<MTLComputePipelineState> strata_metal_cache_make_pso(
    StrataMetalPsoCache*       cache,
    const char*                fn_name,
    MTLFunctionConstantValues* const_values,
    NSError* __autoreleasing*  err_out
);

/*
 * Same as `strata_metal_cache_make_pso()`, with an optional `label_tag` (e.g. `"M=4"`) appended ONLY to the
 * reflection audit display name (`fn_name[label_tag]`) so specialized variants are distinct in reflect logs.
 */
id<MTLComputePipelineState> strata_metal_cache_make_pso_labeled(
    StrataMetalPsoCache*       cache,
    const char*                fn_name,
    const char*                label_tag,
    MTLFunctionConstantValues* const_values,
    NSError* __autoreleasing*  err_out
);

/*
 * Convenience helper for single `uint32_t` function-constant specialization (e.g., `M` at index 0),
 * automatically attaching reflection label `"c<idx>=<val>"` (or `"M=<val>"` when `const_index == 0`).
 */
id<MTLComputePipelineState> strata_metal_cache_make_pso_const_u32(
    StrataMetalPsoCache*      cache,
    const char*               fn_name,
    uint32_t                  const_index,
    uint32_t                  const_value,
    NSError* __autoreleasing* err_out
);

/* Convenience helper accepting `NSString*` function names. */
static inline id<MTLComputePipelineState> strata_metal_cache_make_pso_ns(
    StrataMetalPsoCache*       cache,
    NSString*                  fn_name,
    MTLFunctionConstantValues* const_values,
    NSError* __autoreleasing*  err_out
) {
    return strata_metal_cache_make_pso(
        cache, fn_name ? [fn_name UTF8String] : NULL, const_values, err_out);
}

/*
 * Finalizes the cache session:
 *   - If `dirty && !disabled` and at least one PSO was recorded, atomically serializes the archive to disk
 *     via `<cache_path>.tmp.<pid>.<seq>` -> `rename()` so parallel processes never corrupt the file.
 *   - Finalizes `cache->elapsed_ms` and `cache->serialize_ms`.
 *   - Prints sorted worst-first reflection report to `stderr` when `STRATA_METAL_REFLECT=1`.
 * Does NOT free `cache` so callers can read `cache->cache_hit`, `cache->elapsed_ms`, etc. before destroying.
 * Returns `true` on success (including warm hit no-op or disabled mode), `false` on serialization error.
 */
bool strata_metal_cache_finish(StrataMetalPsoCache* cache);

/* Releases all resources owned by `cache` (safe to call with NULL). */
void strata_metal_cache_destroy(StrataMetalPsoCache* cache);

#endif /* __OBJC__ */

#endif /* STRATA_METAL_CACHE_H */
