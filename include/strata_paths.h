#ifndef STRATA_PATHS_H
#define STRATA_PATHS_H

/*
 * STRATA-APEX — Portable Cache & Bundle Path Resolver
 *
 * Resolution order:
 *   1. `STRATA_APEX_CACHE_DIR` environment variable (if set and non-empty)
 *   2. `STRATA_CACHE_DIR` environment variable (if set and non-empty)
 *   3. `$HOME/.cache/strata-apex` (primary default for new installs)
 *   4. `$HOME/.cache/strata` (automatic read fallback for existing local bundles)
 */

#ifndef __METAL_VERSION__

#define STRATA_APEX_VERSION "0.1.0"

#ifdef __cplusplus
extern "C" {
#endif

/* Primary cache root directory (returns a stable process-lifetime C string). */
const char* strata_apex_cache_root(void);

/* Resolve a relative bundle or file path inside the cache root, preferring an
 * existing file/dir in `$STRATA_APEX_CACHE_DIR`, `$STRATA_CACHE_DIR`,
 * `$HOME/.cache/strata-apex`, or `$HOME/.cache/strata` (in that order). */
const char* strata_apex_bundle_path(const char* rel_subpath);

/* Resolve the Metal binary archive cache directory (`STRATA_METAL_CACHE_DIR` or
 * `<cache_root>/metal_cache`). */
const char* strata_apex_metal_cache_dir(void);

#ifdef __cplusplus
}
#endif

#endif /* !__METAL_VERSION__ */

#endif /* STRATA_PATHS_H */
