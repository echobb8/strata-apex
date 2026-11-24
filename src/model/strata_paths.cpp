#include "strata_paths.h"

#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

std::mutex g_paths_mu;
std::unordered_map<std::string, std::unique_ptr<std::string>> g_paths_intern;

const char* intern_str(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_paths_mu);
    auto it = g_paths_intern.find(s);
    if (it != g_paths_intern.end()) return it->second->c_str();
    auto ptr = std::make_unique<std::string>(s);
    const char* out = ptr->c_str();
    g_paths_intern.emplace(s, std::move(ptr));
    return out;
}

bool path_exists(const std::string& p) {
    struct stat st{};
    return (!p.empty() && ::stat(p.c_str(), &st) == 0);
}

std::string strip_trailing_slash(std::string s) {
    while (s.size() > 1U && s.back() == '/') s.pop_back();
    return s;
}

std::string get_home_dir() {
    const char* h = std::getenv("HOME");
    if (h && h[0] != '\0') return strip_trailing_slash(h);
    const struct passwd* pw = ::getpwuid(::getuid());
    if (pw && pw->pw_dir && pw->pw_dir[0] != '\0') return strip_trailing_slash(pw->pw_dir);
    /* Private per-UID 0700 fallback directory instead of shared /tmp/.cache */
    const std::string fallback = "/tmp/strata-apex-uid-" + std::to_string(static_cast<unsigned long>(::getuid()));
    (void)::mkdir(fallback.c_str(), 0700);
    struct stat st{};
    if (::lstat(fallback.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && st.st_uid == ::getuid()) {
        (void)::chmod(fallback.c_str(), 0700);
        return fallback;
    }
    char tmpl[] = "/tmp/strata-apex-XXXXXX";
    if (char* created = ::mkdtemp(tmpl)) {
        (void)::chmod(created, 0700);
        return std::string(created);
    }
    return fallback;
}

std::string resolve_cache_root_str() {
    const char* env_apex = std::getenv("STRATA_APEX_CACHE_DIR");
    if (env_apex && env_apex[0] != '\0') return strip_trailing_slash(env_apex);
    const char* env_leg = std::getenv("STRATA_CACHE_DIR");
    if (env_leg && env_leg[0] != '\0') return strip_trailing_slash(env_leg);

    const std::string home   = get_home_dir();
    const std::string apex   = home + "/.cache/strata-apex";
    const std::string legacy = home + "/.cache/strata";
    if (path_exists(apex)) return apex;
    if (path_exists(legacy)) return legacy;
    return apex;
}

} // namespace

extern "C" const char* strata_apex_cache_root(void) {
    return intern_str(resolve_cache_root_str());
}

extern "C" const char* strata_apex_bundle_path(const char* rel_subpath) {
    const std::string rel = (rel_subpath && rel_subpath[0] != '\0') ? std::string(rel_subpath) : std::string();
    if (rel.empty()) return strata_apex_cache_root();

    const char* env_apex = std::getenv("STRATA_APEX_CACHE_DIR");
    if (env_apex && env_apex[0] != '\0') {
        const std::string cand = strip_trailing_slash(env_apex) + "/" + rel;
        if (path_exists(cand)) return intern_str(cand);
    }
    const char* env_leg = std::getenv("STRATA_CACHE_DIR");
    if (env_leg && env_leg[0] != '\0') {
        const std::string cand = strip_trailing_slash(env_leg) + "/" + rel;
        if (path_exists(cand)) return intern_str(cand);
    }

    const std::string home        = get_home_dir();
    const std::string apex_cand   = home + "/.cache/strata-apex/" + rel;
    const std::string legacy_cand = home + "/.cache/strata/" + rel;
    if (path_exists(apex_cand)) return intern_str(apex_cand);
    if (path_exists(legacy_cand)) return intern_str(legacy_cand);

    return intern_str(resolve_cache_root_str() + "/" + rel);
}

extern "C" const char* strata_apex_metal_cache_dir(void) {
    const char* env_mc = std::getenv("STRATA_METAL_CACHE_DIR");
    if (env_mc && env_mc[0] != '\0') return intern_str(strip_trailing_slash(env_mc));
    return intern_str(resolve_cache_root_str() + "/metal_cache");
}
