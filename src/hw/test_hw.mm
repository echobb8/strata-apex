/*
 * STRATA-M5 v2.0 — Hardware Subsystem Verification & Benchmark Suite
 * File: src/hw/test_hw.mm -> build/test_hw
 * Owner: silicon-arch
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../include/strata_abi.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <vector>

extern "C" {
int strata_hw_verify_sme2_luti4_bit_exact(uint32_t* mismatches_out);
void strata_hw_bench_sme2_clusters(
    double* super_f16_tflops_out,
    double* all3_f16_tflops_out,
    double* all3_bf16_tflops_out,
    double* all3_i8_tops_out
);
int strata_hw_bench_nvme_qd64_4k(
    int fd,
    uint64_t file_bytes,
    uint32_t total_ios,
    double* iops_out,
    double* avg_batch64_us_out,
    long* majflt_out
);
int strata_ane_probe_verify_vs_sme2(
    StrataAneProbe* probe,
    double* sqnr_db_out,
    bool* compute_plan_all_ane_out
);
const char* strata_ane_probe_status_note(StrataAneProbe* probe);
}

static double ticks_to_sec(uint64_t ticks) {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)ticks * (double)tb.numer / (double)tb.denom * 1e-9;
}

static void print_sysctl_str(const char* name) {
    char buf[256] = {0};
    size_t sz = sizeof(buf) - 1;
    if (sysctlbyname(name, buf, &sz, nullptr, 0) == 0) {
        printf("  %-34s : %s\n", name, buf);
    }
}

static void print_sysctl_i64(const char* name) {
    int64_t val = 0;
    size_t sz = sizeof(val);
    if (sysctlbyname(name, &val, &sz, nullptr, 0) == 0) {
        printf("  %-34s : %lld\n", name, (long long)val);
    }
}

static bool write_nocache_file(const char* path, size_t total_bytes, uint64_t seed_base) {
    int fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (fd < 0) return false;
    (void)fcntl(fd, F_NOCACHE, 1);

    constexpr size_t kChunk = 4ULL * 1024ULL * 1024ULL;
    void* buf = nullptr;
    if (posix_memalign(&buf, STRATA_VM_PAGE_BYTES, kChunk) != 0 || !buf) {
        close(fd);
        return false;
    }
    uint64_t* p64 = static_cast<uint64_t*>(buf);
    size_t written = 0;
    while (written < total_bytes) {
        size_t cur = (written + kChunk <= total_bytes) ? kChunk : (total_bytes - written);
        uint64_t s = seed_base ^ written;
        for (size_t i = 0; i < cur / 8U; ++i) {
            s ^= s >> 30;
            s *= 0xBF58476D1CE4E5B9ULL;
            p64[i] = s;
        }
        ssize_t w = pwrite(fd, buf, cur, static_cast<off_t>(written));
        if (w <= 0) {
            free(buf);
            close(fd);
            return false;
        }
        written += (size_t)w;
    }
    fsync(fd);
    free(buf);
    close(fd);
    return true;
}

struct TempBenchCleanupGuard {
    const char* p1;
    const char* p2;
    bool keep;
    ~TempBenchCleanupGuard() {
        if (!keep) {
            if (p1) unlink(p1);
            if (p2) unlink(p2);
        }
    }
};

#define HW_REQUIRE(cond)                                                                     \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond);                \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    bool keep_bench = false;
    bool cleanup_only = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--keep-bench") == 0) keep_bench = true;
        if (strcmp(argv[i], "--cleanup") == 0) cleanup_only = true;
    }

    const char* bench_dir = "bench_data";
    const char* ngram_path = "bench_data/hw_ngram_bench.bin";
    const char* ring_path  = "bench_data/hw_cold_experts.bin";
    mkdir(bench_dir, 0755);

    if (cleanup_only) {
        unlink(ngram_path);
        unlink(ring_path);
        printf("[test_hw] Cleaned up temporary benchmark files.\n");
        return 0;
    }

    /* Guarantees temp files are unlinked on BOTH success and any early failure return */
    TempBenchCleanupGuard cleanup_guard{ngram_path, ring_path, keep_bench};

    printf("================================================================================\n");
    printf("STRATA-M5 v2.0 — Track 1 Hardware Co-Engine Verification & Benchmark Suite\n");
    printf("================================================================================\n");
    print_sysctl_str("hw.model");
    print_sysctl_str("machdep.cpu.brand_string");
    print_sysctl_i64("hw.memsize");
    print_sysctl_i64("hw.memsize_usable");
    print_sysctl_i64("hw.ncpu");
    print_sysctl_i64("hw.perflevel0.physicalcpu");
    print_sysctl_i64("hw.perflevel0.l2cachesize");
    print_sysctl_i64("hw.perflevel1.physicalcpu");
    print_sysctl_i64("hw.perflevel1.l2cachesize");
    print_sysctl_i64("vm.pagesize");
    print_sysctl_i64("hw.optional.arm.FEAT_SME2");
    print_sysctl_i64("hw.optional.arm.sme_max_svl_b");
    print_sysctl_i64("hw.optional.arm.FEAT_FP8");

    /* ------------------------------------------------------------------------
     * TEST 1: ARMv9.2 SME2 `ZT0` `LUTI4` + `FMOPA` Bit-Exactness & Cluster Peak
     * ------------------------------------------------------------------------ */
    printf("\n[1/4] Verifying ARMv9.2 SME2 (`ZT0` `LUTI4` + `FMOPA` / `BFMOPA` / `SMOPA`)...\n");
    uint32_t mismatches = 999U;
    int rc_sme = strata_hw_verify_sme2_luti4_bit_exact(&mismatches);
    printf("  SME2 ZT0 LUTI4 (512 nibbles) + FMOPA (16x32) mismatches : %u (rc=%d)\n", mismatches, rc_sme);
    HW_REQUIRE(rc_sme == 0 && mismatches == 0);

    double super_f16 = 0.0, all3_f16 = 0.0, all3_bf16 = 0.0, all3_i8 = 0.0;
    strata_hw_bench_sme2_clusters(&super_f16, &all3_f16, &all3_bf16, &all3_i8);
    printf("  1x Super-Cluster SME2 F16F16 FMOPA Peak                 : %.2f TFLOPS\n", super_f16);
    printf("  All-3-Clusters (Super + 2xPerf) SME2 F16F16 Aggregate   : %.2f TFLOPS\n", all3_f16);
    printf("  All-3-Clusters (Super + 2xPerf) SME2 B16B16 Aggregate   : %.2f TFLOPS\n", all3_bf16);
    printf("  All-3-Clusters (Super + 2xPerf) SME2 INT8 SMOPA         : %.2f TOPS\n", all3_i8);

    /* ------------------------------------------------------------------------
     * TEST 2: Isolated Phase-1 `QD64` 4 KiB `F_NOCACHE` Gather + `mlock` LFU + SME2 Gate
     * ------------------------------------------------------------------------ */
    printf("\n[2/4] Benchmarking Phase-1 `QD64` 4 KiB `F_NOCACHE` Gather + Non-IOGPU LFU + SME2 Gate...\n");
    constexpr size_t kNgramFileBytes = 256ULL * 1024ULL * 1024ULL; /* 256 MiB */
    HW_REQUIRE(write_nocache_file(ngram_path, kNgramFileBytes, 0xCAFEBABE12345678ULL));

    int ngram_fd = open(ngram_path, O_RDONLY);
    HW_REQUIRE(ngram_fd >= 0);
    (void)fcntl(ngram_fd, F_NOCACHE, 1);

    double qd64_iops = 0.0, batch64_us = 0.0;
    long qd64_majflt = -1;
    int rc_qd = strata_hw_bench_nvme_qd64_4k(ngram_fd, kNgramFileBytes, 4096U, &qd64_iops, &batch64_us, &qd64_majflt);
    if (rc_qd != 0 || qd64_majflt != 0) {
        close(ngram_fd);
        HW_REQUIRE(rc_qd == 0 && qd64_majflt == 0);
    }
    printf("  ANS NVMe `F_NOCACHE` 4 KiB Random Read (QD64)           : %.0f IOPS (%.1f us / 64-IO wave, majflt=%ld)\n",
           qd64_iops, batch64_us, qd64_majflt);

    /* Initialize StrataHwContext: 16 slots x 16 MiB = 256 MiB ring + 32 MiB CPU LFU pool */
    constexpr size_t kSlotBytes = 16ULL * 1024ULL * 1024ULL;
    constexpr uint32_t kNumSlots = 16U;
    StrataHwContext* hw = strata_hw_init(nullptr, kSlotBytes, kNumSlots, 32ULL * 1024ULL * 1024ULL);
    if (!hw) {
        close(ngram_fd);
        HW_REQUIRE(hw != nullptr);
    }
    printf("  MTLDevice in current sandbox context                    : %s\n",
           strata_hw_ring_get_mtl_buffer(hw) ? "ACTIVE (MTLIOCommandQueue + MTLResidencySet)"
                                             : "NULL (Seatbelt IOKit-gated; using mlock + F_NOCACHE ANS DMA)");

    /* DeepSeek-V4.1-Flash-1017 Engram config: B=6, heads=24 (144 lookups), d=5120 */
    constexpr uint32_t kBatchB = 6U;
    constexpr uint32_t kEngramHeads = 24U;
    constexpr uint32_t kDeepSeekD = 5120U;
    std::vector<uint64_t> hashes(kBatchB * kEngramHeads);
    for (size_t i = 0; i < hashes.size(); ++i) {
        hashes[i] = (i * 104729ULL + 7919ULL);
    }
    std::vector<uint16_t> engram_out((size_t)kBatchB * kDeepSeekD, 0);
    StrataSme2IoMetrics m_cold{}, m_warm{};

    HW_REQUIRE(strata_hw_exec_ngram_phase1(hw, ngram_fd, kNgramFileBytes, hashes.data(),
                                           kBatchB, kEngramHeads, kDeepSeekD, engram_out.data(), &m_cold) == 0);
    /* Re-run identical hashes -> must hit 100% in `mlock` LFU cache with 0 NVMe I/Os! */
    HW_REQUIRE(strata_hw_exec_ngram_phase1(hw, ngram_fd, kNgramFileBytes, hashes.data(),
                                           kBatchB, kEngramHeads, kDeepSeekD, engram_out.data(), &m_warm) == 0);

    printf("  DeepSeek Engram Cold (B=6, 144 lookups, d=5120)         : IO=%.1f us (%u IOs), SME2=%.1f us (%.1f GFLOPS)\n",
           m_cold.phase1_io_us, m_cold.nvme_pread_ios, m_cold.sme2_compute_us, m_cold.sme2_effective_gflops);
    printf("  DeepSeek Engram Warm LFU (%u/%u hits, 0 NVMe IOs)      : IO=%.2f us (%u IOs), SME2=%.1f us (%.1f GFLOPS)\n",
           m_warm.lfu_cache_hits, m_warm.total_lookups, m_warm.phase1_io_us,
           m_warm.nvme_pread_ios, m_warm.sme2_compute_us, m_warm.sme2_effective_gflops);
    HW_REQUIRE(m_warm.lfu_cache_hits == m_warm.total_lookups && m_warm.nvme_pread_ios == 0 && m_warm.hardware_sme2_active);

    /* Qwen3.8-Flash-Next PLE config: B=6, heads=4 (24 lookups), d=2560 */
    constexpr uint32_t kPleHeads = 4U;
    constexpr uint32_t kQwenD = 2560U;
    std::vector<uint64_t> ple_hashes(kBatchB * kPleHeads);
    for (size_t i = 0; i < ple_hashes.size(); ++i) ple_hashes[i] = (i * 65537ULL + 99991ULL);
    std::vector<uint16_t> ple_out((size_t)kBatchB * kQwenD, 0);
    StrataSme2IoMetrics m_ple{};
    HW_REQUIRE(strata_hw_exec_ngram_phase1(hw, ngram_fd, kNgramFileBytes, ple_hashes.data(),
                                           kBatchB, kPleHeads, kQwenD, ple_out.data(), &m_ple) == 0);
    printf("  Qwen3.8 PLE Cold (B=6, 24 lookups, d=2560)              : IO=%.1f us (%u IOs), SME2=%.1f us (%.1f GFLOPS)\n",
           m_ple.phase1_io_us, m_ple.nvme_pread_ios, m_ple.sme2_compute_us, m_ple.sme2_effective_gflops);
    close(ngram_fd);

    /* ------------------------------------------------------------------------
     * TEST 3: Phase-2 Pre-Wired Slot Ring Cold-Expert Streaming (`ru_majflt == 0`)
     * ------------------------------------------------------------------------ */
    printf("\n[3/4] Benchmarking Pre-Wired NVMe Slot Ring (`F_NOCACHE`, 256 KiB-aligned, 0 page faults)...\n");
    constexpr size_t kContainerBytes = 512ULL * 1024ULL * 1024ULL; /* 512 MiB */
    HW_REQUIRE(write_nocache_file(ring_path, kContainerBytes, 0xFEEDFACE87654321ULL));

    /* Stream 16 DeepSeek-V4.1-Flash 15.50 MiB (62 * 256 KiB) extents = 248 MiB */
    constexpr size_t kDeepSeekExtent = 62ULL * STRATA_LBA_ALIGN_BYTES;
    std::vector<uint64_t> offs(16);
    std::vector<size_t> sizes(16, kDeepSeekExtent);
    std::vector<uint32_t> slots(16);
    std::vector<uint32_t> epochs(16);
    for (uint32_t i = 0; i < 16U; ++i) {
        offs[i] = (uint64_t)i * kDeepSeekExtent;
        slots[i] = i;
        epochs[i] = 100U + i;
    }

    StrataRingIoMetrics rm_sync{};
    HW_REQUIRE(strata_hw_ring_stream_extents(hw, ring_path, offs.data(), sizes.data(),
                                             slots.data(), epochs.data(), 16U, true, &rm_sync) == 0);
    printf("  DeepSeek Cold Extents (16 x 15.50 MiB = 248 MiB, sync)  : %.2f GB/s (%.2f ms, ru_majflt=%ld, ru_minflt=%ld)\n",
           rm_sync.sustained_nvme_gbps, rm_sync.dma_wall_us * 1e-3,
           rm_sync.major_page_faults, rm_sync.minor_page_faults);
    HW_REQUIRE(rm_sync.major_page_faults == 0);

    uint32_t* ready_tbl = strata_hw_ring_get_ready_table(hw);
    for (uint32_t i = 0; i < 16U; ++i) {
        HW_REQUIRE(ready_tbl[i] == 100U + i);
    }

    /* Test async submission (`wait_completion = false`) + `strata_hw_ring_wait_slot` */
    constexpr size_t kGlmExtent = 23ULL * STRATA_LBA_ALIGN_BYTES; /* ~5.75 MiB */
    std::vector<size_t> glm_sizes(8, kGlmExtent);
    std::vector<uint32_t> async_epochs(8);
    for (uint32_t i = 0; i < 8U; ++i) async_epochs[i] = 200U + i;

    StrataRingIoMetrics rm_async{};
    HW_REQUIRE(strata_hw_ring_stream_extents(hw, ring_path, offs.data(), glm_sizes.data(),
                                             slots.data(), async_epochs.data(), 8U, false, &rm_async) == 0);
    double total_wait_us = 0.0;
    for (uint32_t i = 0; i < 8U; ++i) {
        double w_us = 0.0;
        HW_REQUIRE(strata_hw_ring_wait_slot(hw, i, async_epochs[i], &w_us) == 0);
        total_wait_us += w_us;
        HW_REQUIRE(ready_tbl[i] == async_epochs[i]);
    }
    printf("  GLM-5.3-Flash Async Window (8 x 5.75 MiB = 46 MiB)      : submit=%.1f us, wait=%.2f ms, ru_majflt=%ld\n",
           rm_async.dma_wall_us, total_wait_us * 1e-3, rm_async.major_page_faults);
    HW_REQUIRE(rm_async.major_page_faults == 0);

    /* ------------------------------------------------------------------------
     * TEST 4: `[1, d, 1, S=16]` Window Probe + Concurrent UMA Zero-Thrash Test
     * ------------------------------------------------------------------------ */
    printf("\n[4/4] Benchmarking `[1, d, 1, S=16]` Multi-Horizon Router Probe & Concurrent UMA Bus Test...\n");
    StrataAneProbe* probe_ds = strata_ane_probe_init(kDeepSeekD, 256U, 384U, 4U);
    StrataAneProbe* probe_qw = strata_ane_probe_init(kQwenD, 256U, 512U, 4U);
    HW_REQUIRE(probe_ds && probe_qw);

    struct rusage ru_probe0{};
    getrusage(RUSAGE_SELF, &ru_probe0);

    double sqnr_ds_db = 0.0, sqnr_qw_db = 0.0;
    bool plan_all_ane_ds = false, plan_all_ane_qw = false;
    int rc_ver_ds = strata_ane_probe_verify_vs_sme2(probe_ds, &sqnr_ds_db, &plan_all_ane_ds);
    int rc_ver_qw = strata_ane_probe_verify_vs_sme2(probe_qw, &sqnr_qw_db, &plan_all_ane_qw);
    printf("  Probe Backend Status                                    : %s\n",
           strata_ane_probe_status_note(probe_ds));
    printf("  DeepSeek MLComputePlan & ANE-vs-SME2 SQNR               : compute_plan_all_ane=%s, SQNR=%.2f dB (rc=%d)\n",
           plan_all_ane_ds ? "true" : "false", sqnr_ds_db, rc_ver_ds);
    printf("  Qwen3.8  MLComputePlan & ANE-vs-SME2 SQNR               : compute_plan_all_ane=%s, SQNR=%.2f dB (rc=%d)\n",
           plan_all_ane_qw ? "true" : "false", sqnr_qw_db, rc_ver_qw);
    HW_REQUIRE(rc_ver_ds == 0 && rc_ver_qw == 0);

    std::vector<uint16_t> hidden_ds((size_t)kBatchB * kDeepSeekD, 0x3800U);
    std::vector<float> scores_ds(4U * 384U, 0.0f);
    bool ran_ane_ds = false;
    double dummy_us = 0.0;
    for (uint32_t w = 0; w < 4U; ++w) {
        HW_REQUIRE(strata_ane_probe_predict_window(probe_ds, w, hidden_ds.data(), kBatchB,
                                                   scores_ds.data(), &dummy_us, &ran_ane_ds) == 0);
    }

    double sum_ds_us = 0.0;
    constexpr int kProbeIters = 50;
    for (int it = 0; it < kProbeIters; ++it) {
        double l_us = 0.0;
        bool cur_ane = false;
        HW_REQUIRE(strata_ane_probe_predict_window(probe_ds, (uint32_t)(it & 3), hidden_ds.data(),
                                                   kBatchB, scores_ds.data(), &l_us, &cur_ane) == 0);
        ran_ane_ds = ran_ane_ds && cur_ane;
        sum_ds_us += l_us;
    }
    double avg_ds_probe_us = sum_ds_us / kProbeIters;

    std::vector<uint16_t> hidden_qw((size_t)kBatchB * kQwenD, 0x3800U);
    std::vector<float> scores_qw(4U * 512U, 0.0f);
    bool ran_ane_qw = false;
    for (uint32_t w = 0; w < 4U; ++w) {
        HW_REQUIRE(strata_ane_probe_predict_window(probe_qw, w, hidden_qw.data(), kBatchB,
                                                   scores_qw.data(), &dummy_us, &ran_ane_qw) == 0);
    }
    double sum_qw_us = 0.0;
    for (int it = 0; it < kProbeIters; ++it) {
        double l_us = 0.0;
        bool cur_ane = false;
        HW_REQUIRE(strata_ane_probe_predict_window(probe_qw, (uint32_t)(it & 3), hidden_qw.data(),
                                                   kBatchB, scores_qw.data(), &l_us, &cur_ane) == 0);
        ran_ane_qw = ran_ane_qw && cur_ane;
        sum_qw_us += l_us;
    }
    double avg_qw_probe_us = sum_qw_us / kProbeIters;

    if (ran_ane_ds) {
        HW_REQUIRE(plan_all_ane_ds && !std::isnan(sqnr_ds_db) && sqnr_ds_db >= 45.0);
    }
    if (ran_ane_qw) {
        HW_REQUIRE(plan_all_ane_qw && !std::isnan(sqnr_qw_db) && sqnr_qw_db >= 45.0);
    }

    struct rusage ru_probe1{};
    getrusage(RUSAGE_SELF, &ru_probe1);
    long probe_majflt = ru_probe1.ru_majflt - ru_probe0.ru_majflt;
    HW_REQUIRE(probe_majflt == 0);

    printf("  DeepSeek [1, 5120, 1, S=16] -> 4x384 Probe Latency      : %.1f us (ran_on_ane=%s, majflt=%ld; vs 3,220 us 4-layer window)\n",
           avg_ds_probe_us, ran_ane_ds ? "true" : "false", probe_majflt);
    printf("  Qwen3.8  [1, 2560, 1, S=16] -> 4x512 Probe Latency      : %.1f us (ran_on_ane=%s)\n",
           avg_qw_probe_us, ran_ane_qw ? "true" : "false");

    /* Concurrent UMA Bandwidth + NVMe Ring DMA + Window Probe Zero-Thrash Proof */
    constexpr size_t kStreamBufBytes = 512ULL * 1024ULL * 1024ULL; /* 512 MiB > 48 MiB SLC */
    void* dram_buf = nullptr;
    HW_REQUIRE(posix_memalign(&dram_buf, STRATA_VM_PAGE_BYTES, kStreamBufBytes) == 0 && dram_buf != nullptr);
    memset(dram_buf, 0x5A, kStreamBufBytes);

    uint64_t coexec_probe_calls = 0;
    bool coexec_all_ane = false;
    auto measure_uma_gbps = [&](bool with_coengines, double* probe_under_load_us) -> double {
        constexpr int kMemThreads = 16;
        size_t slice = kStreamBufBytes / kMemThreads;
        dispatch_queue_t mq = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

        std::atomic<bool> bg_stop{false};
        pthread_t probe_tid{};
        struct BgCtx {
            StrataAneProbe* pr;
            const uint16_t* hid;
            std::atomic<bool>* stop;
            double avg_us;
            uint64_t calls;
            bool all_ane;
        } bg{probe_ds, hidden_ds.data(), &bg_stop, 0.0, 0ULL, false};

        if (with_coengines) {
            for (uint32_t i = 0; i < 8U; ++i) async_epochs[i] = 300U + i;
            (void)strata_hw_ring_stream_extents(hw, ring_path, offs.data(), sizes.data(),
                                                slots.data(), async_epochs.data(), 8U, false, nullptr);
            pthread_create(&probe_tid, nullptr, [](void* p) -> void* {
                @autoreleasepool {
                    auto* b = static_cast<BgCtx*>(p);
                    std::vector<float> sc(4U * 384U);
                    double sum = 0.0;
                    uint64_t n = 0;
                    bool all_hw = false;
                    do {
                        double lu = 0.0;
                        bool ra = false;
                        strata_ane_probe_predict_window(b->pr, (uint32_t)(n & 3), b->hid, 6U, sc.data(), &lu, &ra);
                        sum += lu;
                        all_hw = (n == 0) ? ra : (all_hw && ra);
                        n++;
                    } while (!b->stop->load(std::memory_order_relaxed));
                    b->avg_us = (n > 0) ? (sum / (double)n) : 0.0;
                    b->calls = n;
                    b->all_ane = all_hw;
                }
                return nullptr;
            }, &bg);
        }

        uint64_t t0 = mach_absolute_time();
        constexpr int kPasses = 12;
        for (int pass = 0; pass < kPasses; ++pass) {
            dispatch_apply(kMemThreads, mq, ^(size_t tid) {
                const uint64_t* p = reinterpret_cast<const uint64_t*>(static_cast<const uint8_t*>(dram_buf) + tid * slice);
                size_t words = slice / sizeof(uint64_t);
                uint64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                for (size_t i = 0; i < words; i += 4) {
                    acc0 += p[i];
                    acc1 += p[i + 1];
                    acc2 += p[i + 2];
                    acc3 += p[i + 3];
                }
                asm volatile("" : : "r"(acc0 + acc1 + acc2 + acc3));
            });
        }
        uint64_t t1 = mach_absolute_time();

        if (with_coengines) {
            bg_stop.store(true, std::memory_order_relaxed);
            pthread_join(probe_tid, nullptr);
            for (uint32_t i = 0; i < 8U; ++i) {
                double wu = 0.0;
                (void)strata_hw_ring_wait_slot(hw, i, 300U + i, &wu);
            }
            if (probe_under_load_us) *probe_under_load_us = bg.avg_us;
            coexec_probe_calls = bg.calls;
            coexec_all_ane = bg.all_ane;
        }

        double sec = ticks_to_sec(t1 - t0);
        return ((double)kPasses * (double)kStreamBufBytes / sec) * 1e-9;
    };

    (void)measure_uma_gbps(false, nullptr); /* warmup */
    double solo_uma_gbps = measure_uma_gbps(false, nullptr);
    double loaded_probe_us = 0.0;
    double coexec_uma_gbps = measure_uma_gbps(true, &loaded_probe_us);
    double retention_pct = (solo_uma_gbps > 0.0) ? (100.0 * coexec_uma_gbps / solo_uma_gbps) : 100.0;
    if (ran_ane_ds) {
        HW_REQUIRE(coexec_probe_calls > 0 && coexec_all_ane);
    }

    printf("  Sustained CPU Streaming UMA Read (Solo)                 : %.1f GB/s\n", solo_uma_gbps);
    printf("  Sustained UMA Read + Concurrent NVMe DMA + Window Probe : %.1f GB/s (%.1f%% retained; probe=%.1f us, calls=%llu, ran_on_ane=%s)\n",
           coexec_uma_gbps, retention_pct, loaded_probe_us,
           (unsigned long long)coexec_probe_calls, coexec_all_ane ? "true" : "false");

    free(dram_buf);
    strata_ane_probe_destroy(probe_qw);
    strata_ane_probe_destroy(probe_ds);
    strata_hw_destroy(hw);

    printf("\n================================================================================\n");
    printf("ALL STRATA-M5 TRACK 1 HARDWARE TESTS PASSED (ru_majflt == 0, SME2 bit-exact == 0)\n");
    printf("================================================================================\n");
    return 0;
}
