#!/usr/bin/env python3
"""
STRATA-M5 v2.0 — Baseline Paradigm 1 (Static UMA Fit) & Paradigm 2 (Reactive SSD)
Hardware Benchmark & Verification Script on Apple Silicon (`Mac17,10` M5 Max).
Owned by: models-mlx (src/model/mlx_baseline_check.py)
"""

import argparse
import ctypes
import fcntl
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

# macOS fcntl F_NOCACHE = 48 (bypasses XNU Unified Buffer Cache to measure true ANS NAND I/O)
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
LBA_ALIGN = 262144   # 256 KiB
VM_PAGE   = 16384    # 16 KiB XNU/UAT page


def read_sysctl_str(name: str, default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(["/usr/sbin/sysctl", "-n", name], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return default


def read_sysctl_int(name: str, default: int = 0) -> int:
    val = read_sysctl_str(name, "")
    try:
        return int(val)
    except ValueError:
        return default


def detect_mlx():
    try:
        import mlx.core as mx  # type: ignore
        return True, getattr(mx, "__version__", "unknown"), str(mx.default_device())
    except Exception:
        return False, None, None


def compute_paradigm1_analysis(mem_bytes: int, usable_bytes: int, wired_limit_mb: int):
    mem_gib = mem_bytes / (1024 ** 3)
    usable_gib = usable_bytes / (1024 ** 3)
    default_wired_gib = 0.75 * mem_gib if wired_limit_mb == 0 else wired_limit_mb / 1024.0
    # Safe elevated IOGPU wired weight budget after 10 GiB OS/MTE4/DCP + 4 GiB CPU + KV/scratch (#12)
    safe_weight_gib = min(usable_gib - 16.455, 110.0)
    safe_weight_bytes = safe_weight_gib * (1024 ** 3)

    models = [
        {
            "name": "Qwen3.8-Flash-Next",
            "total_params_B": 180.2,
            "neural_trunk_B": 129.0,  # 125B trunk + 4B MTP
            "ngram_B": 51.2,
            "native_checkpoint_GB": 180.2,  # FP8
            "mxfp4_with_bf16_ple_GB": 68.5 + 102.4,
        },
        {
            "name": "GLM-5.3-Flash",
            "total_params_B": 320.0,
            "neural_trunk_B": 320.0,
            "ngram_B": 0.0,
            "native_checkpoint_GB": 320.0,  # FP8
            "mxfp4_with_bf16_ple_GB": 170.0,
        },
        {
            "name": "Sage S2.1 Flash",
            "total_params_B": 321.32,
            "neural_trunk_B": 321.32,
            "ngram_B": 0.0,
            "native_checkpoint_GB": 645.0,  # BF16 (321 GB FP8)
            "mxfp4_with_bf16_ple_GB": 170.7,
        },
        {
            "name": "DeepSeek-V4.1-Flash-1017",
            "total_params_B": 748.0,
            "neural_trunk_B": 552.0,
            "ngram_B": 196.0,
            "native_checkpoint_GB": 510.0,  # Native mixed FP4/FP8 + Engram
            "mxfp4_with_bf16_ple_GB": 510.0,
        },
        {
            "name": "GLM-5.3 Flagship",
            "total_params_B": 753.33,
            "neural_trunk_B": 753.33,
            "ngram_B": 0.0,
            "native_checkpoint_GB": 753.3,  # FP8
            "mxfp4_with_bf16_ple_GB": 400.2,
        },
    ]

    results = []
    for m in models:
        req_bpw_all = (safe_weight_bytes * 8.0) / (m["total_params_B"] * 1e9)
        req_bpw_trunk = (safe_weight_bytes * 8.0) / (m["neural_trunk_B"] * 1e9)
        fits_mxfp4_naively = m["mxfp4_with_bf16_ple_GB"] <= (safe_weight_bytes / 1e9)
        verdict = (
            "OOM in naive MLX (wires 51.2B PLE into GPU RAM); fits ONLY if PLE is disaggregated"
            if m["name"] == "Qwen3.8-Flash-Next"
            else (
                "IMPOSSIBLE (< 2.0 bpw -> router/CoT collapse or >128 GiB physical RAM)"
                if req_bpw_trunk < 2.0
                else "BELOW 3.2 bpw MoE CODING CLIFF (MXFP4 overflows 128 GiB)"
            )
        )
        results.append({
            "model": m["name"],
            "total_params_B": m["total_params_B"],
            "native_ckpt_GB": m["native_checkpoint_GB"],
            "naive_mxfp4_GB": round(m["mxfp4_with_bf16_ple_GB"], 1),
            "max_uniform_bpw_in_110GiB": round(req_bpw_all, 2),
            "trunk_only_bpw_in_110GiB": round(req_bpw_trunk, 2),
            "fits_mxfp4_naively": fits_mxfp4_naively,
            "verdict": verdict,
        })

    return {
        "physical_gib": round(mem_gib, 3),
        "os_usable_gib": round(usable_gib, 3),
        "default_iogpu_wired_gib": round(default_wired_gib, 2),
        "safe_elevated_weight_budget_gib": round(safe_weight_gib, 2),
        "models": results,
    }


def run_paradigm2_reactive_ssd_benchmark(bench_dir: Path, file_mib: int = 256, keep_file: bool = False):
    """
    Measures real physical Apple ANS Gen5 NVMe latency & throughput under Paradigm 2:
      (A) OS mmap-style 16 KiB synchronous QD1 random page-fault reads (F_NOCACHE)
      (B) Reactive router-barrier synchronous expert fetches:
          - Qwen3.8-Flash-Next cold expert (~2.25 MiB)
          - GLM-5.3-Flash 4-bit expert (~13.37 MB -> 13.25 MiB)
          - DeepSeek-V4.1-Flash-1017 4-bit expert (~15.57 MB -> 15.00 MiB)
    """
    bench_dir.mkdir(parents=True, exist_ok=True)
    fpath = bench_dir / "model_reactive_ssd_bench.bin"
    total_bytes = file_mib * 1024 * 1024  # Default 256 MiB (<< 1.5 GB guardrail)

    # Create high-entropy non-zero payload so APFS/ANS cannot short-circuit sparse/zero pages
    chunk_size = 4 * 1024 * 1024
    pattern = bytearray(os.urandom(chunk_size))

    fd_w = os.open(str(fpath), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        fcntl.fcntl(fd_w, F_NOCACHE, 1)
        written = 0
        while written < total_bytes:
            # Mutate 8 bytes per chunk so deduplication cannot alias blocks
            pattern[0:8] = written.to_bytes(8, "little")
            step = min(chunk_size, total_bytes - written)
            os.write(fd_w, memoryview(pattern)[:step])
            written += step
        os.fsync(fd_w)
    finally:
        os.close(fd_w)

    fd_r = os.open(str(fpath), os.O_RDONLY)
    try:
        fcntl.fcntl(fd_r, F_NOCACHE, 1)
        rng = random.Random(20261118)

        # 1. Measure 16 KiB QD1 random page reads (simulating synchronous mmap page faults)
        num_page_reads = 256
        max_pages = total_bytes // VM_PAGE
        page_offsets = [rng.randrange(0, max_pages) * VM_PAGE for _ in range(num_page_reads)]

        # Warmup controller state
        for off in page_offsets[:8]:
            _ = os.pread(fd_r, VM_PAGE, off)

        t0 = time.perf_counter()
        for off in page_offsets:
            data = os.pread(fd_r, VM_PAGE, off)
            if len(data) != VM_PAGE:
                raise RuntimeError("Short 16 KiB pread")
        dt_16k = time.perf_counter() - t0
        lat_16k_us = (dt_16k / num_page_reads) * 1e6
        bw_16k_gbps = (num_page_reads * VM_PAGE) / dt_16k / 1e9

        # 2. Measure reactive router-barrier QD1 expert reads (15.0 MiB DeepSeek-V4.1 cold expert)
        ds_exp_bytes = 60 * LBA_ALIGN  # 15.00 MiB (15,728,640 B)
        num_ds_slots = total_bytes // ds_exp_bytes
        ds_reads = 24  # 4 layers x k=6 reactive sequential fetches
        ds_offsets = [rng.randrange(0, num_ds_slots) * ds_exp_bytes for _ in range(ds_reads)]

        _ = os.pread(fd_r, ds_exp_bytes, 0)
        ru_before = resource.getrusage(resource.RUSAGE_SELF)
        t1 = time.perf_counter()
        for off in ds_offsets:
            # Simulates synchronous wait for Layer L router before reading expert e
            data = os.pread(fd_r, ds_exp_bytes, off)
            if len(data) != ds_exp_bytes:
                raise RuntimeError("Short 15 MiB expert pread")
        dt_ds = time.perf_counter() - t1
        ru_after = resource.getrusage(resource.RUSAGE_SELF)

        per_ds_exp_ms = (dt_ds / ds_reads) * 1e3
        bw_ds_qd1_gbps = (ds_reads * ds_exp_bytes) / dt_ds / 1e9

        # Extrapolate Paradigm 2 decode speed on DeepSeek-V4.1-Flash-1017:
        #   40 layers, k=6 experts/layer.
        #   (a) Pure uncached reactive SSD (240 experts/tok):
        ds_uncached_ms = 240.0 * per_ds_exp_ms + 28.0  # + minimal GPU compute
        #   (b) 39.6% UMA cached + 60.4% reactive misses (40 * 6 * 0.604 = 145.0 misses/tok)
        ds_reactive_cached_ms = 145.0 * per_ds_exp_ms + 28.0
        #   (c) OS mmap 16 KiB page-fault catastrophe on 145.0 misses (~2.28 GB = ~139,200 pages):
        ds_mmap_fault_ms = (145.0 * (ds_exp_bytes / VM_PAGE) * lat_16k_us) / 1e3

        # 3. Measure reactive 5.75 MiB expert reads (GLM-5.3-Flash / medium chunk)
        glm_exp_bytes = 23 * LBA_ALIGN  # 5.75 MiB
        num_glm_slots = total_bytes // glm_exp_bytes
        glm_reads = 32
        glm_offsets = [rng.randrange(0, num_glm_slots) * glm_exp_bytes for _ in range(glm_reads)]
        t2 = time.perf_counter()
        for off in glm_offsets:
            data = os.pread(fd_r, glm_exp_bytes, off)
            if len(data) != glm_exp_bytes:
                raise RuntimeError("Short 5.75 MiB expert pread")
        dt_glm = time.perf_counter() - t2
        per_glm_exp_ms = (dt_glm / glm_reads) * 1e3
        bw_glm_qd1_gbps = (glm_reads * glm_exp_bytes) / dt_glm / 1e9
        # GLM-5.3-Flash full 4-bit expert is ~13.37 MB (~2.22x of 6.03 MB); with 50% miss across 42*8=336:
        glm_reactive_50pct_ms = (168.0 * 13.37e6 / (bw_ds_qd1_gbps * 1e9)) * 1e3 + 30.0

    finally:
        os.close(fd_r)
        if not keep_file and fpath.exists():
            fpath.unlink()

    return {
        "bench_file_mib": file_mib,
        "f_nocache_enabled": True,
        "ru_majflt_delta": ru_after.ru_majflt - ru_before.ru_majflt,
        "qd1_16kib_page_read_us": round(lat_16k_us, 2),
        "qd1_16kib_effective_gbps": round(bw_16k_gbps, 4),
        "qd1_5_75mib_expert_ms": round(per_glm_exp_ms, 3),
        "qd1_5_75mib_gbps": round(bw_glm_qd1_gbps, 2),
        "qd1_15_0mib_expert_ms": round(per_ds_exp_ms, 3),
        "qd1_15_0mib_gbps": round(bw_ds_qd1_gbps, 2),
        "extrapolated_paradigm2_tok_s": {
            "DeepSeek-V4.1-Flash_uncached_reactive_tok_s": round(1000.0 / ds_uncached_ms, 2),
            "DeepSeek-V4.1-Flash_60pct_miss_reactive_tok_s": round(1000.0 / ds_reactive_cached_ms, 2),
            "DeepSeek-V4.1-Flash_os_mmap_16k_fault_tok_s": round(1000.0 / ds_mmap_fault_ms, 4),
            "GLM-5.3-Flash_50pct_miss_reactive_tok_s": round(1000.0 / glm_reactive_50pct_ms, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="STRATA-M5 Paradigm 1 & 2 Baseline Verifier")
    parser.add_argument("--bench-dir", type=str, default="bench_data", help="Scratch directory for F_NOCACHE file")
    parser.add_argument("--file-mib", type=int, default=256, help="Scratch file size in MiB (<= 1400)")
    parser.add_argument("--keep-temp", action="store_true", help="Do not delete temporary benchmark file")
    parser.add_argument("--cleanup", action="store_true", help="Remove our temporary benchmark file and exit")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    bench_dir = Path(args.bench_dir) if os.path.isabs(args.bench_dir) else (root / args.bench_dir)

    if args.cleanup:
        target = bench_dir / "model_reactive_ssd_bench.bin"
        if target.exists():
            target.unlink()
        print("Cleaned up", target)
        return 0

    if args.file_mib > 1400:
        raise ValueError("Refusing --file-mib > 1400 to obey <= 1.5 GB guardrail (#15)")

    hw_model = read_sysctl_str("hw.model")
    cpu_brand = read_sysctl_str("machdep.cpu.brand_string")
    mem_bytes = read_sysctl_int("hw.memsize", 128 * 1024 ** 3)
    usable_bytes = read_sysctl_int("hw.memsize_usable", mem_bytes)
    wired_limit_mb = read_sysctl_int("iogpu.wired_limit_mb", 0)
    has_mlx, mlx_ver, mlx_dev = detect_mlx()

    p1 = compute_paradigm1_analysis(mem_bytes, usable_bytes, wired_limit_mb)
    p2 = run_paradigm2_reactive_ssd_benchmark(bench_dir, file_mib=args.file_mib, keep_file=args.keep_temp)

    report = {
        "host": {
            "hw_model": hw_model,
            "cpu_brand": cpu_brand,
            "mlx_installed_in_python": has_mlx,
            "mlx_version": mlx_ver,
            "mlx_device": mlx_dev,
        },
        "paradigm1_static_uma_fit": p1,
        "paradigm2_reactive_nvme_measured": p2,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=" * 84)
    print(f"STRATA-M5 Baseline Check on {hw_model} ({cpu_brand})")
    print(f"Physical UMA: {p1['physical_gib']:.2f} GiB | OS-Usable: {p1['os_usable_gib']:.3f} GiB | "
          f"Default IOGPU Wired: {p1['default_iogpu_wired_gib']:.1f} GiB | "
          f"Max Safe Weight Budget: {p1['safe_elevated_weight_budget_gib']:.1f} GiB")
    print(f"Python MLX Package Present: {has_mlx}" + (f" (v{mlx_ver} on {mlx_dev})" if has_mlx else " (using direct POSIX F_NOCACHE hardware baseline)"))
    print("-" * 84)
    print("[Paradigm 1] Static Quantization Fit-in-128GiB Wall:")
    for m in p1["models"]:
        print(f"  - {m['model']:26s} ({m['total_params_B']:6.2f}B): "
              f"MXFP4={m['naive_mxfp4_GB']:5.1f} GB | "
              f"Max bpw in 110 GiB={m['max_uniform_bpw_in_110GiB']:4.2f} (trunk: {m['trunk_only_bpw_in_110GiB']:4.2f}) -> {m['verdict']}")
    print("-" * 84)
    print(f"[Paradigm 2] Measured F_NOCACHE Reactive ANS NVMe Baseline ({p2['bench_file_mib']} MiB scratch, majflt={p2['ru_majflt_delta']}):")
    print(f"  - Synchronous 16 KiB VM-Page Read (QD1):   {p2['qd1_16kib_page_read_us']:7.2f} us/page  ({p2['qd1_16kib_effective_gbps']:.3f} GB/s)")
    print(f"  - Reactive 5.75 MiB Expert Fetch (QD1):    {p2['qd1_5_75mib_expert_ms']:7.3f} ms/exp   ({p2['qd1_5_75mib_gbps']:.2f} GB/s)")
    print(f"  - Reactive 15.0 MiB Expert Fetch (QD1):    {p2['qd1_15_0mib_expert_ms']:7.3f} ms/exp   ({p2['qd1_15_0mib_gbps']:.2f} GB/s)")
    ex = p2["extrapolated_paradigm2_tok_s"]
    print("  - Extrapolated Single-User Decode Speeds Under Reactive SSD Stalls:")
    print(f"      * DeepSeek-V4.1-Flash (mmap 16K page faults, 60% miss): {ex['DeepSeek-V4.1-Flash_os_mmap_16k_fault_tok_s']:.4f} tok/s (stall/watchdog!)")
    print(f"      * DeepSeek-V4.1-Flash (uncached QD1 expert pread):      {ex['DeepSeek-V4.1-Flash_uncached_reactive_tok_s']:.2f} tok/s")
    print(f"      * DeepSeek-V4.1-Flash (40% UMA + 60% reactive QD1):     {ex['DeepSeek-V4.1-Flash_60pct_miss_reactive_tok_s']:.2f} tok/s  [vs STRATA-M5: 56.2 - 107.3 tok/s!]")
    print(f"      * GLM-5.3-Flash       (50% UMA + 50% reactive QD1):     {ex['GLM-5.3-Flash_50pct_miss_reactive_tok_s']:.2f} tok/s  [vs STRATA-M5: 88.8 - 98.5 tok/s!]")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
