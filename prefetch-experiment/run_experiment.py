import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Docker image build
# ─────────────────────────────────────────────────────────────────────────────

def build_image(image_name, disable_prefetch=False, python_version="3.10-slim"):
    print(f"Building Docker image: {image_name}")
    # Platform is pinned to linux/amd64 so that __SSE__ is defined by the compiler.
    # On ARM64 (Apple Silicon default), __SSE__ is absent, hnswlib guards all
    # _mm_prefetch calls behind #ifdef USE_SSE, and they are compiled out entirely —
    # making the ON/OFF builds identical.  amd64 guarantees USE_SSE is active.
    #
    # The injection marker is '#include <memory>' — the last #include in hnswalg.h.
    dockerfile_content = f"""FROM python:{python_version}
RUN apt-get update && apt-get install -y build-essential python3-dev git
ARG DISABLE_PREFETCH=0
RUN pip install numpy
RUN git clone --depth 1 https://github.com/nmslib/hnswlib.git /tmp/hnswlib
RUN if [ "$DISABLE_PREFETCH" = "1" ]; then \\
      python -c "from pathlib import Path; p=Path('/tmp/hnswlib/hnswlib/hnswalg.h'); t=p.read_text(); marker='#include <memory>'; inject='\\\\n#ifdef DISABLE_HNSW_PREFETCH\\\\n#define _mm_prefetch(a, sel) ((void)0)\\\\n#endif\\\\n'; t=t.replace(marker, marker+inject, 1) if marker in t and 'DISABLE_HNSW_PREFETCH' not in t else t; p.write_text(t)"; \\
      CXXFLAGS='-DDISABLE_HNSW_PREFETCH' pip install /tmp/hnswlib; \\
    else \\
      pip install /tmp/hnswlib; \\
    fi && rm -rf /tmp/hnswlib
COPY worker.py /app/worker.py
WORKDIR /app
CMD ["python", "worker.py"]
"""
    Path("Dockerfile").write_text(dockerfile_content)

    cmd = ["docker", "build", "--platform", "linux/amd64", "-t", image_name, "."]
    if disable_prefetch:
        cmd.extend(["--build-arg", "DISABLE_PREFETCH=1"])
    subprocess.run(cmd, check=True)


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_result(stdout):
    result = {
        "latency_ms": None,
        "vm_rss_kb":  -1,
        "vm_swap_kb": -1,
        "recall":     -1.0,
        "p50_ms":     -1.0,
        "p95_ms":     -1.0,
        "p99_ms":     -1.0,
    }
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("RESULT,"):
            parts = line.split(",")
            result["latency_ms"] = float(parts[1])
            if len(parts) > 2:
                result["vm_rss_kb"] = int(parts[2])
            if len(parts) > 3:
                result["vm_swap_kb"] = int(parts[3])
        elif line.startswith("RECALL,"):
            result["recall"] = float(line.split(",")[1])
        elif line.startswith("LATENCY_STATS,"):
            parts = line.split(",")
            result["p50_ms"] = float(parts[1])
            result["p95_ms"] = float(parts[2])
            result["p99_ms"] = float(parts[3])
    return result if result["latency_ms"] is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Single container run
# ─────────────────────────────────────────────────────────────────────────────

def run_once(
    image_name,
    dataset_path,
    timeout_s,
    num_queries,
    ef,
    k,
    num_threads,
    batch_size,
    query_noise,
    memory_limit,
    memory_swap,
    seed,
    chunk_size,
    ef_construction,
    M,
    recall_samples,
    index_size,
):
    dataset_abs = str(dataset_path.resolve())
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "-e", f"HNSW_NUM_QUERIES={num_queries}",
        "-e", f"HNSW_EF={ef}",
        "-e", f"HNSW_K={k}",
        "-e", f"HNSW_NUM_THREADS={num_threads}",
        "-e", f"HNSW_BATCH_SIZE={batch_size}",
        "-e", f"HNSW_QUERY_NOISE={query_noise}",
        "-e", f"HNSW_SEED={seed}",
        "-e", f"HNSW_CHUNK_SIZE={chunk_size}",
        "-e", f"HNSW_EF_CONSTRUCTION={ef_construction}",
        "-e", f"HNSW_M={M}",
        "-e", f"HNSW_RECALL_SAMPLES={recall_samples}",
        "-e", f"HNSW_INDEX_SIZE={index_size}",
        "-v", f"{dataset_abs}:/app/real_world_dataset.npy:ro",
    ]
    if memory_limit:
        cmd.extend(["--memory", memory_limit])
    if memory_swap:
        cmd.extend(["--memory-swap", memory_swap])
    cmd.append(image_name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or f"Container failed: {image_name}")
    parsed = parse_result(result.stdout)
    if parsed is None:
        raise RuntimeError("Could not parse RESULT line from worker output.")
    return parsed, result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def save_raw_log(out_dir, mode, cycle, text):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{mode}_cycle_{cycle:03d}.log"
    path.write_text(text)
    return path


def run_benchmark(
    image_on,
    image_off,
    dataset_path,
    runs,
    warmup,
    timeout_s,
    out_dir,
    num_queries,
    ef,
    k,
    num_threads,
    batch_size,
    query_noise,
    memory_limit,
    memory_swap,
    seed,
    chunk_size,
    ef_construction,
    M,
    recall_samples,
    index_size=0,
):
    _run_args = dict(
        dataset_path=dataset_path, timeout_s=timeout_s,
        num_queries=num_queries, ef=ef, k=k, num_threads=num_threads,
        batch_size=batch_size, query_noise=query_noise,
        memory_limit=memory_limit, memory_swap=memory_swap,
        seed=seed, chunk_size=chunk_size, ef_construction=ef_construction,
        M=M, recall_samples=recall_samples, index_size=index_size,
    )

    rows = []

    # Warmup runs are discarded to reduce first-run cold-start bias.
    for i in range(warmup):
        print(f"Warmup {i + 1}/{warmup}: prefetch ON")
        run_once(image_on, **_run_args)
        print(f"Warmup {i + 1}/{warmup}: prefetch OFF")
        run_once(image_off, **_run_args)

    # Alternate ON/OFF order each cycle to reduce thermal/order bias.
    for cycle in range(1, runs + 1):
        order = ["on", "off"] if cycle % 2 == 1 else ["off", "on"]
        print(f"Cycle {cycle}/{runs} order: {order[0].upper()} -> {order[1].upper()}")
        for mode in order:
            image  = image_on if mode == "on" else image_off
            parsed, raw = run_once(image, **_run_args)
            log_path = save_raw_log(out_dir, mode, cycle, raw)
            rows.append({
                "cycle":       cycle,
                "order_first": order[0],
                "mode":        mode,
                "latency_ms":  parsed["latency_ms"],
                "p50_ms":      parsed["p50_ms"],
                "p95_ms":      parsed["p95_ms"],
                "p99_ms":      parsed["p99_ms"],
                "recall":      parsed["recall"],
                "vm_rss_kb":   parsed["vm_rss_kb"],
                "vm_swap_kb":  parsed["vm_swap_kb"],
                "raw_log":     str(log_path),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Scaling sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_scaling_sweep(
    image_on, image_off, dataset_path, index_sizes,
    timeout_s, num_queries, ef, k, num_threads, batch_size,
    query_noise, memory_limit, memory_swap, seed, chunk_size,
    ef_construction, M, recall_samples, sweep_runs, sweep_warmup, out_dir,
):
    import statistics
    scaling_data = []
    for idx_size in index_sizes:
        print(f"\n--- Scaling sweep: index_size={idx_size:,} ---")
        sweep_dir = out_dir / f"sweep_{idx_size}"
        rows = run_benchmark(
            image_on, image_off, dataset_path,
            sweep_runs, sweep_warmup, timeout_s, sweep_dir,
            num_queries, ef, k, num_threads, batch_size, query_noise,
            memory_limit, memory_swap, seed, chunk_size, ef_construction, M,
            recall_samples, idx_size,
        )
        on_vals  = [r["latency_ms"] for r in rows if r["mode"] == "on"]
        off_vals = [r["latency_ms"] for r in rows if r["mode"] == "off"]
        on_mean  = statistics.mean(on_vals)
        off_mean = statistics.mean(off_vals)
        speedup  = (off_mean - on_mean) / on_mean * 100
        print(f"  ON mean: {on_mean:.4f} ms  OFF mean: {off_mean:.4f} ms  speedup: +{speedup:.1f}%")
        scaling_data.append({
            "index_size":  idx_size,
            "on_mean":     on_mean,
            "off_mean":    off_mean,
            "speedup_pct": speedup,
        })
    return scaling_data


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows, csv_path):
    fields = [
        "cycle", "order_first", "mode",
        "latency_ms", "p50_ms", "p95_ms", "p99_ms",
        "recall", "vm_rss_kb", "vm_swap_kb", "raw_log",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_scaling_csv(scaling_data, csv_path):
    fields = ["index_size", "on_mean", "off_mean", "speedup_pct"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scaling_data)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(vals):
    valid = [v for v in vals if v >= 0]
    import statistics
    return statistics.mean(valid) if valid else 0.0


def plot_comparison(rows, out_path, ef, num_elements, dim):
    import statistics
    import subprocess
    import sys

    on_rows  = [r for r in rows if r["mode"] == "on"]
    off_rows = [r for r in rows if r["mode"] == "off"]
    on_vals  = [r["latency_ms"] for r in on_rows]
    off_vals = [r["latency_ms"] for r in off_rows]

    on_mean  = statistics.mean(on_vals)
    off_mean = statistics.mean(off_vals)
    pct_slower = (off_mean - on_mean) / on_mean * 100

    on_p50  = _safe_mean([r["p50_ms"] for r in on_rows])
    on_p95  = _safe_mean([r["p95_ms"] for r in on_rows])
    on_p99  = _safe_mean([r["p99_ms"] for r in on_rows])
    off_p50 = _safe_mean([r["p50_ms"] for r in off_rows])
    off_p95 = _safe_mean([r["p95_ms"] for r in off_rows])
    off_p99 = _safe_mean([r["p99_ms"] for r in off_rows])

    on_recall  = _safe_mean([r["recall"] for r in on_rows])
    off_recall = _safe_mean([r["recall"] for r in off_rows])
    has_percentiles = on_p50 > 0
    has_recall      = on_recall > 0

    color_on  = "#2196F3"
    color_off = "#F44336"

    n_label = f"{num_elements // 1000} K" if num_elements >= 1000 else str(num_elements)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "HNSW Prefetch ON vs OFF — Per-query Latency Benchmark\n"
        f"({n_label} vectors · {dim}-dim · ef={ef} · real measured runs)",
        fontsize=11,
    )

    # ── Panel 1: Grouped bar chart (mean, p50, p95, p99) ─────────────────────
    ax1 = axes[0]
    metrics      = ["mean", "p50", "p95", "p99"]
    on_bars_val  = [on_mean,  on_p50,  on_p95,  on_p99]
    off_bars_val = [off_mean, off_p50, off_p95, off_p99]
    x = np.arange(len(metrics))
    w = 0.35
    b_on  = ax1.bar(x - w / 2, on_bars_val,  w, label="Prefetch ON",  color=color_on,  alpha=0.85)
    b_off = ax1.bar(x + w / 2, off_bars_val, w, label="Prefetch OFF", color=color_off, alpha=0.85)
    for bar in list(b_on) + list(b_off):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(on_bars_val + off_bars_val) * 0.01,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=7,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel("Latency per query (ms)")
    ax1.set_title(f"Latency percentiles  (+{pct_slower:.1f}% slower without prefetch)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(on_bars_val + off_bars_val) * 1.25)

    # ── Panel 2: Per-cycle mean latency line ──────────────────────────────────
    ax2 = axes[1]
    on_idx  = list(range(1, len(on_vals) + 1))
    off_idx = list(range(1, len(off_vals) + 1))
    ax2.plot(on_idx,  on_vals,  "o-",  color=color_on,  linewidth=2, markersize=7, label="Prefetch ON")
    ax2.plot(off_idx, off_vals, "s--", color=color_off, linewidth=2, markersize=7, label="Prefetch OFF")
    ax2.axhline(on_mean,  color=color_on,  linestyle=":", linewidth=1,
                alpha=0.7, label=f"ON mean {on_mean:.4f} ms")
    ax2.axhline(off_mean, color=color_off, linestyle=":", linewidth=1,
                alpha=0.7, label=f"OFF mean {off_mean:.4f} ms")
    ax2.set_xlabel("Run index")
    ax2.set_ylabel("Mean latency per query (ms)")
    ax2.set_title("Per-run mean latency (all measured cycles)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(on_idx)

    # ── Panel 3: Recall@k parity ──────────────────────────────────────────────
    ax3 = axes[2]
    if has_recall:
        k_val = int(round(1 / max(on_recall, off_recall, 0.0001)))  # rough estimate; label comes from row
        r_bars = ax3.bar(
            ["Prefetch ON", "Prefetch OFF"],
            [on_recall, off_recall],
            color=[color_on, color_off],
            width=0.45,
            alpha=0.85,
        )
        for bar, val in zip(r_bars, [on_recall, off_recall]):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax3.set_ylim(0, 1.15)
        ax3.set_ylabel("Recall@k")
        ax3.set_title("Search quality parity\n(prefetch must not change results)")
        ax3.grid(axis="y", alpha=0.3)
        ax3.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Perfect recall")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Recall data not available\n(update worker to latest version)",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=10)
        ax3.set_title("Recall@k")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(out_path)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(out_path)])
        elif sys.platform == "win32":
            subprocess.Popen(["start", str(out_path)], shell=True)
    except Exception:
        pass


def plot_scaling(scaling_data, out_path):
    import subprocess
    import sys

    sizes    = [d["index_size"] for d in scaling_data]
    speedups = [d["speedup_pct"] for d in scaling_data]
    on_vals  = [d["on_mean"] for d in scaling_data]
    off_vals = [d["off_mean"] for d in scaling_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("HNSW Prefetch Benefit vs. Index Size", fontsize=12)

    color_speedup = "#4CAF50"
    color_on      = "#2196F3"
    color_off     = "#F44336"

    # Panel 1: speedup %
    ax1.plot(sizes, speedups, "o-", color=color_speedup, linewidth=2.5, markersize=9, zorder=3)
    ax1.fill_between(sizes, 0, speedups, alpha=0.12, color=color_speedup)
    for s, sp in zip(sizes, speedups):
        ax1.annotate(
            f"+{sp:.1f}%",
            (s, sp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Index size (vectors)")
    ax1.set_ylabel("Prefetch speedup (%)\n(positive = prefetch ON is faster)")
    ax1.set_title("How prefetch benefit scales with graph size")
    ax1.grid(alpha=0.3)

    # Panel 2: absolute mean latency for ON and OFF
    ax2.plot(sizes, on_vals,  "o-",  color=color_on,  linewidth=2, markersize=8, label="Prefetch ON")
    ax2.plot(sizes, off_vals, "s--", color=color_off, linewidth=2, markersize=8, label="Prefetch OFF")
    ax2.set_xlabel("Index size (vectors)")
    ax2.set_ylabel("Mean latency per query (ms)")
    ax2.set_title("Absolute latency vs. index size")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Scaling plot saved → {out_path}")

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(out_path)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(out_path)])
        elif sys.platform == "win32":
            subprocess.Popen(["start", str(out_path)], shell=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────

def print_raw_results(rows):
    print("\n=== Raw Measured Results ===")
    header = "cycle,order_first,mode,latency_ms,p50_ms,p95_ms,p99_ms,recall,vm_rss_kb,vm_swap_kb"
    print(header)
    for r in rows:
        print(
            f"{r['cycle']},{r['order_first']},{r['mode']},"
            f"{r['latency_ms']},{r['p50_ms']},{r['p95_ms']},{r['p99_ms']},"
            f"{r['recall']},{r['vm_rss_kb']},{r['vm_swap_kb']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real A/B benchmark for HNSW prefetch ON vs OFF.")
    parser.add_argument("--dataset",      default="real_world_dataset.npy", help="Path to .npy dataset")
    parser.add_argument("--runs",         type=int,   default=4,     help="Measured cycles; each cycle runs ON and OFF once")
    parser.add_argument("--warmup",       type=int,   default=1,     help="Warmup cycles not included in results")
    parser.add_argument("--timeout",      type=int,   default=1800,  help="Timeout seconds per container run")
    parser.add_argument("--num-queries",  type=int,   default=50000, help="Queries per run inside worker")
    parser.add_argument("--ef",           type=int,   default=200,   help="HNSW ef search parameter")
    parser.add_argument("--k",            type=int,   default=20,    help="Top-k for knn query")
    parser.add_argument("--num-threads",  type=int,   default=1,     help="Threads used by hnsw knn_query")
    parser.add_argument("--batch-size",   type=int,   default=2000,  help="Batch size per knn_query call")
    parser.add_argument("--query-noise",  type=float, default=0.01,  help="Noise added to sampled queries")
    parser.add_argument("--memory-limit", default="",  help="Optional docker --memory (e.g. 3g)")
    parser.add_argument("--memory-swap",  default="",  help="Optional docker --memory-swap (e.g. 5g)")
    parser.add_argument("--seed",             type=int,   default=42,    help="RNG seed for query generation inside worker")
    parser.add_argument("--chunk-size",       type=int,   default=50000, help="Query generation chunk size inside worker")
    parser.add_argument("--ef-construction",  type=int,   default=100,   help="HNSW ef_construction parameter (graph build quality)")
    parser.add_argument("--hnsw-m",           type=int,   default=32,    help="HNSW M parameter (graph connectivity)")
    parser.add_argument("--recall-samples",   type=int,   default=200,   help="Queries used for recall@k ground-truth check")
    parser.add_argument("--index-size",       type=int,   default=0,     help="Subset the dataset to this many vectors (0 = full)")
    parser.add_argument("--python-version",   default="3.10-slim",       help="Python Docker image tag (e.g. 3.11-slim)")
    # Scaling sweep
    parser.add_argument(
        "--index-sizes",
        default="",
        help="Comma-separated list of index sizes for a scaling sweep (e.g. 10000,50000,100000,150000)",
    )
    parser.add_argument("--sweep-runs",   type=int, default=2, help="Benchmark cycles per scaling-sweep point")
    parser.add_argument("--sweep-warmup", type=int, default=0, help="Warmup cycles per scaling-sweep point")
    parser.add_argument(
        "--scenario",
        choices=["default", "prefetch_friendly", "prefetch_advantage"],
        default="default",
        help="Preset knobs to test a prefetch-favorable but real workload",
    )
    parser.add_argument("--out-dir", default="results", help="Directory for raw logs/CSV/plot")
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.num_queries < 1000:
        raise ValueError("--num-queries must be >= 1000")
    if args.ef < 1:
        raise ValueError("--ef must be >= 1")
    if args.k < 1:
        raise ValueError("--k must be >= 1")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.query_noise < 0:
        raise ValueError("--query-noise must be >= 0")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.ef_construction < 1:
        raise ValueError("--ef-construction must be >= 1")
    if args.hnsw_m < 1:
        raise ValueError("--hnsw-m must be >= 1")
    if args.recall_samples < 1:
        raise ValueError("--recall-samples must be >= 1")
    if args.sweep_runs < 1:
        raise ValueError("--sweep-runs must be >= 1")

    if args.scenario == "prefetch_friendly":
        args.ef           = max(args.ef, 250)
        args.num_queries  = max(args.num_queries, 500000)
        args.query_noise  = max(args.query_noise, 0.05)
        args.batch_size   = min(args.batch_size, 2000)
        args.num_threads  = 1
    elif args.scenario == "prefetch_advantage":
        args.ef           = max(args.ef, 200)
        args.k            = max(args.k, 40)
        args.num_queries  = max(args.num_queries, 150000)
        args.query_noise  = max(args.query_noise, 0.20)
        args.batch_size   = min(args.batch_size, 512)
        args.num_threads  = 1
        args.runs         = max(args.runs, 2)
        args.warmup       = max(args.warmup, 1)
        if not args.memory_limit:
            args.memory_limit = "2g"
        if not args.memory_swap:
            args.memory_swap = "4g"

    dataset_arg = args.dataset.strip()
    if dataset_arg == "/absolute/path/to/your_real_dataset.npy":
        local_default = Path("real_world_dataset.npy")
        if local_default.exists():
            print("Detected placeholder dataset path; using ./real_world_dataset.npy")
            dataset_arg = str(local_default)

    dataset_path = Path(dataset_arg).expanduser()
    if dataset_path.suffix.lower() != ".npy":
        raise ValueError("Dataset must be a .npy file")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            "Please pass a real dataset via --dataset /absolute/path/to/file.npy"
        )

    dataset_meta = np.load(dataset_path, mmap_mode="r")
    if dataset_meta.ndim == 1:
        dataset_meta = dataset_meta.reshape(1, -1)
    full_num_elements, dim = dataset_meta.shape
    del dataset_meta
    num_elements = args.index_size if 0 < args.index_size < full_num_elements else full_num_elements
    print(f"Using dataset: {dataset_path.resolve()} ({full_num_elements} vectors · {dim}-dim)")
    if num_elements != full_num_elements:
        print(f"Index size limited to {num_elements} vectors via --index-size")

    image_on  = "hnsw-prefetch-on"
    image_off = "hnsw-prefetch-off"
    build_image(image_on,  disable_prefetch=False, python_version=args.python_version)
    build_image(image_off, disable_prefetch=True,  python_version=args.python_version)

    out_dir = Path(args.out_dir)

    # ── Common kwargs threaded through all benchmark calls ────────────────────
    bench_kwargs = dict(
        dataset_path=dataset_path,
        timeout_s=args.timeout,
        num_queries=args.num_queries,
        ef=args.ef,
        k=args.k,
        num_threads=args.num_threads,
        batch_size=args.batch_size,
        query_noise=args.query_noise,
        memory_limit=args.memory_limit,
        memory_swap=args.memory_swap,
        seed=args.seed,
        chunk_size=args.chunk_size,
        ef_construction=args.ef_construction,
        M=args.hnsw_m,
        recall_samples=args.recall_samples,
    )

    # ── Main benchmark ────────────────────────────────────────────────────────
    rows = run_benchmark(
        image_on, image_off,
        runs=args.runs, warmup=args.warmup, out_dir=out_dir,
        index_size=args.index_size,
        **bench_kwargs,
    )
    csv_path = out_dir / "raw_results.csv"
    write_csv(rows, csv_path)

    n = 1
    while (out_dir / f"output_{n}.png").exists():
        n += 1
    plot_path = out_dir / f"output_{n}.png"
    plot_comparison(rows, plot_path, ef=args.ef, num_elements=num_elements, dim=dim)
    print_raw_results(rows)

    print("\nNo aggregated math is printed. Only raw measured runs are reported.")
    print(f"Raw CSV:      {csv_path}")
    print(f"Raw logs dir: {out_dir}")
    print(f"Plot:         {plot_path}")

    # ── Scaling sweep (optional) ──────────────────────────────────────────────
    if args.index_sizes.strip():
        index_sizes = [int(s.strip()) for s in args.index_sizes.split(",") if s.strip()]
        if index_sizes:
            print(f"\nRunning scaling sweep over index sizes: {index_sizes}")
            scaling_data = run_scaling_sweep(
                image_on, image_off,
                index_sizes=index_sizes,
                sweep_runs=args.sweep_runs,
                sweep_warmup=args.sweep_warmup,
                out_dir=out_dir,
                **bench_kwargs,
            )
            scaling_csv = out_dir / "scaling_results.csv"
            write_scaling_csv(scaling_data, scaling_csv)
            scaling_plot = out_dir / "scaling_plot.png"
            plot_scaling(scaling_data, scaling_plot)
            print(f"Scaling CSV:  {scaling_csv}")
            print(f"Scaling plot: {scaling_plot}")


if __name__ == "__main__":
    main()
