import os
import subprocess
import sys
import time
import argparse
import csv
import numpy as np
import hnswlib
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# CPU cache detection (Linux + macOS)
# ─────────────────────────────────────────────────────────────────────────────

def get_cache_sizes():
    """Return {'L1d': KB, 'L2': KB, 'L3': KB}. Works on Linux and macOS."""
    caches = {"L1d": 0, "L2": 0, "L3": 0}
    # Linux sysfs
    try:
        for idx in range(8):
            base = f"/sys/devices/system/cpu/cpu0/cache/index{idx}"
            if not os.path.exists(base):
                continue
            with open(f"{base}/level") as f:
                level = int(f.read().strip())
            with open(f"{base}/type") as f:
                ctype = f.read().strip()
            with open(f"{base}/size") as f:
                s = f.read().strip()
                size_kb = int(s[:-1]) * (1024 if s.endswith("M") else 1) if s[-1] in "KM" else int(s) // 1024
            if level == 1 and ctype == "Data":
                caches["L1d"] = size_kb
            elif level == 2:
                caches["L2"] = size_kb
            elif level == 3:
                caches["L3"] = size_kb
    except Exception:
        pass
    # macOS sysctl fallback
    if caches["L1d"] == 0 and sys.platform == "darwin":
        for key, ctl in [("L1d", "hw.l1dcachesize"),
                         ("L2",  "hw.l2cachesize"),
                         ("L3",  "hw.l3cachesize")]:
            try:
                r = subprocess.run(["sysctl", "-n", ctl], capture_output=True, text=True)
                if r.returncode == 0 and r.stdout.strip().isdigit():
                    caches[key] = int(r.stdout.strip()) // 1024
            except Exception:
                pass
    # getconf fallback
    if caches["L1d"] == 0:
        for key, param in [("L1d", "LEVEL1_DCACHE_SIZE"),
                           ("L2",  "LEVEL2_CACHE_SIZE"),
                           ("L3",  "LEVEL3_CACHE_SIZE")]:
            try:
                r = subprocess.run(["getconf", param], capture_output=True, text=True)
                if r.returncode == 0 and r.stdout.strip().isdigit():
                    caches[key] = int(r.stdout.strip()) // 1024
            except Exception:
                pass
    return caches


def cache_fit_label(ws_kb, caches):
    """Return e.g. 'fits L1d' or 'spills to RAM'."""
    l1d, l2, l3 = caches["L1d"], caches["L2"], caches["L3"]
    if l1d == 0 and l2 == 0 and l3 == 0:
        return "?"
    if l1d > 0 and ws_kb <= l1d:
        return "fits L1d"
    if l2 > 0 and ws_kb <= l2:
        return "fits L2"
    if l3 > 0 and ws_kb <= l3:
        return "fits L3"
    return "spills to RAM"


CACHES = get_cache_sizes()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def measure_latency(index, queries, k, num_threads=1, runs=5):
    """Measure query latency (ms/query) over multiple runs. Returns (mean, std)."""
    latencies = []
    # Warmup run (discarded) to stabilize CPU caches
    index.knn_query(queries[:min(1000, len(queries))], k=k, num_threads=num_threads)
    for _ in range(runs):
        t0 = time.perf_counter()
        index.knn_query(queries, k=k, num_threads=num_threads)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed / len(queries) * 1000)
    return float(np.mean(latencies)), float(np.std(latencies))


def build_index(data, ef_construction=100, M=16, ef=100):
    """Build and return an HNSW index."""
    num_elements, dim = data.shape
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    index.add_items(data, np.arange(num_elements))
    index.set_ef(ef)
    return index


def generate_queries(data, num_queries=5000, noise=0.01, seed=42):
    """Generate queries by adding small noise to random data points."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=num_queries, replace=True)
    return data[indices] + rng.normal(scale=noise, size=(num_queries, data.shape[1])).astype(np.float32)


def find_dataset():
    """Look for real_world_dataset.npy in sibling experiment folders."""
    candidates = [
        "../prefetch-experiment/real_world_dataset.npy",
        "../memory-stress-experiment/real_world_dataset.npy",
        "real_world_dataset.npy",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sweeps
# ─────────────────────────────────────────────────────────────────────────────

def _per_query_ws_kb(ef, dim, M):
    """Working set touched per query: ~ef nodes * (vector + link list bytes)."""
    bytes_per_vector = dim * 4
    bytes_per_link_list = 2 * M * 4  # level 0 has 2M connections
    return (ef * (bytes_per_vector + bytes_per_link_list)) / 1024


def _index_total_mb(N, dim, M):
    bytes_per_vector = dim * 4
    bytes_per_link_list = 2 * M * 4
    return (N * (bytes_per_vector + bytes_per_link_list)) / (1024 * 1024)


def sweep_k(data, queries, ef=100, M=16, ef_construction=100, runs=5):
    """How does the number of requested neighbors (k) affect latency?

    k does not change traversal cost (same ef nodes are visited). Working set
    is constant across k values.
    """
    k_values = [1, 5, 10, 20, 50, 100]
    dim = data.shape[1]
    index = build_index(data, ef_construction=ef_construction, M=M, ef=ef)
    ws_kb = _per_query_ws_kb(ef, dim, M)
    fit = cache_fit_label(ws_kb, CACHES)
    print(f"  per-query working set: {ws_kb:.1f} KB ({fit}) — constant across k")
    results = []
    for k in k_values:
        mean, std = measure_latency(index, queries, k=k, runs=runs)
        print(f"  k={k:<5}  latency={mean:.4f} +/- {std:.4f} ms/query")
        results.append({"param": "k", "value": k, "latency_ms": mean, "std_ms": std,
                        "ws_kb": ws_kb, "cache_fit": fit})
    return results


def sweep_ef(data, queries, k=20, M=16, ef_construction=100, runs=5):
    """How does the search beam width (ef) affect latency?

    Working set scales linearly with ef: more candidates = more cache lines
    touched per query.
    """
    ef_values = [10, 50, 100, 200, 400, 800]
    dim = data.shape[1]
    index = build_index(data, ef_construction=ef_construction, M=M, ef=10)
    results = []
    for ef in ef_values:
        index.set_ef(ef)
        mean, std = measure_latency(index, queries, k=k, runs=runs)
        ws_kb = _per_query_ws_kb(ef, dim, M)
        fit = cache_fit_label(ws_kb, CACHES)
        print(f"  ef={ef:<5}  latency={mean:.4f} +/- {std:.4f} ms/query  "
              f"ws={ws_kb:.1f} KB ({fit})")
        results.append({"param": "ef", "value": ef, "latency_ms": mean, "std_ms": std,
                        "ws_kb": ws_kb, "cache_fit": fit})
    return results


def sweep_M(data, queries, k=20, ef=100, ef_construction=100, runs=5):
    """How does graph connectivity (M) affect latency?

    M widens each node's link list — more bytes per node = bigger cache
    footprint per hop.
    """
    M_values = [4, 8, 16, 32, 64]
    dim = data.shape[1]
    results = []
    for M in M_values:
        index = build_index(data, ef_construction=ef_construction, M=M, ef=ef)
        mean, std = measure_latency(index, queries, k=k, runs=runs)
        ws_kb = _per_query_ws_kb(ef, dim, M)
        fit = cache_fit_label(ws_kb, CACHES)
        print(f"  M={M:<5}  latency={mean:.4f} +/- {std:.4f} ms/query  "
              f"ws={ws_kb:.1f} KB ({fit})")
        results.append({"param": "M", "value": M, "latency_ms": mean, "std_ms": std,
                        "ws_kb": ws_kb, "cache_fit": fit})
    return results


def sweep_dataset_size(data, k=20, ef=100, M=16, ef_construction=100,
                       num_queries=10000, runs=5):
    """How does dataset size (N) affect latency?

    Per-query working set is ~constant (ef*node_size). What changes is total
    index footprint vs L3 — once index > L3, every miss goes to DRAM.
    """
    max_n = len(data)
    N_values = [s for s in [5000, 10000, 25000, 50000, 100000, 150000] if s <= max_n]
    dim = data.shape[1]
    results = []
    for N in N_values:
        subset = data[:N]
        q = generate_queries(subset, num_queries=num_queries)
        index = build_index(subset, ef_construction=ef_construction, M=M, ef=ef)
        mean, std = measure_latency(index, q, k=k, runs=runs)
        ws_kb = _per_query_ws_kb(ef, dim, M)
        idx_mb = _index_total_mb(N, dim, M)
        l3_mb = CACHES["L3"] / 1024 if CACHES["L3"] > 0 else 0
        l2_mb = CACHES["L2"] / 1024 if CACHES["L2"] > 0 else 0
        # Index fit: try L3 first, fall back to L2 (Apple Silicon has no L3)
        if l3_mb > 0:
            idx_fit = "fits L3" if idx_mb <= l3_mb else "spills to RAM"
        elif l2_mb > 0:
            idx_fit = "fits L2" if idx_mb <= l2_mb else "spills to RAM"
        else:
            idx_fit = "?"
        print(f"  N={N:<8}  latency={mean:.4f} +/- {std:.4f} ms/query  "
              f"index={idx_mb:.1f} MB ({idx_fit})")
        results.append({"param": "N", "value": N, "latency_ms": mean, "std_ms": std,
                        "ws_kb": ws_kb, "cache_fit": idx_fit, "index_mb": idx_mb})
    return results


def sweep_dimension(k=20, ef=100, M=16, ef_construction=100, N=50000,
                    num_queries=10000, runs=5, seed=42):
    """How does vector dimension (d) affect latency? Uses synthetic data.

    Larger dim = more bytes per vector = more cache lines per distance comp +
    bigger SIMD work per comparison.
    """
    dim_values = [16, 32, 64, 128, 256, 512]
    rng = np.random.default_rng(seed)
    results = []
    for dim in dim_values:
        data = rng.random((N, dim)).astype(np.float32)
        q = generate_queries(data, num_queries=num_queries, seed=seed)
        index = build_index(data, ef_construction=ef_construction, M=M, ef=ef)
        mean, std = measure_latency(index, q, k=k, runs=runs)
        mem_per_vector_kb = dim * 4 / 1024
        ws_kb = _per_query_ws_kb(ef, dim, M)
        fit = cache_fit_label(ws_kb, CACHES)
        print(f"  dim={dim:<5}  latency={mean:.4f} +/- {std:.4f} ms/query  "
              f"bytes/vector={dim * 4}  ws={ws_kb:.1f} KB ({fit})")
        results.append({"param": "dim", "value": dim, "latency_ms": mean, "std_ms": std,
                        "mem_per_vector_kb": mem_per_vector_kb,
                        "ws_kb": ws_kb, "cache_fit": fit})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results, out_path, defaults):
    """Create a multi-panel plot showing latency vs each parameter."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cache_str = (
        f"L1d={CACHES['L1d']} KB · L2={CACHES['L2']} KB · L3={CACHES['L3']} KB"
        if CACHES['L1d'] > 0 else "cache sizes unavailable"
    )
    fig.suptitle(
        "HNSW Parameter Sensitivity: What Factors Affect Query Latency?\n"
        f"(defaults: N={defaults['N']}, dim={defaults['dim']}, k={defaults['k']}, "
        f"ef={defaults['ef']}, M={defaults['M']})  |  CPU caches: {cache_str}",
        fontsize=12, fontweight='bold',
    )

    color_lat = "#E65100"

    def _draw(ax, data, marker, xlabel, title, fit_key="cache_fit"):
        if not data:
            return
        xs = [r["value"] for r in data]
        ys = [r["latency_ms"] for r in data]
        ax.plot(xs, ys, marker + "-", color=color_lat, linewidth=2.5, markersize=8)
        for r in data:
            extra = r.get(fit_key, "") if fit_key else ""
            label = f"{r['latency_ms']:.3f}" + (f"\n{extra}" if extra and extra != "?" else "")
            ax.annotate(label, (r["value"], r["latency_ms"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7)
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel("Latency (ms/query)", fontweight='bold')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ymax = max(ys) if ys else 1
        ax.set_ylim(0, ymax * 1.4 if ymax > 0 else 1)

    _draw(axes[0, 0], [r for r in all_results if r["param"] == "k"],
          "o", "k (top-k returned results)", "Effect of k on Latency",
          fit_key=None)
    _draw(axes[0, 1], [r for r in all_results if r["param"] == "ef"],
          "s", "ef (search beam width)", "Effect of ef on Latency")
    _draw(axes[0, 2], [r for r in all_results if r["param"] == "M"],
          "D", "M (graph connectivity)", "Effect of M on Latency")
    _draw(axes[1, 0], [r for r in all_results if r["param"] == "N"],
          "o", "Dataset Size (N vectors)", "Effect of Dataset Size on Latency")
    _draw(axes[1, 1], [r for r in all_results if r["param"] == "dim"],
          "s", "Dimension (d)", "Effect of Dimension on Latency\n(synthetic data, N=50K)")

    # Panel 6: Summary computed from actual measured data
    ax = axes[1, 2]
    ax.axis("off")

    def _latency_range(param_name):
        d = [r for r in all_results if r["param"] == param_name]
        if not d:
            return 0, 0, 0
        lats = [r["latency_ms"] for r in d]
        return min(lats), max(lats), (max(lats) - min(lats)) / min(lats) * 100

    lines = ["Measured Impact Summary", "─" * 28, ""]
    for name, label in [("k", "k (top-k returned results)"), ("ef", "ef (search beam)"),
                         ("M", "M (connectivity)"), ("N", "Dataset size"),
                         ("dim", "Dimension")]:
        lo, hi, pct = _latency_range(name)
        if lo > 0:
            lines.append(f"{label}:")
            lines.append(f"  {lo:.3f} → {hi:.3f} ms")
            lines.append(f"  ({pct:.0f}% increase across range)")
            lines.append("")

    summary = "\n".join(lines)
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(all_results, csv_path):
    fields = ["param", "value", "latency_ms", "std_ms",
              "ws_kb", "cache_fit", "index_mb", "mem_per_vector_kb"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_results)
    print(f"CSV saved: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HNSW parameter sensitivity analysis — sweeps k, ef, M, "
                    "dataset size, and dimension to show how each affects latency."
    )
    parser.add_argument("--dataset", default="", help="Path to .npy dataset (auto-detected if omitted)")
    parser.add_argument("--num-queries", type=int, default=10000, help="Queries per measurement (default: 10000)")
    parser.add_argument("--runs", type=int, default=5, help="Runs per measurement for averaging (default: 5)")
    parser.add_argument("--k", type=int, default=20, help="Default k (default: 20)")
    parser.add_argument("--ef", type=int, default=100, help="Default ef (default: 100)")
    parser.add_argument("--hnsw-m", type=int, default=16, help="Default M (default: 16)")
    parser.add_argument("--ef-construction", type=int, default=100, help="ef_construction (default: 100)")
    args = parser.parse_args()

    # Find dataset
    dataset_path = args.dataset or find_dataset()
    if dataset_path is None or not os.path.exists(dataset_path):
        print("ERROR: Could not find real_world_dataset.npy.")
        print("Pass --dataset /path/to/your_dataset.npy")
        return

    data = np.load(dataset_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    num_elements, dim = data.shape
    print(f"Dataset: {dataset_path} ({num_elements} vectors, {dim}-dim)")
    if CACHES["L1d"] > 0:
        print(f"CPU caches: L1d={CACHES['L1d']} KB, L2={CACHES['L2']} KB, L3={CACHES['L3']} KB")
    else:
        print("CPU caches: unable to detect (cache fit/spill labels disabled)")
    queries = generate_queries(data, num_queries=args.num_queries)

    defaults = {"N": num_elements, "dim": dim, "k": args.k, "ef": args.ef, "M": args.hnsw_m}
    all_results = []

    print(f"\n=== Sweep 1/5: k (top-k returned results) ===")
    all_results += sweep_k(data, queries, ef=args.ef, M=args.hnsw_m,
                           ef_construction=args.ef_construction, runs=args.runs)

    print(f"\n=== Sweep 2/5: ef (search beam width) ===")
    all_results += sweep_ef(data, queries, k=args.k, M=args.hnsw_m,
                            ef_construction=args.ef_construction, runs=args.runs)

    print(f"\n=== Sweep 3/5: M (graph connectivity) ===")
    all_results += sweep_M(data, queries, k=args.k, ef=args.ef,
                           ef_construction=args.ef_construction, runs=args.runs)

    print(f"\n=== Sweep 4/5: Dataset size (N) ===")
    all_results += sweep_dataset_size(data, k=args.k, ef=args.ef, M=args.hnsw_m,
                                      ef_construction=args.ef_construction,
                                      num_queries=args.num_queries, runs=args.runs)

    print(f"\n=== Sweep 5/5: Dimension (d) — synthetic data ===")
    all_results += sweep_dimension(k=args.k, ef=args.ef, M=args.hnsw_m,
                                   ef_construction=args.ef_construction,
                                   num_queries=args.num_queries, runs=args.runs)

    # Save outputs
    os.makedirs("results", exist_ok=True)
    save_csv(all_results, "results/parameter_sweep.csv")
    plot_results(all_results, "results/parameter_sweep.png", defaults)


if __name__ == "__main__":
    main()
