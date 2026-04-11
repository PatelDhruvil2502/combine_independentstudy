import os
import math
import time
import argparse
import csv
import numpy as np
import hnswlib
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def measure_latency(index, queries, k, num_threads=1, runs=3):
    """Measure mean query latency (ms/query) averaged over multiple runs."""
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        index.knn_query(queries, k=k, num_threads=num_threads)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed / len(queries) * 1000)
    return np.mean(latencies)


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


def expected_max_level(N, M):
    """Expected max HNSW level: ln(N) / ln(M)."""
    if N <= 0 or M <= 1:
        return 0
    return math.log(N) / math.log(M)


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

def sweep_k(data, queries, ef=100, M=16, ef_construction=100, runs=3):
    """How does the number of requested neighbors (k) affect latency?"""
    k_values = [1, 5, 10, 20, 50, 100]
    index = build_index(data, ef_construction=ef_construction, M=M, ef=ef)
    results = []
    for k in k_values:
        lat = measure_latency(index, queries, k=k, runs=runs)
        print(f"  k={k:<5}  latency={lat:.4f} ms/query")
        results.append({"param": "k", "value": k, "latency_ms": lat})
    return results


def sweep_ef(data, queries, k=20, M=16, ef_construction=100, runs=3):
    """How does the search beam width (ef) affect latency?"""
    ef_values = [10, 50, 100, 200, 400, 800]
    index = build_index(data, ef_construction=ef_construction, M=M, ef=10)
    results = []
    for ef in ef_values:
        index.set_ef(ef)
        lat = measure_latency(index, queries, k=k, runs=runs)
        print(f"  ef={ef:<5}  latency={lat:.4f} ms/query")
        results.append({"param": "ef", "value": ef, "latency_ms": lat})
    return results


def sweep_M(data, queries, k=20, ef=100, ef_construction=100, runs=3):
    """How does graph connectivity (M) affect latency and HNSW levels?"""
    M_values = [4, 8, 16, 32, 64]
    results = []
    for M in M_values:
        index = build_index(data, ef_construction=ef_construction, M=M, ef=ef)
        lat = measure_latency(index, queries, k=k, runs=runs)
        levels = expected_max_level(len(data), M)
        print(f"  M={M:<5}  latency={lat:.4f} ms/query  expected_levels={levels:.1f}")
        results.append({"param": "M", "value": M, "latency_ms": lat, "levels": levels})
    return results


def sweep_dataset_size(data, k=20, ef=100, M=16, ef_construction=100,
                       num_queries=5000, runs=3):
    """How does dataset size (N) affect latency and HNSW levels?"""
    max_n = len(data)
    N_values = [s for s in [5000, 10000, 25000, 50000, 100000, 150000] if s <= max_n]
    results = []
    for N in N_values:
        subset = data[:N]
        q = generate_queries(subset, num_queries=num_queries)
        index = build_index(subset, ef_construction=ef_construction, M=M, ef=ef)
        lat = measure_latency(index, q, k=k, runs=runs)
        levels = expected_max_level(N, M)
        print(f"  N={N:<8}  latency={lat:.4f} ms/query  expected_levels={levels:.1f}")
        results.append({"param": "N", "value": N, "latency_ms": lat, "levels": levels})
    return results


def sweep_dimension(k=20, ef=100, M=16, ef_construction=100, N=50000,
                    num_queries=5000, runs=3, seed=42):
    """How does vector dimension (d) affect latency? Uses synthetic data."""
    dim_values = [16, 32, 64, 128, 256, 512]
    rng = np.random.default_rng(seed)
    results = []
    for dim in dim_values:
        data = rng.random((N, dim)).astype(np.float32)
        q = generate_queries(data, num_queries=num_queries, seed=seed)
        index = build_index(data, ef_construction=ef_construction, M=M, ef=ef)
        lat = measure_latency(index, q, k=k, runs=runs)
        mem_per_vector_kb = dim * 4 / 1024
        print(f"  dim={dim:<5}  latency={lat:.4f} ms/query  bytes/vector={dim * 4}")
        results.append({"param": "dim", "value": dim, "latency_ms": lat,
                         "mem_per_vector_kb": mem_per_vector_kb})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results, out_path, defaults):
    """Create a multi-panel plot showing latency vs each parameter."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "HNSW Parameter Sensitivity: What Factors Affect Query Latency?\n"
        f"(defaults: N={defaults['N']}, dim={defaults['dim']}, k={defaults['k']}, "
        f"ef={defaults['ef']}, M={defaults['M']})",
        fontsize=13, fontweight='bold',
    )

    color_lat = "#E65100"
    color_sec = "#1565C0"

    # Panel 1: k sweep
    ax = axes[0, 0]
    k_data = [r for r in all_results if r["param"] == "k"]
    xs = [r["value"] for r in k_data]
    ys = [r["latency_ms"] for r in k_data]
    ax.plot(xs, ys, "o-", color=color_lat, linewidth=2.5, markersize=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xlabel("k (number of neighbors)", fontweight='bold')
    ax.set_ylabel("Latency (ms/query)", fontweight='bold')
    ax.set_title("Effect of k on Latency")
    ax.grid(True, alpha=0.3)

    # Panel 2: ef sweep
    ax = axes[0, 1]
    ef_data = [r for r in all_results if r["param"] == "ef"]
    xs = [r["value"] for r in ef_data]
    ys = [r["latency_ms"] for r in ef_data]
    ax.plot(xs, ys, "s-", color=color_lat, linewidth=2.5, markersize=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xlabel("ef (search beam width)", fontweight='bold')
    ax.set_ylabel("Latency (ms/query)", fontweight='bold')
    ax.set_title("Effect of ef on Latency")
    ax.grid(True, alpha=0.3)

    # Panel 3: M sweep (with expected levels on secondary axis)
    ax = axes[0, 2]
    m_data = [r for r in all_results if r["param"] == "M"]
    xs = [r["value"] for r in m_data]
    ys = [r["latency_ms"] for r in m_data]
    lvls = [r.get("levels", 0) for r in m_data]
    ax.plot(xs, ys, "D-", color=color_lat, linewidth=2.5, markersize=8, label="Latency")
    ax.set_xlabel("M (graph connectivity)", fontweight='bold')
    ax.set_ylabel("Latency (ms/query)", color=color_lat, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=color_lat)
    ax.grid(True, alpha=0.3)
    if any(l > 0 for l in lvls):
        ax2 = ax.twinx()
        ax2.plot(xs, lvls, "^--", color=color_sec, linewidth=1.5, markersize=7, label="Expected levels")
        ax2.set_ylabel("Expected HNSW Levels", color=color_sec, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_sec)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.set_title("Effect of M on Latency & Levels")

    # Panel 4: Dataset size sweep (with expected levels)
    ax = axes[1, 0]
    n_data = [r for r in all_results if r["param"] == "N"]
    xs = [r["value"] for r in n_data]
    ys = [r["latency_ms"] for r in n_data]
    lvls = [r.get("levels", 0) for r in n_data]
    ax.plot(xs, ys, "o-", color=color_lat, linewidth=2.5, markersize=8, label="Latency")
    ax.set_xlabel("Dataset Size (N vectors)", fontweight='bold')
    ax.set_ylabel("Latency (ms/query)", color=color_lat, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=color_lat)
    ax.grid(True, alpha=0.3)
    if any(l > 0 for l in lvls):
        ax2 = ax.twinx()
        ax2.plot(xs, lvls, "^--", color=color_sec, linewidth=1.5, markersize=7, label="Expected levels")
        ax2.set_ylabel("Expected HNSW Levels", color=color_sec, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_sec)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax.set_title("Effect of Dataset Size on Latency & Levels")

    # Panel 5: Dimension sweep
    ax = axes[1, 1]
    dim_data = [r for r in all_results if r["param"] == "dim"]
    xs = [r["value"] for r in dim_data]
    ys = [r["latency_ms"] for r in dim_data]
    ax.plot(xs, ys, "s-", color=color_lat, linewidth=2.5, markersize=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xlabel("Dimension (d)", fontweight='bold')
    ax.set_ylabel("Latency (ms/query)", fontweight='bold')
    ax.set_title("Effect of Dimension on Latency\n(synthetic data, N=50K)")
    ax.grid(True, alpha=0.3)

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
    for name, label in [("k", "k (neighbors)"), ("ef", "ef (search beam)"),
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
    fields = ["param", "value", "latency_ms", "levels", "mem_per_vector_kb"]
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
    parser.add_argument("--num-queries", type=int, default=5000, help="Queries per measurement (default: 5000)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per measurement for averaging (default: 3)")
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
    queries = generate_queries(data, num_queries=args.num_queries)

    defaults = {"N": num_elements, "dim": dim, "k": args.k, "ef": args.ef, "M": args.hnsw_m}
    all_results = []

    print(f"\n=== Sweep 1/5: k (number of neighbors) ===")
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
