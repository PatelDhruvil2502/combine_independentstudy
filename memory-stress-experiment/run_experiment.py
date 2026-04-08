import os
import subprocess
import matplotlib.pyplot as plt
import argparse
import csv


def run_one(ram_mb, swap_mb, image_name, env_vars=None):
    import uuid
    container_name = f"stress_test_{uuid.uuid4().hex[:8]}"
    cmd = [
        "docker",
        "run",
        "--platform", "linux/amd64",
        "--name",
        container_name,
        "--rm",
        f"--memory={ram_mb}m",
        f"--memory-swap={swap_mb}m",
    ]
    for k, v in (env_vars or {}).items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.append(image_name)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired as e:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        class TimeoutResult:
            returncode = 124
            stdout = e.stdout if e.stdout else ""
            stderr = e.stderr if e.stderr else "Command timed out after 1200s"
        return TimeoutResult()


def parse_output(output_lines):
    """Parse all structured output lines from the worker."""
    data = {}
    for line in output_lines:
        if line.startswith("RESULT,"):
            parts = line.split(",")
            data["latency_ms"] = float(parts[1])
            data["vm_rss_kb"] = int(parts[2]) if len(parts) > 2 else 0
            data["vm_swap_kb"] = int(parts[3]) if len(parts) > 3 else 0
        elif line.startswith("LATENCY_STATS,"):
            parts = line.split(",")
            data["p50"] = float(parts[1])
            data["p95"] = float(parts[2])
            data["p99"] = float(parts[3])
        elif line.startswith("PAGE_FAULTS,"):
            parts = line.split(",")
            data["majflt"] = int(parts[1])
            data["minflt"] = int(parts[2]) if len(parts) > 2 else 0
        elif line.startswith("IO_STATS,"):
            parts = line.split(",")
            data["io_read_mb"] = float(parts[1])
            data["io_write_mb"] = float(parts[2])
        elif line.startswith("MEMORY_LAYOUT,"):
            parts = line.split(",")
            data["q_start"] = parts[1]
        elif line.startswith("DATASET_INFO,"):
            parts = line.split(",")
            data["num_elements"] = int(parts[1])
            data["dim"] = int(parts[2])
    return data


def build_image(image_name):
    print(f"Building Docker image ({image_name})...")
    dockerfile_content = """FROM python:3.10-slim
RUN apt-get update && apt-get install -y build-essential python3-dev
RUN pip install hnswlib numpy
COPY worker.py /app/worker.py
COPY real_world_dataset.npy /app/real_world_dataset.npy
WORKDIR /app
CMD ["python", "worker.py"]
"""
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    build_cmd = ["docker", "build", "--platform", "linux/amd64", "-t", image_name, "."]
    subprocess.run(build_cmd, check=True)


def run_experiment(image_name, args):
    env_vars = {
        "HNSW_K": str(args.k),
        "HNSW_EF": str(args.ef),
        "HNSW_M": str(args.hnsw_m),
        "HNSW_EF_CONSTRUCTION": str(args.ef_construction),
        "HNSW_NUM_QUERIES": str(args.num_queries),
        "HNSW_BATCH_SIZE": str(args.batch_size),
        "HNSW_WARMUP_BATCHES": str(args.warmup_batches),
    }

    all_stage_data = []
    crash_point = None

    # ── Probe: run once with generous memory to discover real RSS ────────────
    print("\nProbe run (discover real memory use from container /proc)...")
    probe_ram, probe_swap = 4096, 8192
    result = run_one(probe_ram, probe_swap, image_name, env_vars)
    output = result.stdout.strip().splitlines()
    data = parse_output(output)

    if "latency_ms" not in data or result.returncode != 0:
        print("Probe failed; cannot derive limits from real data.")
        if output:
            print(result.stderr or "\n".join(output[-20:]))
        return None

    rss_mb = max(50, data["vm_rss_kb"] // 1024)
    print(f"  Probe RSS: {rss_mb} MB")
    print(f"  Probe latency: {data['latency_ms']:.4f} ms/query")
    print(f"  Probe majflt: {data.get('majflt', 0)} | minflt: {data.get('minflt', 0)}")

    # ── Design stages to guarantee: RAM-only -> Swapping -> OOM Crash ────────
    #
    # The key insight: we need the HNSW index pages to get evicted from RAM.
    # With 50K queries (~24MB) and 150K vectors (~73MB data + ~170MB index),
    # the working set is ~270-350MB. Stages are percentages of measured RSS.
    #
    # Each non-crash stage gets swap = RAM + RSS (enough total to survive).
    # The crash stage gets swap = 0 (total = RAM only) to guarantee OOM.
    stages = [
        {"pct": 1.10, "label": "Comfortable (110% RSS)", "crash": False},
        {"pct": 0.75, "label": "Mild pressure (75% RSS)", "crash": False},
        {"pct": 0.50, "label": "Moderate swap (50% RSS)", "crash": False},
        {"pct": 0.30, "label": "Heavy swap (30% RSS)", "crash": False},
        {"pct": 0.15, "label": "Severe thrashing (15% RSS)", "crash": False},
        {"pct": 0.05, "label": "Crash (5% RSS, no swap)", "crash": True},
    ]

    header = (
        f"{'Stage':<30} | {'RAM':<7} | {'Total':<7} | {'Status':<20} | "
        f"{'Latency':<10} | {'p50':<8} | {'p95':<8} | {'p99':<8} | "
        f"{'MajFlt':<8} | {'IO Read':<8}"
    )
    print(f"\n--- Memory Stress Test ({len(stages)} stages) ---")
    print(header)
    print("-" * len(header))

    plot_limits, plot_latencies, plot_majflt, plot_io_read = [], [], [], []

    for stage in stages:
        current_ram = max(16, int(rss_mb * stage["pct"]))
        if stage["crash"]:
            current_total = current_ram  # no swap → guaranteed OOM
        else:
            current_total = current_ram + rss_mb  # enough swap to survive

        result = run_one(current_ram, current_total, image_name, env_vars)
        output = result.stdout.strip().splitlines()
        data = parse_output(output)

        if result.returncode != 0 or "latency_ms" not in data:
            status = "OOM Killed" if result.returncode == 137 else "Failed/Timeout"
            print(
                f"{stage['label']:<30} | {current_ram:<7} | {current_total:<7} | "
                f"{status:<20} | {'-':<10} | {'-':<8} | {'-':<8} | {'-':<8} | "
                f"{'-':<8} | {'-':<8}"
            )
            crash_point = current_ram
            all_stage_data.append({
                "stage": stage["label"], "ram_mb": current_ram,
                "total_mb": current_total, "status": status,
                "latency_ms": None, "p50": None, "p95": None, "p99": None,
                "majflt": None, "minflt": None,
                "io_read_mb": None, "io_write_mb": None,
                "vm_rss_kb": None, "vm_swap_kb": None,
            })
            break

        latency = data["latency_ms"]
        swap_kb = data.get("vm_swap_kb", 0)
        p50 = data.get("p50", 0)
        p95 = data.get("p95", 0)
        p99 = data.get("p99", 0)
        majflt = data.get("majflt", 0)
        minflt = data.get("minflt", 0)
        io_read = data.get("io_read_mb", 0)
        io_write = data.get("io_write_mb", 0)
        status = "RAM only" if swap_kb == 0 else f"Swapping ({swap_kb // 1024} MB)"

        print(
            f"{stage['label']:<30} | {current_ram:<7} | {current_total:<7} | "
            f"{status:<20} | {latency:<10.4f} | {p50:<8.4f} | {p95:<8.4f} | {p99:<8.4f} | "
            f"{majflt:<8} | {io_read:<8.1f}"
        )

        plot_limits.append(current_ram)
        plot_latencies.append(latency)
        plot_majflt.append(majflt)
        plot_io_read.append(io_read)
        all_stage_data.append({
            "stage": stage["label"], "ram_mb": current_ram,
            "total_mb": current_total, "status": status,
            "latency_ms": latency, "p50": p50, "p95": p95, "p99": p99,
            "majflt": majflt, "minflt": minflt,
            "io_read_mb": io_read, "io_write_mb": io_write,
            "vm_rss_kb": data.get("vm_rss_kb", 0), "vm_swap_kb": swap_kb,
        })

    # ── Save outputs to results/ folder ────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = "results/results.csv"
    if all_stage_data:
        keys = list(all_stage_data[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_stage_data)
        print(f"\nCSV saved: {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if len(plot_limits) < 2:
        print("Not enough data points for a meaningful plot.")
        return

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Latency (mean + p95) vs RAM with page faults on secondary axis
    ax1.invert_xaxis()
    ax1.set_xlabel("Physical RAM Allowed (MB) ->", fontweight='bold')
    ax1.set_ylabel("Query Latency (ms)", color="tab:orange", fontweight='bold')

    # Plot mean + p95 latency
    p95_vals = [s["p95"] for s in all_stage_data if s["latency_ms"] is not None]
    ax1.plot(plot_limits, plot_latencies, "s-", color="tab:orange", linewidth=2.5,
             markersize=8, label="Mean latency", zorder=3)
    if p95_vals:
        ax1.plot(plot_limits, p95_vals, "^--", color="tab:red", linewidth=1.5,
                 markersize=7, alpha=0.8, label="p95 latency", zorder=3)
    ax1.fill_between(plot_limits, plot_latencies, color="tab:orange", alpha=0.1)
    lat_max = max(plot_latencies + p95_vals) if p95_vals else max(plot_latencies)
    ax1.set_ylim(0, lat_max * 1.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # Major page faults on secondary axis
    if any(f > 0 for f in plot_majflt):
        ax2 = ax1.twinx()
        ax2.set_ylabel("Major Page Faults (during search)", color="tab:blue", fontweight='bold')
        bar_width = max(1, (max(plot_limits) - min(plot_limits)) / (len(plot_limits) * 2))
        ax2.bar(plot_limits, plot_majflt, width=bar_width,
                alpha=0.25, color="tab:blue", label="Major page faults", zorder=1)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

    # OOM crash marker
    if crash_point is not None:
        ax1.axvline(x=crash_point, color="red", linestyle="--", linewidth=3)
        if plot_limits and plot_latencies:
            ax1.plot([plot_limits[-1], crash_point],
                     [plot_latencies[-1], plot_latencies[-1]], "r:", linewidth=2)
        ax1.text(
            crash_point, lat_max * 0.6,
            " OOM CRASH\n (System Killed)",
            color="white",
            bbox=dict(facecolor='red', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.5'),
            fontweight="bold", va="center", fontsize=10,
        )

    ax1.set_title("Latency & Page Faults vs Available RAM", fontweight='bold')

    # Panel 2: Disk I/O during search vs RAM
    ax3.invert_xaxis()
    ax3.set_xlabel("Physical RAM Allowed (MB) ->", fontweight='bold')
    ax3.set_ylabel("Disk I/O During Search (MB)", fontweight='bold')
    ax3.bar(plot_limits, plot_io_read, width=bar_width if any(f > 0 for f in plot_majflt) else 5,
            alpha=0.7, color="tab:green", label="Bytes read from disk")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    ax3.set_title("Disk Read I/O vs Available RAM", fontweight='bold')

    fig.suptitle(
        f"HNSW Memory Contention: How Latency Degrades Under RAM Pressure\n"
        f"(ef={args.ef}, M={args.hnsw_m}, k={args.k}, queries={args.num_queries})",
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    MAX_IMAGES = 5
    results_dir = "results"
    from pathlib import Path
    existing = sorted(Path(results_dir).glob("memory_stress_plot_*.png"))
    if len(existing) < MAX_IMAGES:
        n = len(existing) + 1
    else:
        oldest = min(existing, key=lambda p: p.stat().st_mtime)
        n = int(oldest.stem.split("_")[-1])
    plot_path = f"{results_dir}/memory_stress_plot_{n}.png"
    fig.savefig(plot_path, dpi=300)
    print(f"\nPlot saved: {plot_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="HNSW memory stress experiment — progressively reduces RAM to observe "
                    "latency spikes, page faults, and OOM crashes under memory contention."
    )
    parser.add_argument("--ef", type=int, default=100, help="HNSW ef search parameter (default: 100)")
    parser.add_argument("--k", type=int, default=20, help="Number of nearest neighbors (default: 20)")
    parser.add_argument("--hnsw-m", type=int, default=16, help="HNSW M parameter (default: 16)")
    parser.add_argument("--ef-construction", type=int, default=64, help="HNSW ef_construction (default: 64)")
    parser.add_argument("--num-queries", type=int, default=50000, help="Total queries to run (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Query batch size (default: 2000)")
    parser.add_argument("--warmup-batches", type=int, default=3, help="Warmup batches before timed search (default: 3)")
    args = parser.parse_args()

    image_name = "hnsw-memory-stress"
    build_image(image_name)
    run_experiment(image_name, args)


if __name__ == "__main__":
    main()
