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
        elif line.startswith("RECALL,"):
            data["recall"] = float(line.split(",")[1])
        elif line.startswith("LATENCY_STATS,"):
            parts = line.split(",")
            data["p50"] = float(parts[1])
            data["p95"] = float(parts[2])
            data["p99"] = float(parts[3])
        elif line.startswith("PAGE_FAULTS,"):
            data["page_faults"] = int(line.split(",")[1])
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
        "HNSW_RECALL_SAMPLES": str(args.recall_samples),
    }

    plot_limits, plot_latencies, plot_page_faults = [], [], []
    all_stage_data = []
    crash_point = None

    # Probe: one run with high limit to get real memory use
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
    probe_recall = data.get("recall", 0)
    probe_faults = data.get("page_faults", 0)
    print(f"  Probe RSS: {rss_mb} MB | Recall@{args.k}: {probe_recall:.4f} | Page faults: {probe_faults}")

    # 5 stages from comfortable RAM to guaranteed OOM
    stages = [
        int(rss_mb * 1.05),  # 1. RAM Only
        int(rss_mb * 0.60),  # 2. Sequential Query Swapping
        int(rss_mb * 0.30),  # 3. Outer Graph Boundary Swapping
        int(rss_mb * 0.15),  # 4. Core Graph Thrashing
        int(rss_mb * 0.05),  # 5. Guaranteed Organic Crash
    ]

    header = (
        f"{'RAM (MB)':<10} | {'Total (MB)':<11} | {'Status':<20} | "
        f"{'Latency (ms)':<14} | {'p50':<8} | {'p95':<8} | {'p99':<8} | "
        f"{'Recall':<8} | {'Page Faults':<12}"
    )
    print(f"\n--- Running 5-stage memory stress test ---")
    print(header)
    print("-" * len(header))

    for current_ram in stages:
        if current_ram == stages[-1]:
            current_total = current_ram
        else:
            current_total = current_ram + rss_mb

        result = run_one(current_ram, current_total, image_name, env_vars)
        output = result.stdout.strip().splitlines()
        data = parse_output(output)

        if result.returncode != 0 or "latency_ms" not in data:
            status = "Crashed (OOM)" if result.returncode == 137 else "Failed/Timeout"
            print(f"{current_ram:<10} | {current_total:<11} | {status:<20} | {'-':<14} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<12}")
            crash_point = current_ram
            all_stage_data.append({
                "ram_mb": current_ram, "total_mb": current_total,
                "status": status, "latency_ms": None,
            })
            break

        latency = data["latency_ms"]
        swap_kb = data.get("vm_swap_kb", 0)
        recall = data.get("recall", 0)
        p50 = data.get("p50", 0)
        p95 = data.get("p95", 0)
        p99 = data.get("p99", 0)
        faults = data.get("page_faults", 0)
        status = "RAM only" if swap_kb == 0 else f"Swapping ({swap_kb/1024:.1f} MB)"

        print(
            f"{current_ram:<10} | {current_total:<11} | {status:<20} | "
            f"{latency:<14.4f} | {p50:<8.4f} | {p95:<8.4f} | {p99:<8.4f} | "
            f"{recall:<8.4f} | {faults:<12}"
        )

        plot_limits.append(current_ram)
        plot_latencies.append(latency)
        plot_page_faults.append(faults)
        all_stage_data.append({
            "ram_mb": current_ram, "total_mb": current_total,
            "status": status, "latency_ms": latency,
            "p50": p50, "p95": p95, "p99": p99,
            "recall": recall, "page_faults": faults,
            "vm_rss_kb": data.get("vm_rss_kb", 0),
            "vm_swap_kb": swap_kb,
        })

    # --- Save CSV ---
    csv_path = "results.csv"
    if all_stage_data:
        keys = all_stage_data[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_stage_data)
        print(f"\nCSV saved: {csv_path}")

    # --- Plot: latency + page faults vs RAM ---
    if not plot_limits:
        print("No data points to plot.")
        return None

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.invert_xaxis()
    ax1.set_xlabel("Physical RAM Allowed (MB) — Decreasing ->", fontweight='bold')
    ax1.set_ylabel("Avg Query Latency (ms)", color="tab:orange", fontweight='bold')
    ax1.plot(plot_limits, plot_latencies, "s-", color="tab:orange", linewidth=2.5, markersize=8, label="Latency")
    lat_max = max(plot_latencies)
    ax1.set_ylim(0, lat_max * 1.2)
    ax1.fill_between(plot_limits, plot_latencies, color="tab:orange", alpha=0.1)
    ax1.grid(True, alpha=0.3)

    # Page faults on secondary axis
    if any(f > 0 for f in plot_page_faults):
        ax2 = ax1.twinx()
        ax2.set_ylabel("Major Page Faults", color="tab:blue", fontweight='bold')
        ax2.bar(plot_limits, plot_page_faults, width=max(1, (max(plot_limits) - min(plot_limits)) / 15),
                alpha=0.3, color="tab:blue", label="Page Faults")
        ax2.tick_params(axis='y', labelcolor='tab:blue')

    if crash_point is not None:
        ax1.axvline(x=crash_point, color="red", linestyle="--", linewidth=3)
        if plot_limits and plot_latencies:
            ax1.plot([plot_limits[-1], crash_point], [plot_latencies[-1], plot_latencies[-1]], "r:", linewidth=2)
        ax1.text(
            crash_point, lat_max * 0.5,
            " OOM CRASH\n (System Killed)",
            color="white",
            bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'),
            fontweight="bold", va="center"
        )

    fig.suptitle(
        f"HNSW Memory Stress: Latency & Page Faults vs RAM\n"
        f"(ef={args.ef}, M={args.hnsw_m}, k={args.k})",
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig("memory_stress_plot.png", dpi=300)
    print(f"\nPlot saved: memory_stress_plot.png")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Run HNSW memory stress experiment — measures latency, tail percentiles, "
                    "recall, and page faults as RAM is progressively reduced."
    )
    parser.add_argument("--ef", type=int, default=100, help="HNSW ef search parameter (default: 100)")
    parser.add_argument("--k", type=int, default=20, help="Number of nearest neighbors (default: 20)")
    parser.add_argument("--hnsw-m", type=int, default=16, help="HNSW M parameter (default: 16)")
    parser.add_argument("--ef-construction", type=int, default=64, help="HNSW ef_construction (default: 64)")
    parser.add_argument("--num-queries", type=int, default=1000000, help="Total queries to run (default: 1000000)")
    parser.add_argument("--batch-size", type=int, default=5000, help="Query batch size (default: 5000)")
    parser.add_argument("--recall-samples", type=int, default=200, help="Queries for recall measurement (default: 200)")
    args = parser.parse_args()

    image_name = "hnsw-memory-stress"
    build_image(image_name)
    run_experiment(image_name, args)

if __name__ == "__main__":
    main()
