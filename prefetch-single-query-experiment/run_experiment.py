import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Docker image build (same prefetch ON/OFF mechanism as prefetch-experiment)
# ─────────────────────────────────────────────────────────────────────────────

def build_image(image_name, disable_prefetch=False, python_version="3.10-slim"):
    print(f"Building Docker image: {image_name}")
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


def image_exists(image_name):
    r = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery and query construction
# ─────────────────────────────────────────────────────────────────────────────

def find_dataset():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "real_world_dataset.npy",
        here.parent / "prefetch-experiment" / "real_world_dataset.npy",
        here.parent / "memory-stress-experiment" / "real_world_dataset.npy",
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return None


def _prompt(msg, default=None):
    suffix = f" [default {default}]" if default is not None else ""
    raw = input(f"{msg}{suffix}: ").strip()
    return raw if raw else (str(default) if default is not None else "")


def prompt_queries(dataset_path, default_num=1, default_noise=0.0, default_seed=42):
    data = np.load(dataset_path, mmap_mode="r")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n, dim = data.shape

    raw = _prompt("How many queries do you want to run?", default_num)
    num = int(raw)
    if num < 1:
        raise ValueError("Number of queries must be >= 1")

    idx_raw = _prompt(
        f"Enter {num} dataset indices (comma-separated, 0..{n-1}), or 'r' for random",
        "r",
    ).lower()

    if idx_raw == "r":
        seed = int(_prompt("Random seed", default_seed))
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=num, replace=True)
    else:
        indices = np.array([int(x.strip()) for x in idx_raw.split(",") if x.strip()])
        if len(indices) != num:
            raise ValueError(f"Expected {num} indices, got {len(indices)}")
        if (indices < 0).any() or (indices >= n).any():
            raise ValueError(f"Indices must be within [0, {n - 1}]")

    noise = float(_prompt("Query noise (gaussian scale, 0 = use exact vector)", default_noise))
    noise_seed = int(_prompt("Noise seed", default_seed))

    queries = data[indices].astype(np.float32).copy()
    if noise > 0:
        rng = np.random.default_rng(noise_seed)
        queries += rng.normal(scale=noise, size=queries.shape).astype(np.float32)

    return queries, indices.tolist(), noise, (n, dim)


# ─────────────────────────────────────────────────────────────────────────────
# Container execution and output parsing
# ─────────────────────────────────────────────────────────────────────────────

def run_worker(image_name, dataset_abs, queries_abs, k, ef, num_threads,
               M, ef_construction, index_size, timeout_s):
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "-e", f"HNSW_K={k}",
        "-e", f"HNSW_EF={ef}",
        "-e", f"HNSW_NUM_THREADS={num_threads}",
        "-e", f"HNSW_M={M}",
        "-e", f"HNSW_EF_CONSTRUCTION={ef_construction}",
        "-e", f"HNSW_INDEX_SIZE={index_size}",
        "-v", f"{dataset_abs}:/app/real_world_dataset.npy:ro",
        "-v", f"{queries_abs}:/app/queries.npy:ro",
        image_name,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or f"Container failed: {image_name}")
    return r.stdout


def parse_worker_output(stdout):
    per_query = []
    overall_latency = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("QUERY,"):
            parts = line.split(",")
            idx = int(parts[1])
            lat_ms = float(parts[2])
            distances = [float(x) for x in parts[3].split(";")] if parts[3] else []
            ids = [int(x) for x in parts[4].split(";")] if parts[4] else []
            per_query.append({
                "idx": idx,
                "latency_ms": lat_ms,
                "distances": distances,
                "ids": ids,
            })
        elif line.startswith("RESULT,"):
            parts = line.split(",")
            overall_latency = float(parts[1])
    return per_query, overall_latency


# ─────────────────────────────────────────────────────────────────────────────
# Comparison output
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(pq_on, pq_off, indices_used, overall_on, overall_off):
    print("\n" + "=" * 72)
    print("Per-Query Comparison: Prefetch ON vs Prefetch OFF")
    print("=" * 72)

    for i, (a, b) in enumerate(zip(pq_on, pq_off)):
        diff_ms = b["latency_ms"] - a["latency_ms"]
        same_order = a["ids"] == b["ids"]
        overlap = (len(set(a["ids"]) & set(b["ids"])) / max(1, len(a["ids"])))
        src_idx = indices_used[i] if i < len(indices_used) else "?"
        print(f"\nQuery #{i} (dataset index {src_idx})")
        print(f"  Latency ON : {a['latency_ms']:.4f} ms")
        print(f"  Latency OFF: {b['latency_ms']:.4f} ms  (Δ = {diff_ms:+.4f} ms)")
        print(f"  Neighbors ON : {a['ids']}")
        print(f"  Neighbors OFF: {b['ids']}")
        print(f"  Same neighbor order? {'YES' if same_order else 'NO'}   "
              f"Set overlap: {overlap:.2%}")

    if overall_on is not None and overall_off is not None:
        pct = (overall_off - overall_on) / overall_on * 100 if overall_on > 0 else 0.0
        print("\n" + "-" * 72)
        print(f"Overall mean latency ON : {overall_on:.4f} ms/query")
        print(f"Overall mean latency OFF: {overall_off:.4f} ms/query")
        direction = "slower" if pct > 0 else "faster"
        print(f"OFF vs ON: {pct:+.2f}%  ({direction} without prefetch)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive single-query HNSW Prefetch ON vs OFF comparison."
    )
    parser.add_argument("--dataset", default="", help="Path to .npy dataset (auto-detected if omitted)")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors returned per query")
    parser.add_argument("--ef", type=int, default=200, help="HNSW search width")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--hnsw-m", type=int, default=32, help="HNSW M (graph connectivity)")
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--index-size", type=int, default=0, help="Subset dataset to first N vectors (0 = full)")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--python-version", default="3.10-slim")
    parser.add_argument("--skip-build", action="store_true",
                        help="Reuse existing images if they already exist")
    args = parser.parse_args()

    dataset_path = args.dataset or find_dataset()
    if not dataset_path or not Path(dataset_path).exists():
        print("ERROR: Could not find dataset. Pass --dataset /path/to/file.npy")
        sys.exit(1)
    dataset_abs = str(Path(dataset_path).resolve())

    image_on  = "hnsw-prefetch-single-on"
    image_off = "hnsw-prefetch-single-off"
    need_build = not (args.skip_build and image_exists(image_on) and image_exists(image_off))
    if need_build:
        build_image(image_on,  disable_prefetch=False, python_version=args.python_version)
        build_image(image_off, disable_prefetch=True,  python_version=args.python_version)
    else:
        print("Reusing existing Docker images (--skip-build).")

    print(f"\nDataset: {dataset_abs}")

    while True:
        queries, indices_used, noise, (n, dim) = prompt_queries(dataset_abs)
        print(f"Using {len(queries)} query/queries "
              f"(dim={dim}, dataset size={n}, noise={noise})")

        tmp = tempfile.NamedTemporaryFile(prefix="queries_", suffix=".npy", delete=False)
        tmp.close()
        np.save(tmp.name, queries)
        queries_abs = str(Path(tmp.name).resolve())

        try:
            print("\nRunning Prefetch ON ...")
            out_on = run_worker(
                image_on, dataset_abs, queries_abs,
                args.k, args.ef, args.num_threads, args.hnsw_m,
                args.ef_construction, args.index_size, args.timeout,
            )
            print("Running Prefetch OFF ...")
            out_off = run_worker(
                image_off, dataset_abs, queries_abs,
                args.k, args.ef, args.num_threads, args.hnsw_m,
                args.ef_construction, args.index_size, args.timeout,
            )
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except Exception:
                pass

        pq_on,  overall_on  = parse_worker_output(out_on)
        pq_off, overall_off = parse_worker_output(out_off)

        if len(pq_on) != len(pq_off):
            print(f"WARN: number of query results differs ON={len(pq_on)} OFF={len(pq_off)}")

        print_comparison(pq_on, pq_off, indices_used, overall_on, overall_off)

        again = input("\nRun another batch of queries? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
