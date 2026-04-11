import os
import time
import subprocess
import numpy as np
import hnswlib


def get_cache_sizes():
    """Detect CPU cache sizes (L1d, L2, L3) in KB from sysfs or getconf."""
    caches = {"L1d": 0, "L2": 0, "L3": 0}
    # Try sysfs (works in Linux containers)
    try:
        for idx in range(4):
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
    # Fallback: getconf
    if caches["L1d"] == 0:
        for key, param in [("L1d", "LEVEL1_DCACHE_SIZE"), ("L2", "LEVEL2_CACHE_SIZE"), ("L3", "LEVEL3_CACHE_SIZE")]:
            try:
                r = subprocess.run(["getconf", param], capture_output=True, text=True)
                if r.returncode == 0 and r.stdout.strip().isdigit():
                    caches[key] = int(r.stdout.strip()) // 1024
            except Exception:
                pass
    return caches


def main():
    # --- Configurable parameters via environment variables ---
    k = int(os.getenv("HNSW_K", "20"))
    ef = int(os.getenv("HNSW_EF", "100"))
    M = int(os.getenv("HNSW_M", "16"))
    ef_construction = int(os.getenv("HNSW_EF_CONSTRUCTION", "64"))
    num_queries_total = int(os.getenv("HNSW_NUM_QUERIES", "50000"))
    num_threads = int(os.getenv("HNSW_NUM_THREADS", "1"))
    eval_batch_size = int(os.getenv("HNSW_BATCH_SIZE", "2000"))
    seed = int(os.getenv("HNSW_SEED", "42"))
    warmup_batches = int(os.getenv("HNSW_WARMUP_BATCHES", "3"))

    print("Loading real_world_dataset.npy...")
    train_data = np.load("/app/real_world_dataset.npy")
    if train_data.ndim == 1:
        train_data = train_data.reshape(1, -1)
    num_elements, dim = train_data.shape
    print(f"DATASET_INFO,{num_elements},{dim}")

    rng = np.random.default_rng(seed)
    print("Generating queries...")
    indices = rng.choice(num_elements, size=num_queries_total, replace=True)
    queries = np.empty((num_queries_total, dim), dtype=np.float32)
    chunk_size = 50000
    for i in range(0, num_queries_total, chunk_size):
        end = min(i + chunk_size, num_queries_total)
        idx = indices[i:end]
        queries[i:end] = train_data[idx] + rng.normal(scale=0.01, size=(len(idx), dim)).astype(np.float32)

    print("Building HNSW graph...")
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    index.add_items(train_data, np.arange(num_elements))
    index.set_ef(ef)

    # --- CPU cache analysis ---
    caches = get_cache_sizes()
    # Per-query working set: each query visits ~ef candidate nodes.
    # For each node: read the vector (dim * 4 bytes) + read link list (2*M * 4 bytes at level 0)
    bytes_per_vector = dim * 4
    bytes_per_link_list = 2 * M * 4  # level 0 has 2*M connections
    bytes_per_node = bytes_per_vector + bytes_per_link_list
    per_query_ws_kb = (ef * bytes_per_node) / 1024
    # Total index footprint estimate
    index_data_mb = (num_elements * bytes_per_vector) / (1024 * 1024)
    index_graph_mb = (num_elements * bytes_per_link_list) / (1024 * 1024)
    total_index_mb = index_data_mb + index_graph_mb
    print(f"CACHE_INFO,{caches['L1d']},{caches['L2']},{caches['L3']}")
    print(f"WORKING_SET,{per_query_ws_kb:.1f},{total_index_mb:.1f},{index_data_mb:.1f},{index_graph_mb:.1f}")

    # --- Warmup: run a few batches to stabilize caches, not timed ---
    if warmup_batches > 0:
        print(f"Running {warmup_batches} warmup batches...")
        warmup_order = rng.permutation(num_queries_total)
        for i in range(warmup_batches):
            start = (i * eval_batch_size) % num_queries_total
            end = min(start + eval_batch_size, num_queries_total)
            batch_idx = warmup_order[start:end]
            index.knn_query(queries[batch_idx], k=k, num_threads=num_threads)

    # --- Snapshot page faults + I/O BEFORE timed search ---
    majflt_before = 0
    minflt_before = 0
    io_read_before = 0
    io_write_before = 0
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().split()
            minflt_before = int(fields[9])   # field 10 (0-indexed 9) = minflt
            majflt_before = int(fields[11])  # field 12 (0-indexed 11) = majflt
    except Exception:
        pass
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    io_read_before = int(line.split()[1])
                elif line.startswith("write_bytes:"):
                    io_write_before = int(line.split()[1])
    except Exception:
        pass

    # --- Timed search with per-batch latency tracking ---
    print("Starting timed search...")
    random_order = rng.permutation(num_queries_total)
    batch_latencies_ms = []

    t0 = time.perf_counter()
    for i in range(0, num_queries_total, eval_batch_size):
        batch_idx = random_order[i : i + eval_batch_size]
        batch_queries = queries[batch_idx]
        bt0 = time.perf_counter()
        index.knn_query(batch_queries, k=k, num_threads=num_threads)
        bt_ms = (time.perf_counter() - bt0) / len(batch_idx) * 1000
        batch_latencies_ms.append(bt_ms)

    elapsed_s = time.perf_counter() - t0
    latency_ms = (elapsed_s / num_queries_total) * 1000

    # --- Tail latency percentiles ---
    arr = np.array(batch_latencies_ms)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    print(f"LATENCY_STATS,{p50:.6f},{p95:.6f},{p99:.6f}")

    # --- Snapshot page faults + I/O AFTER timed search ---
    majflt_after = 0
    minflt_after = 0
    io_read_after = 0
    io_write_after = 0
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().split()
            minflt_after = int(fields[9])
            majflt_after = int(fields[11])
    except Exception:
        pass
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    io_read_after = int(line.split()[1])
                elif line.startswith("write_bytes:"):
                    io_write_after = int(line.split()[1])
    except Exception:
        pass

    majflt_delta = majflt_after - majflt_before
    minflt_delta = minflt_after - minflt_before
    io_read_mb = (io_read_after - io_read_before) / (1024 * 1024)
    io_write_mb = (io_write_after - io_write_before) / (1024 * 1024)
    print(f"PAGE_FAULTS,{majflt_delta},{minflt_delta}")
    print(f"IO_STATS,{io_read_mb:.2f},{io_write_mb:.2f}")

    # --- Memory stats ---
    vm_rss_kb = 0
    vm_swap_kb = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    vm_rss_kb = int(line.split()[1])
                elif line.startswith("VmSwap:"):
                    vm_swap_kb = int(line.split()[1])
                    break
    except Exception:
        pass

    # --- RAM hit ratio ---
    # Fraction of the process's memory pages that are in physical RAM vs swapped to disk.
    # hit_ratio = 1.0 means everything is in RAM (no swap pressure).
    # hit_ratio < 1.0 means some pages were evicted to disk → page faults on access.
    total_mem_kb = vm_rss_kb + vm_swap_kb
    ram_hit_ratio = vm_rss_kb / total_mem_kb if total_mem_kb > 0 else 1.0
    print(f"HIT_RATIO,{ram_hit_ratio:.6f}")

    # Memory layout info
    q_start = hex(queries.__array_interface__['data'][0])
    q_size_mb = queries.nbytes / (1024 * 1024)
    t_start = hex(train_data.__array_interface__['data'][0])
    t_size_mb = train_data.nbytes / (1024 * 1024)

    print(f"MEMORY_LAYOUT,{q_start},{q_size_mb:.1f},{t_start},{t_size_mb:.1f}")
    print(f"RESULT,{latency_ms},{vm_rss_kb},{vm_swap_kb}")


if __name__ == "__main__":
    main()
