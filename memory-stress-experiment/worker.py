import os
import time
import numpy as np
import hnswlib


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

    # Memory layout info
    q_start = hex(queries.__array_interface__['data'][0])
    q_size_mb = queries.nbytes / (1024 * 1024)
    t_start = hex(train_data.__array_interface__['data'][0])
    t_size_mb = train_data.nbytes / (1024 * 1024)

    print(f"MEMORY_LAYOUT,{q_start},{q_size_mb:.1f},{t_start},{t_size_mb:.1f}")
    print(f"RESULT,{latency_ms},{vm_rss_kb},{vm_swap_kb}")


if __name__ == "__main__":
    main()
