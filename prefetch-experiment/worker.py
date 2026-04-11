import time
import os
import numpy as np
import hnswlib


def main():
    dataset_path = os.getenv("HNSW_DATASET_PATH", "/app/real_world_dataset.npy")
    print(f"Loading {dataset_path}...")
    train_data = np.load(dataset_path)
    if train_data.ndim == 1:
        train_data = train_data.reshape(1, -1)

    index_size = int(os.getenv("HNSW_INDEX_SIZE", "0"))
    if 0 < index_size < train_data.shape[0]:
        train_data = train_data[:index_size].copy()

    num_elements, dim = train_data.shape
    k                 = int(os.getenv("HNSW_K",               "20"))
    ef                = int(os.getenv("HNSW_EF",              "100"))
    num_threads       = int(os.getenv("HNSW_NUM_THREADS",     "1"))
    eval_batch_size   = int(os.getenv("HNSW_BATCH_SIZE",      "5000"))
    query_noise       = float(os.getenv("HNSW_QUERY_NOISE",   "0.01"))
    num_queries_total = int(os.getenv("HNSW_NUM_QUERIES",     "300000"))
    seed              = int(os.getenv("HNSW_SEED",            "42"))
    chunk_size        = int(os.getenv("HNSW_CHUNK_SIZE",      "50000"))
    ef_construction   = int(os.getenv("HNSW_EF_CONSTRUCTION", "100"))
    M                 = int(os.getenv("HNSW_M",               "32"))
    rng = np.random.default_rng(seed)
    print("Generating queries...")
    indices = rng.choice(num_elements, size=num_queries_total, replace=True)
    queries = np.empty((num_queries_total, dim), dtype=np.float32)
    for i in range(0, num_queries_total, chunk_size):
        end = min(i + chunk_size, num_queries_total)
        idx = indices[i:end]
        queries[i:end] = (
            train_data[idx]
            + rng.normal(scale=query_noise, size=(len(idx), dim)).astype(np.float32)
        )

    print("Building HNSW graph...")
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    index.add_items(train_data, np.arange(num_elements))
    index.set_ef(ef)

    print("Starting timed search...")
    # Randomise batch order to defeat sequential OS prefetching and
    # ensure each batch touches a scattered region of the graph.
    random_order = rng.permutation(num_queries_total)

    # Time every individual batch so we can derive tail-latency percentiles.
    batch_latencies_ms = []
    t0 = time.perf_counter()
    for i in range(0, num_queries_total, eval_batch_size):
        batch_idx = random_order[i: i + eval_batch_size]
        batch_q   = queries[batch_idx]
        bt0 = time.perf_counter()
        index.knn_query(batch_q, k=k, num_threads=num_threads)
        bt_ms = (time.perf_counter() - bt0) / len(batch_idx) * 1000
        batch_latencies_ms.append(bt_ms)

    elapsed_s  = time.perf_counter() - t0
    latency_ms = elapsed_s / num_queries_total * 1000

    lats = np.array(batch_latencies_ms)
    p50  = float(np.percentile(lats, 50))
    p95  = float(np.percentile(lats, 95))
    p99  = float(np.percentile(lats, 99))

    vm_rss_kb = vm_swap_kb = 0
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

    q_start   = hex(queries.__array_interface__["data"][0])
    q_size_mb = queries.nbytes / (1024 * 1024)
    t_start   = hex(train_data.__array_interface__["data"][0])
    t_size_mb = train_data.nbytes / (1024 * 1024)

    print(f"DATASET_INFO,{num_elements},{dim}")
    print(f"LATENCY_STATS,{p50:.6f},{p95:.6f},{p99:.6f}")
    print(f"MEMORY_LAYOUT,{q_start},{q_size_mb:.1f},{t_start},{t_size_mb:.1f}")
    print(f"RESULT,{latency_ms},{vm_rss_kb},{vm_swap_kb}")


if __name__ == "__main__":
    main()
