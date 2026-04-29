import os
import time

import numpy as np
import hnswlib


def main():
    dataset_path = os.getenv("HNSW_DATASET_PATH", "/app/real_world_dataset.npy")
    queries_path = os.getenv("HNSW_QUERIES_PATH", "/app/queries.npy")

    train_data = np.load(dataset_path)
    if train_data.ndim == 1:
        train_data = train_data.reshape(1, -1)

    index_size = int(os.getenv("HNSW_INDEX_SIZE", "0"))
    if 0 < index_size < train_data.shape[0]:
        train_data = train_data[:index_size].copy()

    num_elements, dim = train_data.shape

    k               = int(os.getenv("HNSW_K",               "10"))
    ef              = int(os.getenv("HNSW_EF",              "200"))
    num_threads     = int(os.getenv("HNSW_NUM_THREADS",     "1"))
    ef_construction = int(os.getenv("HNSW_EF_CONSTRUCTION", "100"))
    M               = int(os.getenv("HNSW_M",               "32"))

    queries = np.load(queries_path)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.shape[1] != dim:
        raise ValueError(f"queries dim {queries.shape[1]} != index dim {dim}")

    k = min(k, num_elements)

    print("Building HNSW graph...")
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    index.add_items(train_data, np.arange(num_elements))
    index.set_ef(ef)

    print(f"DATASET_INFO,{num_elements},{dim}")

    # Single-query warmup reduces first-query cold-start bias. Use the first
    # real query (not a zero vector — zero vectors land in unrepresentative
    # regions of the graph and skew the warmup's traversal pattern).
    warmup_q = queries[0:1].astype(np.float32, copy=False)
    index.knn_query(warmup_q, k=k, num_threads=num_threads)

    per_query_ms = []
    for i in range(queries.shape[0]):
        q = queries[i:i + 1].astype(np.float32, copy=False)
        t0 = time.perf_counter()
        ids, dists = index.knn_query(q, k=k, num_threads=num_threads)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        per_query_ms.append(lat_ms)

        ids_flat   = ids[0].tolist()
        dists_flat = [float(x) for x in dists[0].tolist()]
        ids_str    = ";".join(str(x) for x in ids_flat)
        dists_str  = ";".join(f"{x:.6f}" for x in dists_flat)
        print(f"QUERY,{i},{lat_ms:.6f},{dists_str},{ids_str}")

    mean_ms = float(np.mean(per_query_ms)) if per_query_ms else 0.0

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

    print(f"RESULT,{mean_ms:.6f},{vm_rss_kb},{vm_swap_kb}")


if __name__ == "__main__":
    main()
