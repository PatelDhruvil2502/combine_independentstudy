# Comprehensive Line-by-Line Code Explanation: HNSW Prefetch Experiment

This document breaks down **every single line of code** in the HNSW prefetch experiment, covering both `run_experiment.py` (the orchestrator) and `worker.py` (the execution environment).

---

## 1. `worker.py` - The Execution Environment

This script runs *inside* the Docker container. It defines the memory bounds and measures the latency internally to avoid Docker overhead. 

### Imports and Exact Search
```python
1: import time
2: import os
3: import numpy as np
4: import hnswlib
```
**Lines 1-4:** Imports the standard `time` for timers, `os` for reading environment variables, `numpy` for multi-dimensional array math, and `hnswlib` as the core C++ vector search library that we are benchmarking.

```python
7: def brute_force_knn(train_data, queries, k):
8:     """Exact L2 nearest neighbours via matrix arithmetic."""
9:     train_sq = (train_data ** 2).sum(axis=1)           # (N,)
10:     query_sq = (queries ** 2).sum(axis=1)              # (n,)
11:     dot      = queries @ train_data.T                  # (n, N)
12:     dists    = train_sq[np.newaxis, :] + query_sq[:, np.newaxis] - 2.0 * dot
13:     return np.argsort(dists, axis=1)[:, :k]
```
**Lines 7-13:** This function provides a 100% exact "ground truth" nearest neighbor search by doing standard matrix multiplication. 
- Lines 9-10 find the squared magnitudes of the dataset and the queries.
- Line 11 does a dot product between the queries and the dataset.
- Line 12 uses algebraic expansion `(a-b)^2 = a^2 + b^2 - 2ab` to calculate exactly the squared L2 distances in a hyper-optimized, vectorized numpy manner without loops. 
- Line 13 returns the indices ("argsorts") of the smallest `k` distances (the nearest neighbors).

### Loading Data and Configurations
```python
16: def main():
17:     dataset_path = os.getenv("HNSW_DATASET_PATH", "/app/real_world_dataset.npy")
18:     print(f"Loading {dataset_path}...")
19:     train_data = np.load(dataset_path)
20:     if train_data.ndim == 1:
21:         train_data = train_data.reshape(1, -1)
```
**Lines 16-21:** The entry point. Reads the dataset path from the environment (defaulting to the volume-mounted docker path). Loads it using `numpy.load`. Lines 20-21 ensure that if the dataset only has one vector (1D array), we reshape it to 2D shape `(1, dim)` to avoid shape errors later.

```python
23:     index_size = int(os.getenv("HNSW_INDEX_SIZE", "0"))
24:     if 0 < index_size < train_data.shape[0]:
25:         train_data = train_data[:index_size].copy()
```
**Lines 23-25:** Captures `HNSW_INDEX_SIZE` to slice the dataset down. If it is greater than zero and smaller than the full dataset size, we safely truncate the memory structure returning a `.copy()` to ensure contiguous clean memory.

```python
27:     num_elements, dim = train_data.shape
28-38:  ... # os.getenv parsing for k, ef, threads, batch, queries, etc.
```
**Lines 27-38:** We capture all configuration parameters that the exterior script (`run_experiment.py`) populated through docker `docker run -e` environment variables. Example config items include: `k` (neighbors to find), `ef` (exploration depth), `num_threads` (CPU parallelism), and `M` (maximum connections per node in graph).

### Synthesizing the Query Workload
```python
40:     rng = np.random.default_rng(seed)
41:     print("Generating queries...")
42:     indices = rng.choice(num_elements, size=num_queries_total, replace=True)
43:     queries = np.empty((num_queries_total, dim), dtype=np.float32)
```
**Lines 40-43:** Initializes random number generator deterministically using our `seed`. We first randomly pick dataset node array indices. Then we allocate the bare memory `np.empty` that our newly synthesized queries will live in.

```python
44:     for i in range(0, num_queries_total, chunk_size):
45:         end = min(i + chunk_size, num_queries_total)
46:         idx = indices[i:end]
47:         queries[i:end] = (
48:             train_data[idx]
49:             + rng.normal(scale=query_noise, size=(len(idx), dim)).astype(np.float32)
50:         )
```
**Lines 44-50:** We don't want tests looking for vectors *exactly* inside the graph. This loops in chunks, taking a chunk of existing real vectors `train_data[idx]`, and adds artificial `normal` random noise to them. The model now searches for variations of existing data—forming realistic queries.

### Building the Graph
```python
52:     print("Building HNSW graph...")
53:     index = hnswlib.Index(space="l2", dim=dim)
54:     index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
55:     index.add_items(train_data, np.arange(num_elements))
56:     index.set_ef(ef)
```
**Lines 52-56:** We instantiate an `hnswlib` structure using L2 (Euclidean) distance space. We initialize memory parameters, and dump the entire dense `train_data` into the tree (`add_items`), mapping them linearly to integer IDs (`np.arange()`). Lastly, we configure `ef` for querying time.

### Recall Safety Net
```python
58-62:  ... # Comments and variable min capping
63:     sample_q  = queries[:recall_samples]
64:     true_nn   = brute_force_knn(train_data, sample_q, k)
65:     hnsw_nn, _ = index.knn_query(sample_q, k=k, num_threads=num_threads)
66:     recall = float(np.mean([
67:         len(set(true_nn[i].tolist()) & set(hnsw_nn[i].tolist())) / k
68:         for i in range(recall_samples)
69:     ]))
```
**Lines 58-69:** Takes a tiny test subset (`sample_q`) of our queries. It runs the exact math (Line 64) and the approximate C++ math (Line 65). Lines 66-69 compute the 'Overlap' (Recall). By taking the intersect `&` of the sets of IDs, divided by K, it asserts whether or not prefetching "broke" accurate results!

### Benchmarking Core Loop
```python
71:     print("Starting timed search...")
74:     random_order = rng.permutation(num_queries_total)
77:     batch_latencies_ms = []
78:     t0 = time.perf_counter()
```
**Lines 71-78:** Randomizes the order we ask queries to completely defeat any passive OS "Next Page" auto-prefetching. `batch_latencies_ms` array will store how long specific micro-segments of runs take. `t0` records wall time using the highly-precise Python `perf_counter()`.

```python
79:     for i in range(0, num_queries_total, eval_batch_size):
80:         batch_idx = random_order[i: i + eval_batch_size]
81:         batch_q   = queries[batch_idx]
82:         bt0 = time.perf_counter()
83:         index.knn_query(batch_q, k=k, num_threads=num_threads)
84:         bt_ms = (time.perf_counter() - bt0) / len(batch_idx) * 1000
85:         batch_latencies_ms.append(bt_ms)
```
**Lines 79-85:** Iterates queries by batches. It isolates `batch_q` (Line 81), starts a micro-timer, queries the index against the C++ layer (Line 83), and finishes the mini-timer. Note on Line 84: Multiplying by `1000` converts standard Seconds into Milliseconds per-query.

```python
87:     elapsed_s  = time.perf_counter() - t0
88:     latency_ms = elapsed_s / num_queries_total * 1000
90:     lats = np.array(batch_latencies_ms)
91:     p50  = float(np.percentile(lats, 50))
92:     p95  = float(np.percentile(lats, 95))
93:     p99  = float(np.percentile(lats, 99))
```
**Lines 87-93:** Finishes the macro-timer `t0`, calculating full batch latency span. Uses the `batch_latencies_ms` to calculate robust Percentiles (Median=P50, P95, P99). Extreme spikes in memory are usually caught at P99.

### Environmental Info
```python
95:     vm_rss_kb = vm_swap_kb = 0
96:     try:
97:         with open("/proc/self/status") as f:
98:             for line in f:
99:                 if line.startswith("VmRSS:"):
100:                     vm_rss_kb = int(line.split()[1])
101:                 elif line.startswith("VmSwap:"):
102:                     vm_swap_kb = int(line.split()[1])
103:                     break
104:     except Exception:
105:         pass
```
**Lines 95-105:** Linux specific telemetry reading `/proc/self/status`. Safely sniffs how much raw RAM (`VmRSS`) and logical Swap (`VmSwap`) memory that Python + C++ + The Index are currently consuming right before process death.

```python
107:     q_start   = hex(queries.__array_interface__["data"][0])
108:     q_size_mb = queries.nbytes / (1024 * 1024)
109:     t_start   = hex(train_data.__array_interface__["data"][0])
110:     t_size_mb = train_data.nbytes / (1024 * 1024)
```
**Lines 107-110:** Purely educational/diagnostic math tracking exactly where in the C memory address plane `__array_interface__['data']` the matrices are laying, and sizes in MB.

```python
112:     print(f"DATASET_INFO,{num_elements},{dim}")
113:     print(f"RECALL,{recall:.6f}")
114:     print(f"LATENCY_STATS,{p50:.6f},{p95:.6f},{p99:.6f}")
115:     print(f"MEMORY_LAYOUT,{q_start},{q_size_mb:.1f},{t_start},{t_size_mb:.1f}")
116:     print(f"RESULT,{latency_ms},{vm_rss_kb},{vm_swap_kb}")
```
**Lines 112-116:** Standardized print outputs. It communicates by writing explicitly formatted text to `stdout` that the python script outside docker will safely parse.
**Line 119-121:** Python `__main__` entrypoint invoking `main()`.

---

## 2. `run_experiment.py` - Orchestrator and Benchmarker

This Python file exists purely to manage isolation boundaries. It builds environments, issues requests to them, catches responses, and plots results. 

### Initial Setup and Dockerfile Synthesis
```python
1: import argparse
2: import csv
3: import subprocess
4: from pathlib import Path
6: import matplotlib.pyplot as plt
7: import numpy as np
```
**Lines 1-7:** Core libraries: Argparsing strings, writing CSV datasets, executing Docker subprocesses, graphing with MPL.

```python
14: def build_image(image_name, disable_prefetch=False, python_version="3.10-slim"):
...
22:     dockerfile_content = f"""FROM python:{python_version}
23: RUN apt-get update && apt-get install -y build-essential python3-dev git
24: ARG DISABLE_PREFETCH=0
25: RUN pip install numpy
26: RUN git clone --depth 1 https://github.com/nmslib/hnswlib.git /tmp/hnswlib
27: RUN if [ "$DISABLE_PREFETCH" = "1" ]; then \\
28:       python -c "from pathlib import Path; p=Path('/tmp/hnswlib/hnswlib/hnswalg.h'); t=p.read_text(); marker='#include <memory>'; inject='\\\\n#ifdef DISABLE_HNSW_PREFETCH\\\\n#define _mm_prefetch(a, sel) ((void)0)\\\\n#endif\\\\n'; t=t.replace(marker, marker+inject, 1) if marker in t and 'DISABLE_HNSW_PREFETCH' not in t else t; p.write_text(t)"; \\
29:       CXXFLAGS='-DDISABLE_HNSW_PREFETCH' pip install /tmp/hnswlib; \\
30:     else \\
31:       pip install /tmp/hnswlib; \\
32:     fi && rm -rf /tmp/hnswlib
...
37: """
```
**Lines 14-37:** The Docker image build function dynamically generates a text string corresponding to a Dockerfile.
- Line 26: Downloads the source code for hnswlib directly from git.
- Line 27-28 is **critical**. This is the mechanism that disables prefetch. It searches for `#include <memory>` inside `hnswalg.h` and *injects* a C++ Macro redefined over the `_mm_prefetch()` call: `#define _mm_prefetch(a, sel) ((void)0)`. This forces the compiler to ignore any prefetch statements if `-DDISABLE_HNSW_PREFETCH` is passed.
- Line 29 tests that parameter during `pip install`, replacing the prefetch headers completely when built into python binaries.
- Line 31 just installs the default unbroken library.

```python
39:     cmd = ["docker", "build", "--platform", "linux/amd64", "-t", image_name, "."]
40:     if disable_prefetch:
41:         cmd.extend(["--build-arg", "DISABLE_PREFETCH=1"])
42:     subprocess.run(cmd, check=True)
```
**Lines 39-42:** Constructs and executes the `docker build` terminal command. The `--platform linux/amd64` constraint ensures that CPUs support Intel SSE intrinsics (since HNSW disables them completely on default Apple ARM chips anyways, we simulate AMD for accurate prefetching conditions).

### Parsing from stdout
```python
49: def parse_result(stdout):
...
59:     for line in stdout.splitlines():
60:         line = line.strip()
61:         if line.startswith("RESULT,"):
...
75:     return result if result["latency_ms"] is not None else None
```
**Lines 49-75:** Matches the `print` formats from `worker.py` (like `RESULT,`, `RECALL,`, and `LATENCY_STATS,`), separates them on `,` logic mapping them back into native python floats/dictionaries. Returns None if it failed.

### Isolating individual Test Executions
```python
82: def run_once(image_name, dataset_path, timeout_s, num_queries, ... # arguments...
...
101:     dataset_abs = str(dataset_path.resolve())
102:     cmd = [
103:         "docker", "run", "--rm", "--platform", "linux/amd64",
104:         "-e", f"HNSW_NUM_QUERIES={num_queries}",
 ... # environment binds 
116:         "-v", f"{dataset_abs}:/app/real_world_dataset.npy:ro",
117:     ]
118:     if memory_limit:
119:         cmd.extend(["--memory", memory_limit])
...
123:     result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
...
129:     return parsed, result.stdout
```
**Lines 82-129:** Exposes standard Python arguments and crafts a safe `docker run` command wrapper. Crucial details include:
- `--rm`: Safely destroys the container after it finishes so layers and temp files do not bleed over environments.
- `-v {dataset_abs} : {path} : ro`: Volume maps the user's `.npy` dataset safely as `ro` (Read Only), ensuring worker script cannot touch/ruin user data.
- Processes stdout by parsing it using the `parse_results()` above, then returns it.

### Benchmarking Loops
```python
136: def save_raw_log(out_dir, mode, cycle, text): ... # Saves execution prints to a .log tracker folder
```
**Lines 136-140:** Basic file-io mechanism saving stdout logs in case something failed.

```python
143: def run_benchmark(image_on, image_off, dataset_path, runs, warmup, timeout_s, out_dir, ...
...
177:     for i in range(warmup):
178:         print(f"Warmup {i + 1}/{warmup}: prefetch ON")
179:         run_once(image_on, **_run_args)
...
184:     for cycle in range(1, runs + 1):
185:         order = ["on", "off"] if cycle % 2 == 1 else ["off", "on"]
186:         print(f"Cycle {cycle}/{runs} order: {order[0].upper()} -> {order[1].upper()}")
```
**Lines 143-189.** The core benchmark loop. 
- Lines 177-179 execute "warmup cycles". Often memory pages or container startup latency are much slower the very first time. We buffer it.
- Line 185 toggles the runtime order. Cycle 1 does (ON then OFF). Cycle 2 does (OFF then ON). This isolates thermal throttling of the CPU or sequential heat degradation issues.

### CSV Writers
```python
249: def write_csv(rows, csv_path): ... (Writes columns headers and dicts)
```
**Lines 249-266:** Utilizes pure Python `csv.DictWriter` logic taking array structure loops into standard formats. It exports 1 main file `raw_results.csv` and an optional scaling file.

### Graphing Utilities (Matplotlib)
```python
278: def plot_comparison(rows, out_path, ef, num_elements, dim):
...
289:     on_mean  = statistics.mean(on_vals)
290:     off_mean = statistics.mean(off_vals)
291:     pct_slower = (off_mean - on_mean) / on_mean * 100
```
**Lines 278-291:** Groups dictionary outputs depending on the `mode=="on"` parameters, and determines safely calculated statistical means utilizing the `statistics` python standard library. Mathematically defines how much slower the code runs because `off - on / on` returns proportional difference.

```python
309:     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
...
323:     b_on  = ax1.bar(x - w / 2, on_bars_val,  w, label="Prefetch ON",  color=color_on,  alpha=0.85)
...
344:     ax2.plot(on_idx,  on_vals,  "o-",  color=color_on,  linewidth=2, markersize=7, label="Prefetch ON")
...
361:         r_bars = ax3.bar(["Prefetch ON", "Prefetch OFF"], [on_recall, off_recall] ...
```
**Lines 309-389:** Composes the complex UI panel for the image result. It defines a grid of 1 Row, 3 Columns (`ax1, ax2, ax3`).  
- `ax1` graphs bar metrics comparing exact latencies.
- `ax2` uses a line chart plotting latencies against runtime cycles tracking thermal variances.
- `ax3` ensures `on_recall` perfectly equals `off_recall`.

```python
391:     try:
392:         if sys.platform == "darwin":  ... Open native image apps ..
```
**Lines 391-458 (`plot_scaling` Included):** Attempts to open the system viewer based on Operating System. Mac operates via `subprocess.Popen(["open"])`, Linux via `xdg-open`. It safely tries and skips if UI interfaces are unavailable during the execution mode.

### Script Configurations
```python
495: def main():
496:     parser = argparse.ArgumentParser(description="Real A/B benchmark for HNSW prefetch ON vs OFF.")
497:     parser.add_argument("--dataset",      default="real_world_dataset.npy", help="Path to .npy dataset")
...
560:     if args.scenario == "prefetch_friendly":
...
566:     elif args.scenario == "prefetch_advantage":
...
580:     dataset_arg = args.dataset.strip()
581:     if dataset_arg == "/absolute/path/to/your_real_dataset.npy": ...
```
**Lines 495-606:** Main command-block parsing logic. Validates min and max bound conditions around threading, integers (E.g. num_queries cannot logically be `<1000`). It includes "scenarios" (Line 560-578) where it artificially bloats or changes queries parameters forcing workloads that expose Prefetch dependencies better depending on user demands. Defines logic to gracefully recover if a user utilizes standard filler strings like `/absolute/path/to` by checking relative scopes instead.

```python
606:     image_on  = "hnsw-prefetch-on"
607:     image_off = "hnsw-prefetch-off"
608:     build_image(image_on,  disable_prefetch=False, python_version=args.python_version)
609:     build_image(image_off, disable_prefetch=True,  python_version=args.python_version)
```
**Lines 606-609:** Coordinates building both separate docker container boundaries utilizing the above injection blocks, allowing isolated memory runtime and dependency sets for the benchmark toggling later.

```python
632:     # ── Main benchmark ────────────────────────────────────────────────────────
633:     rows = run_benchmark(
634:         image_on, image_off,
635:         runs=args.runs, warmup=args.warmup, out_dir=out_dir,
...
640:     write_csv(rows, csv_path)
...
651:     plot_comparison(rows, plot_path, ef=args.ef, num_elements=num_elements, dim=dim)
652:     print_raw_results(rows)
```
**Lines 632-658:** Triggers the loop. Stores iterations, commits the writes, builds the plots, and invokes `print_raw_results(rows)` to dump the ASCII layout locally to the terminal.

```python
659:     # ── Scaling sweep (optional) ──────────────────────────────────────────────
660:     if args.index_sizes.strip():
661:         index_sizes = [int(s.strip()) for s in args.index_sizes.split(",") if s.strip()]
```
**Lines 659-682:** Analyzes if the user asked to track latency vs index size variations (`--index-sizes 10000,50000`). If so, runs the whole script *again* truncating arrays incrementally, charting exact performance variations onto a different artifact logic block!
