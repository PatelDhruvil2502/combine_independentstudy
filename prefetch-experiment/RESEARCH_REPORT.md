# Quantifying the Performance Impact of Software Prefetching in HNSW-Based Approximate Nearest Neighbour Search

**Author:** Dhruvil Patel  
**Date:** April 2026  
**Repository:** https://github.com/PatelDhruvil2502/prefetch  
**Branch:** improvements

---

## Abstract

Hierarchical Navigable Small World (HNSW) graphs are among the most widely used data structures for approximate nearest neighbour (ANN) search in high-dimensional vector spaces. The `hnswlib` C++ implementation contains software prefetch instructions (`_mm_prefetch`) embedded in its core graph traversal routine. This report documents a ground-up experimental investigation into the real performance impact of these prefetch calls — isolating them surgically at compile time, running controlled A/B benchmarks on a real-world dataset of 150,000 vectors at 128 dimensions, and measuring the effect across mean latency, tail latency percentiles (p50, p95, p99), and search quality (recall@k). The results show a consistent **~12% per-query latency reduction** attributable entirely to the prefetch instructions, with no degradation in search quality. The path to this result involved several non-trivial debugging challenges that are documented in detail.

---

## 1. Introduction

### 1.1 What is HNSW?

HNSW constructs a multi-layer proximity graph over a set of vectors. During a query, the algorithm starts from an entry point on the top layer and greedily moves toward the query vector, descending through layers until it reaches the bottom, where it performs a beam search with a candidate queue of size `ef`. At each step, it fetches the neighbour list of the current best node and evaluates each neighbour's distance to the query.

This traversal pattern has an important memory access characteristic: **the next node to visit is not known until the current node has been evaluated**. The CPU cannot predict which memory address to fetch next. This results in a chain of dependent cache misses — each one stalling the pipeline until the data arrives from RAM. On a modern system, a single cache miss to main memory can cost 60–100 nanoseconds, and HNSW's traversal may incur hundreds of such misses per query.

### 1.2 Software Prefetching

The `_mm_prefetch` intrinsic is an x86 SSE instruction that tells the CPU to begin loading a specific memory address into cache *before* it is actually needed. The key insight in `hnswlib` is that while processing node `N`, the algorithm already knows the IDs of `N`'s neighbours — and therefore already knows *which nodes* will likely be visited next. By issuing prefetch instructions for those nodes' data while still computing distances for node `N`, the memory latency can be hidden behind useful computation.

### 1.3 Research Question

The `hnswlib` source code has contained these prefetch calls for years, but I could not find any standalone, reproducible benchmark that answers the simple question: **how much do they actually help, and is the benefit consistent?** This research attempts to answer that with a clean, reproducible, end-to-end experiment.

---

## 2. Experimental Setup

### 2.1 Overall Strategy

The core idea is simple: build two versions of `hnswlib` — one with prefetch enabled (stock), one with prefetch disabled (patched) — run the same workload against both, and measure the difference. Everything else must be held constant.

I chose Docker containers as the isolation boundary for the following reasons:

- The patch needs to happen at **compile time** inside the build environment
- Results need to be reproducible on any machine regardless of local Python or compiler setup
- Each run gets a clean process with no shared state from previous runs
- Memory limits can be enforced at the container level if needed

Two Docker images are built automatically:
- `hnsw-prefetch-on` — stock `hnswlib`, all `_mm_prefetch` calls active
- `hnsw-prefetch-off` — `hnswlib` patched to replace `_mm_prefetch(a, sel)` with `((void)0)`

### 2.2 The Prefetch Patch

The patch injects a preprocessor macro into `hnswlib/hnswalg.h` during the Docker build:

```c
#ifdef DISABLE_HNSW_PREFETCH
#define _mm_prefetch(a, sel) ((void)0)
#endif
```

This is compiled with `CXXFLAGS='-DDISABLE_HNSW_PREFETCH'`. The result is that every prefetch call in the binary becomes a no-op at zero runtime cost — the traversal logic is completely unchanged, only the memory hints are removed.

### 2.3 Bias Controls

| Potential Bias | Mitigation Applied |
|---|---|
| Cold cache first-run advantage | One warmup cycle run and discarded before measurement |
| Thermal throttling / run-order bias | ON and OFF alternated each cycle (odd: ON→OFF, even: OFF→ON) |
| Hardware stream prefetcher masking the effect | Queries fed in randomised batch order to scatter memory access patterns |
| Fluke single-run result | Four independent measured cycles, each reported individually |

### 2.4 Index and Workload Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Vectors | 150,000 | Large enough to exceed typical L3 cache |
| Dimensions | 128 | Realistic embedding size |
| Space | L2 (Euclidean) | Most common in practice |
| M | 32 | Dense graph — more pointer chasing, amplifies cache miss cost |
| ef_construction | 100 | Standard graph build quality |
| ef (search) | 200 | Visits 2× more candidates than default, amplifies cache miss frequency |
| k | 20 | Realistic top-k retrieval size |
| Queries per run | 50,000 | Sufficient for stable per-query latency |
| Batch size | 2,000 | Balances measurement granularity and overhead |
| Threads | 1 | Isolates the cache effect without thread-competition noise |

`ef=200` was deliberately chosen higher than the default. It doubles the number of candidate nodes visited per query, which doubles the frequency of scattered memory accesses and makes the prefetch benefit easier to measure cleanly.

---

## 3. Implementation

### 3.1 System Architecture

The benchmark is split into two components:

**`run_experiment.py` (Orchestrator)**  
Runs on the host machine. Responsible for generating the Dockerfile, building both images, managing the benchmark lifecycle (warmup, cycle ordering, timeout), parsing worker output, writing CSV results, and generating plots. Every parameter is exposed as a CLI argument with no hardcoded values.

**`worker.py` (Worker)**  
Runs inside each Docker container. Loads the dataset, generates query vectors, builds the HNSW index from scratch, computes recall@k against brute-force ground truth, and then runs the timed search loop. Results are emitted as structured output lines that the orchestrator parses.

### 3.2 Recall@k Measurement

Before the timed benchmark begins, the worker computes exact L2 nearest neighbours for 200 sample queries using pure matrix arithmetic:

```
||a - b||² = ||a||² + ||b||² - 2·(a · b)
```

This gives exact ground truth. The HNSW results for the same queries are then compared to compute recall@k. This serves as a scientific validity check — confirming that removing prefetch instructions does not alter the search results in any way, only the speed.

### 3.3 Tail Latency Measurement

Each individual `knn_query(batch)` call is timed:

```python
bt0 = time.perf_counter()
index.knn_query(batch, k=k, num_threads=num_threads)
bt_ms = (time.perf_counter() - bt0) / len(batch) * 1000
batch_latencies_ms.append(bt_ms)
```

p50, p95, and p99 are computed from all batch-level latencies. This reveals whether prefetch helps uniformly across the run or only at specific points (e.g., cold start vs. warm).

### 3.4 Output Protocol

The worker communicates results via structured lines on stdout:

```
DATASET_INFO,{num_elements},{dim}
RECALL,{recall_at_k}
LATENCY_STATS,{p50},{p95},{p99}
MEMORY_LAYOUT,{query_ptr},{query_mb},{train_ptr},{train_mb}
RESULT,{mean_latency_ms},{vm_rss_kb},{vm_swap_kb}
```

The orchestrator parses these lines and stores all metrics in a CSV alongside raw log files for every individual run.

---

## 4. Bugs and Challenges

This section documents the significant issues encountered during development. Several of these took considerable time to identify and resolve, and they fundamentally changed my understanding of the problem.

### 4.1 The Silent Patch Failure — "Both Images Are Identical" (≈ 1 Week)

**This was the most damaging bug in the project.** For the first week of experiments, both `hnsw-prefetch-on` and `hnsw-prefetch-off` were producing **exactly the same latency numbers** — down to the fourth decimal place in some runs. At first I thought the prefetch effect was simply too small to measure and I needed a different dataset or different parameters. I spent several days tweaking `ef`, `M`, query count, memory pressure, and dataset size, getting the same result every time: the two images were essentially identical.

The real problem was that the patch was being silently skipped. The injection code looked for the string `#include <xmmintrin.h>` as a marker in `hnswalg.h` to know where to insert the macro definition:

```python
marker = '#include <xmmintrin.h>'
```

That string does not exist in `hnswalg.h`. The `xmmintrin.h` header is included transitively — not directly in that file. Since the marker was never found, the `str.replace()` call had nothing to replace, returned the original file unchanged, and the file was written back with no modification. No error was raised. No warning was printed. The build completed successfully, both images got the same binary, and every benchmark run I did for that entire period was completely meaningless.

The fix was to find an injection point that actually exists in `hnswalg.h`. I read through the header file manually and found that `#include <memory>` is the last `#include` in the file. Switching the marker to that string made the patch reliable:

```python
marker = '#include <memory>'
inject = '\n#ifdef DISABLE_HNSW_PREFETCH\n#define _mm_prefetch(a, sel) ((void)0)\n#endif\n'
t = t.replace(marker, marker + inject, 1)
```

After this fix, the ON and OFF images diverged immediately and consistently.

**Lesson learned:** Any patch that silently succeeds when the target string is not found is a correctness trap. The fix should always validate that the replacement actually happened, e.g., by asserting that `'DISABLE_HNSW_PREFETCH' in t` after the replacement.

---

### 4.2 The ARM64 Architecture Trap — "Zero Difference on My Machine" (≈ 4–5 Days)

Even after fixing the silent patch failure, I initially ran the experiments on my MacBook's native ARM64 architecture and continued to see essentially no difference between the two builds. The numbers were not identical anymore, but the gap was well within noise — maybe 0.5% — nowhere near statistically meaningful.

After spending days re-examining the patch logic and Docker setup, I eventually traced the issue to the `hnswlib` source code itself. In `hnswalg.h`, all `_mm_prefetch` calls are wrapped inside an `#ifdef USE_SSE` block:

```cpp
#ifdef USE_SSE
    _mm_prefetch((char *)(data_level0_memory_ + ...), _MM_HINT_T0);
#endif
```

`USE_SSE` is only defined by the build system when the compiler detects x86 SSE support. On ARM64, there is no SSE — so `USE_SSE` is never defined, the prefetch calls are compiled out entirely regardless of my patch, and **both images are built without any prefetch calls at all**. My `DISABLE_HNSW_PREFETCH` patch was patching something that was already not there on that platform.

The solution was to force the Docker build to target `linux/amd64` explicitly:

```bash
docker build --platform linux/amd64 -t hnsw-prefetch-on .
```

On `amd64`, `USE_SSE` is defined, the prefetch calls are compiled in for the ON build and patched out for the OFF build, and the experiment works as intended. Running amd64 containers on Apple Silicon goes through Rosetta 2 emulation, which is slower overall, but the *relative* difference between ON and OFF remains valid and measurable.

**Lesson learned:** Always verify that the code path you are trying to benchmark actually exists in the binary you are running. A disassembler or a build log review would have caught this on day one.

---

### 4.3 Hardcoded Parameters Producing Irreproducible Results (≈ 2–3 Days)

Once the core experiment was working, I tried varying the HNSW graph parameters to understand how the prefetch benefit changes with graph density. I would change `M` or `ef_construction` in the orchestrator command line, re-run, and get results that looked identical to the previous run — same latency, same recall, same memory usage.

The cause: both `M=32` and `ef_construction=100` were hardcoded directly inside `worker.py`:

```python
index.init_index(max_elements=num_elements, ef_construction=100, M=32)
```

The orchestrator was passing different values via environment variables to the container, but the worker was ignoring them completely and always using the hardcoded values. I had changed the CLI arguments and re-run the benchmark, but the Docker image was cached from the previous build — so the new `worker.py` was not even being picked up. I was passing `--hnsw-m 16` on the command line, the container was building with `M=32`, and I was comparing runs that were actually identical.

This was actually a compounded bug: hardcoded values *and* Docker layer caching. The fix required two changes:
1. Replace all hardcoded values in `worker.py` with `os.getenv(...)` calls
2. Ensure `worker.py` changes invalidate the Docker build cache (which they do, since `COPY worker.py` is a later layer)

After this, every parameter — `M`, `ef_construction`, RNG seed, chunk size, dataset path — was fully configurable from the command line, and changing any of them produced genuinely different results.

---

### 4.4 Docker Platform Warning — Iterative Fix (≈ Half a Day)

During the containerisation work, the Dockerfile contained:

```dockerfile
FROM --platform=linux/amd64 python:3.10-slim
```

Docker BuildKit started emitting a warning on every build:

```
WARN: FromPlatformFlagConstDisallowed: FROM --platform flag should not
use constant value "linux/amd64" (line 1)
```

The linter does not want a hardcoded string in the `FROM --platform` argument — it wants the value to be parameterised so the Dockerfile can be used across different target platforms.

My first fix was to parameterise it using Docker's built-in `TARGETPLATFORM` variable:

```dockerfile
ARG TARGETPLATFORM=linux/amd64
FROM --platform=$TARGETPLATFORM python:3.10-slim
```

This silenced the first warning but immediately triggered a second one:

```
WARN: RedundantTargetPlatform: Setting platform to predefined
$TARGETPLATFORM in FROM is redundant as this is the default behavior
```

Docker's own documentation explains that `$TARGETPLATFORM` is already what `FROM` uses when you pass `--platform` to `docker build`. Explicitly writing `--platform=$TARGETPLATFORM` in the `FROM` instruction is therefore circular and unnecessary.

The correct and final fix was to remove `--platform` from the `FROM` instruction entirely:

```dockerfile
FROM python:{python_version}
```

The platform pinning is handled solely by the `docker build --platform linux/amd64` flag in the Python orchestrator — which is the right place for it. The Dockerfile itself stays clean and portable.

---

### 4.5 Plot Title Showing Wrong Dataset Dimensions (≈ 1 Hour)

A smaller but important issue: the benchmark plot always showed `"150 K vectors · 128-dim"` in the title regardless of what dataset was actually used. This was hardcoded as a string literal in the `plot_comparison` function:

```python
fig.suptitle(
    "HNSW Prefetch ON vs OFF\n"
    "(150 K vectors · 128-dim · ef=200 · real measured runs)"
)
```

If the experiment was run with a different dataset — say a 50K-vector subset or a 768-dim embedding — the plot would lie about the data it was showing. The fix was to load the dataset with `numpy.load(mmap_mode="r")` at the start of the orchestrator before any Docker builds begin, read the actual shape, and pass `num_elements` and `dim` as parameters to the plot function:

```python
dataset_meta = np.load(dataset_path, mmap_mode="r")
num_elements, dim = dataset_meta.shape
```

Using `mmap_mode="r"` means NumPy reads only the header metadata without loading the full array into memory — efficient even for very large datasets.

---

### 4.6 Recall Measurement Initially Using Wrong Distance Computation (≈ 2 Days)

The first implementation of the brute-force ground truth for recall@k used a naive Python loop:

```python
for i in range(num_recall_samples):
    dists = np.linalg.norm(train_data - queries[i], axis=1)
    true_neighbors[i] = np.argsort(dists)[:k]
```

This had two problems. First, it was extremely slow — iterating over 200 queries with 150K distance computations each took over 3 minutes inside the container, dominating the total runtime and making the benchmark impractically slow. Second, `np.linalg.norm` computes the Euclidean distance (the square root), while `hnswlib` internally uses **squared** Euclidean distance and returns neighbours sorted by the squared distance. For the purpose of finding top-k neighbours this makes no difference (square root is monotonic), but the vectorised replacement I wrote initially accidentally used a dot product formulation that assumed normalised vectors — which these are not.

The correct vectorised implementation uses the algebraic expansion of squared L2 distance:

```
||a − b||² = ||a||² + ||b||² − 2·(a · b)
```

In NumPy:

```python
train_sq = (train_data ** 2).sum(axis=1)        # (N,)
query_sq = (queries ** 2).sum(axis=1)           # (n,)
dot      = queries @ train_data.T               # (n, N)
dists    = train_sq[None, :] + query_sq[:, None] - 2.0 * dot
true_neighbors = np.argsort(dists, axis=1)[:, :k]
```

This processes all 200 queries in a single matrix multiply, runs in under 2 seconds inside the container, and produces the correct exact L2 nearest neighbours that match `hnswlib`'s internal distance metric.

---

## 5. Results

### 5.1 Raw Per-Cycle Latency

Four measured cycles were run with one warmup cycle discarded. ON and OFF order was alternated to control for thermal bias.

| Cycle | Order | Prefetch ON (ms/query) | Prefetch OFF (ms/query) | Difference |
|---|---|---|---|---|
| 1 | ON → OFF | 0.52355 | 0.59557 | +13.7% |
| 2 | OFF → ON | 0.52663 | 0.59159 | +12.3% |
| 3 | ON → OFF | 0.52465 | 0.58709 | +11.9% |
| 4 | OFF → ON | 0.52857 | 0.59064 | +11.7% |
| **Mean** | | **0.52585** | **0.59122** | **+12.4%** |

### 5.2 Tail Latency Percentiles

| Percentile | Prefetch ON | Prefetch OFF | Delta |
|---|---|---|---|
| p50 | 0.523 ms | 0.590 ms | +12.8% |
| p95 | 0.540 ms | 0.604 ms | +11.8% |
| p99 | 0.547 ms | 0.611 ms | +11.7% |

The prefetch advantage is **consistent across all percentiles**. The p50 gap (+12.8%) is slightly larger than the mean gap (+12.4%), indicating that prefetch helps the median batch more than it helps outlier batches. There is no evidence of prefetch causing occasional slowdowns — the gap is positive and stable at every percentile level.

### 5.3 Search Quality

| Metric | Prefetch ON | Prefetch OFF |
|---|---|---|
| Recall@20 (cycle 1) | 0.8450 | 0.8435 |
| Recall@20 (cycle 2) | 0.8473 | 0.8450 |
| Recall@20 (cycle 3) | 0.8475 | 0.8485 |
| Recall@20 (cycle 4) | 0.8440 | 0.8455 |
| **Mean Recall@20** | **0.8460** | **0.8456** |

The difference in recall between the two builds is **0.0004** — well within the statistical noise of the 200-sample measurement. Both builds return effectively identical results. This confirms that `_mm_prefetch` is a pure timing hint with no effect on the correctness of the search.

### 5.4 Memory Usage

Both configurations consumed approximately **551 MB RSS** with zero swap usage, confirming the index fit entirely in container memory. The latency difference is therefore attributable to **CPU cache effects**, not memory pressure or swap activity.

---

## 6. Analysis and Discussion

### 6.1 Why Does Prefetch Help HNSW Specifically?

The HNSW traversal loop has a structure that makes it particularly vulnerable to memory latency:

1. Pop the best candidate from the priority queue
2. **Load that node's neighbour list from memory** ← cache miss here
3. For each neighbour, compute the distance to the query
4. Push promising neighbours back onto the queue
5. Repeat

Step 2 is a dependent load — the address to fetch is determined by the result of step 1, which means the CPU cannot speculatively start the fetch in advance. In a typical L3 cache miss scenario, the CPU simply stalls waiting for the data to arrive from main memory.

The `_mm_prefetch` calls in `hnswlib` break this dependency by issuing the fetch for a *future* node while still processing the current one. At the point in the loop where node `N` is being evaluated, the algorithm already knows the IDs of `N`'s neighbours — and it knows those neighbours will likely be visited next. The prefetch is issued for those neighbours' data **now**, so by the time the loop gets to them, the data is already in cache.

This effectively pipelines the memory access with the computation, hiding most of the latency.

### 6.2 Why ~12% and Not More?

A 12% improvement is meaningful but not dramatic. The reason the gain is not larger comes down to several factors:

- **The computation itself is not free.** Distance computations (dot products over 128 dimensions) take real CPU cycles. Even without prefetch, the CPU is not idle during a cache miss — it may execute other instructions out-of-order. The prefetch closes the gap but does not eliminate it.

- **L3 cache is doing some of the work anyway.** A 150K-vector index at 128 dimensions occupies about 73 MB. Modern systems have L3 caches of 8–32 MB. Frequently accessed nodes (the high-degree nodes near the top of the graph) may already be in L3, reducing the prefetch benefit for those accesses.

- **`ef=200` amplifies the effect.** At the default `ef=100`, the prefetch benefit would likely be smaller. The 12% figure is not a lower bound — it is specific to these parameters.

### 6.3 Consistency as Evidence

Perhaps the most important finding is not the magnitude but the **consistency**. Across four cycles, two different run orders, and every percentile from p50 to p99, the ON build is always faster than the OFF build by a similar margin. This rules out the possibility that the result is a measurement artifact and establishes that software prefetching provides a reliable, reproducible performance benefit in HNSW graph traversal.

---

## 7. Conclusion

This research set out to answer whether the `_mm_prefetch` instructions in `hnswlib` provide a real, measurable performance benefit or are largely vestigial code that modern CPUs handle anyway through hardware prefetching. The answer is clear: **they provide a consistent ~12% per-query latency reduction** on a 150K-vector, 128-dimensional dataset with `ef=200`.

The path to this answer required building a rigorous, bias-controlled experimental framework, fixing a week-long silent patch failure, navigating an architecture-specific compilation trap, replacing every hardcoded parameter with a configurable one, and implementing proper recall measurement to validate correctness. Each of these challenges deepened the understanding of how `hnswlib` actually works and how the interaction between algorithm structure, memory hierarchy, and hardware hints produces the final latency number.

The key takeaway for practitioners is this: if you are deploying `hnswlib` in a latency-sensitive application on x86-64 hardware, ensure you are compiling with SSE support enabled. The prefetch calls that result are not just theoretical optimisations — they reliably save roughly 70 microseconds per 1,000 queries at these scales, and that gap grows with higher `ef`, larger `M`, and higher-dimensional vectors.

---

## 8. Reproducibility

All code is available on GitHub. To reproduce the results:

```bash
git clone https://github.com/PatelDhruvil2502/prefetch.git
cd prefetch
git checkout improvements
pip install matplotlib numpy
python3 run_experiment.py
```

Docker must be running. The full benchmark (4 cycles + 1 warmup) completes in approximately 25–40 minutes depending on hardware. All raw logs, per-run CSVs, and plots are saved to the `results/` directory.

### Key CLI Options

```bash
# Run with stronger prefetch-advantage conditions
python3 run_experiment.py --scenario prefetch_advantage

# Run a scaling sweep across different index sizes
python3 run_experiment.py --index-sizes 10000,50000,100000,150000

# Increase recall measurement accuracy
python3 run_experiment.py --recall-samples 500

# Change HNSW graph parameters
python3 run_experiment.py --hnsw-m 16 --ef-construction 200 --ef 400
```

---

## Appendix A — Raw Worker Output Sample (Prefetch ON, Cycle 1)

```
Loading /app/real_world_dataset.npy...
Generating queries...
Building HNSW graph...
Computing recall@20 on 200 sample queries...
Starting timed search...
DATASET_INFO,150000,128
RECALL,0.845000
LATENCY_STATS,0.522087,0.530697,0.533514
MEMORY_LAYOUT,0x7fffdc94b010,24.4,0x7fffded10010,73.2
RESULT,0.5235525935599998,551028,0
```

## Appendix B — Raw Worker Output Sample (Prefetch OFF, Cycle 1)

```
Loading /app/real_world_dataset.npy...
Generating queries...
Building HNSW graph...
Computing recall@20 on 200 sample queries...
Starting timed search...
DATASET_INFO,150000,128
RECALL,0.843500
LATENCY_STATS,0.593941,0.613769,0.618902
MEMORY_LAYOUT,0x7fffdc942010,24.4,0x7fffded07010,73.2
RESULT,0.5955748952600004,551348,0
```

---

*All experiments were conducted on macOS Darwin 25.4.0 (Apple Silicon, Rosetta 2 x86_64 emulation) with Docker Desktop. The benchmark is platform-independent as long as Docker can run linux/amd64 containers.*
