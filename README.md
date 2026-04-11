# Independent Study: HNSW Indexing with File-based Vector Data Storage

**Student:** Dhruvil Patel (dp86)
**Faculty Mentor:** Prof. Feng Chen
**Course:** CSCI-Y790, Spring 2026

This repository contains three experiments investigating HNSW performance under different conditions.

## Experiments

### 1. Memory Stress Experiment (`memory-stress-experiment/`)

Measures how HNSW query latency degrades as available RAM is progressively reduced below the index size. Uses Docker containers with capped memory to force OS paging and observe swap thrashing, tail-latency spikes, and eventual OOM crashes.

**Metrics collected:** mean latency, p50/p95/p99 tail latency, major/minor page faults, disk I/O, VmRSS/VmSwap.

```bash
cd memory-stress-experiment
python3 run_experiment.py
```

### 2. Prefetch Experiment (`prefetch-experiment/`)

Isolates the performance impact of `_mm_prefetch` SSE instructions in hnswlib's HNSW graph traversal. Builds two Docker images (prefetch ON vs OFF) and runs controlled A/B benchmarks.

**Key result:** ~12% per-query latency reduction from software prefetching, consistent across all tail percentiles.

```bash
cd prefetch-experiment
python3 run_experiment.py
```

### 3. Parameter Analysis Experiment (`parameter-analysis-experiment/`)

Sweeps over HNSW parameters (k, ef, M, dataset size, dimension) to show how each factor affects query latency. Also computes expected HNSW levels for M and N sweeps. Runs locally without Docker.

**Factors analyzed:** number of neighbors (k), search beam width (ef), graph connectivity (M), dataset size (N), vector dimension (d).

```bash
cd parameter-analysis-experiment
pip install hnswlib
python3 run_experiment.py
```

## Requirements

- Docker Desktop (with `linux/amd64` support / Rosetta 2 on Apple Silicon) — for experiments 1 & 2
- Python 3.10+ with `matplotlib`, `numpy`, and `hnswlib`
