"""
generate_figure.py — reads results/single_query_results.csv produced by
run_experiment.py and generates a bar-chart figure for the report.

Usage:
    python3 generate_figure.py [--results-dir results] [--out results/single_query_figure.png]
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(csv_path):
    """Load per-query rows; skip OVERALL summary row."""
    rows = []
    overall = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["query_num"] == "OVERALL":
                overall = {
                    "latency_on_ms":  float(r["latency_on_ms"]),
                    "latency_off_ms": float(r["latency_off_ms"]),
                    "delta_ms":       float(r["delta_ms"]),
                    "speedup_pct":    float(r["speedup_pct"]),
                }
            else:
                rows.append({
                    "query_num":      int(r["query_num"]),
                    "dataset_idx":    r["dataset_idx"],
                    "latency_on_ms":  float(r["latency_on_ms"]),
                    "latency_off_ms": float(r["latency_off_ms"]),
                    "delta_ms":       float(r["delta_ms"]),
                    "speedup_pct":    float(r["speedup_pct"]),
                    "same_order":     r["same_order"],
                    "overlap_pct":    float(r["overlap_pct"]) if r["overlap_pct"] else 100.0,
                })
    return rows, overall


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure(rows, overall, out_path):
    n = len(rows)
    if n == 0:
        print("ERROR: No per-query rows found in CSV.")
        sys.exit(1)

    color_on  = "#1565C0"   # deep blue  → prefetch ON
    color_off = "#B71C1C"   # deep red   → prefetch OFF
    color_delta = "#2E7D32" # green      → speedup delta bar

    x = np.arange(n)
    width = 0.35

    # ── Layout: 2 rows
    #   Top: grouped bar chart (ON vs OFF latency per query)
    #   Bottom: delta bar chart (OFF − ON)
    fig, (ax_main, ax_delta) = plt.subplots(
        2, 1, figsize=(max(8, 3 * n), 10),
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    lats_on  = [r["latency_on_ms"]  for r in rows]
    lats_off = [r["latency_off_ms"] for r in rows]
    deltas   = [r["delta_ms"]       for r in rows]   # positive = OFF slower = prefetch helps
    labels   = [f"Q{r['query_num']}\n(idx {r['dataset_idx']})" for r in rows]

    # ── Top panel: grouped bars
    bars_on  = ax_main.bar(x - width / 2, lats_on,  width, label="Prefetch ON",  color=color_on,  alpha=0.85)
    bars_off = ax_main.bar(x + width / 2, lats_off, width, label="Prefetch OFF", color=color_off, alpha=0.85)

    # Annotate exact values above bars
    for bar in bars_on:
        h = bar.get_height()
        ax_main.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=color_on, fontweight="bold")
    for bar in bars_off:
        h = bar.get_height()
        ax_main.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=color_off, fontweight="bold")

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(labels, fontsize=9)
    ax_main.set_ylabel("Latency (ms)", fontweight="bold")
    ax_main.set_title(
        "Single-Query Latency: Prefetch ON vs Prefetch OFF\n"
        "Each bar = one query issued to a freshly built HNSW index (no batch warm-up)",
        fontsize=12, fontweight="bold",
    )
    ax_main.legend(fontsize=10)
    ax_main.grid(True, axis="y", alpha=0.3)
    ax_main.set_xlim(-0.6, n - 0.4)

    # Overall mean line
    if overall:
        ax_main.axhline(overall["latency_on_ms"],  color=color_on,  linestyle="--", linewidth=1.2,
                        label=f"Mean ON  {overall['latency_on_ms']:.3f} ms")
        ax_main.axhline(overall["latency_off_ms"], color=color_off, linestyle="--", linewidth=1.2,
                        label=f"Mean OFF {overall['latency_off_ms']:.3f} ms")
        ax_main.legend(fontsize=9)

    # ── Bottom panel: delta bars (OFF − ON)
    bar_colors = [color_delta if d >= 0 else color_off for d in deltas]
    ax_delta.bar(x, deltas, width=0.6, color=bar_colors, alpha=0.85)
    for xi, d in zip(x, deltas):
        ax_delta.text(xi, d + (0.0005 if d >= 0 else -0.001),
                      f"{d:+.3f}", ha="center",
                      va="bottom" if d >= 0 else "top",
                      fontsize=8, fontweight="bold")
    ax_delta.axhline(0, color="black", linewidth=0.8)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(labels, fontsize=9)
    ax_delta.set_ylabel("Δ Latency\n(OFF − ON, ms)", fontweight="bold")
    ax_delta.set_title("Latency Saved by Prefetch per Query  (positive = prefetch helped)", fontsize=10)
    ax_delta.grid(True, axis="y", alpha=0.3)
    ax_delta.set_xlim(-0.6, n - 0.4)

    # Green = helped, red = hurt
    patch_help = mpatches.Patch(color=color_delta, alpha=0.85, label="Prefetch helped (OFF > ON)")
    patch_hurt = mpatches.Patch(color=color_off,   alpha=0.85, label="Prefetch hurt   (OFF < ON)")
    ax_delta.legend(handles=[patch_help, patch_hurt], fontsize=8, loc="upper right")

    # ── Overall summary text box
    if overall:
        pct = overall["speedup_pct"]
        direction = "slower" if pct > 0 else "faster"
        summary = (
            f"Overall mean  ON : {overall['latency_on_ms']:.4f} ms\n"
            f"Overall mean OFF : {overall['latency_off_ms']:.4f} ms\n"
            f"OFF vs ON : {pct:+.2f}%  ({direction} without prefetch)"
        )
        ax_delta.text(
            0.99, 0.05, summary, transform=ax_delta.transAxes,
            fontsize=8, va="bottom", ha="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
        )

    fig.tight_layout(pad=2.0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate single-query prefetch figure from saved CSV results."
    )
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing single_query_results.csv (default: results/)")
    parser.add_argument("--out", default="",
                        help="Output PNG path (default: <results-dir>/single_query_figure.png)")
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "single_query_results.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run run_experiment.py first — it saves results automatically after each session.")
        sys.exit(1)

    out_path = args.out or os.path.join(args.results_dir, "single_query_figure.png")
    rows, overall = load_results(csv_path)
    print(f"Loaded {len(rows)} per-query result(s) from {csv_path}")
    plot_figure(rows, overall, out_path)


if __name__ == "__main__":
    main()
