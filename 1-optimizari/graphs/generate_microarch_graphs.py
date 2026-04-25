from __future__ import annotations

import matplotlib.pyplot as plt

from common import (
    COLORS,
    DISPLAY_NAMES,
    branch_mispredicts,
    ensure_graphs_dir,
    ll_misses,
    load_cachegrind,
)


def plot_cache_metrics(cachegrind):
    implementations = ["neopt_equiv", "opt_base", "opt_inline_b32", "opt_avx"]
    metrics = [
        ("D1mr", "D1 misses"),
        ("ll", "Last-level cache misses"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)

    for axis, (metric_key, title) in zip(axes, metrics, strict=True):
        for impl in implementations:
            sizes = sorted(cachegrind[impl])
            if metric_key == "ll":
                values = [ll_misses(cachegrind[impl][n]) for n in sizes]
            else:
                values = [cachegrind[impl][n][metric_key] for n in sizes]
            axis.plot(
                sizes,
                values,
                marker="o",
                linewidth=2,
                color=COLORS[impl],
                label=DISPLAY_NAMES[impl],
            )
        axis.set_title(title)
        axis.set_xlabel("N")
        axis.set_yscale("log")
        axis.grid(True, which="both", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Events")
    axes[1].legend()
    fig.suptitle("Cache Behavior Along the Optimization Path")
    fig.tight_layout()
    fig.savefig(ensure_graphs_dir() / "cache_metrics_optimization.svg")
    plt.close(fig)


def plot_instruction_branch_metrics(cachegrind):
    implementations = ["neopt_equiv", "opt_base", "opt_inline_b32", "opt_avx"]
    metrics = [
        ("Ir", "Instruction references"),
        ("Dr", "Data references"),
        ("Bc", "Conditional branches"),
        ("bm", "Branch mispredicts"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    flat_axes = axes.flatten()

    for axis, (metric_key, title) in zip(flat_axes, metrics, strict=True):
        for impl in implementations:
            sizes = sorted(cachegrind[impl])
            if metric_key == "bm":
                values = [branch_mispredicts(cachegrind[impl][n]) for n in sizes]
            else:
                values = [cachegrind[impl][n][metric_key] for n in sizes]
            axis.plot(
                sizes,
                values,
                marker="o",
                linewidth=2,
                color=COLORS[impl],
                label=DISPLAY_NAMES[impl],
            )
        axis.set_title(title)
        axis.set_xlabel("N")
        axis.set_yscale("log")
        axis.grid(True, which="both", linestyle="--", alpha=0.35)

    flat_axes[0].set_ylabel("Events")
    flat_axes[2].set_ylabel("Events")
    flat_axes[1].legend()
    fig.suptitle("Instruction and Branch Metrics Along the Optimization Path")
    fig.tight_layout()
    fig.savefig(ensure_graphs_dir() / "instruction_branch_metrics.svg")
    plt.close(fig)


def plot_overall_microarch(cachegrind):
    implementations = ["blas", "opt_avx", "neopt_equiv", "neopt"]
    metrics = [
        ("Ir", "Instruction references"),
        ("D1mr", "D1 misses"),
        ("ll", "Last-level cache misses"),
        ("bm", "Branch mispredicts"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    flat_axes = axes.flatten()

    for axis, (metric_key, title) in zip(flat_axes, metrics, strict=True):
        for impl in implementations:
            sizes = sorted(cachegrind[impl])
            if metric_key == "ll":
                values = [ll_misses(cachegrind[impl][n]) for n in sizes]
            elif metric_key == "bm":
                values = [branch_mispredicts(cachegrind[impl][n]) for n in sizes]
            else:
                values = [cachegrind[impl][n][metric_key] for n in sizes]
            axis.plot(
                sizes,
                values,
                marker="o",
                linewidth=2,
                color=COLORS[impl],
                label=DISPLAY_NAMES[impl],
            )
        axis.set_title(title)
        axis.set_xlabel("N")
        axis.set_yscale("log")
        axis.grid(True, which="both", linestyle="--", alpha=0.35)

    flat_axes[0].set_ylabel("Events")
    flat_axes[2].set_ylabel("Events")
    flat_axes[1].legend()
    fig.suptitle("Overall Low-Level Metrics")
    fig.tight_layout()
    fig.savefig(ensure_graphs_dir() / "overall_microarch_comparison.svg")
    plt.close(fig)


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    cachegrind = load_cachegrind()
    plot_cache_metrics(cachegrind)
    plot_instruction_branch_metrics(cachegrind)
    plot_overall_microarch(cachegrind)


if __name__ == "__main__":
    main()
