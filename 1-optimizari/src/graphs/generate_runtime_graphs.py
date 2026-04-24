from __future__ import annotations

import matplotlib.pyplot as plt

from common import COLORS, DISPLAY_NAMES, ensure_graphs_dir, load_benchmarks, ns_to_seconds


def plot_runtime_optimization_steps(benchmarks):
    sizes = sorted(benchmarks["neopt_equiv"])
    implementations = [
        "neopt_equiv",
        "opt_base",
        "opt_inline_b24",
        "opt_inline_b32",
        "opt_inline_b40",
        "opt_avx",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for impl in implementations:
        ax.plot(
            sizes,
            [ns_to_seconds(benchmarks[impl][n].mean) for n in sizes],
            marker="o",
            linewidth=2,
            color=COLORS[impl],
            label=DISPLAY_NAMES[impl],
        )

    ax.set_title("Runtime Evolution Across Optimization Steps")
    ax.set_xlabel("N")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_graphs_dir() / "runtime_optimization_steps.svg")
    plt.close(fig)


def plot_step_speedups(benchmarks):
    sizes = sorted(benchmarks["neopt_equiv"])
    speedup_vs_equiv = [
        benchmarks["neopt_equiv"][n].mean / benchmarks["opt_base"][n].mean for n in sizes
    ]
    speedup_vs_base = [
        benchmarks["opt_base"][n].mean / benchmarks["opt_inline_b32"][n].mean for n in sizes
    ]
    speedup_vs_inline = [
        benchmarks["opt_inline_b32"][n].mean / benchmarks["opt_avx"][n].mean for n in sizes
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, speedup_vs_equiv, marker="o", linewidth=2, color=COLORS["opt_base"], label="opt_base vs neopt_equiv")
    ax.plot(sizes, speedup_vs_base, marker="o", linewidth=2, color=COLORS["opt_inline_b32"], label="opt_inline b=32 vs opt_base")
    ax.plot(sizes, speedup_vs_inline, marker="o", linewidth=2, color=COLORS["opt_avx"], label="opt_avx vs opt_inline b=32")

    ax.set_title("Per-Step Speedups Along the Chosen Path")
    ax.set_xlabel("N")
    ax.set_ylabel("Speedup (x)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_graphs_dir() / "step_speedups.svg")
    plt.close(fig)


def plot_block_sizes(benchmarks):
    sizes = sorted(benchmarks["opt_inline_b24"])
    implementations = ["opt_inline_b24", "opt_inline_b32", "opt_inline_b40"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for impl in implementations:
        ax.plot(
            sizes,
            [ns_to_seconds(benchmarks[impl][n].mean) for n in sizes],
            marker="o",
            linewidth=2,
            color=COLORS[impl],
            label=DISPLAY_NAMES[impl],
        )

    ax.set_title("Blocked Inline Variant: Runtime by Block Size")
    ax.set_xlabel("N")
    ax.set_ylabel("Mean runtime (s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_graphs_dir() / "opt_inline_block_sizes.svg")
    plt.close(fig)


def plot_runtime_all_versions(benchmarks):
    sizes = sorted(benchmarks["blas"])
    implementations = ["blas", "opt_avx", "neopt_equiv", "neopt"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for impl in implementations:
        ax.plot(
            sizes,
            [ns_to_seconds(benchmarks[impl][n].mean) for n in sizes],
            marker="o",
            linewidth=2,
            color=COLORS[impl],
            label=DISPLAY_NAMES[impl],
        )

    ax.set_title("Final Runtime Comparison")
    ax.set_xlabel("N")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_graphs_dir() / "runtime_all_versions.svg")
    plt.close(fig)


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    benchmarks = load_benchmarks()
    plot_runtime_optimization_steps(benchmarks)
    plot_step_speedups(benchmarks)
    plot_block_sizes(benchmarks)
    plot_runtime_all_versions(benchmarks)


if __name__ == "__main__":
    main()
