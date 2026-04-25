from __future__ import annotations

import matplotlib.pyplot as plt

from common import (
    COLORS,
    DISPLAY_NAMES,
    ensure_graphs_dir,
    load_haswell_benchmarks,
    load_ice_benchmarks,
    ns_to_seconds,
)


def plot_ice_runtime_comparison(benchmarks):
    sizes = sorted(benchmarks["blas"])
    implementations = ["blas", "opt", "neopt_equiv", "neopt"]

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

    ax.set_title("Ice Lake Runtime Comparison")
    ax.set_xlabel("N")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_graphs_dir() / "runtime_ice_lake.svg")
    plt.close(fig)


def plot_architecture_comparison(ice, haswell, impl):
    sizes = sorted(set(ice[impl]) & set(haswell[impl]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        sizes,
        [ns_to_seconds(haswell[impl][n].mean) for n in sizes],
        marker="o",
        linewidth=2,
        color=COLORS["haswell"],
        label=DISPLAY_NAMES["haswell"],
    )
    ax.plot(
        sizes,
        [ns_to_seconds(ice[impl][n].mean) for n in sizes],
        marker="o",
        linewidth=2,
        color=COLORS["ice_lake"],
        label=DISPLAY_NAMES["ice_lake"],
    )

    ax.set_title(f"{DISPLAY_NAMES[impl]}: Haswell vs Ice Lake")
    ax.set_xlabel("N")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_graphs_dir() / f"arch_compare_{impl}.svg")
    plt.close(fig)


def plot_speedup_comparison(ice, haswell):
    comparisons = [
        ("neopt_equiv", "neopt", "neopt_equiv vs neopt"),
        ("opt", "neopt_equiv", "opt vs neopt_equiv"),
        ("blas", "opt", "blas vs opt"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for axis, (faster, slower, title) in zip(axes, comparisons, strict=True):
        sizes = sorted(set(ice[faster]) & set(ice[slower]) & set(haswell[faster]) & set(haswell[slower]))
        haswell_speedup_pct = [
            (haswell[slower][n].mean / haswell[faster][n].mean - 1.0) * 100.0 for n in sizes
        ]
        ice_speedup_pct = [
            (ice[slower][n].mean / ice[faster][n].mean - 1.0) * 100.0 for n in sizes
        ]

        axis.plot(
            sizes,
            haswell_speedup_pct,
            marker="o",
            linewidth=2,
            color=COLORS["haswell"],
            label=DISPLAY_NAMES["haswell"],
        )
        axis.plot(
            sizes,
            ice_speedup_pct,
            marker="o",
            linewidth=2,
            color=COLORS["ice_lake"],
            label=DISPLAY_NAMES["ice_lake"],
        )
        axis.set_title(title)
        axis.set_ylabel("Speedup (%)")
        axis.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("N")
    axes[0].legend()
    fig.suptitle("Speedup Comparison: Haswell vs Ice Lake")
    fig.tight_layout()
    fig.savefig(ensure_graphs_dir() / "speedup_compare.svg")
    plt.close(fig)


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    ice = load_ice_benchmarks()
    haswell = load_haswell_benchmarks()

    plot_ice_runtime_comparison(ice)
    for impl in ["blas", "neopt", "neopt_equiv", "opt"]:
        plot_architecture_comparison(ice, haswell, impl)
    plot_speedup_comparison(ice, haswell)


if __name__ == "__main__":
    main()
