from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


BONUS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BONUS_ROOT.parent
GRAPHS_DIR = BONUS_ROOT / "graphs"

ICE_BENCHMARK_FILES = {
    "blas": BONUS_ROOT / "results/blas/bench_results",
    "neopt": BONUS_ROOT / "results/neopt/bench_results",
    "neopt_equiv": BONUS_ROOT / "results/neopt_equiv/bench_results",
    "opt": BONUS_ROOT / "results/opt/bench_results",
}

HASWELL_BENCHMARK_FILES = {
    "blas": PROJECT_ROOT / "results/blas/bench_results",
    "neopt": PROJECT_ROOT / "results/neopt/bench_results",
    "neopt_equiv": PROJECT_ROOT / "results/neopt_equiv/bench_results",
    "opt": PROJECT_ROOT / "results/opt_avx/bench_results",
}

DISPLAY_NAMES = {
    "blas": "BLAS",
    "neopt": "neopt",
    "neopt_equiv": "neopt_equiv",
    "opt": "opt",
    "haswell": "Haswell",
    "ice_lake": "Ice Lake",
}

COLORS = {
    "blas": "#1f77b4",
    "neopt": "#8c564b",
    "neopt_equiv": "#d62728",
    "opt": "#9467bd",
    "haswell": "#ff7f0e",
    "ice_lake": "#2ca02c",
}


@dataclass(frozen=True)
class BenchmarkPoint:
    n: int
    runs: int
    mean: float
    stddev: float
    min: float
    max: float
    p50: float
    p90: float
    p99: float


def ensure_graphs_dir() -> Path:
    GRAPHS_DIR.mkdir(exist_ok=True)
    return GRAPHS_DIR


def _load_table(path: Path) -> dict[int, BenchmarkPoint]:
    with path.open() as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["N"]): BenchmarkPoint(
                n=int(row["N"]),
                runs=int(row["runs"]),
                mean=float(row["mean"]),
                stddev=float(row["stddev"]),
                min=float(row["min"]),
                max=float(row["max"]),
                p50=float(row["p50"]),
                p90=float(row["p90"]),
                p99=float(row["p99"]),
            )
            for row in rows
        }


def load_ice_benchmarks() -> dict[str, dict[int, BenchmarkPoint]]:
    return {impl: _load_table(path) for impl, path in ICE_BENCHMARK_FILES.items()}


def load_haswell_benchmarks() -> dict[str, dict[int, BenchmarkPoint]]:
    return {impl: _load_table(path) for impl, path in HASWELL_BENCHMARK_FILES.items()}


def ns_to_seconds(value: float) -> float:
    return value / 1_000_000_000.0
