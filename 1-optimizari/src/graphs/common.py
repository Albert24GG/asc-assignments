from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
GRAPHS_DIR = ROOT / "graphs"

BENCHMARK_FILES = {
    "blas": ROOT / "results/blas/bench_results",
    "neopt": ROOT / "results/neopt/bench_results",
    "neopt_equiv": ROOT / "results/neopt_equiv/bench_results",
    "opt_base": ROOT / "results/opt_base/bench_results",
    "opt_inline_b24": ROOT / "results/opt_inline/block24/bench_results",
    "opt_inline_b32": ROOT / "results/opt_inline/block32/bench_results",
    "opt_inline_b40": ROOT / "results/opt_inline/block40/bench_results",
    "opt_avx": ROOT / "results/opt_avx/bench_results",
}

CACHEGRIND_GLOBS = {
    "blas": "results/blas/cachegrind_results/n*/cachegrind_blas_n*",
    "neopt": "results/neopt/cachegrind_results/n*/cachegrind_neopt_n*",
    "neopt_equiv": "results/neopt_equiv/cachegrind_results/n*/cachegrind_neopt_equiv_n*",
    "opt_base": "results/opt_base/cachegrind_results/n*/cachegrind_opt_base_n*",
    "opt_inline_b24": "results/opt_inline/block24/cachegrind_results/n*/cachegrind_opt_inline_block24_n*",
    "opt_inline_b32": "results/opt_inline/block32/cachegrind_results/n*/cachegrind_opt_inline_block32_n*",
    "opt_inline_b40": "results/opt_inline/block40/cachegrind_results/n*/cachegrind_opt_inline_block40_n*",
    "opt_avx": "results/opt_avx/cachegrind_results/n*/cachegrind_opt_avx_n*",
}

DISPLAY_NAMES = {
    "blas": "BLAS",
    "neopt": "neopt",
    "neopt_equiv": "neopt_equiv",
    "opt_base": "opt_base",
    "opt_inline_b24": "opt_inline b=24",
    "opt_inline_b32": "opt_inline b=32",
    "opt_inline_b40": "opt_inline b=40",
    "opt_avx": "opt_avx",
}

COLORS = {
    "blas": "#1f77b4",
    "neopt": "#8c564b",
    "neopt_equiv": "#d62728",
    "opt_base": "#ff7f0e",
    "opt_inline_b24": "#bcbd22",
    "opt_inline_b32": "#2ca02c",
    "opt_inline_b40": "#17becf",
    "opt_avx": "#9467bd",
}

CACHEGRIND_LABELS = [
    "Ir",
    "I1mr",
    "ILmr",
    "Dr",
    "D1mr",
    "DLmr",
    "Dw",
    "D1mw",
    "DLmw",
    "Bc",
    "Bcm",
    "Bi",
    "Bim",
]

INVALID_CACHE_POINTS = {
    ("neopt", 1400): "Excluded because Ir is lower than the N=1200 run, which is impossible for a larger cubic workload and indicates truncated output.",
}

SUMMARY_NUMBER_RE = re.compile(r"([0-9,]+) \(100\.0%\)")


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


def load_benchmarks() -> dict[str, dict[int, BenchmarkPoint]]:
    result: dict[str, dict[int, BenchmarkPoint]] = {}
    for impl, path in BENCHMARK_FILES.items():
        with path.open() as handle:
            rows = csv.DictReader(handle)
            result[impl] = {
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
    return result


def load_cachegrind() -> dict[str, dict[int, dict[str, int]]]:
    result: dict[str, dict[int, dict[str, int]]] = {}
    for impl, rel_glob in CACHEGRIND_GLOBS.items():
        points: dict[int, dict[str, int]] = {}
        for path in sorted(ROOT.glob(rel_glob)):
            match = re.search(r"n(\d+)$", path.name)
            if match is None:
                continue
            n = int(match.group(1))
            if (impl, n) in INVALID_CACHE_POINTS:
                continue
            summary = extract_cachegrind_summary(path)
            if summary is not None:
                points[n] = summary
        result[impl] = points
    return result


def extract_cachegrind_summary(path: Path) -> dict[str, int] | None:
    text = path.read_text(errors="ignore")
    summary_line = next((line for line in text.splitlines() if "PROGRAM TOTALS" in line), None)
    if summary_line is None:
        return None
    numbers = [int(value.replace(",", "")) for value in SUMMARY_NUMBER_RE.findall(summary_line)]
    if len(numbers) != len(CACHEGRIND_LABELS):
        return None
    return dict(zip(CACHEGRIND_LABELS, numbers, strict=True))


def ns_to_seconds(value: float) -> float:
    return value / 1_000_000_000.0


def ll_misses(point: dict[str, int]) -> int:
    return point["DLmr"] + point["DLmw"]


def branch_mispredicts(point: dict[str, int]) -> int:
    return point["Bcm"] + point["Bim"]
