# Optimization Report

## Scope and methodology

This report analyzes the three required implementations of

```text
C = A^T * B
D = C * C^T
y = 0
for each row c_i of C:
    y += D * c_i
y += x
```

using the code in `solver_neopt.c`, `impl_variants/solver_neopt_equiv.c`, `impl_variants/solver_opt_base.c`, `impl_variants/solver_opt_inline.c`, `impl_variants/solver_opt_avx.c`, and `solver_blas.c`.

Runtime data comes from the `bench_results` CSV files and always uses the `mean` column, as requested. Low-level metrics come from the `cg_annotate` summaries stored under `results/**/cachegrind_results/`.

The benchmark command was:

```bash
<bench_program> -n <N> -s 1234 -w 10 -t 30 -a 16
```

The cachegrind command was:

```bash
valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes \
  --I1=32768,8,64 --D1=32768,8,64 --LL=20971520,20,64 \
  <bench_program> -n <N> -s 1234 -t 1 -a 16
```

The graph scripts are:

- `graphs/generate_runtime_graphs.py`
- `graphs/generate_microarch_graphs.py`

and they generate the SVG files referenced below in `graphs/`.

## Data validation

All runtime datasets are monotonic and usable for every `N`.

For cachegrind, I excluded the `neopt` data point at `N = 1400` from low-level graphs. Its instruction count is **120.50G**, which is lower than the `N = 1200` run at **127.99G** even though the workload is larger; that is not plausible for this cubic algorithm and is consistent with the prompt's warning that large unoptimized cachegrind runs may be truncated. I kept the `neopt_equiv` large-`N` cachegrind points because they remain monotonic and match the runtime trend.

## Code-level differences between versions

### `neopt`

`solver_neopt.c` is the literal implementation of the pseudocode. It:

- materializes `A^T` and `C^T` explicitly,
- performs dense `A^T * B`,
- computes only the upper triangle of `D`,
- but still applies `D * row(C, i)` separately for every row of `C`.

This preserves the original access pattern, but it is extremely expensive because it repeats a symmetric matrix-vector product `N` times.

### `neopt_equiv`

`impl_variants/solver_neopt_equiv.c` keeps the same straightforward coding style but uses

```text
D * c_0 + ... + D * c_{N-1} = D * (c_0 + ... + c_{N-1})
```

This is the correct baseline for the optimization path because it keeps the same high-level computation structure as the optimized versions.

### `opt_base`

`impl_variants/solver_opt_base.c` introduces the first real systems-level improvements:

- blocked matrix multiplication for `C = A^T * B`,
- `k-i-j` loop order so the inner loop streams through both `B` and `C`,
- contiguous allocation for the working buffers,
- packed storage for only the upper triangular part of `D`,
- pointer-walking instead of repeated index arithmetic,
- reuse of the last row of `C` to hold `sum_rows_c`.

This is the version where locality changes materially.

### `opt_inline`

`impl_variants/solver_opt_inline.c` keeps the same algorithm as `opt_base`, but forces the hot helpers to inline under `-O0`. The block-size study (`24`, `32`, `40`) also lives in this family; `BLOCK_SIZE = 32` is the main candidate because it matches the L1-based estimate

```text
floor_to_multiple_of_8(sqrt(32768 / (3 * 8))) = 32
```

### `opt_avx`

`impl_variants/solver_opt_avx.c` keeps the same blocked layout as the inline version and adds:

- AVX/FMA vectorization for dot products,
- AVX/FMA vectorization for AXPY-style updates inside the blocked multiply,
- AVX vectorization for row accumulation,
- 32-byte aligned allocation via `posix_memalign`.

This version attacks execution throughput rather than memory layout.

### `blas`

`solver_blas.c` delegates the heavy work to optimized kernels:

- `dgemm` for `A^T * B`,
- `dsyrk` for `C * C^T`,
- `dsymv` for the symmetric matrix-vector step,
- `daxpy` for `y += x`.

It is the reference for how far a hand-written solution still is from a tuned library.

## Optimization path: `neopt_equiv -> opt_base -> opt_inline -> opt_avx`

### 1. From `neopt_equiv` to `opt_base`

The biggest structural improvement is the move from naive dense traversal to blocked, cache-aware multiplication plus packed storage of `D`.

Runtime speedup:

- `2.73x` at `N=400`
- `2.50x` at `N=1000`
- `2.49x` at `N=1400`

Graph:

- [Runtime evolution](graphs/runtime_optimization_steps.svg)
- [Per-step speedups](graphs/step_speedups.svg)

The low-level counters explain this well. At `N=1000`:

- instruction references drop from `39.26G` to `25.27G` (`1.55x` fewer),
- D1 misses drop from `1.69B` to `78.9M` (`21.4x` fewer),
- last-level misses drop from `1.05M` to `0.89M`.

Graphs:

- [Cache metrics along the optimization path](graphs/cache_metrics_optimization.svg)
- [Instruction and branch metrics](graphs/instruction_branch_metrics.svg)

This is the clearest cache-locality win in the whole report. The blocked `k-i-j` multiply keeps the current `B[k][j]` strip and the updated `C[i][j]` block hot in L1/L2 much longer than the naive version. Packing `D` also avoids touching an almost-half-empty symmetric matrix representation.

One interesting detail is that `opt_base` increases branch activity relative to `neopt_equiv`. That does not contradict the runtime improvement: the added strip/block loops introduce more control flow, but the reduction in cache stalls is much more important than the extra branch work.

### 2. From `opt_base` to `opt_inline`

Inlining improves runtime consistently, but this step is smaller because it does not change data movement.

Using the `BLOCK_SIZE=32` build as the main inline candidate, the speedup over `opt_base` is:

- `1.33x` at `N=400`
- `1.36x` at `N=1000`
- `1.35x` at `N=1400`

The cache plots show almost no change in D1 or LL misses between `opt_base` and `opt_inline`, which is exactly what should happen: the memory access pattern is the same. The gain comes from fewer executed instructions, because the hot helper calls disappear even though the program is compiled with `-O0`. At `N=1400`, instruction references drop from `68.99G` to `52.92G`, a reduction of about `1.30x`, which matches the runtime improvement closely.

### 3. Choosing the block size inside `opt_inline`

The three inline variants confirm that `BLOCK_SIZE=32` is the best choice on the tested machine.

Average runtime improvement of `b=32`:

- over `b=24`: about `5.4%`
- over `b=40`: about `9.3%`

At `N=1400`:

- `b=24`: `3.919 s`
- `b=32`: `3.731 s`
- `b=40`: `4.083 s`

Graph:

- [Block-size comparison](graphs/opt_inline_block_sizes.svg)

This result matches the L1-based estimate better than either alternative. `b=24` leaves locality on the table because the working tiles are too small, so loop overhead becomes more visible. `b=40` goes in the other direction: it reduces the number of block iterations, but the tiles are large enough to push more pressure onto the effective L1 working set. Even though cachegrind does not fully penalize `b=40` in every metric, the native runtime benchmark consistently shows that `32` is the best balance on this Haswell system.

### 4. From `opt_inline` to `opt_avx`

This is the throughput optimization step. The memory layout barely changes, so the runtime gain must come mainly from SIMD.

Speedup of `opt_avx` over `opt_inline b=32`:

- `2.75x` at `N=400`
- `3.13x` at `N=1000`
- `3.00x` at `N=1400`

At `N=1000`, cachegrind shows:

- D1 misses are essentially unchanged (`78.9M` vs `78.3M`),
- LL misses are unchanged at about `0.89M`,
- instruction references drop from `19.26G` to `3.56G` (`5.41x` fewer),
- branch mispredicts drop from `33.01M` to `3.01M` (`10.95x` fewer).

This is exactly the signature expected from AVX/FMA: the algorithm does the same memory traffic, but each hot loop iteration performs much more arithmetic work per instruction and per branch. The vectorized `vec_dot`, `vec_axpy`, and `vec_add` helpers replace long scalar loops with wide fused multiply-add/update kernels.

## Final comparison of all required versions

The final runtime ordering is stable for every tested size:

```text
BLAS < opt_avx < neopt_equiv < neopt
```

Graph:

- [Final runtime comparison](graphs/runtime_all_versions.svg)

Key speedups:

- `neopt_equiv` vs `neopt`: about `1.94x` to `2.10x`
- `opt_avx` vs `neopt_equiv`: about `9.95x` to `11.55x`
- `opt_avx` vs `neopt`: about `19.31x` to `24.25x`
- `blas` vs `opt_avx`: about `2.19x` to `2.49x`

At `N=1400`, the mean runtimes are:

- `blas`: `0.518 s`
- `opt_avx`: `1.244 s`
- `neopt_equiv`: `12.535 s`
- `neopt`: `25.865 s`

Graphs:

- [Overall low-level metrics](graphs/overall_microarch_comparison.svg)

The comparison with `neopt` highlights why `neopt_equiv` is the right baseline for the optimization path: simply regrouping the symmetric matrix-vector phase already halves the runtime. The hardware-focused optimizations then build on top of that by improving locality, reducing scalar overhead, and finally exploiting AVX.

The comparison with BLAS shows a different lesson. `opt_avx` is much faster than the custom scalar code, but BLAS is still ahead by more than `2x`. That is expected. Vendor-tuned BLAS kernels combine stronger vectorization, better micro-kernel design, better packing, and decades of architecture-specific tuning. For example, at `N=1400` BLAS executes about `3.88G` instructions versus `10.34G` for `opt_avx`, and it also has far fewer branch mispredicts.

## Conclusions

The optimization story is consistent:

1. `neopt_equiv` removes redundant work and gives the fair baseline.
2. `opt_base` delivers the largest cache-locality improvement through blocking, loop reordering, contiguous buffers, and packed symmetric storage.
3. `opt_inline` removes call overhead under `-O0` and adds another steady `~1.35x`.
4. `opt_avx` keeps the same memory behavior but raises arithmetic throughput enough to add another `~3x`.

Overall, the final hand-written optimized version is about `10x` faster than the fair baseline (`neopt_equiv`) and about `20x` faster than the literal unoptimized implementation (`neopt`), while still remaining around `2.2x` to `2.5x` slower than BLAS.
