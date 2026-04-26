# Bonus implementation for UCSX

## UCSX Setup

- Model: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz 
- Architecture: Ice Lake
- Base frequency: 2.90 GHz
- Boost frequency: 3.50 GHz
- Sockets: 2
- Number of cores per socket: 16
- Threads per core: 2
- L1d cache: 48KiB per core => 12-way associative, 64 sets
- L1i cache: 32KiB per core => 8-way associative, 64 sets
- L2 cache: 1.3MiB per core => 20-way associative, 1024 sets
- L3 cache: 24MiB shared between all cores of a socket => 12-way associative, 32768 sets
- RAM size: 512 GiB
- RAM frequency and type: Not found

- Compiler: GCC 15.2
- LIBC: musl 1.2.5
- BLAS: OpenBLAS 0.3.30

## Implementation details

I started from the implementation established for haswell. The `blas`, `neopt`, and `neopt_equiv` variants are the same as before. The `opt` variant (which is the vectorized version) has to be modified slightly so that it takes full advantage of the capabilities of the new architecture.

What gets changed:
- tile block size
- avx instructions and unrolling factor
- importance of data alignment

### Tile block size

Since now the L1d cache is 48KB, we can fit more data in it. As we already know that the theoretical formula for the block size is accurate in practice, we can just plug in the new cache size.

```text
block_size = sqrt(L1d_cache_size / (sizeof(double) * 3)) = sqrt(48KB / (8B * 3)) = 45.25
```

Because we are working with vectors of 8 doubles, we need to round down the block size to a multiple of 8, so we get a block size of 40.

### AVX instructions and unrolling factor

The new architecture supports AVX512, which means that we can use 512-bit wide vector registers. That allows us to process 8 doubles at a time instead of 4.

The AVX512 instructions are a bit more expensive than the AVX2 ones. For example, both `_mm512_fmadd_pd` and `_mm512_add_pd` instructions have a latency of 11 and a reciprocal throughput of 1. Therefore, we cannot be as aggressive with the unrolling factor, especially in our functions where the pattern is `Load-Compute[-Store]`, otherwise we are going to hit a bottleneck in the memory bandwidth, and possibly oversaturate the decoded instruction buffer.

Because the general consensus is to choose unroll factors that are powers of 2, I tested both 4 and 8. The results showed that 4 generally performs better, which confirms the analysis above.

For `vec_axpy`, which is used in the tiled matrix multiplication, going from an unroll factor of 4 (and tile size of 32) to an unroll factor of 5 (and tile size of 40) actually brings visible performance improvements, so I decided to go with this one.


### Importance of data alignment

AVX512, like AVX2, works best when the data is aligned to the size of the vector registers, which is 64 bytes for AVX512. Although the `_mm512_loadu_pd` and `_mm512_storeu_pd` intrinsics can handle unaligned data, they benefit when the data is aligned. AVX512 presents an even higher sensitivity to data alignment than AVX2, because any unaligned access guarantees that 2 cache lines are going to be fetches, whereas with AVX2 only in 50% of the cases an unaligned access would cross a cache line boundary. Therefore, I made sure that all the data, including the input matrices `A` and `B` (which I copied to aligned buffers), is aligned to 64 bytes by using `posix_memalign`.


## Testing methodology

I used the exact same testing methodology as for the haswell implementation, which is described in the `README.md` file in the main directory. All commands and parameters are identical.


## Results and Graphs

This section contains the analysis of the results obtained for the Ice Lake implementation, available in the `results/` directory, as well as the comparison with the results obtained for the Haswell implementation. 

The graphs and parts of the analysis are once again generated using GPT-5.4 (via Codex), using the prompt from `prompt2.txt`. Its full response is available in `REPORT.md`.


### UCSX Results

Because valgrind does not support binaries compiled for AVX512-enabled architectures, the `-march=native` flag prevents me from running `cachegrind` on these binaries. Technically, I could not use `-march=native` on the unoptimized versions, but that would possibly prohibit the use of some hardware-specific instructions. To keep the process consistent and the comparison with Haswell fair, I decided to keep the `-march=native` flag and skip the `cachegrind` analysis for this implementation altogether. 

Graph:

- [Ice Lake runtime comparison](graphs/runtime_ice_lake.svg)


The ranking is stable for all tested sizes:

```text
BLAS < opt < neopt_equiv < neopt
```

At `N=1400`, the mean runtimes are:

- `blas`: `0.534 s`
- `opt`: `1.011 s`
- `neopt_equiv`: `12.007 s`
- `neopt`: `24.853 s`


As we can see, the graphs very closely resemble the ones obtained for the Haswell implementation.

The plots once again prove that the time complexity of the entire algorithm is dominated by the matrix-matrix multiplication, since the gap between the plots remain relatively constant as the input size increases.

The most significant improvement is in the `opt` version, which was to be expected given the new architectural improvements. As a consequence, the gap between `opt` and `blas` shrinked compared to the Haswell implementation.

### Comparison with Haswell

The comparisons are based solely on the execution times, using the same exact methodology for both implementations.

- [BLAS: Haswell vs Ice Lake](graphs/arch_compare_blas.svg)
- [neopt: Haswell vs Ice Lake](graphs/arch_compare_neopt.svg)
- [neopt_equiv: Haswell vs Ice Lake](graphs/arch_compare_neopt_equiv.svg)
- [opt: Haswell vs Ice Lake](graphs/arch_compare_opt.svg)


In this section, `opt` on Ice Lake is compared with `opt_avx` from Haswell, as requested.

### Speedup comparison between architectures

To make the cross-architecture effect more explicit, it is useful to compare the *internal* speedups of each platform:

- `neopt_equiv` relative to `neopt`
- `opt` relative to `neopt_equiv`
- `blas` relative to `opt`

Here I express speedup as a percentage improvement over the slower version:

```text
speedup_pct = (runtime_slower / runtime_faster - 1) * 100
```

Graph:

- [Speedup comparison: Haswell vs Ice Lake](graphs/speedup_compare.svg)

Table:

| N | `neopt_equiv vs neopt` Haswell | `neopt_equiv vs neopt` Ice Lake | `opt vs neopt_equiv` Haswell | `opt vs neopt_equiv` Ice Lake | `blas vs opt` Haswell | `blas vs opt` Ice Lake |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 94.1% | 101.8% | 895.2% | 1331.9% | 149.1% | 118.9% |
| 600 | 99.5% | 98.3% | 870.1% | 1120.1% | 149.2% | 92.9% |
| 800 | 109.9% | 104.9% | 1055.1% | 1148.5% | 119.3% | 75.8% |
| 1000 | 104.7% | 106.3% | 965.8% | 1089.2% | 130.2% | 88.0% |
| 1200 | 104.8% | 107.0% | 935.7% | 1097.5% | 135.6% | 87.4% |
| 1400 | 106.3% | 107.0% | 907.8% | 1087.2% | 140.0% | 89.4% |

Three patterns stand out.

### 1. `neopt_equiv vs neopt` is almost architecture-independent

The percentage gain stays close to `~100%` on both machines. That is the expected result, because this improvement comes from regrouping the final symmetric matrix-vector phase and roughly halving the amount of work there. Since this is mostly an algorithm-structure effect, the relative gain is nearly the same on Haswell and Ice Lake.

### 2. `opt vs neopt_equiv` is stronger on Ice Lake

This is where Ice Lake clearly pulls ahead. The speedup grows from about `8.7x` to `11.6x` on Haswell (`870%` to `1055%`) to about `11.9x` to `14.3x` on Ice Lake (`1087%` to `1332%`). In other words, the optimized code benefits more from the newer machine than the scalar fair baseline does.

That matches the code and the hardware:

- AVX-512 doubles vector width relative to AVX2,
- the larger L1 supports `BLOCK_SIZE=40`,
- the much larger L2 helps blocked reuse,
- 64-byte alignment makes the wide vector loads/stores more effective, although this is less of an issue for the AVX2 version.

### 3. `blas vs opt` is smaller on Ice Lake

The gap between BLAS and the hand-written optimized version shrinks on Ice Lake. On Haswell, BLAS is roughly `119%` to `149%` faster than `opt_avx`; on Ice Lake, the gap is only `76%` to `119%`.

This does **not** mean BLAS got worse. It means the custom optimized version improved more than BLAS when moving from Haswell to Ice Lake. That is consistent with the earlier architecture comparison: the naive versions barely improved, BLAS improved only modestly, but the AVX-enabled optimized version gained the most from the new vector width and larger caches.
