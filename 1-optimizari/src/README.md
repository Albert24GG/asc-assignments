
## Haswell specifications

- Model: Intel(R) Xeon(R) CPU E5-2640 v3
- Base frequency: 2.60GHz
- Boost frequency: 3.40GHz
- Socket: 2
- Number of cores per socket: 8
- Threads per core: 2
- L1d cache: 32KB per core => 8-way associative, 64 sets
- L1i cache: 32KB per core => 8-way associative, 64 sets
- L2 cache: 256KB per core => 8-way associative, 512 sets
- L3 cache: 20MB shared between all cores of a socket => 20-way associative, 16384 sets


## Benchmarking methodology

In order to benchmark the performance of the implementation, without introducing noise of reading/writing from/to files, I wrote a separate program (`benchmark.c`) that does exactly the same thing as the main program, but more focused on benchmarking: generate random matrices, optionally run some warmup iterations (e.g. to make sure all libs/instructions are loaded in memory), and then run the solver for the specified number of times, measuring the execution time for each call. At the end, it calculates some statistics (average, median, standard deviation, percentiles) and prints them to the console in csv format. The benchmark program also has the option to pin the execution to a specific CPU core, which is useful for avoiding noise. For example, running on core 0 seems to generally be a [bad idea](https://kish1n.io/posts/is-core-0-sabotaging-your-performance/).


``` bash
./bench -h
Usage:
Benchmark for my_solver

    -h, --help            show this help message and exit
    -n, --size=<int>      size of the matrices
    -s, --seed=<int>      random seed for data generation, default: 0
    -w, --warmup=<int>    number of warmup runs, default: 0
    -t, --test=<int>      number of test runs, default: 1
    -a, --affinity=<int>  CPU affinity for the benchmark, default: 0
    -p, --print-times     print individual run times, default: false
```


In the following sections, any use of the benchmark command will be use the following arguments, unless specified otherwise:
```bash
<bench_program> -n <N> -s 1234 -w 10 -t 30 -a 16

```

In other words, the benchmark will be run with matrices of size `N`, a random seed of `1234`, 10 warmup runs, 30 test runs, and pinned to CPU core 16.

## Valgrind command

For detailed performance analysis, I used the following command:

```bash
valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes --collect-jumps=yes --dump-instr=yes --I1=32768,8,64 --D1=32768,8,64 --LL=20971520,20,64 <benchmark_command>
```

The benchmark command arguments used for running callgrind are the following:
```bash
<bench_program> -n <N> -s 1234 -t 1 -a 16
```

Even if valgrind does not care about cpu affinity, I still set it to 16, to be consistent with the benchmarking methodology.

This is not a perfect simulation of the haswell architecture, since valgrind runs in its own environment, but is the closest I could get without access to `perf`.

## Implementation details

I first started by implementing the unoptimized version. It it the simplest way to implement the provided algorithm. The only 2 "optimizations" that I did were:

1. Take advantage of the distributive property of `D * c_0 + D * c_1 + ... + D * c_n` to compute `D * (c_0 + c_1 + ... + c_n)` (`c_i` = `row(c, i)`). This decreases the overall complexity of the operation from `O(n^3)` to `O(n^2)`. The overall complexity of the algorithm is still `O(n^3)` because of the initial matrix multiplications;

2. Use the symmetry of the matrix `D` and compute only its upper triangular part. Because of this, when computing `D * (c_0 + c_1 + ... + c_n)`, the resulting elements corresponding to `D[i][j]` and `D[j][i]` are computed at the same time.


The blas version again just implements the raw algorithm, without any mathematical trick other than D's symmetry:

- `dgemm` for the regular matrix multiplication (`A^t * B`);
- `dsyrk` for the symmetric rank-k update (`C * C^T`);
- `dsymv` for the symmetric matrix-vector multiplication (`D * C_row_sum`, `D` - symmetric, `C_row_sum` - column vector);
- `daxpy` for the vector addition (`y += x`).

One more mention for BLAS is that I allocated the memory for the matrices in a contiguous block rather than separate allocations for each matrix.



For the optimized version, I worked in an incremental manner. All versions are available in the `opt_variants` directory.

The first iteration is `solver_opt_base.c`. It implements all basic optimizations discussed during the labs, without any AI assistance or without necessarily doing anything hardware-specific. The optimizations I implemented are:

- Since we are compiling with `-O0`, the compiler won't even bother placing the variables in registers, so I manually specified the `register` keyword for any variable used. This indeed cluttered the code a lot, but I could not see any way around it. This, however, does not guarantee all variables will be placed in registers, for example if there are more variables than registers (a.k.a register spilling), but it still seems to have a significant impact on the performance. The decompiled output also shows variables previously stored on the stack are now stored in registers.

- Used the same distributive property trick as in the unoptimized version, so the complexities are the same. What I changed is how I store the data. Instead of storing `D` as a full `NxN` matrix, I only store its upper triangular part, so each line decreases in size by 1 element: first N elements represent the first line, the following N-1 elements represent the second line, and so on. This way, I save half of the memory, and now the matrix is no longer sparse, meaning that it should be more cache-friendly.

- Similar to what I did in the BLAS implementation, I allocated the memory for the matrices in a contiguous block rather than separate allocations for each matrix. For the `C_row_sum` vector, I reuse the memory allocated for `C`, since after computing `C * C^T`, it is no longer needed. The way I now compute `C_row_sum` is to iterate over each line of `C` and add it to the next one. This way, in the end, the last line of `C` will contain `c_0 + c_1 + ... + c_n`, which is exactly `C_row_sum`.

- The pointers used during loop iterations were stored in separate variables, which were incremented during the iterations, instead of using the original pointers and doing pointer arithmetic on them. This should save at least one multiplication (`m[i * n + j]`), which adds up when doing it millions of times.

- Used blocked matrix multiplication for the initial `C = A^t * B` computation. As seen during the labs, this improves cache performance. For choosing the right block size, I applied the formula `block_size = sqrt(L1_cache_size / (sizeof(double) * 3))`. Since the L1 cache on haswell is 32KB, the result is 36.9, which rounded to a multiple of 8 (`8 * sizeof(double)` = 64 bytes, which is the cache line size) gives us a block size of 32. Since N is a multiple of 40, I had to split the computation in 4 parts: the main part where we do blocked matrix multiplication for `N/BLOCK_SIZE` blocks, and then the remaining 3 parts that were not covered: the right strip, the bottom strip and the contribution of these strips to the main large block.

- Implemented loop reordering for the `C = A^t * B` computation. Since the formula is essentially `C[i][j] += A[k][i] * B[k][j]`, the theoretically optimal loop order is `k-i-j`, because `A[k][i]` is a constant during the inner loop, and `B[k][j]` and `C[i][j]` are accessed sequentially.


## Optimized versions

### Inlining

First I wanted to see the impact of just inlining the function calls. I used the `__attribute__((always_inline))` attribute for this. The modified version is `solver_opt_inline.c`. 

```bash
./bench_opt_inline -n 1200 -s 1234 -w 10 -t 30 -a 16

```


