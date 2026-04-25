
## Haswell Setup

- Model: Intel(R) Xeon(R) CPU E5-2640 v3
- Base frequency: 2.60GHz
- Boost frequency: 3.40GHz
- Socket: 2
- Number of cores per socket: 8
- Threads per core: 2
- L1d cache: 32KiB per core => 8-way associative, 64 sets
- L1i cache: 32KiB per core => 8-way associative, 64 sets
- L2 cache: 256KiB per core => 8-way associative, 512 sets
- L3 cache: 20MiB shared between all cores of a socket => 20-way associative, 16384 sets
- RAM size: 128 GiB

- Compiler: GCC 15.2
- LIBC: musl 1.2.5
- BLAS: OpenBLAS 0.3.30


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
valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes --I1=32768,8,64 --D1=32768,8,64 --LL=20971520,20,64 <benchmark_command>
```

The benchmark command arguments used for running cachegrind are the following:
```bash
<bench_program> -n <N> -s 1234 -t 1 -a 16
```

Even if valgrind does not care about cpu affinity, I still set it to 16, to be consistent with the benchmarking methodology.

This is not a perfect simulation of the haswell architecture, since valgrind runs in its own environment, but is the closest I could get without access to `perf` or other hardware-level profilers.

## Implementation details

### Unoptimized version

It it the simplest way to implement the provided algorithm. The only "optimization" implemented, which has been stated as a requirement in the assignment description, is the use of the symmetric property of matrix `D`. Therefor, I only compute and use its upper triangular part. Additionally, when computing `y += D * row(C, i)`, the resulting elements corresponding to `D[i][j]` and `D[j][i]` are equal, and thus are computed at the same time.

### Blas version

The blas version again just implements the raw algorithm, without any mathematical tricks other than D's symmetry:

- `dgemm` for the regular matrix multiplication (`A^t * B`);
- `dsyrk` for the symmetric rank-k update (`C * C^T`);
- `dsymv` for the symmetric matrix-vector multiplication (`D * C_row_sum`, `D` - symmetric, `C_row_sum` - column vector);
- `daxpy` for the vector addition (`y += x`).

One more mention for BLAS is that I allocated the memory for the matrices in a contiguous block rather than separate allocations for each matrix.

### Optimized version

For the optimized version, I adopted an  incremental approach. All versions are available in the `impl_variants/` directory, and are prefixed by `solver_opt`.

The first iteration is `solver_opt_base.c`. It implements all basic optimizations discussed during the labs, without any AI assistance or without necessarily doing anything hardware-specific. The optimizations I implemented are:

- Since we are compiling with `-O0`, the compiler won't even bother placing the variables in registers, so I manually specified the `register` keyword for any variable used. This indeed cluttered the code a lot, but I could not see any way around it. This, however, does not guarantee all variables will be placed in registers, for example if there are more variables than registers (a.k.a register spilling), but it still seems to have a significant impact on the performance. The decompiled output also shows variables previously stored on the stack are now stored in registers.

- The transposes are no longer explicitly computed and stored in memory, but are instead accessed on the fly. For example, `A^t[i][k]` is accessed as `A[k][i]`. This saves both time and memory, and since we only access the transposed matrices only once, there is no advantage of storing it in memory.

- I took advantage of the distributive property of `D * c_0 + D * c_1 + ... + D * c_n` and instead compute `D * (c_0 + c_1 + ... + c_n) = D * sum_rows_c`. This decreases the time complexity of this operation from `O(n^3) to O(n^2)`, but the overall complexity of algorithm is still dominated by the matrix-matrix multiplication.

- Similar to what I did in the BLAS implementation, I allocated the memory for the matrices in a large contiguous block rather than separate allocations for each matrix. For the `C_row_sum` vector, I reuse the memory allocated for `C`, since after computing `C * C^T`, it is no longer needed. The way I now compute `C_row_sum` is to iterate over each line of `C` and add it to the next one. This way, in the end, the last line of `C` will contain `c_0 + c_1 + ... + c_n`, which is exactly `C_row_sum`. In addition, instead of storing `D` as a full `NxN` matrix, I only store its upper triangular part `N * (N + 1) / 2` elements, so each line decreases in size by 1 element: first `N` elements represent the first line, the following `N-1` elements represent the second line, and so on. This way, I save half of the memory, and now the matrix is no longer sparse, meaning that it should be more cache-friendly.

- The pointers used during loop iterations were stored in separate variables, which were incremented during the iterations, instead of using the original pointers and doing pointer arithmetic on them. This should save at least one multiplication (`m[i * n + j]`), which adds up when doing it millions of times.

- I used blocked matrix multiplication for the initial `C = A^t * B` computation. As seen during the labs, this improves cache performance. For choosing the right block size, I applied the formula `block_size = sqrt(L1_cache_size / (sizeof(double) * 3))`. Since the L1 cache on haswell is 32KB, the result is 36.9, which rounded to a multiple of 8 (`8 * sizeof(double)` = 64 bytes, which is the cache line size) gives us a block size of 32. Since N is a multiple of 40, I had to split the computation in 4 parts: the main part where we do blocked matrix multiplication for `N/BLOCK_SIZE` blocks, and then the remaining 3 parts that were not covered: the right strip, the bottom strip and the contribution of these strips to the main large block.

- I implemented loop reordering for the `C = A^t * B` computation. Since the formula is essentially `C[i][j] += A[k][i] * B[k][j]`, the optimal loop order is `k-i-j`, because `A[k][i]` is a constant during the inner loop, and `B[k][j]` and `C[i][j]` are accessed sequentially.


### Unoptimized equivalent version

To keep the comparison between the unoptimized and optimized versions fair (i.e. same complexity), I also implemented an equivalent unoptimized version which also takes advantage of the redunction in number of operations (although the overall time complexity stays the same). Otherwise, this is identical to the unoptimized version

## Optimized versions

### Inlining

Given that the programs are compiled using `-O0`, there is absolutely no inlining performed by default. However, functions get called millions of times inside the loops in the current implementation, so the overhead of a function call, including saving registers, stackframe allocation and jumping to the function, is probably a large time-consumer. Therefore, I wanted to see the impact of just inlining the function calls by using the `__attribute__((always_inline))` function attribute. The source code can be found at `impl_variants/solver_opt_inline.c`. 



### AVX

To integrate AVX SIMD instructions, I extracted the obvious candidate loops into separate functions:

- `vec_dot` - which computes `dot(A, B)`
- `vec_axpy` - which computes `Y = alpha * X + Y`
- `vec_add` - which computes `Y = X + Y`

All these operations can benefit from SIMD instructions, because their loop body consists of just one operations, with no branches and no data dependencies.

For documentation, I used both the [Intel Intrinsics Guide](https://www.felixcloutier.com/x86/v)

The documentation process included selecting the right instruction form through the [x86 instruction reference](https://www.felixcloutier.com/x86/) and finding the corresponding AVX intrinsics using [Intel's Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) (+ selecting the correct instruction form from), as well as extracting hardware-specific measurements for the selected instructions from [uops.info](uops.info). The main instructions used are `vfmadd231pd` for the fused multiply-add operation and `vaddpd` for the vector addition. 

Additionally, I have also used `posix_memalign` to allocate 32-byte aligned memory for the buffers used in the computations, which should improve the performance of the AVX instructions. Despite this, I used the unaligned load/store intrinsics (`_mm256_loadu_pd` and `_mm256_storeu_pd`) instead of the aligned ones (`_mm256_load_pd` and `_mm256_store_pd`), to simplify the code and avoid additional checks.

### Vec_dot

The dot product is just a fused-multiply-add operation with an accumulator, so the central intrinsic used is `_mm256_fmadd_pd`.
The fmadd instruction used is the memory fold one (which takes one operand from memory and the other two from YMM registers). On haswell, this one has a latency of 12 cycles, and a reciprocal throughput of 0.5 cycles, meaning that there are two ports which can execute this instruction, and therefore two instructions can be issued per cycle. The other two operands are manually loaded into registers using the `_mm256_loadu_pd` instruction (latency 7, reciprocal throughput 0.5). Since the latency of the fmadd instruction is dominant, we want to calculate the loop unroll factor such that we can hide the latency of the fmadd instruction. The theoretically optimal unroll factor is `ceil(latency_fmadd / reciprocal_throughput_fmadd) = ceil(12 / 0.5) = 24`, meaning that we would have to have 24 independent fmadd instructions in the loop body in order to fully hide the latency. However, this is not possible due to the limited number of registers (16 YMM registers). Given that we use 2 different operands loaded into registers (the other one is directly moved from memory), I figured the maximum unroll factor would be `16 / 2 = 8`. 

Initial results showed little to no improvement from these optimizations, so I decided to take a loop at the generated assembly code. This is the part of the assembly code corresponding to the main loop.

```asm
.L35:
	mov	QWORD PTR 224[rsp], rbx
	mov	rax, QWORD PTR 224[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	ymm2, ymm0
	lea	rax, 32[rbx]
	mov	QWORD PTR 216[rsp], rax
	mov	rax, QWORD PTR 216[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	ymm4, ymm0
	lea	rax, 64[rbx]
	mov	QWORD PTR 208[rsp], rax
	mov	rax, QWORD PTR 208[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	ymm6, ymm0
	lea	rax, 96[rbx]
	mov	QWORD PTR 200[rsp], rax
	mov	rax, QWORD PTR 200[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	ymm8, ymm0
	lea	rax, 128[rbx]
	mov	QWORD PTR 192[rsp], rax
	mov	rax, QWORD PTR 192[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	ymm9, ymm0
	lea	rax, 160[rbx]
	mov	QWORD PTR 184[rsp], rax
	mov	rax, QWORD PTR 184[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	ymm10, ymm0
	lea	rax, 192[rbx]
	mov	QWORD PTR 176[rsp], rax
	mov	rax, QWORD PTR 176[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	ymm11, ymm0
	lea	rax, 224[rbx]
	mov	QWORD PTR 168[rsp], rax
	mov	rax, QWORD PTR 168[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	ymm12, ymm0
	mov	QWORD PTR 160[rsp], r12
	mov	rax, QWORD PTR 160[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	YMMWORD PTR 904[rsp], ymm2
	vmovapd	YMMWORD PTR 936[rsp], ymm0
	vmovapd	ymm3, ymm13
	vmovapd	YMMWORD PTR 968[rsp], ymm3
	vmovapd	ymm1, YMMWORD PTR 936[rsp]
	vmovapd	ymm0, YMMWORD PTR 968[rsp]
	vfmadd231pd	ymm0, ymm1, YMMWORD PTR 904[rsp]
	nop
	vmovapd	ymm13, ymm0
	lea	rax, 32[r12]
	mov	QWORD PTR 152[rsp], rax
	mov	rax, QWORD PTR 152[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	YMMWORD PTR 808[rsp], ymm4
	vmovapd	YMMWORD PTR 840[rsp], ymm0
	vmovapd	ymm5, ymm14
	vmovapd	YMMWORD PTR 872[rsp], ymm5
	vmovapd	ymm1, YMMWORD PTR 840[rsp]
	vmovapd	ymm0, YMMWORD PTR 872[rsp]
	vfmadd231pd	ymm0, ymm1, YMMWORD PTR 808[rsp]
	nop
	vmovapd	ymm14, ymm0
	lea	rax, 64[r12]
	mov	QWORD PTR 144[rsp], rax
	mov	rax, QWORD PTR 144[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	YMMWORD PTR 712[rsp], ymm6
	vmovapd	YMMWORD PTR 744[rsp], ymm0
	vmovapd	ymm7, ymm15
	vmovapd	YMMWORD PTR 776[rsp], ymm7
	vmovapd	ymm1, YMMWORD PTR 744[rsp]
	vmovapd	ymm0, YMMWORD PTR 776[rsp]
	vfmadd231pd	ymm0, ymm1, YMMWORD PTR 712[rsp]
	nop
	vmovapd	ymm15, ymm0
	lea	rax, 96[r12]
	mov	QWORD PTR 136[rsp], rax
	mov	rax, QWORD PTR 136[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	YMMWORD PTR 616[rsp], ymm8
	vmovapd	YMMWORD PTR 648[rsp], ymm0
	vmovapd	ymm3, YMMWORD PTR 8[rsp]
	vmovapd	YMMWORD PTR 680[rsp], ymm3
	vmovapd	ymm1, YMMWORD PTR 648[rsp]
	vmovapd	ymm0, YMMWORD PTR 680[rsp]
	vfmadd231pd	ymm0, ymm1, YMMWORD PTR 616[rsp]
	nop
	vmovapd	YMMWORD PTR 8[rsp], ymm0
	lea	rax, 128[r12]
	mov	QWORD PTR 128[rsp], rax
	mov	rax, QWORD PTR 128[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	YMMWORD PTR 520[rsp], ymm9
	vmovapd	YMMWORD PTR 552[rsp], ymm0
	vmovapd	ymm5, YMMWORD PTR -24[rsp]
	vmovapd	YMMWORD PTR 584[rsp], ymm5
	vmovapd	ymm1, YMMWORD PTR 552[rsp]
	vmovapd	ymm0, YMMWORD PTR 584[rsp]
	vfmadd231pd	ymm0, ymm1, YMMWORD PTR 520[rsp]
	nop
	vmovapd	YMMWORD PTR -24[rsp], ymm0
	lea	rax, 160[r12]
	mov	QWORD PTR 120[rsp], rax
	mov	rax, QWORD PTR 120[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	YMMWORD PTR 424[rsp], ymm10
	vmovapd	YMMWORD PTR 456[rsp], ymm0
	vmovapd	ymm7, YMMWORD PTR -56[rsp]
	vmovapd	YMMWORD PTR 488[rsp], ymm7
	vmovapd	ymm1, YMMWORD PTR 456[rsp]
	vmovapd	ymm0, YMMWORD PTR 488[rsp]
	vfmadd231pd	ymm0, ymm1, YMMWORD PTR 424[rsp]
	nop
	vmovapd	YMMWORD PTR -56[rsp], ymm0
	lea	rax, 192[r12]
	mov	QWORD PTR 112[rsp], rax
	mov	rax, QWORD PTR 112[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	YMMWORD PTR 328[rsp], ymm11
	vmovapd	YMMWORD PTR 360[rsp], ymm0
	vmovapd	ymm2, YMMWORD PTR -88[rsp]
	vmovapd	YMMWORD PTR 392[rsp], ymm2
	vmovapd	ymm1, YMMWORD PTR 360[rsp]
	vmovapd	ymm0, YMMWORD PTR 392[rsp]
	vfmadd231pd	ymm0, ymm1, YMMWORD PTR 328[rsp]
	nop
	vmovapd	YMMWORD PTR -88[rsp], ymm0
	lea	rax, 224[r12]
	mov	QWORD PTR 104[rsp], rax
	mov	rax, QWORD PTR 104[rsp]
	vmovupd	ymm0, YMMWORD PTR [rax]
	vmovapd	YMMWORD PTR 232[rsp], ymm12
	vmovapd	YMMWORD PTR 264[rsp], ymm0
	vmovapd	ymm3, YMMWORD PTR -120[rsp]
	vmovapd	YMMWORD PTR 296[rsp], ymm3
	vmovapd	ymm1, YMMWORD PTR 264[rsp]
	vmovapd	ymm0, YMMWORD PTR 296[rsp]
	vfmadd231pd	ymm0, ymm1, YMMWORD PTR 232[rsp]
	nop
	vmovapd	YMMWORD PTR -120[rsp], ymm0
	add	r13d, 32
	add	rbx, 256
	add	r12, 256
.L10:
	cmp	r13d, DWORD PTR 100[rsp]
	jl	.L35
```

As we can see, there are indeed 8 independent fmadd instructions and all 16 YMM registers are being used. However, there is a huge problem: the vectors are repeatedly moved back and forth between stack memory and YMM registers. Upon further investigation, this issue stems from the use of `-O0` flag. According to the GCC man page, `-O0 Reduce compilation time and make debugging produce the expected results`, the flag prioritizes **debuggability** and producing the expected results over performance, which means that it moves all the results to stack so that they can be easily inspected and modified during debugging. This creates a huge overhead, transforming the loop into a memory-bound operation. To visualize how much this affects the performance, I ran `llvm-mca` so that we can see how the Reservation Station (a.k.a the CPU's backend scheduler) would actually schedule the instructions:

```
[0,37]    .    .    .   D========eER    .    .    .    .    .    .    .    .    .    .   .   mov	qword ptr [rsp + 224], rbx
[0,38]    .    .    .   DeeeeeE----R    .    .    .    .    .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 224]
[0,39]    .    .    .    D====eeeeeeeER .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,40]    .    .    .    D===========eER.    .    .    .    .    .    .    .    .    .   .   vmovapd	ymm2, ymm0
[0,41]    .    .    .    D===eE--------R.    .    .    .    .    .    .    .    .    .   .   lea	rax, [rbx + 32]
[0,42]    .    .    .    D========eE---R.    .    .    .    .    .    .    .    .    .   .   mov	qword ptr [rsp + 216], rax
[0,43]    .    .    .    .DeeeeeE------R.    .    .    .    .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 216]
[0,44]    .    .    .    .D=====eeeeeeeER    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,45]    .    .    .    .D============eER   .    .    .    .    .    .    .    .    .   .   vmovapd	ymm4, ymm0
[0,46]    .    .    .    .D==eE----------R   .    .    .    .    .    .    .    .    .   .   lea	rax, [rbx + 64]
[0,47]    .    .    .    . D=======eE----R   .    .    .    .    .    .    .    .    .   .   mov	qword ptr [rsp + 208], rax
[0,48]    .    .    .    . DeeeeeE-------R   .    .    .    .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 208]
[0,49]    .    .    .    . D=====eeeeeeeER   .    .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,50]    .    .    .    . D============eER  .    .    .    .    .    .    .    .    .   .   vmovapd	ymm6, ymm0
[0,51]    .    .    .    .  D=eE----------R  .    .    .    .    .    .    .    .    .   .   lea	rax, [rbx + 96]
[0,52]    .    .    .    .  D=======eE----R  .    .    .    .    .    .    .    .    .   .   mov	qword ptr [rsp + 200], rax
[0,53]    .    .    .    .  DeeeeeE-------R  .    .    .    .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 200]
[0,54]    .    .    .    .  D=====eeeeeeeER  .    .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,55]    .    .    .    .   D===========eER .    .    .    .    .    .    .    .    .   .   vmovapd	ymm8, ymm0
[0,56]    .    .    .    .   DeE-----------R .    .    .    .    .    .    .    .    .   .   lea	rax, [rbx + 128]
[0,57]    .    .    .    .   D=======eE----R .    .    .    .    .    .    .    .    .   .   mov	qword ptr [rsp + 192], rax
[0,58]    .    .    .    .   DeeeeeE-------R .    .    .    .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 192]
[0,59]    .    .    .    .    D====eeeeeeeER .    .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,60]    .    .    .    .    D===========eER.    .    .    .    .    .    .    .    .   .   vmovapd	ymm9, ymm0
[0,61]    .    .    .    .    DeE-----------R.    .    .    .    .    .    .    .    .   .   lea	rax, [rbx + 160]
[0,62]    .    .    .    .    D=======eE----R.    .    .    .    .    .    .    .    .   .   mov	qword ptr [rsp + 184], rax
[0,63]    .    .    .    .    .DeeeeeE------R.    .    .    .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 184]
[0,64]    .    .    .    .    .D=====eeeeeeeER    .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,65]    .    .    .    .    .D============eER   .    .    .    .    .    .    .    .   .   vmovapd	ymm10, ymm0
[0,66]    .    .    .    .    .DeE------------R   .    .    .    .    .    .    .    .   .   lea	rax, [rbx + 192]
[0,67]    .    .    .    .    . D======eE-----R   .    .    .    .    .    .    .    .   .   mov	qword ptr [rsp + 176], rax
[0,68]    .    .    .    .    . DeeeeeE-------R   .    .    .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 176]
[0,69]    .    .    .    .    . D=====eeeeeeeER   .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,70]    .    .    .    .    . D============eER  .    .    .    .    .    .    .    .   .   vmovapd	ymm11, ymm0
[0,71]    .    .    .    .    .  DeE-----------R  .    .    .    .    .    .    .    .   .   lea	rax, [rbx + 224]
[0,72]    .    .    .    .    .  D======eE-----R  .    .    .    .    .    .    .    .   .   mov	qword ptr [rsp + 168], rax
[0,73]    .    .    .    .    .  DeeeeeE-------R  .    .    .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 168]
[0,74]    .    .    .    .    .  D=====eeeeeeeER  .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,75]    .    .    .    .    .   D===========eER .    .    .    .    .    .    .    .   .   vmovapd	ymm12, ymm0
[0,76]    .    .    .    .    .   D======eE-----R .    .    .    .    .    .    .    .   .   mov	qword ptr [rsp + 160], r12
[0,77]    .    .    .    .    .   DeeeeeE-------R .    .    .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 160]
[0,78]    .    .    .    .    .   D=====eeeeeeeER .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,79]    .    .    .    .    .    D======eE----R .    .    .    .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 904], ymm2
[0,80]    .    .    .    .    .    D===========eER.    .    .    .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 936], ymm0
[0,81]    .    .    .    .    .    .D=eE---------R.    .    .    .    .    .    .    .   .   vmovapd	ymm3, ymm13
[0,82]    .    .    .    .    .    .D===========eER    .    .    .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 968], ymm3
[0,83]    .    .    .    .    .    .DeeeeeeeE-----R    .    .    .    .    .    .    .   .   vmovapd	ymm1, ymmword ptr [rsp + 936]
[0,84]    .    .    .    .    .    . DeeeeeeeE----R    .    .    .    .    .    .    .   .   vmovapd	ymm0, ymmword ptr [rsp + 968]
[0,85]    .    .    .    .    .    . D=eeeeeeeeeeeeER  .    .    .    .    .    .    .   .   vfmadd231pd	ymm0, ymm1, ymmword ptr [rsp + 904]
[0,86]    .    .    .    .    .    . DeE------------R  .    .    .    .    .    .    .   .   nop
[0,87]    .    .    .    .    .    .  D============eER .    .    .    .    .    .    .   .   vmovapd	ymm13, ymm0
[0,88]    .    .    .    .    .    .  DeE------------R .    .    .    .    .    .    .   .   lea	rax, [r12 + 32]
[0,89]    .    .    .    .    .    .  D==========eE--R .    .    .    .    .    .    .   .   mov	qword ptr [rsp + 152], rax
[0,90]    .    .    .    .    .    .  D=eeeeeE-------R .    .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 152]
[0,91]    .    .    .    .    .    .   D=====eeeeeeeER .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,92]    .    .    .    .    .    .   D==========eE-R .    .    .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 808], ymm4
[0,93]    .    .    .    .    .    .    D===========eER.    .    .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 840], ymm0
[0,94]    .    .    .    .    .    .    D==eE---------R.    .    .    .    .    .    .   .   vmovapd	ymm5, ymm14
[0,95]    .    .    .    .    .    .    .D===========eER    .    .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 872], ymm5
[0,96]    .    .    .    .    .    .    .DeeeeeeeE-----R    .    .    .    .    .    .   .   vmovapd	ymm1, ymmword ptr [rsp + 840]
[0,97]    .    .    .    .    .    .    .D=eeeeeeeE----R    .    .    .    .    .    .   .   vmovapd	ymm0, ymmword ptr [rsp + 872]
[0,98]    .    .    .    .    .    .    . DeeeeeeeeeeeeER   .    .    .    .    .    .   .   vfmadd231pd	ymm0, ymm1, ymmword ptr [rsp + 808]
[0,99]    .    .    .    .    .    .    . DeE-----------R   .    .    .    .    .    .   .   nop
[0,100]   .    .    .    .    .    .    . D============eER  .    .    .    .    .    .   .   vmovapd	ymm14, ymm0
[0,101]   .    .    .    .    .    .    .  DeE-----------R  .    .    .    .    .    .   .   lea	rax, [r12 + 64]
[0,102]   .    .    .    .    .    .    .  D==========eE-R  .    .    .    .    .    .   .   mov	qword ptr [rsp + 144], rax
[0,103]   .    .    .    .    .    .    .  DeeeeeE-------R  .    .    .    .    .    .   .   mov	rax, qword ptr [rsp + 144]
[0,104]   .    .    .    .    .    .    .  D=====eeeeeeeER  .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,105]   .    .    .    .    .    .    .   D==========eER  .    .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 712], ymm6
[0,106]   .    .    .    .    .    .    .   D===========eER .    .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 744], ymm0
[0,107]   .    .    .    .    .    .    .    D=eE---------R .    .    .    .    .    .   .   vmovapd	ymm7, ymm15
[0,108]   .    .    .    .    .    .    .    D===========eER.    .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 776], ymm7
[0,109]   .    .    .    .    .    .    .    DeeeeeeeE-----R.    .    .    .    .    .   .   vmovapd	ymm1, ymmword ptr [rsp + 744]
[0,110]   .    .    .    .    .    .    .    .DeeeeeeeE----R.    .    .    .    .    .   .   vmovapd	ymm0, ymmword ptr [rsp + 776]
[0,111]   .    .    .    .    .    .    .    .DeeeeeeeeeeeeER    .    .    .    .    .   .   vfmadd231pd	ymm0, ymm1, ymmword ptr [rsp + 712]
[0,112]   .    .    .    .    .    .    .    .DeE-----------R    .    .    .    .    .   .   nop
[0,113]   .    .    .    .    .    .    .    . D===========eER   .    .    .    .    .   .   vmovapd	ymm15, ymm0
[0,114]   .    .    .    .    .    .    .    . DeE-----------R   .    .    .    .    .   .   lea	rax, [r12 + 96]
[0,115]   .    .    .    .    .    .    .    . D==========eE-R   .    .    .    .    .   .   mov	qword ptr [rsp + 136], rax
[0,116]   .    .    .    .    .    .    .    . DeeeeeE-------R   .    .    .    .    .   .   mov	rax, qword ptr [rsp + 136]
[0,117]   .    .    .    .    .    .    .    .  D====eeeeeeeER   .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,118]   .    .    .    .    .    .    .    .  D==========eER   .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 616], ymm8
[0,119]   .    .    .    .    .    .    .    .   D==========eER  .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 648], ymm0
[0,120]   .    .    .    .    .    .    .    .   DeeeeeeeE----R  .    .    .    .    .   .   vmovapd	ymm3, ymmword ptr [rsp + 8]
[0,121]   .    .    .    .    .    .    .    .    D==========eER .    .    .    .    .   .   vmovapd	ymmword ptr [rsp + 680], ymm3
[0,122]   .    .    .    .    .    .    .    .    DeeeeeeeE----R .    .    .    .    .   .   vmovapd	ymm1, ymmword ptr [rsp + 648]
[0,123]   .    .    .    .    .    .    .    .    DeeeeeeeE----R .    .    .    .    .   .   vmovapd	ymm0, ymmword ptr [rsp + 680]
[0,124]   .    .    .    .    .    .    .    .    .DeeeeeeeeeeeeER    .    .    .    .   .   vfmadd231pd	ymm0, ymm1, ymmword ptr [rsp + 616]
[0,125]   .    .    .    .    .    .    .    .    .DeE-----------R    .    .    .    .   .   nop
[0,126]   .    .    .    .    .    .    .    .    . D===========eER   .    .    .    .   .   vmovapd	ymmword ptr [rsp + 8], ymm0
[0,127]   .    .    .    .    .    .    .    .    . DeE-----------R   .    .    .    .   .   lea	rax, [r12 + 128]
[0,128]   .    .    .    .    .    .    .    .    . D============eER  .    .    .    .   .   mov	qword ptr [rsp + 128], rax
[0,129]   .    .    .    .    .    .    .    .    .  DeeeeeE-------R  .    .    .    .   .   mov	rax, qword ptr [rsp + 128]
[0,130]   .    .    .    .    .    .    .    .    .  D=====eeeeeeeER  .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,131]   .    .    .    .    .    .    .    .    .  D============eER .    .    .    .   .   vmovapd	ymmword ptr [rsp + 520], ymm9
[0,132]   .    .    .    .    .    .    .    .    .   D============eER.    .    .    .   .   vmovapd	ymmword ptr [rsp + 552], ymm0
[0,133]   .    .    .    .    .    .    .    .    .   DeeeeeeeE------R.    .    .    .   .   vmovapd	ymm5, ymmword ptr [rsp - 24]
[0,134]   .    .    .    .    .    .    .    .    .    D============eER    .    .    .   .   vmovapd	ymmword ptr [rsp + 584], ymm5
[0,135]   .    .    .    .    .    .    .    .    .    DeeeeeeeE------R    .    .    .   .   vmovapd	ymm1, ymmword ptr [rsp + 552]
[0,136]   .    .    .    .    .    .    .    .    .    DeeeeeeeE------R    .    .    .   .   vmovapd	ymm0, ymmword ptr [rsp + 584]
[0,137]   .    .    .    .    .    .    .    .    .    .DeeeeeeeeeeeeER    .    .    .   .   vfmadd231pd	ymm0, ymm1, ymmword ptr [rsp + 520]
[0,138]   .    .    .    .    .    .    .    .    .    .DeE-----------R    .    .    .   .   nop
[0,139]   .    .    .    .    .    .    .    .    .    . D===========eER   .    .    .   .   vmovapd	ymmword ptr [rsp - 24], ymm0
[0,140]   .    .    .    .    .    .    .    .    .    . DeE-----------R   .    .    .   .   lea	rax, [r12 + 160]
[0,141]   .    .    .    .    .    .    .    .    .    . D============eER  .    .    .   .   mov	qword ptr [rsp + 120], rax
[0,142]   .    .    .    .    .    .    .    .    .    .  DeeeeeE-------R  .    .    .   .   mov	rax, qword ptr [rsp + 120]
[0,143]   .    .    .    .    .    .    .    .    .    .  D=====eeeeeeeER  .    .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,144]   .    .    .    .    .    .    .    .    .    .  D============eER .    .    .   .   vmovapd	ymmword ptr [rsp + 424], ymm10
[0,145]   .    .    .    .    .    .    .    .    .    .   D============eER.    .    .   .   vmovapd	ymmword ptr [rsp + 456], ymm0
[0,146]   .    .    .    .    .    .    .    .    .    .   DeeeeeeeE------R.    .    .   .   vmovapd	ymm7, ymmword ptr [rsp - 56]
[0,147]   .    .    .    .    .    .    .    .    .    .    D============eER    .    .   .   vmovapd	ymmword ptr [rsp + 488], ymm7
[0,148]   .    .    .    .    .    .    .    .    .    .    DeeeeeeeE------R    .    .   .   vmovapd	ymm1, ymmword ptr [rsp + 456]
[0,149]   .    .    .    .    .    .    .    .    .    .    D=eeeeeeeE-----R    .    .   .   vmovapd	ymm0, ymmword ptr [rsp + 488]
[0,150]   .    .    .    .    .    .    .    .    .    .    .DeeeeeeeeeeeeER    .    .   .   vfmadd231pd	ymm0, ymm1, ymmword ptr [rsp + 424]
[0,151]   .    .    .    .    .    .    .    .    .    .    .DeE-----------R    .    .   .   nop
[0,152]   .    .    .    .    .    .    .    .    .    .    . D===========eER   .    .   .   vmovapd	ymmword ptr [rsp - 56], ymm0
[0,153]   .    .    .    .    .    .    .    .    .    .    . DeE-----------R   .    .   .   lea	rax, [r12 + 192]
[0,154]   .    .    .    .    .    .    .    .    .    .    . D============eER  .    .   .   mov	qword ptr [rsp + 112], rax
[0,155]   .    .    .    .    .    .    .    .    .    .    .  DeeeeeE-------R  .    .   .   mov	rax, qword ptr [rsp + 112]
[0,156]   .    .    .    .    .    .    .    .    .    .    .  D=====eeeeeeeER  .    .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,157]   .    .    .    .    .    .    .    .    .    .    .  D============eER .    .   .   vmovapd	ymmword ptr [rsp + 328], ymm11
[0,158]   .    .    .    .    .    .    .    .    .    .    .   D============eER.    .   .   vmovapd	ymmword ptr [rsp + 360], ymm0
[0,159]   .    .    .    .    .    .    .    .    .    .    .   DeeeeeeeE------R.    .   .   vmovapd	ymm2, ymmword ptr [rsp - 88]
[0,160]   .    .    .    .    .    .    .    .    .    .    .    D============eER    .   .   vmovapd	ymmword ptr [rsp + 392], ymm2
[0,161]   .    .    .    .    .    .    .    .    .    .    .    DeeeeeeeE------R    .   .   vmovapd	ymm1, ymmword ptr [rsp + 360]
[0,162]   .    .    .    .    .    .    .    .    .    .    .    DeeeeeeeE------R    .   .   vmovapd	ymm0, ymmword ptr [rsp + 392]
[0,163]   .    .    .    .    .    .    .    .    .    .    .    .DeeeeeeeeeeeeER    .   .   vfmadd231pd	ymm0, ymm1, ymmword ptr [rsp + 328]
[0,164]   .    .    .    .    .    .    .    .    .    .    .    .DeE-----------R    .   .   nop
[0,165]   .    .    .    .    .    .    .    .    .    .    .    . D===========eER   .   .   vmovapd	ymmword ptr [rsp - 88], ymm0
[0,166]   .    .    .    .    .    .    .    .    .    .    .    . DeE-----------R   .   .   lea	rax, [r12 + 224]
[0,167]   .    .    .    .    .    .    .    .    .    .    .    . D============eER  .   .   mov	qword ptr [rsp + 104], rax
[0,168]   .    .    .    .    .    .    .    .    .    .    .    .  D=eeeeeE------R  .   .   mov	rax, qword ptr [rsp + 104]
[0,169]   .    .    .    .    .    .    .    .    .    .    .    .  D======eeeeeeeER .   .   vmovupd	ymm0, ymmword ptr [rax]
[0,170]   .    .    .    .    .    .    .    .    .    .    .    .  D============eER .   .   vmovapd	ymmword ptr [rsp + 232], ymm12
[0,171]   .    .    .    .    .    .    .    .    .    .    .    .   D============eER.   .   vmovapd	ymmword ptr [rsp + 264], ymm0
[0,172]   .    .    .    .    .    .    .    .    .    .    .    .   DeeeeeeeE------R.   .   vmovapd	ymm3, ymmword ptr [rsp - 120]
[0,173]   .    .    .    .    .    .    .    .    .    .    .    .    D============eER   .   vmovapd	ymmword ptr [rsp + 296], ymm3
[0,174]   .    .    .    .    .    .    .    .    .    .    .    .    DeeeeeeeE------R   .   vmovapd	ymm1, ymmword ptr [rsp + 264]
[0,175]   .    .    .    .    .    .    .    .    .    .    .    .    DeeeeeeeE------R   .   vmovapd	ymm0, ymmword ptr [rsp + 296]
[0,176]   .    .    .    .    .    .    .    .    .    .    .    .    .DeeeeeeeeeeeeER   .   vfmadd231pd	ymm0, ymm1, ymmword ptr [rsp + 232]
[0,177]   .    .    .    .    .    .    .    .    .    .    .    .    .DeE-----------R   .   nop
[0,178]   .    .    .    .    .    .    .    .    .    .    .    .    . D===========eER  .   vmovapd	ymmword ptr [rsp - 120], ymm0
[0,179]   .    .    .    .    .    .    .    .    .    .    .    .    . DeE-----------R  .   add	r13d, 32
[0,180]   .    .    .    .    .    .    .    .    .    .    .    .    . DeE-----------R  .   add	rbx, 256
[0,181]   .    .    .    .    .    .    .    .    .    .    .    .    .  DeE----------R  .   add	r12, 256
[0,182]   .    .    .    .    .    .    .    .    .    .    .    .    .  DeeeeeeE-----R  .   cmp	r13d, dword ptr [rsp + 100]
[0,183]   .    .    .    .    .    .    .    .    .    .    .    .    .  D======eE----R  .   jl	.L35
[0,184]   .    .    .    .    .    .    .    .    .    .    .    .    .   DeE---------R  .   vmovapd	ymm4, ymm13
[0,185]   .    .    .    .    .    .    .    .    .    .    .    .    .   D==========eER .   vmovapd	ymmword ptr [rsp + 1416], ymm4
[0,186]   .    .    .    .    .    .    .    .    .    .    .    .    .   D=eE---------R .   vmovapd	ymm5, ymm14
[0,187]   .    .    .    .    .    .    .    .    .    .    .    .    .    D==========eER.   vmovapd	ymmword ptr [rsp + 1448], ymm5
[0,188]   .    .    .    .    .    .    .    .    .    .    .    .    .    DeeeeeeeE----R.   vmovapd	ymm0, ymmword ptr [rsp + 1416]
[0,189]   .    .    .    .    .    .    .    .    .    .    .    .    .    .DeeeeeeeeeeER.   vaddpd	ymm0, ymm0, ymmword ptr [rsp + 1448]
[0,190]   .    .    .    .    .    .    .    .    .    .    .    .    .    .D==========eER   vmovapd	ymm1, ymm0
[0,191]   .    .    .    .    .    .    .    .    .    .    .    .    .    .DeE----------R   vmovapd	ymm6, ymm15
[0,192]   .    .    .    .    .    .    .    .    .    .    .    .    .    . D=========eER   vmovapd	ymmword ptr [rsp + 1352], ymm6
[0,193]   .    .    .    .    .    .    .    .    .    .    .    .    .    . DeeeeeeeE---R   vmovapd	ymm7, ymmword ptr [rsp + 8]
```

Legend:
- **D**: Instruction dispatched to the Reservation Station
- **e**: Instruction issued to the Execution Unit and started execution
- **E**: Instruction finished execution and is waiting for retirement
- **R**: Instruction retired and committed
- **=**: Instruction dispatched but waiting for operands or resources
- **-**: Instruction executed but waiting to be retired (in-order)

As we can see there are lots of **=** symbols which means a lot of cycles are wasted just on waiting. This is not an efficient utilization of the pipeline.

If we use instead turn on the optimization just for this function via `__attribute__((optimize("O3")))` we get the following:

```asm
.L3:
	add	ecx, 32
	vmovupd	ymm8, YMMWORD PTR [rdx]
	vmovupd	ymm9, YMMWORD PTR 32[rdx]
	add	rax, 256
	vmovupd	ymm10, YMMWORD PTR 64[rdx]
	vmovupd	ymm11, YMMWORD PTR 96[rdx]
	add	rdx, 256
	vmovupd	ymm12, YMMWORD PTR -128[rdx]
	vmovupd	ymm13, YMMWORD PTR -96[rdx]
	vmovupd	ymm14, YMMWORD PTR -64[rdx]
	vmovupd	ymm15, YMMWORD PTR -32[rdx]
	vfmadd231pd	ymm5, ymm8, YMMWORD PTR -256[rax]
	vfmadd231pd	ymm7, ymm12, YMMWORD PTR -128[rax]
	vfmadd231pd	ymm3, ymm9, YMMWORD PTR -224[rax]
	vfmadd231pd	ymm4, ymm13, YMMWORD PTR -96[rax]
	vfmadd231pd	ymm6, ymm10, YMMWORD PTR -192[rax]
	vfmadd231pd	ymm1, ymm14, YMMWORD PTR -64[rax]
	vfmadd231pd	ymm2, ymm11, YMMWORD PTR -160[rax]
	vfmadd231pd	ymm0, ymm15, YMMWORD PTR -32[rax]
	cmp	esi, ecx
	jg	.L3
```

```
[0,20]    .    DeE-----R .    .    .    .    .    .    .    .    .    .    .    .    .   .   add	ecx, 32
[0,21]    .    DeeeeeeeER.    .    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm8, ymmword ptr [rdx]
[0,22]    .    DeeeeeeeER.    .    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm9, ymmword ptr [rdx + 32]
[0,23]    .    .DeE-----R.    .    .    .    .    .    .    .    .    .    .    .    .   .   add	rax, 256
[0,24]    .    .DeeeeeeeER    .    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm10, ymmword ptr [rdx + 64]
[0,25]    .    .DeeeeeeeER    .    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm11, ymmword ptr [rdx + 96]
[0,26]    .    .DeE------R    .    .    .    .    .    .    .    .    .    .    .    .   .   add	rdx, 256
[0,27]    .    . DeeeeeeeER   .    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm12, ymmword ptr [rdx - 128]
[0,28]    .    . DeeeeeeeER   .    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm13, ymmword ptr [rdx - 96]
[0,29]    .    . D=eeeeeeeER  .    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm14, ymmword ptr [rdx - 64]
[0,30]    .    . D=eeeeeeeER  .    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm15, ymmword ptr [rdx - 32]
[0,31]    .    .  D==eeeeeeeeeeeeER.    .    .    .    .    .    .    .    .    .    .   .   vfmadd231pd	ymm5, ymm8, ymmword ptr [rax - 256]
[0,32]    .    .  D=eeeeeeeeeeeeE-R.    .    .    .    .    .    .    .    .    .    .   .   vfmadd231pd	ymm7, ymm12, ymmword ptr [rax - 128]
[0,33]    .    .   DeeeeeeeeeeeeE-R.    .    .    .    .    .    .    .    .    .    .   .   vfmadd231pd	ymm3, ymm9, ymmword ptr [rax - 224]
[0,34]    .    .   D=eeeeeeeeeeeeER.    .    .    .    .    .    .    .    .    .    .   .   vfmadd231pd	ymm4, ymm13, ymmword ptr [rax - 96]
[0,35]    .    .    D=eeeeeeeeeeeeER    .    .    .    .    .    .    .    .    .    .   .   vfmadd231pd	ymm6, ymm10, ymmword ptr [rax - 192]
[0,36]    .    .    D=eeeeeeeeeeeeER    .    .    .    .    .    .    .    .    .    .   .   vfmadd231pd	ymm1, ymm14, ymmword ptr [rax - 64]
[0,37]    .    .    .D=eeeeeeeeeeeeER   .    .    .    .    .    .    .    .    .    .   .   vfmadd231pd	ymm2, ymm11, ymmword ptr [rax - 160]
[0,38]    .    .    .D=eeeeeeeeeeeeER   .    .    .    .    .    .    .    .    .    .   .   vfmadd231pd	ymm0, ymm15, ymmword ptr [rax - 32]
[0,39]    .    .    . DeE-----------R   .    .    .    .    .    .    .    .    .    .   .   cmp	esi, ecx
[0,40]    .    .    . D=eE----------R   .    .    .    .    .    .    .    .    .    .   .   jg	.L3
```

Now we are talking! The assembly is now a lot cleaner and the pipline is exploited almost perfectly. There are just a few stalls left, but I don't think we can do much better than this.

So it seems like there is no much point in using AVX instructions if we don't turn on the optimizations, or if we don't use inline assembly, but that would make the code a lot less portable and unreadable. So the best solution, which I think is justified, is using `-O3` optimizations just for the critical AVX-based functions.


### Vec_axpy

This operation is similar to the previous one, but instead we multiply a vector by a constant and add it to another vector. So now the fmadd instruction involves one constant vector and two vectors from memory. After executing fmadd, the result is stored back in memory via `_mm256_storeu_pd` (latency 1, reciprocal throughput 1). The fmadd is still dominant in latency, so the theoretical loop unrolling factor remains the same.

Using the same unroll factor of 8, we get the following assembly:

```asm
.L3:
	vmovupd	ymm8, YMMWORD PTR [rdx]
	vmovupd	ymm7, YMMWORD PTR 32[rdx]
	add	ecx, 32
	add	rax, 256
	vmovupd	ymm6, YMMWORD PTR 64[rdx]
	vmovupd	ymm5, YMMWORD PTR 96[rdx]
	add	rdx, 256
	vmovupd	ymm4, YMMWORD PTR -128[rdx]
	vmovupd	ymm3, YMMWORD PTR -96[rdx]
	vmovupd	ymm2, YMMWORD PTR -64[rdx]
	vmovupd	ymm1, YMMWORD PTR -32[rdx]
	vfmadd213pd	ymm8, ymm0, YMMWORD PTR -256[rax]
	vfmadd213pd	ymm4, ymm0, YMMWORD PTR -128[rax]
	vfmadd213pd	ymm7, ymm0, YMMWORD PTR -224[rax]
	vfmadd213pd	ymm3, ymm0, YMMWORD PTR -96[rax]
	vfmadd213pd	ymm6, ymm0, YMMWORD PTR -192[rax]
	vfmadd213pd	ymm2, ymm0, YMMWORD PTR -64[rax]
	vfmadd213pd	ymm5, ymm0, YMMWORD PTR -160[rax]
	vfmadd213pd	ymm1, ymm0, YMMWORD PTR -32[rax]
	vmovupd	YMMWORD PTR -256[rax], ymm8
	vmovupd	YMMWORD PTR -224[rax], ymm7
	vmovupd	YMMWORD PTR -192[rax], ymm6
	vmovupd	YMMWORD PTR -160[rax], ymm5
	vmovupd	YMMWORD PTR -128[rax], ymm4
	vmovupd	YMMWORD PTR -96[rax], ymm3
	vmovupd	YMMWORD PTR -64[rax], ymm2
	vmovupd	YMMWORD PTR -32[rax], ymm1
	cmp	esi, ecx
	jg	.L3
```

As we can see, there are 7 ymm registers that are left unused, so I figured it would be a good idea to unroll the loop even more. However, since the tile size is 32, we cannot go in larger strides. I tried an unroll factor of 9 (stride of 36) and a tile size of 36, but there are virtually no performance improvements, so I decided to stick with the unroll factor of 8. Additionally, a stride of 32 doubles (256 bytes) represents a multiple of the cache line size (64 bytes), so it means that we are not bringing in lines that are going to be just half utilized.


### Vec_add


This operation is very similar to the previous one, but we only use the `_mm256_add_pd` instrinsic instead of the `_mm256_fmadd_pd` one. This has a latency of 10 and a reciprocal throughput of 1, so the theoretical unroll factor is 10. However, after testing and analysing using `llvm-mca`, it seems like there is no significant performance improvement after unrolling by 8, and therefore I decided to stick with this one. The following two llvm-mca timelines show there are virtually no improvements in pipeline utilization when going from an unroll factor of 8 to an unroll factor of 10:


- Unroll factor of 8:
```
[0,18]    .  DeE-----------R  .    .    .    .    .    .    .    .    .    .    .    .   .   add	rax, 256
[0,19]    .  D=eeeeeeeE----R  .    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm4, ymmword ptr [rax - 160]
[0,20]    .  D==eeeeeeeE---R  .    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm3, ymmword ptr [rax - 128]
[0,21]    .   D=eeeeeeeeeeeER .    .    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm4, ymm4, ymmword ptr [rdx - 160]
[0,22]    .   D==eeeeeeeeeeeER.    .    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm3, ymm3, ymmword ptr [rdx - 128]
[0,23]    .   D==eeeeeeeE----R.    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm2, ymmword ptr [rax - 96]
[0,24]    .   D===eeeeeeeE---R.    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm1, ymmword ptr [rax - 64]
[0,25]    .    D==eeeeeeeeeeeER    .    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm2, ymm2, ymmword ptr [rdx - 96]
[0,26]    .    D===eeeeeeeeeeeER   .    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm1, ymm1, ymmword ptr [rdx - 64]
[0,27]    .    D===eeeeeeeE----R   .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax - 32]
[0,28]    .    D====eeeeeeeE---R   .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm7, ymmword ptr [rax - 256]
[0,29]    .    .D===eeeeeeeeeeeER  .    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm0, ymm0, ymmword ptr [rdx - 32]
[0,30]    .    .D====eeeeeeeeeeeER .    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm7, ymm7, ymmword ptr [rdx - 256]
[0,31]    .    .D========eE------R .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 224], ymm6
[0,32]    .    . D========eE-----R .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 192], ymm5
[0,33]    .    . D=========eE----R .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 160], ymm4
[0,34]    .    . D==============eER.    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 256], ymm7
[0,35]    .    .  D=============eER.    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 128], ymm3
[0,36]    .    .  D==============eER    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 96], ymm2
[0,37]    .    .  D==============eER    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 64], ymm1
[0,38]    .    .   D==============eER   .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 32], ymm0
[0,39]    .    .   DeE--------------R   .    .    .    .    .    .    .    .    .    .   .   cmp	esi, ecx
[0,40]    .    .   D=eE-------------R   .    .    .    .    .    .    .    .    .    .   .   jg	.L3
```

- Unroll factor of 10:
```
[0,27]    .    .DeE-----------R    .    .    .    .    .    .    .    .    .    .    .   .   add	rax, 320
[0,28]    .    .D=eeeeeeeE----R    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm6, ymmword ptr [rax - 224]
[0,29]    .    .D==eeeeeeeE---R    .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm5, ymmword ptr [rax - 192]
[0,30]    .    . D=eeeeeeeeeeeER   .    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm6, ymm6, ymmword ptr [rdx - 224]
[0,31]    .    . D==eeeeeeeeeeeER  .    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm5, ymm5, ymmword ptr [rdx - 192]
[0,32]    .    . D==eeeeeeeE----R  .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm4, ymmword ptr [rax - 160]
[0,33]    .    . D===eeeeeeeE---R  .    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm3, ymmword ptr [rax - 128]
[0,34]    .    .  D==eeeeeeeeeeeER .    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm4, ymm4, ymmword ptr [rdx - 160]
[0,35]    .    .  D===eeeeeeeeeeeER.    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm3, ymm3, ymmword ptr [rdx - 128]
[0,36]    .    .  D===eeeeeeeE----R.    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm2, ymmword ptr [rax - 96]
[0,37]    .    .  D====eeeeeeeE---R.    .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm1, ymmword ptr [rax - 64]
[0,38]    .    .   D===eeeeeeeeeeeER    .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm2, ymm2, ymmword ptr [rdx - 96]
[0,39]    .    .   D====eeeeeeeeeeeER   .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm1, ymm1, ymmword ptr [rdx - 64]
[0,40]    .    .   D====eeeeeeeE----R   .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm0, ymmword ptr [rax - 32]
[0,41]    .    .   D=====eeeeeeeE---R   .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymm9, ymmword ptr [rax - 320]
[0,42]    .    .    D====eeeeeeeeeeeER  .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm0, ymm0, ymmword ptr [rdx - 32]
[0,43]    .    .    D=====eeeeeeeeeeeER .    .    .    .    .    .    .    .    .    .   .   vaddpd	ymm9, ymm9, ymmword ptr [rdx - 320]
[0,44]    .    .    D=======eE--------R .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 288], ymm8
[0,45]    .    .    .D=======eE-------R .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 256], ymm7
[0,46]    .    .    .D========eE------R .    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 224], ymm6
[0,47]    .    .    .D===============eER.    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 320], ymm9
[0,48]    .    .    . D==============eER.    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 192], ymm5
[0,49]    .    .    . D===============eER    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 160], ymm4
[0,50]    .    .    . D===============eER    .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 128], ymm3
[0,51]    .    .    .  D===============eER   .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 96], ymm2
[0,52]    .    .    .  D===============eER   .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 64], ymm1
[0,53]    .    .    .  D================eER  .    .    .    .    .    .    .    .    .   .   vmovupd	ymmword ptr [rax - 32], ymm0
[0,54]    .    .    .   DeE---------------R  .    .    .    .    .    .    .    .    .   .   cmp	r8d, ecx
[0,55]    .    .    .   D=eE--------------R  .    .    .    .    .    .    .    .    .   .   jg	.L3
```


# Comparisons and Graphs

This section is based on the analysis of the benchmarks that I ran for all the versions discussed, as well as the `cachegrind` reports. All benchmarks and cachegrind reports can be found in the `results/` directory. The values used for `N` are: 400, 800, 1000, 1200, 1400.

To make the process easier, I have prompted GPT-5.4 (via Codex) to analyze the results and create python scripts for generating the graphs. The prompt I used can be found in `prompt.txt`.

## `neopt_equiv` -> `opt_base` -> `opt_inline` -> `opt_avx`

Graphs: 

- [Runtime evolution](graphs/runtime_optimization_steps.svg)
- [Per-step speedups](graphs/step_speedups.svg)
- [Cache metrics along the optimization path](graphs/cache_metrics_optimization.svg)
- [Instruction and branch metrics](graphs/instruction_branch_metrics.svg)

### `neopt_equiv` -> `opt_base`

Besides the AVX implementation, the `opt_base` shows the biggest performance uplift over the non-optimized version. This can be explained by the drop in instruction and data references, D1 and LLC misses.

For example, for `N=1000`:
- instruction references drop from `39.26G` to `25.27G` (`1.55x` fewer),
- D1 misses drop from `1.69B` to `78.9M` (`21.4x` fewer),
- last-level misses drop from `1.05M` to `0.89M`.

This can be mostly attributed to the cache-locality improvements brought by the tiled matrix multiplication, `k-i-j` loop reordering and the packed manner in which `D` is stored in memory. Data references also drop due to optimizations such as avoiding the computation of the transpose.

### `opt_base` -> `opt_inline`

The cache plots show almost no change in D1 or LL misses between `opt_base` and `opt_inline`, which is exactly what should happen: the memory access pattern is the same. The gain comes from fewer executed instructions, because the hot helper calls disappear even though the program is compiled with `-O0`. At `N=1400`, instruction references drop from `68.99G` to `52.92G`, a reduction of about `1.30x`, which matches the runtime improvement closely.

### Choosing the right block size

Graph:

- [Block-size comparison](graphs/opt_inline_block_sizes.svg)

This result matches the L1-based estimate better than either alternative. `b=24` leaves locality on the table because the working tiles are too small, so loop overhead becomes more visible. `b=40` goes in the other direction: it reduces the number of block iterations, but the tiles are large enough to push more pressure onto the effective L1 working set. Even though cachegrind does not fully penalize `b=40` in every metric, the native runtime benchmark consistently shows that `32` is the best balance on this Haswell system.


### `opt_inline` -> `opt_avx`

The AVX implementation shows a significant performance improvements in runtime. As we can see in the cache graphs, the cache-related metrics remain essentially unchanged, since we did not modify the memory access patterns. The gain comes from fewer instruction/data references (since a single SIMD instruction processes in bulk) and fewer branches and brach misses due to the loop unrolling.

Speedup of `opt_avx` over `opt_inline b=32`:

- `2.75x` at `N=400`
- `3.13x` at `N=1000`
- `3.00x` at `N=1400`

For example, at `N=1000`, cachegrind shows:

- D1 misses are essentially unchanged (`78.9M` vs `78.3M`),
- LL misses are unchanged at about `0.89M`,
- instruction references drop from `19.26G` to `3.56G` (`5.41x` fewer),
- branch mispredicts drop from `33.01M` to `3.01M` (`10.95x` fewer).

This is exactly how AVX/FMA algorithms behave: the memory traffic remains unchanged, but each loop iteration performs more work.

## Final Comparison

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

The comparison with BLAS shows a something different. `opt_avx` is much faster than the custom scalar code, but BLAS is still ahead by more than `2x`. That is expected. Vendor-tuned BLAS kernels combine stronger vectorization, better micro-kernel design, better packing, and decades of architecture-specific tuning. For example, at `N=1400` BLAS executes about `3.88G` instructions versus `10.34G` for `opt_avx`, and it also has far fewer branch mispredicts.
