# Assignment 2 - GPU Miner

## General overview

The assignment was realized entirely within `gpu_miner/utils.cu`, as required by the assignment instructions. The main goal was to parallelize the logic of the `construct_merkle_root` and `find_nonce` functions from `cpu_miner` using CUDA. The implementation of the sha256 hashing was left untouched.

## Implementation details

### `construct_merkle_root`

For the Merkle root construction, I first analyzed the structure of the Merkle tree and how the hashes are combined. Since the Merkle tree is a binary tree (almost), distinct subtrees are completely independent of each other, which allows the parallelization at a subtree level. Each subtree reduces in the end to a single node, provided that the number of leaves in that subtree is a power of 2. That means that the number of nodes gets divided by the number of leaves in the subtree, ending up with either a single node or a much smaller tree. For choosing the thread block size and grid size, instead of hardcoding the values, I decided to make use of the CUDA occupancy API, which provides the optimal block size and grid size for a given kernel, based on the resource usage of the kernel and the hardware specifications of the GPU. Because the grid size cannot stay fixed (there is a 1:2 correspondence between the number of threads and the number of transactions processed), I computed it as `ceil(num_transactions / transactions_per_block)`, where `transactions_per_block` is (initially) `2 * blockSize`. As we established earlier, the number of transactions assigned per block should be a power of 2, so I rounded down the `blockSize` value returned by the occupancy API to the nearest power of 2. 

Each thread in a block is responsible for hashing and combining two transactions, and after each level of the tree is processed, the number of nodes halves, and so does the number of threads needed (so half of the active threads become idle). This process continues until we are left with a single node, which is the Merkle root. At the end, the thread with thread block id `0` copies the Merkle root back into the hashes output array on the position `blockIdx.x`.

Inside the kernel, each thread first computes the hash of two transactions (because they are not provided as hashes, but rather as raw transactions), and stores them in the shared buffer. I used a shared buffer for the hashes processed by a thread block to avoid the overhead of global memory accesses. Then each thread proceeds as discussed, combining the hashes and storing the results back in the shared buffer until we are left with a single hash, which is the Merkle root. At each iteration (level of the tree), the threads in the block synchronize to ensure that all threads have completed their work before proceeding to the next level.

A "trick" that I used to avoid thread divergence as much was based on the observation that only the last block would ever need to process a number of transactions that is odd. Therefore, for the rest of the blocks I assumed that I can always combine two consecutive transactions. The last block processes them normally, by checking if the second transaction exists or not.

To summarize, the workflow is the following:
1. Split the transactions into blocks of `MAX_TRANSACTIONS_PER_BLOCK` transactions, and launch a kernel containing `num_blocks` blocks, where `num_blocks` is the number of blocks needed to process all transactions. The assumption was that based on the provided tests, the number of blocks in the grids does not exceed the maximum number of blocks allowed by the GPU (which is 2^31 - 1 for the devices I tested on).
2. Each thread in a block computes the hash of two transactions and stores them in the shared buffer.
3. Each thread in a block combines the hashes of two transactions and stores the result back in the shared buffer, until we are left with a single hash, which is the Merkle root.
4. The thread with thread block id `0` copies the Merkle root back into the hashes output array on the position `blockIdx.x`.
5. The CPU waits for the kernel to finish and then it computes the Merkle root of the Merkle roots (which is a much smaller tree) using the CPU implementation.

Empirical testing showed that the `blockSize` value returned by the occupancy API is indeed the optimal value. Manually varying the block size did not lead to significant improvements, and in some cases even led to worse performance.

The existence of the occupancy API was brought to my attention by GPT 5.5.


##  `find_nonce`

This function is more straightforward to parallelize. Since for every number in a range we have to do the same work, that means they are completely independent of each other, making it embarrassingly parallel. 

The first idea for the implementation would be to have each thread responsible for checking a single nonce. However, since the range of nonces can be very large (up to 2^32), this would lead to a very large number of threads, which is not feasible. Therefore, the chosen approach was to have each thread handle 1 or more nonces, in a grid-strided manner (i.e. each thread processes a nonce, then jumps by the total number of threads in the grid to process another nonce, and so on). When it comes to choosing the optimal grid size and block size, I proceeded as with the `construct_merkle_root` function, by using the CUDA occupancy API to determine the optimal block size and grid size. This time, however, the grid size can stay fixed, since a single thread does more work. The API returns the `minGridSize`, which is the minimum grid size needed to achieve the maximum occupancy, but that does not mean that a larger grid size won't lead to better performance. Therefore, I decided to use a grid size of `minGridSize * k`, where `k` is a small constant integer that allows for fine-tuning. Testing multiple values of `k` showed that a value in the range of `8-16` lead to the best performance, and also that these block/grid sizes outperform or at least match the performance of hardcoded values, with the added benefit of being adaptable to different GPU architectures.

For the `difficulty` array, I decided to use the constant memory, which is the best option here since the length is fixed and small, and all threads read the same values (and implicitly the same address), which is more efficient than using shared memory or global memory (+caching).

The "synchronization" between threads is done using an atomic variable `d_found`, which is initially `0`, and is set to `1` by the first thread that finds a valid nonce. Each thread checks this variable before doing the work for a nonce, and if it is set to `1`, it stops.

A useful optimization that I implemented is based on the observation that the sha256 hash is computed by hashing `concat(block_content, nonce)`, and the `block_content` is fixed for all nonces. The sha256 context computation allows to incrementally add strings before finalizing the hash through the `sha256_update` function. This function basically appends the provided string to the context. Therefore, istead of completely recomputing the sha256 context for each nonce, the context can be precomputed once for the `block_content`, and then for each nonce, it can just be copied over and updated with the nonce, which is much faster. For the same reason as before, I used contant memory to store the precomputed context, since it is read-only and shared by all threads.

The kernel is very simple and straightforward. Each thread starts with the nonce equal to its thread id in the grid, and then while the current nonce is less than the `max_nonce` and the `d_found` variable is not set, it computes the hash for the current nonce, checks if it is valid, and if it is, it sets the `d_found` variable and copies the nonce to the output variable. Then it increments the nonce by the stride (total number of threads in the grid) and continues. The memory order used for `d_found` is `memory_order_relaxed`, since we don't need any ordering guarantees between threads; we just need to ensure atomicity and ensure that the compiler does not optimize away the loads/stores from/to this variable. The value likely does not propagate immediately to other threads, but that is not a problem, since the threads will eventually see the updated value and stop.


## Common optimizations

For both functions, I slightly modified the `apply_sha256` logic. I noticed that an intermediate buffer was used for storing the hash output, which was then copied to the final output buffer. This is not necessary, since the raw hash can be written directly to the output buffer, and then the hash in hex format can be constructed by iterating over the raw hash in the output buffer in reverse order.


# AI Conversations

AI was used briefly for research/conceptual undestanding purposes. The model used was GPT 5.5.

## ChatGPT-Merkle_Root_CUDA_Optimization

````md
# Merkle Root CUDA Optimization

#### You:
Analyze `utils.c`. It contains the implementation of the Merkle root finding algorithm for a list of elements, but using hashes in their hex format (so SHA256_HASH_SIZE= 65 = 32 * 2 + 1 for the null terminator). Now I want to implement the exact same algorithm but in Cuda. My idea is that because this seems to be easily split into independent subproblems (subtrees), that means that  I could compute the Merkle root for subtrees, which then become leaves again and I apply the algorithm again on them. To be able to do this, I would have to initially choose chunks of N elements (where N is a power of 2 so that it perfectly reduces to 1 root), except for the last chunk which can have less elements and will be filled with duplicates as we go. Now this would translate to having either a kernel launched for each subtree (perhaps in distinct streams), or a thread block in a grid taking care of a subtree, but I am not sure which would be more optimal. What do you think, is this algorithm idea acceptable? Which approach would you recommend, 1 kernel launch per each subtree or a single kernel launch, having one large grid with each block computing the root for a subtree? Are there any other things that I should consider, such as using shared memory? Provide a concise response, without any code.

#### ChatGPT:
I’ll quickly inspect `utils.c` to ground the advice in the actual CPU algorithm, then I’ll focus only on the CUDA design trade-offs.

Yes, the idea is acceptable, but I would **not** launch one kernel per subtree. I would use **one kernel launch with a grid**, where each block computes one subtree/chunk.

In your CPU code, the Merkle root algorithm is:

1. Hash each transaction into a 64-char hex SHA256 string.
2. Repeatedly combine pairs as `hash[i] || hash[i+1]`.
3. If the level has an odd number of hashes, duplicate the last one.
4. Hash the concatenated 128-char string.
5. Continue until one hash remains.

### Recommended CUDA structure

Use a **single kernel launch** where:

- each block handles one chunk/subtree;
- each thread initially hashes one or more transactions;
- the block reduces the chunk level by level;
- the final root of each chunk is written to global memory;
- then you repeat the process on the CPU/GPU over the chunk roots until one root remains.

This is better than “one kernel launch per subtree” because kernel launches have non-trivial overhead. Streams will not help much unless each subtree is large enough, and for small chunks the overhead dominates.

So prefer:

```text
kernel<<<num_chunks, threads_per_block>>>
    block 0 -> subtree 0
    block 1 -> subtree 1
    block 2 -> subtree 2
    ...
```

not:

```text
launch kernel for subtree 0
launch kernel for subtree 1
launch kernel for subtree 2
...
```

### Chunk size

Choose a power-of-two chunk size, for example 128, 256, or 512 leaves per subtree.

A power of two simplifies the reduction, but you do **not** need to physically pad the input. You can just apply the “duplicate last hash” rule during reduction.

For small inputs, like ≤10 transactions, CUDA is probably not worth it. The CPU version may be faster because SHA256 + Merkle reduction is too small to amortize GPU overhead.

### Shared memory

Yes, shared memory is useful inside each block.

A good layout is:

```text
shared hashes[chunk_size][65]
```

or better:

```text
shared binary_hashes[chunk_size][32]
```

The second option is much better.

Your current algorithm stores hashes as hex strings, so every internal node hashes 128 ASCII bytes. That matches your CPU behavior, so if you need **exact same output**, you must preserve that behavior.

But internally, hex format is inefficient:

- 32-byte SHA256 digest becomes 64 bytes;
- concatenating two hashes becomes 128 bytes instead of 64;
- more memory traffic;
- more shared/global memory usage;
- more string-like operations.

If exact compatibility with `utils.c` is required, keep hex. If not, use raw 32-byte hashes and convert only the final root to hex.

### Important CUDA considerations

Avoid `strcpy`, `strcat`, `strlen`, `sprintf`, etc. in device code. The lengths are fixed, so use fixed-size copies:

- transaction length is `transaction_size`;
- hash length is 64 chars + null;
- combined internal input is exactly 128 chars if using hex;
- no need for null terminators except for compatibility with your SHA256 wrapper.

Also avoid dynamic allocation inside kernels. Preallocate global buffers.

### Bigger picture

The Merkle reduction itself has limited parallelism as the tree gets smaller. The most parallel part is the **leaf hashing** and the first few reduction levels. That is why one block per subtree is a good mapping: it gives parallelism across chunks and local synchronization inside each subtree.

Final recommendation:

> Use one/few kernel launches over a large grid, with one block per subtree, using shared memory for the subtree hashes. Avoid one kernel launch per subtree. Keep hex only if you need exact compatibility with `utils.c`; otherwise use raw 32-byte hashes internally.
````

## ChatGPT-CUDA_Nonce_Search_Optimization

````md
# CUDA Nonce Search Optimization

#### You:
Analyze `utils.c`. It contains the implementation of the nonce finding algorithm., The hashes are represented in their hex format (so SHA256_HASH_SIZE= 65 = 32 * 2 + 1 for the null terminator). Now I want to implement the exact same algorithm but in Cuda. My idea is to have each thread test 1 or more nonces, and when a threads finds a nonce, it updates a `found` flag in global memory, as well as stores the found nonce. For the `difficulty` and `block_content` arrays, which are the fixed across all threads, I think I can have them stored in shared memory by thread with id 0 in the thread block. Now for the approach itself, I can either have each thread handle just one nonce, in which case there would be a lot of threads, or test several nonces within each thread, which would reduce the number of threads/memory needed, but would likely increase the run time. For computing the hash of the block content with the found nonce, I would say it would be easier to do it on the cpu, after the gpu execution finishes. What do you think? Is this idea acceptable? What approach would you consider better and why? Should I dynamically decide the number of nonces per thread, thread blocks based on the `max_nonce` and the device properties? Provide a concise response, without any code.

#### ChatGPT:
Yes, the idea is acceptable, and it maps much better to CUDA than the Merkle-root computation.

For nonce search, the work is embarrassingly parallel:

```text
nonce -> build block string -> SHA256 -> compare with difficulty
```

Each nonce is independent, so one CUDA thread testing one or more nonces is the right model.

### Recommended approach

I would use **one kernel launch with many threads**, where each thread tests multiple nonces using a grid-stride loop:

```text
global_tid = blockIdx.x * blockDim.x + threadIdx.x
stride = gridDim.x * blockDim.x

for nonce = global_tid; nonce <= max_nonce; nonce += stride:
    if found flag is set: stop
    test nonce
```

Conceptually, this is better than “one thread = one nonce” because:

- it avoids needing an enormous number of threads for large `max_nonce`;
- it keeps the GPU saturated without launching too many blocks;
- it naturally adapts to any `max_nonce`;
- it lets threads exit early when another thread finds the answer.

So yes: **each thread should usually test several nonces**, but not by manually assigning a fixed small chunk. A grid-stride loop is the cleanest version.

### About the `found` flag

Use a global `found` flag and `found_nonce`.

When a thread finds a valid nonce:

- use an atomic operation, for example `atomicCAS`, to ensure only the first successful thread stores the result;
- other threads periodically check `found` and return early.

Without atomic protection, multiple valid threads could race and overwrite `found_nonce`.

### About `difficulty` and `block_content`

Your instinct is mostly right, but I would not necessarily put them in shared memory.

For `difficulty`, if it is small and read-only, good options are:

- **constant memory**, probably best;
- or just global memory, because it will be cached.

For `block_content`, if all threads read the same fixed prefix, it can also go in:

- constant memory, if small enough;
- global memory, if simpler;
- shared memory only if you actually benefit from staging it per block.

Shared memory is not automatically faster here. If thread 0 copies `block_content` into shared memory, you also need synchronization, and every block repeats the copy. For read-only data shared by all threads, **constant memory is often a better fit**.

### About computing the final hash on CPU

Yes, that is reasonable.

The GPU only needs to find a valid nonce. After the kernel finishes, the CPU can recompute:

```text
SHA256(block_content + nonce)
```

This is cheap and also useful for validation/debugging.

### Choosing blocks and threads

Yes, choose the launch configuration based on `max_nonce` and device properties, but keep it simple.

A good practical strategy:

- use 128 or 256 threads per block;
- launch enough blocks to occupy the GPU, not necessarily one block per nonce;
- cap the number of blocks to some multiple of the number of SMs;
- use a grid-stride loop to cover all nonces.

For example, conceptually:

```text
threads_per_block = 256
blocks = min(ceil(max_nonce / threads_per_block), SM_count * some_factor)
```

The `some_factor` could be something like 4, 8, 16, etc. Exact tuning depends on the SHA256 implementation and register usage.

### Main design choice

I would not use:

```text
one CUDA thread per nonce only
```

as the main design, because it scales poorly and gives less control.

I would use:

```text
fixed-size CUDA grid + grid-stride loop
```

because it is simple, scalable, and standard for this kind of brute-force search.

Overall: your idea is good. The main adjustment is to avoid thinking in terms of “one nonce per thread vs fixed multiple per thread” and instead use a **grid-stride nonce search**, with read-only data in constant/global memory and an atomic global `found` flag.


#### You:
In CUDA, how do you dynamically (at runtime) calculate the optimal number of threads in a block and blocks in a grid based on the CUDA device properties? By dynamic I mean that you do not hardcode a values, such as 256 threads per block, but rather you calculate these values based on the device properties and perhaps other information. For example if there is a constraint about the max number of threads in a SM, or the max number of blocks in a SM. This seems like an essential thing to do to ensure the GPU is fully utilized. Are there both manual and automatic ways to find these optimal values? If so, present both of them. Keep your response concise. This time you can show snippets of code, but just for illustrating the API needed.

#### ChatGPT:
Yes, there are **manual** and **automatic** ways.

The important point: “fully utilized” does **not** always mean “maximum threads”. The best launch config depends on:

- register usage per thread
- shared memory usage per block
- memory vs compute intensity
- occupancy
- instruction-level parallelism
- early exits / branch divergence

So the runtime-calculated config is usually a **good starting point**, not guaranteed optimal.

---

## 1. Automatic way: CUDA occupancy API

CUDA provides APIs that suggest block size / grid size based on occupancy.

### Recommended API

```cpp
int minGridSize = 0;
int blockSize = 0;

cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    my_kernel,
    0,      // dynamic shared memory per block
    0       // block size limit, 0 = no explicit limit
);
```

Then compute the grid size from your problem size:

```cpp
int gridSize = (num_items + blockSize - 1) / blockSize;

my_kernel<<<gridSize, blockSize>>>(...);
```

For a grid-stride loop, you can also cap/adjust the grid size:

```cpp
int smCount = 0;
cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

gridSize = min(gridSize, minGridSize * 4);
```

The exact multiplier is workload-dependent. For brute-force nonce search, a few blocks per SM is usually enough because each thread loops over many nonces.

---

## 2. Automatic way with dynamic shared memory

If your kernel uses runtime-sized shared memory:

```cpp
auto sharedMemForBlock = [](int blockSize) {
    return blockSize * sizeof(int);
};

cudaOccupancyMaxPotentialBlockSizeVariableSMem(
    &minGridSize,
    &blockSize,
    my_kernel,
    sharedMemForBlock,
    0
);
```

This lets CUDA account for shared memory limiting occupancy.

---

## 3. Manual way: inspect device limits

You can read device properties yourself:

```cpp
cudaDeviceProp prop{};
cudaGetDeviceProperties(&prop, device);

int smCount = prop.multiProcessorCount;
int maxThreadsPerBlock = prop.maxThreadsPerBlock;
int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
int maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
int warpSize = prop.warpSize;
```

Then choose values based on constraints.

Example heuristic:

```cpp
int threadsPerBlock = min(256, maxThreadsPerBlock);

// round down to multiple of warp size
threadsPerBlock = (threadsPerBlock / warpSize) * warpSize;

int targetBlocksPerSM = 4;
int gridSize = smCount * targetBlocksPerSM;
```

For a grid-stride kernel:

```cpp
my_kernel<<<gridSize, threadsPerBlock>>>(...);
```

This avoids launching a huge grid while still keeping the GPU occupied.

---

## 4. Manual occupancy calculation

You can also ask CUDA how many blocks per SM are possible for a chosen block size:

```cpp
int activeBlocksPerSM = 0;

cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &activeBlocksPerSM,
    my_kernel,
    threadsPerBlock,
    dynamicSharedMemBytes
);
```

Then estimate occupancy:

```cpp
int activeThreadsPerSM = activeBlocksPerSM * threadsPerBlock;

double occupancy =
    double(activeThreadsPerSM) / prop.maxThreadsPerMultiProcessor;
```

This is useful if you want to test several block sizes:

```cpp
for (int blockSize = warpSize; blockSize <= prop.maxThreadsPerBlock; blockSize += warpSize) {
    int activeBlocks = 0;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocks,
        my_kernel,
        blockSize,
        0
    );

    double occ = double(activeBlocks * blockSize)
               / prop.maxThreadsPerMultiProcessor;

    // keep the blockSize with best occupancy
}
```

---

## Practical recommendation

For your nonce search kernel, I would use:

```cpp
cudaOccupancyMaxPotentialBlockSize(...)
```

to get `blockSize`, then use a **grid-stride loop** inside the kernel.

Something like:

```cpp
int minGridSize = 0;
int blockSize = 0;

cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    find_nonce_kernel,
    0,
    0
);

int gridSize = minGridSize;
find_nonce_kernel<<<gridSize, blockSize>>>(...);
```

Inside the kernel:

```cpp
for (uint64_t nonce = global_tid;
     nonce <= max_nonce;
     nonce += gridDim.x * blockDim.x) {
    // test nonce
}
```

This is usually better than trying to launch one CUDA thread per nonce.

---

## Summary

Use **automatic occupancy APIs** when you want a good default:

```cpp
cudaOccupancyMaxPotentialBlockSize
cudaOccupancyMaxPotentialBlockSizeVariableSMem
```

Use **manual device properties** when you want more control:

```cpp
cudaGetDeviceProperties
cudaOccupancyMaxActiveBlocksPerMultiprocessor
```

For your case, the best design is probably:

```text
runtime-selected block size + enough blocks to fill SMs + grid-stride loop
```

not a massive grid with one thread per nonce.


#### You:
I have implemented the nonce finding algorithm:
```
__constant__ BYTE d_constant_difficulty[SHA256_HASH_SIZE];
__constant__ BYTE d_constant_block_content[2 * SHA256_HASH_SIZE];

__global__ void find_nonce_kernel(uint32_t max_nonce, size_t current_length,
                                  BYTE *result_hash, uint32_t* found_nonce, uint32_t *found) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;

  if (tid >= max_nonce || *found == 1) {
    return;
  }

  BYTE nonce_string[2 * SHA256_HASH_SIZE + NONCE_SIZE] = {};
  BYTE hash[SHA256_HASH_SIZE] = {};
  memcpy(nonce_string, d_constant_block_content, current_length);

  for (uint64_t nonce = tid; nonce < max_nonce;
       nonce += stride) {

    int nonce_length =
        intToString(nonce, (char *)nonce_string + current_length);

    apply_sha256(nonce_string, hash);

    if (compare_hashes(hash, d_constant_difficulty) <= 0) {
      if (atomicCAS(found, 0, 1) == 0) {
        memcpy(result_hash, hash, SHA256_HASH_SIZE);
        *found_nonce = nonce;
      }
      return;
    }
  }
}

// TODO 2: Implement this function in CUDA
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content,
               size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
  int minGridSize = 0;
  int blockSize = 0;

  cudaOccupancyMaxPotentialBlockSize(
      &minGridSize,
      &blockSize,
      find_nonce_kernel,
      0,      // dynamic shared memory per block
      0       // block size limit, 0 = no explicit limit
  ); 

  uint8_t *cuda_buf;
  const size_t total_size = 2 * sizeof(uint32_t) + SHA256_HASH_SIZE;
  cudaMalloc((void **)&cuda_buf, total_size);
  cudaMemset(cuda_buf, 0, total_size);
  uint32_t *d_found = (uint32_t *)cuda_buf;
  uint32_t *d_found_nonce = (uint32_t *)(cuda_buf + sizeof(uint32_t));
  BYTE *result_hash = (BYTE *)(cuda_buf + 2 * sizeof(uint32_t));

  cudaMemcpyToSymbol(d_constant_difficulty, difficulty, SHA256_HASH_SIZE);
  cudaMemcpyToSymbol(d_constant_block_content, block_content,
                     current_length + 1);

  uint32_t nonces_per_thread = 1;
  uint32_t num_blocks = ((uint64_t)max_nonce / nonces_per_thread + blockSize - 1) / blockSize;

  find_nonce_kernel<<<num_blocks, blockSize>>>(
      max_nonce, current_length, result_hash, d_found_nonce, d_found);

  cudaDeviceSynchronize();

  uint32_t found;
  cudaMemcpy(&found, d_found, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (found == 1) {
    cudaMemcpy(block_hash, result_hash, SHA256_HASH_SIZE,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(valid_nonce, d_found_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  }

  cudaFree(cuda_buf);

  return 1 - found;
}
```

Analyze it, then provide me with a feedback for it. Are there any obvious bottlenecks in this code? From what I can see it should run faster than it currently does. Point out any errors and inefficiencies. Prefer a concise response.

#### ChatGPT:
I’ll focus on correctness first, then the main CUDA performance issues that likely explain why it is slower than expected.

Main issue: your kernel is probably **not really using the grid-stride loop effectively**.

You compute:

```cpp
num_blocks = ceil(max_nonce / blockSize)
stride = gridDim.x * blockDim.x
```

So `stride >= max_nonce`, meaning almost every thread tests **only one nonce**. That creates a huge grid and removes the benefit of the loop.

For nonce search, better:

```cpp
num_blocks = min(minGridSize * k, ceil(max_nonce / blockSize));
```

where `k` could be `2`, `4`, `8`, etc. Then each thread tests multiple nonces via the grid-stride loop.

---

## Important issues / inefficiencies

### 1. You only check `found` once

Currently:

```cpp
if (tid >= max_nonce || *found == 1) return;
```

But inside the loop, threads do not stop after another thread finds the nonce.

Add a periodic check inside the loop:

```cpp
if (*found) return;
```

Maybe not every iteration if the loop is very small, but for larger searches it matters.

---

### 2. You launch too many blocks

This:

```cpp
num_blocks = ceil(max_nonce / blockSize);
```

can create far more blocks than needed.

For a grid-stride kernel, you usually want enough blocks to occupy the GPU, not one thread per nonce.

Better conceptually:

```cpp
num_blocks = minGridSize;
// or
num_blocks = SM_count * 4;
// or
num_blocks = SM_count * 8;
```

Then let the grid-stride loop cover the full nonce range.

---

### 3. Local arrays are expensive

Inside every thread:

```cpp
BYTE nonce_string[2 * SHA256_HASH_SIZE + NONCE_SIZE] = {};
BYTE hash[SHA256_HASH_SIZE] = {};
```

These may cause high register/local memory usage. If they spill to local memory, performance drops a lot.

Also, zero-initializing them:

```cpp
= {};
```

costs extra work per thread.

You probably only need to initialize the required null terminators / touched bytes, not clear the whole arrays.

---

### 4. `memcpy` from constant memory per thread

This happens in every thread:

```cpp
memcpy(nonce_string, d_constant_block_content, current_length);
```

That is okay for correctness, but expensive if done millions of times.

Since every thread must build its own input string, some copy is unavoidable, but you should keep `current_length` small and avoid copying unused bytes.

For better performance, a custom copy loop or optimized SHA input construction may outperform generic `memcpy`.

---

### 5. `nonce_length` is unused

```cpp
int nonce_length = intToString(...);
```

Unused variable. More importantly, make sure `intToString` properly null-terminates the string.

If `apply_sha256()` relies on `strlen`, then null termination is essential.

---

### 6. Possible off-by-one bug

Your loop is:

```cpp
nonce < max_nonce
```

If the original CPU algorithm tests up to and including `max_nonce`, this is wrong. It should be:

```cpp
nonce <= max_nonce
```

Depends on the original `utils.c`.

---

### 7. `uint64_t nonce` but `found_nonce` is `uint32_t`

You use:

```cpp
for (uint64_t nonce = tid; nonce < max_nonce; nonce += stride)
```

but store into:

```cpp
uint32_t *found_nonce
```

Since `max_nonce` is `uint32_t`, this is currently fine, but using `uint64_t` is not really useful unless you also make `max_nonce` 64-bit.

---

### 8. No CUDA error checking

You should check errors after:

```cpp
cudaMalloc
cudaMemcpyToSymbol
kernel launch
cudaDeviceSynchronize
cudaMemcpy
```

At minimum:

```cpp
find_nonce_kernel<<<...>>>(...);
cudaError_t err = cudaGetLastError();
cudaDeviceSynchronize();
```

Without this, the kernel may be failing silently.

---

## Biggest bottleneck

Most likely bottlenecks are:

```text
1. too many blocks / one nonce per thread
2. expensive per-thread local arrays
3. SHA256 implementation not optimized for CUDA
4. repeated string construction + hex string hashing
```

The biggest algorithmic issue is that you hash **ASCII hex/string data**, not compact binary data. If the assignment requires exact behavior, that is fine, but it is much slower than hashing binary buffers.

---

## What I would change first

Use fewer blocks and actually rely on the grid-stride loop:

```cpp
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    find_nonce_kernel,
    0,
    0
);

num_blocks = minGridSize * 4;
```

or:

```cpp
num_blocks = SM_count * 4;
```

Then inside the kernel:

```cpp
for (uint32_t nonce = tid; nonce < max_nonce; nonce += stride) {
    if (*found) return;

    ...
}
```

This should reduce launch overhead pressure, improve scheduling, and avoid creating a massive number of mostly single-use threads.

Overall, the structure is good, but currently the launch configuration defeats the main advantage of the grid-stride design.


#### You:
Keep responses concise.
1. What advantages does constant memory have over the shared/global memory, when accounting for the cache?
2. Is it usually advised to use cuda::atomic instead of the old atomic API (e.g. `atomicCAS()`) and volatile variables?

#### ChatGPT:
### 1. Constant memory advantages

`__constant__` memory is best when **all threads in a warp read the same address**.

Advantages:

- cached separately in the **constant cache**;
- one read can be **broadcast to the whole warp**;
- does not consume shared memory;
- avoids repeated per-block copies into shared memory;
- good for small read-only data like `difficulty`, fixed prefixes, lookup tables.

But if threads read **different constant-memory addresses**, accesses serialize and can be slower than normal cached global memory.

So:

```text
same address across warp  -> constant memory very good
different addresses       -> global memory may be better
```

For your case, `difficulty` is a very good constant-memory candidate. `block_content` is also reasonable, since every thread copies the same prefix.

---

### 2. `cuda::atomic` vs old atomics / volatile

Yes, in modern CUDA C++, prefer `cuda::atomic` / `cuda::atomic_ref` when possible because they give explicit:

- memory order;
- memory scope;
- clearer semantics;
- more C++-like code.

Example conceptually:

```cpp
cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(*found);
flag.store(1, cuda::memory_order_release);
```

and readers use acquire/relaxed loads depending on need.

But old APIs like:

```cpp
atomicCAS(found, 0, 1)
```

are still common, valid, and often simpler for CUDA kernels.

Avoid relying on `volatile` for synchronization. `volatile` is **not a synchronization primitive**. Use atomics for inter-thread communication.

For your case, `atomicCAS(found, 0, 1)` is perfectly fine. Just read `found` through an atomic/load mechanism if you want clean correctness.
````