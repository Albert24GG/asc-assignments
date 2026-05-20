#include "utils.h"
#include <cstdint>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <sha256.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// CUDA sprintf alternative for nonce finding. Converts integer to its string
// representation. Returns string's length.
__device__ int intToString(uint64_t num, char *out) {
  if (num == 0) {
    out[0] = '0';
    out[1] = '\0';
    return 1;
  }

  int i = 0;
  while (num != 0) {
    int digit = num % 10;
    num /= 10;
    out[i++] = '0' + digit;
  }

  // Reverse the string
  for (int j = 0; j < i / 2; j++) {
    char temp = out[j];
    out[j] = out[i - j - 1];
    out[i - j - 1] = temp;
  }
  out[i] = '\0';
  return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str) {
  size_t len = 0;
  while (str[len] != '\0') {
    len++;
  }
  return len;
}

// CUDA strcpy implementation.
__device__ void d_strcpy(char *dest, const char *src) {
  int i = 0;
  while ((dest[i] = src[i]) != '\0') {
    i++;
  }
}

// CUDA strcat implementation.
__device__ void d_strcat(char *dest, const char *src) {
  while (*dest != '\0') {
    dest++;
  }
  while (*src != '\0') {
    *dest = *src;
    dest++;
    src++;
  }
  *dest = '\0';
}

// Compute SHA256 and convert to hex
__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
  SHA256_CTX ctx;
  static constexpr auto hex_chars = "0123456789abcdef";
  const size_t input_length = d_strlen((const char *)input);

  sha256_init(&ctx);
  sha256_update(&ctx, input, input_length);
  sha256_final(&ctx, output);

  output[SHA256_HASH_SIZE - 1] = '\0'; // Null-terminate
  for (int i = SHA256_BLOCK_SIZE - 1; i >= 0; --i) {
    const BYTE byte = output[i];
    output[i * 2] = hex_chars[(byte >> 4) & 0x0F]; // High nibble
    output[i * 2 + 1] = hex_chars[byte & 0x0F];    // Low nibble
  }
}

// Compare two hashes
__host__ __device__ int compare_hashes(BYTE *hash1, BYTE *hash2) {
  for (int i = 0; i < SHA256_HASH_SIZE - 1; i++) {
    if (hash1[i] < hash2[i]) {
      return -1; // hash1 is lower
    } else if (hash1[i] > hash2[i]) {
      return 1; // hash2 is lower
    }
  }
  return 0; // hashes are equal
}

// TODO 1: Implement this function in CUDA
static void construct_merkle_root_cpu(int hash_size, BYTE *hashes, int n,
                                      BYTE *merkle_root) {

  // Build the Merkle tree
  BYTE combined[SHA256_HASH_SIZE * 2];

  while (n > 1) {
    int new_n = 0;
    int i;
    for (i = 0; i < n - 1; i += 2, ++new_n) {
      strcpy((char *)combined, (const char *)(hashes + i * hash_size));
      strcat((char *)combined, (const char *)(hashes + (i + 1) * hash_size));
      apply_sha256(combined, hashes + new_n * hash_size);
    }

    if (i < n) {
      // If odd number of hashes, duplicate the last one
      strcpy((char *)combined, (const char *)(hashes + i * hash_size));
      strcat((char *)combined, (const char *)(hashes + i * hash_size));
      apply_sha256(combined, hashes + new_n * hash_size);
      ++new_n;
    }
    n = new_n;
  }

  memcpy(merkle_root, hashes, SHA256_HASH_SIZE);
}

__global__ void construct_merkle_root_kernel(BYTE *d_hashes,
                                             BYTE *d_transactions, int n,
                                             int transaction_size) {
  // Thread id in the grid
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

  // Thread id in the block
  const int tid = threadIdx.x;

  static constexpr int COMBINED_SIZE = 2 * SHA256_HASH_SIZE - 1;
  BYTE combined[COMBINED_SIZE];
  extern __shared__ BYTE shared_buf[];

  // Load the transactions from vram, compute their hashes, and store in shared
  // memory
  {
    const int left_idx = 2 * gtid;
    if (left_idx < n) {
      if (left_idx + 1 < n) {
        memcpy(combined, d_transactions + left_idx * transaction_size,
               transaction_size);
        apply_sha256(combined, shared_buf + 2 * tid * SHA256_HASH_SIZE);
        memcpy(combined, d_transactions + (left_idx + 1) * transaction_size,
               transaction_size);
        apply_sha256(combined, shared_buf + (2 * tid + 1) * SHA256_HASH_SIZE);
      } else {
        // If odd number of hashes, duplicate the last one
        memcpy(combined, d_transactions + left_idx * transaction_size,
               transaction_size);
        apply_sha256(combined, shared_buf + 2 * tid * SHA256_HASH_SIZE);
        memcpy(shared_buf + (2 * tid + 1) * SHA256_HASH_SIZE,
               shared_buf + 2 * tid * SHA256_HASH_SIZE, SHA256_HASH_SIZE);
      }
    }
  }

  const int transactions_in_block =
      min(blockDim.x * 2, n - blockIdx.x * blockDim.x * 2);
  int cur_n = transactions_in_block;

  bool active = true;

  // If it is the last block, we have to check if the number of transactions is
  // odd and handle the last hash accordingly Otherwise, we can assume that the
  // number of transactions is even and we can simply combine pairs of hashes

  if (blockIdx.x == gridDim.x - 1) {

    const int left_idx = 2 * tid;
    while (cur_n > 1) {
      const int threads_needed = (cur_n + 1) / 2;

      if (tid < threads_needed) {
        const int right_idx = min(left_idx + 1, cur_n - 1);

        memcpy(combined, shared_buf + left_idx * SHA256_HASH_SIZE,
               SHA256_HASH_SIZE);
        d_strcat((char *)combined,
                 (const char *)(shared_buf + right_idx * SHA256_HASH_SIZE));

        apply_sha256(combined, combined);
      } else {
        active = false;
      }

      __syncthreads();

      if (active) {
        memcpy(shared_buf + tid * SHA256_HASH_SIZE, combined, SHA256_HASH_SIZE);
      }

      __syncthreads();

      cur_n = (cur_n + 1) / 2;
    }
  } else {

    const int left_idx = 2 * tid;
    const int right_idx = left_idx + 1;

    while (cur_n > 1) {
      const int threads_needed = cur_n / 2;

      if (tid < threads_needed) {
        memcpy(combined, shared_buf + left_idx * SHA256_HASH_SIZE,
               SHA256_HASH_SIZE);
        d_strcat((char *)combined,
                 (const char *)(shared_buf + right_idx * SHA256_HASH_SIZE));
        apply_sha256(combined, combined);
      } else {
        active = false;
      }

      __syncthreads();

      if (active) {
        memcpy(shared_buf + tid * SHA256_HASH_SIZE, combined, SHA256_HASH_SIZE);
      }

      __syncthreads();

      cur_n = cur_n / 2;
    }
  }

  if (tid == 0) {
    memcpy(d_hashes + blockIdx.x * SHA256_HASH_SIZE, shared_buf,
           SHA256_HASH_SIZE);
  }
}

void construct_merkle_root(int transaction_size, BYTE *transactions,
                           int max_transactions_in_a_block, int n,
                           BYTE merkle_root[SHA256_HASH_SIZE]) {
  int minGridSize = 0;
  int blockSize = 0;
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(
      &minGridSize, &blockSize, construct_merkle_root_kernel,
      [](int blockSize) { return blockSize * 2 * SHA256_HASH_SIZE; }, 0);
  // Make sure block size is a power of 2
  blockSize = [blockSize] {
    int x = 1;
    while (x * 2 <= blockSize) {
      x *= 2;
    }
    return x;
  }();

  BYTE *cuda_buf;
  cudaMalloc((void **)&cuda_buf, n * transaction_size + n * SHA256_HASH_SIZE);

  BYTE *d_hashes = cuda_buf;
  BYTE *d_transactions = cuda_buf + n * SHA256_HASH_SIZE;
  cudaMemcpy(d_transactions, transactions, n * transaction_size,
             cudaMemcpyHostToDevice);

  const uint32_t transactions_per_block = blockSize * 2;
  const uint32_t num_blocks =
      (static_cast<uint64_t>(n) + transactions_per_block - 1) /
      transactions_per_block;
  const uint32_t share_mem_per_block =
      transactions_per_block * SHA256_HASH_SIZE;

  construct_merkle_root_kernel<<<num_blocks, blockSize, share_mem_per_block>>>(
      d_hashes, d_transactions, n, transaction_size);

  cudaDeviceSynchronize();

  if (num_blocks == 1) {
    cudaMemcpy(merkle_root, d_hashes, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
  } else {
    BYTE *hashes = (BYTE *)malloc(num_blocks * SHA256_HASH_SIZE);
    cudaMemcpy(hashes, d_hashes, num_blocks * SHA256_HASH_SIZE,
               cudaMemcpyDeviceToHost);
    // Perform the merkle root reduction on the CPU on the remaining hashes
    // We assume that the remaining number of hashes is small enough to be
    // processed on the CPU efficiently
    construct_merkle_root_cpu(SHA256_HASH_SIZE, hashes, num_blocks,
                              merkle_root);
    free(hashes);
  }

  cudaFree(cuda_buf);
}

__device__ void find_nonce_apply_sha256(const BYTE *input, size_t input_length,
                                        BYTE *output, SHA256_CTX ctx) {
  static constexpr auto hex_chars = "0123456789abcdef";

  sha256_update(&ctx, input, input_length);
  sha256_final(&ctx, output);

  output[SHA256_HASH_SIZE - 1] = '\0'; // Null-terminate
  for (int i = SHA256_BLOCK_SIZE - 1; i >= 0; --i) {
    const BYTE byte = output[i];
    output[i * 2] = hex_chars[(byte >> 4) & 0x0F]; // High nibble
    output[i * 2 + 1] = hex_chars[byte & 0x0F];    // Low nibble
  }
}

__constant__ BYTE d_constant_difficulty[SHA256_HASH_SIZE];
__constant__ SHA256_CTX d_constant_block_content_shactx;
__device__ BYTE d_hash[SHA256_HASH_SIZE];
__device__ uint32_t d_nonce = 0;
__device__ cuda::atomic<uint32_t, cuda::thread_scope_device> d_found(0);

__global__ void find_nonce_kernel(uint32_t max_nonce, size_t current_length) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;

  if (tid >= max_nonce || d_found.load(cuda::memory_order_acquire) == 1) {
    return;
  }

  BYTE nonce_string[NONCE_SIZE];
  BYTE hash[SHA256_HASH_SIZE];
  const SHA256_CTX local_d_constant_block_content_shactx =
      d_constant_block_content_shactx;

  for (uint64_t nonce = tid;
       nonce < max_nonce && d_found.load(cuda::memory_order_acquire) == 0;
       nonce += stride) {

    const int nonce_length = intToString(nonce, (char *)nonce_string);

    SHA256_CTX ctx = local_d_constant_block_content_shactx;
    find_nonce_apply_sha256(nonce_string, nonce_length, hash, ctx);

    if (compare_hashes(hash, d_constant_difficulty) <= 0) {
      uint32_t expected = 0;
      if (d_found.compare_exchange_strong(expected, 1,
                                          cuda::memory_order_acq_rel)) {
        d_nonce = nonce;
        for (int i = 0; i < SHA256_HASH_SIZE; i++) {
          d_hash[i] = hash[i];
        }
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
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                     find_nonce_kernel, 0, 0);

  SHA256_CTX block_content_ctx;
  sha256_init(&block_content_ctx);
  sha256_update(&block_content_ctx, block_content, current_length);
  cudaMemcpyToSymbol(d_constant_block_content_shactx, &block_content_ctx,
                     sizeof(SHA256_CTX));

  cudaMemcpyToSymbol(d_constant_difficulty, difficulty, SHA256_HASH_SIZE);
  {
    uint32_t zero = 0;
    cudaMemcpyToSymbol(d_found, &zero, sizeof(uint32_t));
  }

  const uint32_t threads_per_block = blockSize;

  const uint32_t k = 8;
  const uint32_t num_blocks = minGridSize * k;

  find_nonce_kernel<<<num_blocks, threads_per_block>>>(max_nonce,
                                                       current_length);

  cudaDeviceSynchronize();

  uint32_t found;
  cudaMemcpyFromSymbol(&found, d_found, sizeof(uint32_t), 0,
                       cudaMemcpyDeviceToHost);
  if (found == 1) {
    cudaMemcpyFromSymbol(valid_nonce, d_nonce, sizeof(uint32_t), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(block_hash, d_hash, SHA256_HASH_SIZE, 0,
                         cudaMemcpyDeviceToHost);
  }

  return 1 - found;
}

__global__ void dummy_kernel() {}

// Warm-up function
void warm_up_gpu() {
  BYTE *dummy_data;
  cudaMalloc((void **)&dummy_data, 256);
  dummy_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  cudaFree(dummy_data);
}
