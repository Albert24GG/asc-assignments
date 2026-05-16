#include "utils.h"
#include <cstdint>
#include <cuda_runtime.h>
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
  BYTE buf[SHA256_BLOCK_SIZE] = {};
  const char hex_chars[] = "0123456789abcdef";
  const size_t input_length = d_strlen((const char *)input);

  sha256_init(&ctx);
  sha256_update(&ctx, input, input_length);
  sha256_final(&ctx, buf);

  for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
    output[i * 2] = hex_chars[(buf[i] >> 4) & 0x0F]; // High nibble
    output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];    // Low nibble
  }
  output[SHA256_HASH_SIZE - 1] = '\0'; // Null-terminate
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
    for (int i = 0; i < n; i += 2) {
      if (i + 1 < n) {
        strcpy((char *)combined, (const char *)(hashes + i * hash_size));
        strcat((char *)combined, (const char *)(hashes + (i + 1) * hash_size));
      } else {
        // If odd number of hashes, duplicate the last one
        strcpy((char *)combined, (const char *)(hashes + i * hash_size));
        strcat((char *)combined, (const char *)(hashes + i * hash_size));
      }
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
  BYTE combined[COMBINED_SIZE] = {};
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

  int transactions_in_block =
      min(blockDim.x * 2, n - blockIdx.x * blockDim.x * 2);
  int cur_n = transactions_in_block;

  __syncthreads();

  bool active = true;

  while (cur_n > 1) {
    const int threads_needed = (cur_n + 1) / 2;

    if (tid < threads_needed) {
      const int left_idx = 2 * tid;
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

  if (tid == 0) {
    memcpy(d_hashes + blockIdx.x * SHA256_HASH_SIZE, shared_buf,
           SHA256_HASH_SIZE);
  }
}

void construct_merkle_root(int transaction_size, BYTE *transactions,
                           int max_transactions_in_a_block, int n,
                           BYTE merkle_root[SHA256_HASH_SIZE]) {
  static constexpr int MAX_THREADS_PER_THREADBLOCK = 256;
  static constexpr int MAX_TRANSACTIONS_PER_BLOCK =
      MAX_THREADS_PER_THREADBLOCK * 2;

  BYTE *cuda_buf;
  cudaMalloc((void **)&cuda_buf, n * transaction_size + n * SHA256_HASH_SIZE);

  BYTE *d_hashes = cuda_buf;
  BYTE *d_transactions = cuda_buf + n * SHA256_HASH_SIZE;
  cudaMemcpy(d_transactions, transactions, n * transaction_size,
             cudaMemcpyHostToDevice);

  const int num_blocks =
      (n + MAX_TRANSACTIONS_PER_BLOCK - 1) / MAX_TRANSACTIONS_PER_BLOCK;
  const int share_mem_per_block = 512 * SHA256_HASH_SIZE;

  construct_merkle_root_kernel<<<num_blocks, MAX_THREADS_PER_THREADBLOCK,
                                 share_mem_per_block>>>(
      d_hashes, d_transactions, n, transaction_size);

  cudaDeviceSynchronize();

  BYTE *hashes = (BYTE *)malloc(num_blocks * SHA256_HASH_SIZE);
  cudaMemcpy(hashes, d_hashes, num_blocks * SHA256_HASH_SIZE,
             cudaMemcpyDeviceToHost);

  if (num_blocks == 1) {
    memcpy(merkle_root, hashes, SHA256_HASH_SIZE);
  } else {
    // Perform the merkle root reduction on the CPU on the remaining hashes
    // We assume that the remaining number of hashes is small enough to be
    // processed on the CPU efficiently
    construct_merkle_root_cpu(SHA256_HASH_SIZE, hashes, num_blocks,
                              merkle_root);
  }

  free(hashes);
  // cudaFree(d_hashes);
  cudaFree(cuda_buf);
}

__constant__ BYTE d_constant_difficulty[SHA256_HASH_SIZE];
__constant__ BYTE d_constant_block_content[2 * SHA256_HASH_SIZE];

__global__ void find_nonce_kernel(uint32_t max_nonce, size_t current_length,
                                  BYTE *result_hash, uint32_t *found) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;

  if (tid >= max_nonce || *found == 1) {
    return;
  }

  BYTE nonce_string[2 * SHA256_HASH_SIZE + NONCE_SIZE] = {};
  BYTE hash[SHA256_HASH_SIZE] = {};
  memcpy(nonce_string, d_constant_block_content, current_length);

  for (uint64_t nonce = tid; nonce < max_nonce && *found == 0;
       nonce += stride) {

    int nonce_length =
        intToString(nonce, (char *)nonce_string + current_length);

    apply_sha256(nonce_string, hash);

    if (compare_hashes(hash, d_constant_difficulty) <= 0) {
      if (atomicCAS(found, 0, 1) == 0) {
        memcpy(result_hash, hash, SHA256_HASH_SIZE);
      }
      return;
    }
  }
}

// TODO 2: Implement this function in CUDA
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content,
               size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {

  uint8_t *cuda_buf;
  const size_t total_size = sizeof(uint32_t) + SHA256_HASH_SIZE;
  cudaMalloc((void **)&cuda_buf, total_size);
  cudaMemset(cuda_buf, 0, total_size);
  uint32_t *d_found = (uint32_t *)cuda_buf;
  BYTE *result_hash = (BYTE *)(cuda_buf + sizeof(uint32_t));

  cudaMemcpyToSymbol(d_constant_difficulty, difficulty, SHA256_HASH_SIZE);
  cudaMemcpyToSymbol(d_constant_block_content, block_content,
                     current_length + 1);

  static constexpr uint32_t threads_per_block = 256;
  uint32_t num_blocks =
      ((uint64_t)max_nonce + threads_per_block - 1) / threads_per_block / 2;

  find_nonce_kernel<<<num_blocks, threads_per_block>>>(
      max_nonce, current_length, result_hash, d_found);

  cudaDeviceSynchronize();

  uint32_t found;
  cudaMemcpy(&found, d_found, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (found == 1) {
    cudaMemcpy(block_hash, result_hash, SHA256_HASH_SIZE,
               cudaMemcpyDeviceToHost);
  }

  cudaFree(cuda_buf);

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
