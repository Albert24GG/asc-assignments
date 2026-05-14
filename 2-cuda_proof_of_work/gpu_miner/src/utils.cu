#include "utils.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

__global__ void construct_merkle_root_kernel(BYTE *d_hashes, int n) {
  const int tid = threadIdx.x;

  static constexpr int COMBINED_SIZE = 2 * SHA256_HASH_SIZE - 1;
  BYTE combined[COMBINED_SIZE] = {};
  extern __shared__ BYTE shared_buf[];
  BYTE *local_buf = shared_buf + SHA256_HASH_SIZE * tid;

  bool active = true;

  while (n > 1) {
    if (active && tid < (n + 1) / 2) {
      if (2 * tid + 1 < n) {
        // normal case: combine two hashes
        d_strcpy((char *)combined,
                 (const char *)(d_hashes + (2 * tid) * SHA256_HASH_SIZE));
        d_strcat((char *)combined,
                 (const char *)(d_hashes + (2 * tid + 1) * SHA256_HASH_SIZE));
      } else {
        // odd case: duplicate the last hash
        d_strcpy((char *)combined,
                 (const char *)(d_hashes + (2 * tid) * SHA256_HASH_SIZE));
        d_strcat((char *)combined,
                 (const char *)(d_hashes + (2 * tid) * SHA256_HASH_SIZE));
      }
      apply_sha256(combined, local_buf);
    } else {
      active = false;
    }

    __syncthreads();

    if (active) {
      d_strcpy((char *)(d_hashes + tid * SHA256_HASH_SIZE),
               (const char *)local_buf);
    }

    __syncthreads();

    n = (n + 1) / 2;
  }
}

void construct_merkle_root(int transaction_size, BYTE *transactions,
                           int max_transactions_in_a_block, int n,
                           BYTE merkle_root[SHA256_HASH_SIZE]) {
  static constexpr int MAX_THREADS_PER_THREADBLOCK = 1024;
  static constexpr int MAX_STREAMS = 20;
  static cudaStream_t streams[MAX_STREAMS];
  static bool initialized_streams = false;

  if (!initialized_streams) {
    for (int i = 0; i < MAX_STREAMS; ++i) {
      cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    initialized_streams = true;
  }

  BYTE *d_hashes;
  BYTE *hashes = (BYTE *)malloc(n * SHA256_HASH_SIZE);

  cudaMalloc((void **)&d_hashes, n * SHA256_HASH_SIZE);
  {
    BYTE *cur_hashes_ptr = hashes;
    BYTE *cur_transactions_ptr = transactions;
    for (int i = 0; i < n; ++i, cur_hashes_ptr += SHA256_HASH_SIZE,
             cur_transactions_ptr += transaction_size) {
      apply_sha256(cur_transactions_ptr, cur_hashes_ptr);
    }
    cudaMemcpy(d_hashes, hashes, n * SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
  }

  const int num_blocks =
      (n + MAX_THREADS_PER_THREADBLOCK - 1) / MAX_THREADS_PER_THREADBLOCK;
  int remaining_transactions = n;

  for (int i = 0; i < num_blocks; ++i) {
    const int transactions_in_block =
        min(remaining_transactions, MAX_THREADS_PER_THREADBLOCK);
    remaining_transactions -= transactions_in_block;

    const int threads_per_block = (transactions_in_block + 1) / 2;
    const int shared_mem_size = threads_per_block * SHA256_HASH_SIZE;
    const int start_index = i * MAX_THREADS_PER_THREADBLOCK * SHA256_HASH_SIZE;
    construct_merkle_root_kernel<<<1, threads_per_block, shared_mem_size,
                                   streams[i]>>>(d_hashes + start_index,
                                                 transactions_in_block);
  }

  cudaDeviceSynchronize();

    if (num_blocks == 1) {
      cudaMemcpy(merkle_root, d_hashes, SHA256_HASH_SIZE,
                 cudaMemcpyDeviceToHost);
    } else {
      // Perform the merkle root reduction on the CPU on the remaining hashes
      for (int i = 0; i < num_blocks; ++i) {
        cudaMemcpy(hashes + i * SHA256_HASH_SIZE,
                   d_hashes +
                       i * SHA256_HASH_SIZE * MAX_THREADS_PER_THREADBLOCK,
                   SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
      }

      construct_merkle_root_cpu(SHA256_HASH_SIZE, hashes, num_blocks,
                                merkle_root);
    }

  free(hashes);
  cudaFree(d_hashes);
}

// TODO 2: Implement this function in CUDA
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content,
               size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
  char nonce_string[NONCE_SIZE];

  for (uint32_t nonce = 0; nonce < max_nonce; nonce++) {
    sprintf(nonce_string, "%u", nonce);
    strcpy((char *)block_content + current_length, nonce_string);
    apply_sha256(block_content, block_hash);

    if (compare_hashes(block_hash, difficulty) <= 0) {
      *valid_nonce = nonce;
      return 0;
    }
  }

  return 1;
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
