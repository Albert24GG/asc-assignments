/*
 * Tema 2 ASC
 * 2026 Spring
 */
#include "utils.h"
#include <immintrin.h>
#include <string.h>

#define MIN(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a < _b ? _a : _b;                                                         \
  })

#define BLOCK_SIZE 32

// L1d - 32KB

/*
 * Add your optimized implementation here
 */

__attribute__((always_inline)) static inline double
vec_dot(int N, const double *A, const double *B) {
  register double sum = 0;
  for (register int i = 0; i < N; ++i) {
    sum += A[i] * B[i];
  }

  return sum;
}

__attribute__((always_inline)) static inline void
compute_block(int N, const double *a_t_ptr, const double *b_ptr,
              double *c_ptr) {

  register const double *local2_a_t_ptr = a_t_ptr;
  register const double *local2_b_ptr = b_ptr;

  for (register int k = 0; k < BLOCK_SIZE;
       ++k, local2_a_t_ptr += N, local2_b_ptr += N) {

    register double *local2_c_ptr = c_ptr;
    register const double *local3_a_t_ptr = local2_a_t_ptr;

    for (register int i = 0; i < BLOCK_SIZE;
         ++i, local2_c_ptr += N, ++local3_a_t_ptr) {

      register const double *local3_b_ptr = local2_b_ptr;
      register double *local3_c_ptr = local2_c_ptr;

      register const double a_t_val = *local3_a_t_ptr;

      for (register int j = 0; j < BLOCK_SIZE;
           ++j, ++local3_c_ptr, ++local3_b_ptr) {
        *local3_c_ptr += a_t_val * *local3_b_ptr;
      }
    }
  }
}

__attribute__((always_inline)) static inline void
remaining_main(int N, const double *A, const double *B, double *C) {
  register const int limit = (N / BLOCK_SIZE) * BLOCK_SIZE;
  register const double *a_t_ptr = A + limit * N;
  register const double *b_ptr = B + limit * N;

  for (register int k = limit; k < N; ++k, a_t_ptr += N, b_ptr += N) {

    register double *c_ptr = C;

    register const double *local_a_t_ptr = a_t_ptr;

    for (register int i = 0; i < limit; ++i, c_ptr += N, ++local_a_t_ptr) {

      register const double a_t_val = *local_a_t_ptr;
      // register const double a_t_val = A[k * N + i];

      register double *local_c_ptr = c_ptr;
      register const double *local_b_ptr = b_ptr;

      for (register int j = 0; j < limit; ++j, ++local_c_ptr, ++local_b_ptr) {
        // C[i * N + j] += a_t_val * B[k * N + j];
        *local_c_ptr += a_t_val * *local_b_ptr;
      }
    }
  }
}

__attribute__((always_inline)) static inline void
remaining_bottom(int N, const double *A, const double *B, double *C) {
  register const int limit = (N / BLOCK_SIZE) * BLOCK_SIZE;
  register const double *a_t_ptr = A + limit;
  register const double *b_ptr = B;

  for (register int k = 0; k < N; ++k, a_t_ptr += N, b_ptr += N) {

    register double *c_ptr = C + limit * N;

    register const double *local_a_t_ptr = a_t_ptr;

    for (register int i = limit; i < N; ++i, c_ptr += N, ++local_a_t_ptr) {

      register const double a_t_val = *local_a_t_ptr;

      register double *local_c_ptr = c_ptr;
      register const double *local_b_ptr = b_ptr;

      for (register int j = 0; j < N; ++j, ++local_c_ptr, ++local_b_ptr) {
        // C[i * N + j] += A[k * N + i] * B[k * N + j];
        *local_c_ptr += a_t_val * *local_b_ptr;
      }
    }
  }
}

__attribute__((always_inline)) static inline void
remaining_right(int N, const double *A, const double *B, double *C) {
  register const int limit = (N / BLOCK_SIZE) * BLOCK_SIZE;
  register const double *a_t_ptr = A;
  register const double *b_ptr = B + limit;

  for (register int k = 0; k < N; ++k, a_t_ptr += N, b_ptr += N) {

    register double *c_ptr = C + limit;

    register const double *local_a_t_ptr = a_t_ptr;

    for (register int i = 0; i < limit; ++i, c_ptr += N, ++local_a_t_ptr) {

      register const double a_t_val = *local_a_t_ptr;

      register double *local_c_ptr = c_ptr;
      register const double *local_b_ptr = b_ptr;

      for (register int j = limit; j < N; ++j, ++local_c_ptr, ++local_b_ptr) {
        // C[i * N + j] += A[k * N + i] * B[k * N + j];
        *local_c_ptr += a_t_val * *local_b_ptr;
      }
    }
  }
}

__attribute__((always_inline)) static inline void
main_block(int N, const double *A, const double *B, double *C) {
  register const int blocks_per_line = N / BLOCK_SIZE;
  register const double *a_t_ptr = A;
  register const double *b_ptr = B;

  for (register int bk = 0; bk < blocks_per_line;
       ++bk, a_t_ptr += BLOCK_SIZE * N, b_ptr += BLOCK_SIZE * N) {

    register double *c_ptr = C;

    register const double *local_a_t_ptr = a_t_ptr;

    for (register int bi = 0; bi < blocks_per_line;
         ++bi, local_a_t_ptr += BLOCK_SIZE, c_ptr += BLOCK_SIZE * N) {

      register double *local_c_ptr = c_ptr;

      register const double *local_b_ptr = b_ptr;
      for (register int bj = 0; bj < blocks_per_line;
           ++bj, local_b_ptr += BLOCK_SIZE, local_c_ptr += BLOCK_SIZE) {

        compute_block(N, local_a_t_ptr, local_b_ptr, local_c_ptr);
      }
    }
  }
}

__attribute__((always_inline)) static inline void
bmm_kij(int N, const double *A, const double *B, double *C) {
  main_block(N, A, B, C);

  if (N % BLOCK_SIZE == 0)
    return;

  // handle remaining elements

  // remaining from line/cols ends that we haven't covered because of k
  // reaching the limit
  remaining_main(N, A, B, C);

  // bottom part
  remaining_bottom(N, A, B, C);

  // right part
  remaining_right(N, A, B, C);
}

double *my_solver(int N, double *A, double *B, double *x) {

  // align to 16 bytes
  const int c_aligned_size = (N * N + 0xF) & ~0xF;
  const int d_aligned_size = (N * (N + 1) / 2 + 0xF) & ~0xF;

  register double *_data =
      calloc(c_aligned_size + d_aligned_size, sizeof(double));

  register double *C = _data;

  // blocked matrix multiplication
  bmm_kij(N, A, B, C);

  register double *d = C + c_aligned_size;
  // d = c * c^T

  // consider c is made of rows c_0, c_1, ..., c_{N-1}
  // each element of d will be  d[i][j] = c_i * c_j, thus symmetric due to
  // commutativity of the dot product
  {
    register const double *c_i = C;

    // since we use only the upper triangular part of d, d will only contain
    // its upper triangular part therefore each row of d will have one element
    // less than the previous one
    register double *d_ptr = d;

    for (register int i = 0; i < N; ++i, c_i += N) {
      register const double *c_j = c_i;
      for (register int j = i; j < N; ++j, ++d_ptr, c_j += N) {
        *d_ptr = vec_dot(N, c_i, c_j);
      }
    }
  }

  register double *y = malloc(N * sizeof(double));
  memcpy(y, x, N * sizeof(double));

  // d * c_0 + d * c_1 + ... + d * c_{N-1} = d * (c_0 + c_1 + ... + c_{N-1})
  // so we compute sum_rows_c = c_0 + c_1 + ... + c_{N-1} and then we do d *
  // sum_rows_c
  {
    register const double *c_rowa_ptr = C;
    register double *c_rowb_ptr = C + N;

    for (register int i = 1; i < N; ++i) {

      // assume N is a multiple of 4, otherwise here should be a switch with 4
      // cases for the remaining elements
      for (register int j = 0; j < N; ++j, ++c_rowa_ptr, ++c_rowb_ptr) {
        *c_rowb_ptr += *c_rowa_ptr;
      }
    }
  }

  {
    register const double *sum_rows_c = C + N * (N - 1);
    register const double *sum_rows_c_i = sum_rows_c;
    register double *y_i = y;

    // register const double *d_ptr = &d[i * N - i * (i - 1) / 2];
    register const double *d_ptr = d;
    for (register int i = 0; i < N;
         d_ptr += N - i, ++i, ++y_i, ++sum_rows_c_i) {
      register const double sum_rows_c_i_val = *sum_rows_c_i;

      register const double *local_d_ptr = d_ptr;
      register double local_y_i_val = *local_d_ptr * sum_rows_c_i_val;
      ++local_d_ptr;

      register const double *sum_rows_c_j = sum_rows_c_i + 1;
      register double *y_j = y_i + 1;

      for (register int j = i + 1; j < N;
           ++j, ++local_d_ptr, ++sum_rows_c_j, ++y_j) {
        const register double d_val = *local_d_ptr;
        // since d is symmetric, we have d[j * N + i] = d[i * N + j], we can
        // perform the update for both elements at the same time

        // when multiplying d with sum_rows_c, row i contributes to y[i] and
        // the element on column j gets multiplied by sum_rows_c[j]
        local_y_i_val += d_val * *sum_rows_c_j;
        *y_j += d_val * sum_rows_c_i_val;
      }

      *y_i += local_y_i_val;
    }
  }

  free(_data);

  return y;
}
