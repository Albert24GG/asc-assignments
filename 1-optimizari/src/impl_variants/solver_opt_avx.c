/*
 * Tema 2 ASC
 * 2026 Spring
 */
#include "utils.h"
#include <immintrin.h>
#include <string.h>

#define BLOCK_SIZE 32

// L1d - 32KB

/*
 * Add your optimized implementation here
 */

__attribute__((always_inline, optimize("O3"))) static inline double
reduce_vec(__m256d vec) {
  register __m128d upper = _mm256_extractf128_pd(vec, 1);
  register __m128d lower = _mm256_castpd256_pd128(vec);

  register __m128d partial_sum = _mm_add_pd(upper, lower);
  register __m128d partial_sum_high = _mm_unpackhi_pd(partial_sum, partial_sum);
  register __m128d result = _mm_add_sd(partial_sum, partial_sum_high);

  return _mm_cvtsd_f64(result);
}

__attribute__((always_inline, optimize("O3"))) static inline double
vec_dot(int N, const double *A, const double *B) {
  // N is a multiple of 40
  register __m256d sum_vec1 = _mm256_setzero_pd();
  register __m256d sum_vec2 = _mm256_setzero_pd();
  register __m256d sum_vec3 = _mm256_setzero_pd();
  register __m256d sum_vec4 = _mm256_setzero_pd();
  register __m256d sum_vec5 = _mm256_setzero_pd();
  register __m256d sum_vec6 = _mm256_setzero_pd();
  register __m256d sum_vec7 = _mm256_setzero_pd();
  register __m256d sum_vec8 = _mm256_setzero_pd();

  const int limit = (N / 32) * 32;
  {
    register const double *a_ptr = A;
    register const double *b_ptr = B;
    for (register int i = 0; i < limit; i += 32, a_ptr += 32, b_ptr += 32) {
      // sum += A[i] * B[i];
      register const __m256d a_vec1 = _mm256_loadu_pd(a_ptr);
      register const __m256d a_vec2 = _mm256_loadu_pd(a_ptr + 4);
      register const __m256d a_vec3 = _mm256_loadu_pd(a_ptr + 8);
      register const __m256d a_vec4 = _mm256_loadu_pd(a_ptr + 12);
      register const __m256d a_vec5 = _mm256_loadu_pd(a_ptr + 16);
      register const __m256d a_vec6 = _mm256_loadu_pd(a_ptr + 20);
      register const __m256d a_vec7 = _mm256_loadu_pd(a_ptr + 24);
      register const __m256d a_vec8 = _mm256_loadu_pd(a_ptr + 28);

      sum_vec1 = _mm256_fmadd_pd(a_vec1, _mm256_loadu_pd(b_ptr), sum_vec1);
      sum_vec2 = _mm256_fmadd_pd(a_vec2, _mm256_loadu_pd(b_ptr + 4), sum_vec2);
      sum_vec3 = _mm256_fmadd_pd(a_vec3, _mm256_loadu_pd(b_ptr + 8), sum_vec3);
      sum_vec4 = _mm256_fmadd_pd(a_vec4, _mm256_loadu_pd(b_ptr + 12), sum_vec4);
      sum_vec5 = _mm256_fmadd_pd(a_vec5, _mm256_loadu_pd(b_ptr + 16), sum_vec5);
      sum_vec6 = _mm256_fmadd_pd(a_vec6, _mm256_loadu_pd(b_ptr + 20), sum_vec6);
      sum_vec7 = _mm256_fmadd_pd(a_vec7, _mm256_loadu_pd(b_ptr + 24), sum_vec7);
      sum_vec8 = _mm256_fmadd_pd(a_vec8, _mm256_loadu_pd(b_ptr + 28), sum_vec8);
    }
  }

  sum_vec1 = _mm256_add_pd(sum_vec1, sum_vec2);
  sum_vec3 = _mm256_add_pd(sum_vec3, sum_vec4);
  sum_vec5 = _mm256_add_pd(sum_vec5, sum_vec6);
  sum_vec7 = _mm256_add_pd(sum_vec7, sum_vec8);

  sum_vec1 = _mm256_add_pd(sum_vec1, sum_vec3);
  sum_vec5 = _mm256_add_pd(sum_vec5, sum_vec7);

  sum_vec1 = _mm256_add_pd(sum_vec1, sum_vec5);

  register double result = reduce_vec(sum_vec1);
  for (register int i = limit; i < N; ++i) {
    result += A[i] * B[i];
  }

  return result;
}

// Y = alpha * X + Y
__attribute__((always_inline, optimize("O3"))) static inline void
vec_axpy(int N, double alpha, const double *X, double *Y) {
  register const __m256d alpha_vec = _mm256_set1_pd(alpha);

  const int limit = (N / 32) * 32;
  {
    register double *y_ptr = Y;
    register const double *x_ptr = X;
    for (register int i = 0; i < limit; i += 32, y_ptr += 32, x_ptr += 32) {

      register __m256d y_vec1 = _mm256_loadu_pd(y_ptr);
      register __m256d y_vec2 = _mm256_loadu_pd(y_ptr + 4);
      register __m256d y_vec3 = _mm256_loadu_pd(y_ptr + 8);
      register __m256d y_vec4 = _mm256_loadu_pd(y_ptr + 12);
      register __m256d y_vec5 = _mm256_loadu_pd(y_ptr + 16);
      register __m256d y_vec6 = _mm256_loadu_pd(y_ptr + 20);
      register __m256d y_vec7 = _mm256_loadu_pd(y_ptr + 24);
      register __m256d y_vec8 = _mm256_loadu_pd(y_ptr + 28);

      y_vec1 = _mm256_fmadd_pd(alpha_vec, _mm256_loadu_pd(x_ptr), y_vec1);
      y_vec2 = _mm256_fmadd_pd(alpha_vec, _mm256_loadu_pd(x_ptr + 4), y_vec2);
      y_vec3 = _mm256_fmadd_pd(alpha_vec, _mm256_loadu_pd(x_ptr + 8), y_vec3);
      y_vec4 = _mm256_fmadd_pd(alpha_vec, _mm256_loadu_pd(x_ptr + 12), y_vec4);
      y_vec5 = _mm256_fmadd_pd(alpha_vec, _mm256_loadu_pd(x_ptr + 16), y_vec5);
      y_vec6 = _mm256_fmadd_pd(alpha_vec, _mm256_loadu_pd(x_ptr + 20), y_vec6);
      y_vec7 = _mm256_fmadd_pd(alpha_vec, _mm256_loadu_pd(x_ptr + 24), y_vec7);
      y_vec8 = _mm256_fmadd_pd(alpha_vec, _mm256_loadu_pd(x_ptr + 28), y_vec8);

      _mm256_storeu_pd(y_ptr, y_vec1);
      _mm256_storeu_pd(y_ptr + 4, y_vec2);
      _mm256_storeu_pd(y_ptr + 8, y_vec3);
      _mm256_storeu_pd(y_ptr + 12, y_vec4);
      _mm256_storeu_pd(y_ptr + 16, y_vec5);
      _mm256_storeu_pd(y_ptr + 20, y_vec6);
      _mm256_storeu_pd(y_ptr + 24, y_vec7);
      _mm256_storeu_pd(y_ptr + 28, y_vec8);
    }
  }

  for (register int i = limit; i < N; ++i) {
    Y[i] += alpha * X[i];
  }
}

// Y = X + Y
__attribute__((always_inline, optimize("O3"))) static inline void
vec_add(int N, const double *X, double *Y) {
  const int limit = (N / 32) * 32;
  {
    register double *y_ptr = Y;
    register const double *x_ptr = X;
    for (register int i = 0; i < limit; i += 32, x_ptr += 32, y_ptr += 32) {
      register __m256d y_vec1 = _mm256_loadu_pd(y_ptr);
      register __m256d y_vec2 = _mm256_loadu_pd(y_ptr + 4);
      register __m256d y_vec3 = _mm256_loadu_pd(y_ptr + 8);
      register __m256d y_vec4 = _mm256_loadu_pd(y_ptr + 12);
      register __m256d y_vec5 = _mm256_loadu_pd(y_ptr + 16);
      register __m256d y_vec6 = _mm256_loadu_pd(y_ptr + 20);
      register __m256d y_vec7 = _mm256_loadu_pd(y_ptr + 24);
      register __m256d y_vec8 = _mm256_loadu_pd(y_ptr + 28);

      y_vec1 = _mm256_add_pd(y_vec1, _mm256_loadu_pd(x_ptr));
      y_vec2 = _mm256_add_pd(y_vec2, _mm256_loadu_pd(x_ptr + 4));
      y_vec3 = _mm256_add_pd(y_vec3, _mm256_loadu_pd(x_ptr + 8));
      y_vec4 = _mm256_add_pd(y_vec4, _mm256_loadu_pd(x_ptr + 12));
      y_vec5 = _mm256_add_pd(y_vec5, _mm256_loadu_pd(x_ptr + 16));
      y_vec6 = _mm256_add_pd(y_vec6, _mm256_loadu_pd(x_ptr + 20));
      y_vec7 = _mm256_add_pd(y_vec7, _mm256_loadu_pd(x_ptr + 24));
      y_vec8 = _mm256_add_pd(y_vec8, _mm256_loadu_pd(x_ptr + 28));

      _mm256_storeu_pd(y_ptr, y_vec1);
      _mm256_storeu_pd(y_ptr + 4, y_vec2);
      _mm256_storeu_pd(y_ptr + 8, y_vec3);
      _mm256_storeu_pd(y_ptr + 12, y_vec4);
      _mm256_storeu_pd(y_ptr + 16, y_vec5);
      _mm256_storeu_pd(y_ptr + 20, y_vec6);
      _mm256_storeu_pd(y_ptr + 24, y_vec7);
      _mm256_storeu_pd(y_ptr + 28, y_vec8);
    }
  }

  for (register int i = limit; i < N; ++i) {
    Y[i] += X[i];
  }
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
      // C[i * N + j] += A[k * N + i] * B[k * N + j];
      vec_axpy(BLOCK_SIZE, *local3_a_t_ptr, local2_b_ptr, local2_c_ptr);
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

      // C[i * N + j] += A[k * N + i] * B[k * N + j];
      for (register int j = 0; j < limit; ++j, ++local_c_ptr, ++local_b_ptr) {
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
      // C[i * N + j] += A[k * N + i] * B[k * N + j];
      vec_axpy(N, *local_a_t_ptr, b_ptr, c_ptr);
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

      // C[i * N + j] += A[k * N + i] * B[k * N + j];
      for (register int j = limit; j < N; ++j, ++local_c_ptr, ++local_b_ptr) {
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

  // align to 32 bytes
  const int c_aligned_size = (N * N + 0x1F) & ~0x1F;
  const int d_aligned_size = (N * (N + 1) / 2 + 0x1F) & ~0x1F;

  double *_data;
  posix_memalign((void **)&_data, 32,
                 (c_aligned_size + d_aligned_size) * sizeof(double));

  if (_data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  double *C = _data;

  // blocked matrix multiplication
  bmm_kij(N, A, B, C);

  double *d = C + c_aligned_size;
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

  double *y = malloc(N * sizeof(double));

  if (y == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(_data);
    return NULL;
  }

  memcpy(y, x, N * sizeof(double));

  // d * c_0 + d * c_1 + ... + d * c_{N-1} = d * (c_0 + c_1 + ... + c_{N-1})
  // so we compute sum_rows_c = c_0 + c_1 + ... + c_{N-1} and then we do d *
  // sum_rows_c
  {
    register const double *c_rowa_ptr = C;
    register double *c_rowb_ptr = C + N;

    for (register int i = 1; i < N; ++i, c_rowa_ptr += N, c_rowb_ptr += N) {

      // C[i * N + j] += C[(i - 1) * N + j];
      vec_add(N, c_rowa_ptr, c_rowb_ptr);
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
