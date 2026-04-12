/*
 * Tema 2 ASC
 * 2026 Spring
 */
#include "utils.h"
#include <immintrin.h>
#include <string.h>

#pragma GCC target("avx2,fma")

// L1d - 32KB

/*
 * Add your optimized implementation here
 */

// double vec_dot(int N, const double *a, const double *b) {
//   double sum = 0;
//   for (int i = 0; i < N; ++i) {
//     sum += a[i] * b[i];
//   }
//   return sum;
// }

double reduce_vec(__m256d vec) {
  __m128d upper = _mm256_extractf128_pd(vec, 1);
  __m128d lower = _mm256_castpd256_pd128(vec);

  __m128d partial_sum = _mm_add_pd(upper, lower);
  __m128d partial_sum_high = _mm_unpackhi_pd(partial_sum, partial_sum);
  __m128d result = _mm_add_sd(partial_sum, partial_sum_high);

  return _mm_cvtsd_f64(result);
}

double vec_dot(int N, const double *a, const double *b) {
  register __m256d sum_vec = _mm256_setzero_pd();

  register int i = 0;
  register const double *a_ptr = a;
  register const double *b_ptr = b;
  const register int limit = N & ~3;
  for (; i < limit; i += 4, a_ptr += 4, b_ptr += 4) {
    __m256d x = _mm256_loadu_pd(a_ptr);
    __m256d y = _mm256_loadu_pd(b_ptr);
    sum_vec = _mm256_fmadd_pd(x, y, sum_vec);
  }

  register double res = 0;
  switch (N - limit) {
  case 3:
    res += a_ptr[2] * b_ptr[2];
  case 2:
    res += a_ptr[1] * b_ptr[1];
  case 1:
    res += a_ptr[0] * b_ptr[0];
  }

  return res + reduce_vec(sum_vec);
}

double *my_solver(int N, double *A, double *B, double *x) {

  register double *_data = calloc(3 * N * (N + 1) / 2, sizeof(double));

  register double *c = _data;

  // matrix_mult(N, aux, B, c);
  // c = A^T * B
  for (register int i = 0; i < N; ++i) {
    for (register int k = 0; k < N; ++k) {
      // instead of transposing, we can change the indexing of A
      register const double a_t_val = A[k * N + i];
      for (register int j = 0; j < N; ++j) {
        c[i * N + j] += a_t_val * B[k * N + j];
      }
    }
  }

  double *d = c + N * N;
  // d = c * c^T

  // consider c is made of rows c_0, c_1, ..., c_{N-1}
  // we compute all multiplications we need then we fill the upper triangular
  // part of d
  {
    register const double *c_i = c;

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
  memcpy(y, x, N * sizeof(double));

  double *sum_rows_c = d + N * (N + 1) / 2;

  // d * c_0 + d * c_1 + ... + d * c_{N-1} = d * (c_0 + c_1 + ... + c_{N-1})
  // so we compute sum_rows_c = c_0 + c_1 + ... + c_{N-1} and then we do d *
  // sum_rows_c
  {
    register const double *c_ptr = c;
    for (register int i = 0; i < N; ++i) {
      register double *sum_rows_c_j = sum_rows_c;
      for (register int j = 0; j < N; ++j, ++c_ptr, ++sum_rows_c_j) {
        *sum_rows_c_j += *c_ptr;
      }
    }
  }

  {
    register double *y_i = y;
    for (register int i = 0; i < N; ++i, ++y_i) {
      register const double *d_ptr = &d[i * N - i * (i - 1) / 2];
      register const double *sum_rows_c_i = sum_rows_c + i;

      *y_i += *d_ptr * *sum_rows_c_i;
      ++d_ptr;

      register double *sum_rows_c_j = sum_rows_c + i + 1;
      register double *y_j = y_i + 1;
      for (register int j = i + 1; j < N; ++j, ++d_ptr, ++sum_rows_c_j, ++y_j) {
        const register double d_val = *d_ptr;
        // since d is symmetric, we have d[j * N + i] = d[i * N + j], we can
        // perform the update for both elements at the same time

        // when multiplying d with sum_rows_c, row i contributes to y[i] and the
        // element on column j gets multiplied by sum_rows_c[j]
        *y_i += d_val * *sum_rows_c_j;
        *y_j += d_val * *sum_rows_c_i;
      }
    }
  }

  free(_data);

  return y;
}
