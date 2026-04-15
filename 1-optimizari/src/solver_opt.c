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

// double vec_dot(int N, const double *a, const double *b) {
//   double sum = 0;
//   for (int i = 0; i < N; ++i) {
//     sum += a[i] * b[i];
//   }
//   return sum;
// }

// double reduce_vec(__m256d vec) {
//   __m128d upper = _mm256_extractf128_pd(vec, 1);
//   __m128d lower = _mm256_castpd256_pd128(vec);
//
//   __m128d partial_sum = _mm_add_pd(upper, lower);
//   __m128d partial_sum_high = _mm_unpackhi_pd(partial_sum, partial_sum);
//   __m128d result = _mm_add_sd(partial_sum, partial_sum_high);
//
//   return _mm_cvtsd_f64(result);
// }

// double vec_dot(int N, const double *a, const double *b) {
//   register __m256d sum_vec = _mm256_setzero_pd();
//
//   register int i = 0;
//   register const double *a_ptr = a;
//   register const double *b_ptr = b;
//   const register int limit = N & ~3;
//   for (; i < limit; i += 4, a_ptr += 4, b_ptr += 4) {
//     __m256d x = _mm256_loadu_pd(a_ptr);
//     __m256d y = _mm256_loadu_pd(b_ptr);
//     sum_vec = _mm256_fmadd_pd(x, y, sum_vec);
//   }
//
//   register double res = 0;
//   switch (N - limit) {
//   case 3:
//     res += a_ptr[2] * b_ptr[2];
//   case 2:
//     res += a_ptr[1] * b_ptr[1];
//   case 1:
//     res += a_ptr[0] * b_ptr[0];
//   }
//
//   return res + reduce_vec(sum_vec);
// }

#define reduce_vec(vec)                                                        \
  ({                                                                           \
    __m128d upper = _mm256_extractf128_pd(vec, 1);                             \
    __m128d lower = _mm256_castpd256_pd128(vec);                               \
                                                                               \
    __m128d partial_sum = _mm_add_pd(upper, lower);                            \
    __m128d partial_sum_high = _mm_unpackhi_pd(partial_sum, partial_sum);      \
    __m128d result = _mm_add_sd(partial_sum, partial_sum_high);                \
                                                                               \
    _mm_cvtsd_f64(result);                                                     \
  })

#define vec_dot(N, a, b)                                                       \
  ({                                                                           \
    register __m256d sum_vec = _mm256_setzero_pd();                            \
                                                                               \
    register const double *a_ptr = a;                                          \
    register const double *b_ptr = b;                                          \
    const register int limit = N & ~3;                                         \
    for (register int i = 0; i < limit; i += 4, a_ptr += 4, b_ptr += 4) {      \
      __m256d x = _mm256_loadu_pd(a_ptr);                                      \
      __m256d y = _mm256_loadu_pd(b_ptr);                                      \
      sum_vec = _mm256_fmadd_pd(x, y, sum_vec);                                \
    }                                                                          \
                                                                               \
    register double res = 0;                                                   \
    switch (N - limit) {                                                       \
    case 3:                                                                    \
      res += a_ptr[2] * b_ptr[2];                                              \
    case 2:                                                                    \
      res += a_ptr[1] * b_ptr[1];                                              \
    case 1:                                                                    \
      res += a_ptr[0] * b_ptr[0];                                              \
    }                                                                          \
                                                                               \
    res + reduce_vec(sum_vec);                                                 \
  })

#define bmm_ikj(N, A, B, C)                                                    \
  ({                                                                           \
    for (register int bi = 0; bi < N; bi += BLOCK_SIZE) {                      \
      for (register int bj = 0; bj < N; bj += BLOCK_SIZE) {                    \
        for (register int bk = 0; bk < N; bk += BLOCK_SIZE) {                  \
                                                                               \
          register const int max_i = MIN(bi + BLOCK_SIZE, N);                  \
          register const int max_j = MIN(bj + BLOCK_SIZE, N);                  \
          register const int max_k = MIN(bk + BLOCK_SIZE, N);                  \
                                                                               \
          register double *c_ptr = &C[bi * N + bj];                            \
                                                                               \
          register const double *a_t_ptr = &A[bk * N + bi];                    \
                                                                               \
          for (register int i = bi; i < max_i; ++i, c_ptr += N, ++a_t_ptr) {   \
                                                                               \
            register const double *local_a_t_ptr = a_t_ptr;                    \
                                                                               \
            register const double *b_ptr = &B[bk * N + bj];                    \
            for (register int k = bk; k < max_k;                               \
                 ++k, local_a_t_ptr += N, b_ptr += N) {                        \
                                                                               \
              register double *local_c_ptr = c_ptr;                            \
              register const double *local_b_ptr = b_ptr;                      \
                                                                               \
              register const double a_t_val = *local_a_t_ptr;                  \
                                                                               \
              for (register int j = bj; j < max_j;                             \
                   ++j, ++local_c_ptr, ++local_b_ptr) {                        \
                *local_c_ptr += a_t_val * *local_b_ptr;                        \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  })

#define bmm_ijk(N, A, B, C)                                                    \
  ({                                                                           \
    for (register int bi = 0; bi < N; bi += BLOCK_SIZE) {                      \
      for (register int bj = 0; bj < N; bj += BLOCK_SIZE) {                    \
        for (register int bk = 0; bk < N; bk += BLOCK_SIZE) {                  \
                                                                               \
          register const int max_i = MIN(bi + BLOCK_SIZE, N);                  \
          register const int max_j = MIN(bj + BLOCK_SIZE, N);                  \
          register const int max_k = MIN(bk + BLOCK_SIZE, N);                  \
                                                                               \
          register double *c_ptr = &C[bi * N + bj];                            \
          register const int c_row_stride = N - (max_j - bj);                  \
                                                                               \
          register const double *a_t_ptr = &A[bk * N + bi];                    \
                                                                               \
          for (register int i = bi; i < max_i;                                 \
               ++i, c_ptr += c_row_stride, ++a_t_ptr) {                        \
                                                                               \
            register double *b_ptr = &B[bk * N + bj];                          \
                                                                               \
            for (register int j = bj; j < max_j; ++j, ++b_ptr, ++c_ptr) {      \
                                                                               \
              register const double *local_b_ptr = b_ptr;                      \
              register const double *local_a_t_ptr = a_t_ptr;                  \
                                                                               \
              for (register int k = bk; k < max_k;                             \
                   ++k, local_b_ptr += N, local_a_t_ptr += N) {                \
                                                                               \
                *c_ptr += *local_a_t_ptr * *local_b_ptr;                       \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  })

void compute_block_fn(int N, double *a_t_ptr, double *b_ptr, double *c_ptr) {

  register double *local2_a_t_ptr = a_t_ptr;
  register double *local2_b_ptr = b_ptr;

  for (register int k = 0; k < BLOCK_SIZE;
       ++k, local2_a_t_ptr += N, local2_b_ptr += N) {

    register double *local2_c_ptr = c_ptr;
    register double *local3_a_t_ptr = local2_a_t_ptr;

    for (register int i = 0; i < BLOCK_SIZE;
         ++i, local2_c_ptr += N, ++local3_a_t_ptr) {

      register double *local3_b_ptr = local2_b_ptr;
      register double *local3_c_ptr = local2_c_ptr;

      register const double a_t_val = *local3_a_t_ptr;

      for (register int j = 0; j < BLOCK_SIZE;
           j += 4, local3_c_ptr += 4, local3_b_ptr += 4) {
        // c_ptr[i * N + j] += a_t_ptr[k * N + i] * b_ptr[k * N + j];

        local3_c_ptr[0] += a_t_val * local3_b_ptr[0];
        local3_c_ptr[1] += a_t_val * local3_b_ptr[1];
        local3_c_ptr[2] += a_t_val * local3_b_ptr[2];
        local3_c_ptr[3] += a_t_val * local3_b_ptr[3];
      }
    }
  }
}

#define compute_block(N, a_t_ptr, b_ptr, c_ptr)                                \
  ({                                                                           \
    register double *local2_a_t_ptr = a_t_ptr;                                 \
    register double *local2_b_ptr = b_ptr;                                     \
                                                                               \
    for (register int k = 0; k < BLOCK_SIZE;                                   \
         ++k, local2_a_t_ptr += N, local2_b_ptr += N) {                        \
                                                                               \
      register double *local2_c_ptr = c_ptr;                                   \
      register double *local3_a_t_ptr = local2_a_t_ptr;                        \
                                                                               \
      for (register int i = 0; i < BLOCK_SIZE;                                 \
           ++i, local2_c_ptr += N, ++local3_a_t_ptr) {                         \
                                                                               \
        register double *local3_b_ptr = local2_b_ptr;                          \
        register double *local3_c_ptr = local2_c_ptr;                          \
                                                                               \
        register const double a_t_val = *local3_a_t_ptr;                       \
                                                                               \
        for (register int j = 0; j < BLOCK_SIZE;                               \
             j += 4, local3_c_ptr += 4, local3_b_ptr += 4) {                   \
          local3_c_ptr[0] += a_t_val * local3_b_ptr[0];                        \
          local3_c_ptr[1] += a_t_val * local3_b_ptr[1];                        \
          local3_c_ptr[2] += a_t_val * local3_b_ptr[2];                        \
          local3_c_ptr[3] += a_t_val * local3_b_ptr[3];                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  })

void bmm_kij_updated_fn(int N, double *A, double *B, double *C) {
  register const int limit = N & ~(BLOCK_SIZE - 1);

  {
    register double *a_t_ptr = A;
    register double *b_ptr = B;

    for (register int bk = 0; bk < limit;
         bk += BLOCK_SIZE, a_t_ptr += BLOCK_SIZE * N, b_ptr += BLOCK_SIZE * N) {

      register double *c_ptr = C;

      register double *local_a_t_ptr = a_t_ptr;

      for (register int bi = 0; bi < limit; bi += BLOCK_SIZE,
                        local_a_t_ptr += BLOCK_SIZE, c_ptr += BLOCK_SIZE * N) {

        register double *local_c_ptr = c_ptr;

        register double *local_b_ptr = b_ptr;
        for (register int bj = 0; bj < limit; bj += BLOCK_SIZE,
                          local_b_ptr += BLOCK_SIZE,
                          local_c_ptr += BLOCK_SIZE) {

          compute_block(N, local_a_t_ptr, local_b_ptr, local_c_ptr);
        }
      }
    }
  }

  if (limit != N) {
    // handle remaining elements

    // remaining from line/cols ends that we haven't covered because of k
    // reaching the limit
    {
      register const double *a_t_ptr = A + limit * N;
      register const double *b_ptr = B + limit * N;

      for (int k = limit; k < N; ++k, a_t_ptr += N, b_ptr += N) {

        register double *c_ptr = C;

        register const double *local_a_t_ptr = a_t_ptr;

        for (int i = 0; i < limit; ++i, c_ptr += N, ++local_a_t_ptr) {

          register const double a_t_val = *local_a_t_ptr;
          // register const double a_t_val = A[k * N + i];

          register double *local_c_ptr = c_ptr;
          register const double *local_b_ptr = b_ptr;

          for (int j = 0; j < limit; ++j, ++local_c_ptr, ++local_b_ptr) {
            // C[i * N + j] += a_t_val * B[k * N + j];
            *local_c_ptr += a_t_val * *local_b_ptr;
          }
        }
      }
    }

    // bottom part
    {
      register const double *a_t_ptr = A + limit;
      register const double *b_ptr = B;

      for (int k = 0; k < N; ++k, a_t_ptr += N, b_ptr += N) {

        register double *c_ptr = C + limit * N;

        register const double *local_a_t_ptr = a_t_ptr;

        for (int i = limit; i < N; ++i, c_ptr += N, ++local_a_t_ptr) {

          register const double a_t_val = *local_a_t_ptr;

          register double *local_c_ptr = c_ptr;
          register const double *local_b_ptr = b_ptr;

          for (int j = 0; j < N; ++j, ++local_c_ptr, ++local_b_ptr) {
            // C[i * N + j] += A[k * N + i] * B[k * N + j];
            *local_c_ptr += a_t_val * *local_b_ptr;
          }
        }
      }
    }

    // right part
    {
      register const double *a_t_ptr = A;
      register const double *b_ptr = B + limit;

      for (int k = 0; k < N; ++k, a_t_ptr += N, b_ptr += N) {

        register double *c_ptr = C + limit;

        register const double *local_a_t_ptr = a_t_ptr;

        for (int i = 0; i < limit; ++i, c_ptr += N, ++local_a_t_ptr) {

          register const double a_t_val = *local_a_t_ptr;

          register double *local_c_ptr = c_ptr;
          register const double *local_b_ptr = b_ptr;

          for (int j = limit; j < N; ++j, ++local_c_ptr, ++local_b_ptr) {
            // C[i * N + j] += A[k * N + i] * B[k * N + j];
            *local_c_ptr += a_t_val * *local_b_ptr;
          }
        }
      }
    }
  }
}

#define bmm_kij_updated(N, A, B, C)                                            \
  ({                                                                           \
    register const int limit = N & ~(BLOCK_SIZE - 1);                          \
                                                                               \
    {                                                                          \
      register double *a_t_ptr = A;                                            \
      register double *b_ptr = B;                                              \
                                                                               \
      for (register int bk = 0; bk < limit; bk += BLOCK_SIZE,                  \
                        a_t_ptr += BLOCK_SIZE * N, b_ptr += BLOCK_SIZE * N) {  \
                                                                               \
        register double *c_ptr = C;                                            \
                                                                               \
        register double *local_a_t_ptr = a_t_ptr;                              \
                                                                               \
        for (register int bi = 0; bi < limit; bi += BLOCK_SIZE,                \
                          local_a_t_ptr += BLOCK_SIZE,                         \
                          c_ptr += BLOCK_SIZE * N) {                           \
                                                                               \
          register double *local_c_ptr = c_ptr;                                \
                                                                               \
          register double *local_b_ptr = b_ptr;                                \
          for (register int bj = 0; bj < limit; bj += BLOCK_SIZE,              \
                            local_b_ptr += BLOCK_SIZE,                         \
                            local_c_ptr += BLOCK_SIZE) {                       \
                                                                               \
            compute_block(N, local_a_t_ptr, local_b_ptr, local_c_ptr);         \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    if (limit != N) {                                                          \
                                                                               \
      {                                                                        \
        register const double *a_t_ptr = A + limit * N;                        \
        register const double *b_ptr = B + limit * N;                          \
                                                                               \
        for (int k = limit; k < N; ++k, a_t_ptr += N, b_ptr += N) {            \
                                                                               \
          register double *c_ptr = C;                                          \
                                                                               \
          register const double *local_a_t_ptr = a_t_ptr;                      \
                                                                               \
          for (int i = 0; i < limit; ++i, c_ptr += N, ++local_a_t_ptr) {       \
                                                                               \
            register const double a_t_val = *local_a_t_ptr;                    \
                                                                               \
            register double *local_c_ptr = c_ptr;                              \
            register const double *local_b_ptr = b_ptr;                        \
                                                                               \
            for (int j = 0; j < limit; ++j, ++local_c_ptr, ++local_b_ptr) {    \
              *local_c_ptr += a_t_val * *local_b_ptr;                          \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
                                                                               \
      {                                                                        \
        register const double *a_t_ptr = A + limit;                            \
        register const double *b_ptr = B;                                      \
                                                                               \
        for (int k = 0; k < N; ++k, a_t_ptr += N, b_ptr += N) {                \
                                                                               \
          register double *c_ptr = C + limit * N;                              \
                                                                               \
          register const double *local_a_t_ptr = a_t_ptr;                      \
                                                                               \
          for (int i = limit; i < N; ++i, c_ptr += N, ++local_a_t_ptr) {       \
                                                                               \
            register const double a_t_val = *local_a_t_ptr;                    \
                                                                               \
            register double *local_c_ptr = c_ptr;                              \
            register const double *local_b_ptr = b_ptr;                        \
                                                                               \
            for (int j = 0; j < N; ++j, ++local_c_ptr, ++local_b_ptr) {        \
              *local_c_ptr += a_t_val * *local_b_ptr;                          \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
                                                                               \
      {                                                                        \
        register const double *a_t_ptr = A;                                    \
        register const double *b_ptr = B + limit;                              \
                                                                               \
        for (int k = 0; k < N; ++k, a_t_ptr += N, b_ptr += N) {                \
                                                                               \
          register double *c_ptr = C + limit;                                  \
                                                                               \
          register const double *local_a_t_ptr = a_t_ptr;                      \
                                                                               \
          for (int i = 0; i < limit; ++i, c_ptr += N, ++local_a_t_ptr) {       \
                                                                               \
            register const double a_t_val = *local_a_t_ptr;                    \
                                                                               \
            register double *local_c_ptr = c_ptr;                              \
            register const double *local_b_ptr = b_ptr;                        \
                                                                               \
            for (int j = limit; j < N; ++j, ++local_c_ptr, ++local_b_ptr) {    \
              *local_c_ptr += a_t_val * *local_b_ptr;                          \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  })

double *my_solver(int N, double *A, double *B, double *x) {

  // align to 16 bytes
  const int c_aligned_size = (N * N + 0xF) & ~0xF;
  const int d_aligned_size = (N * (N + 1) / 2 + 0xF) & ~0xF;

  register double *_data =
      calloc(c_aligned_size + d_aligned_size, sizeof(double));

  register double *C = _data;

  // blocked matrix multiplication
  bmm_kij_updated(N, A, B, C);

  register double *d = C + c_aligned_size;
  // d = c * c^T

  // consider c is made of rows c_0, c_1, ..., c_{N-1}
  // we compute all multiplications we need then we fill the upper triangular
  // part of d
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

  // register double *sum_rows_c = d + d_aligned_size;

  // d * c_0 + d * c_1 + ... + d * c_{N-1} = d * (c_0 + c_1 + ... + c_{N-1})
  // so we compute sum_rows_c = c_0 + c_1 + ... + c_{N-1} and then we do d *
  // sum_rows_c
  {
    register const double *c_rowa_ptr = C;
    register double *c_rowb_ptr = C + N;

    for (register int i = 1; i < N; ++i) {

      // assume N is a multiple of 4, otherwise here should be a switch with 4
      // cases for the remaining elements
      for (register int j = 0; j < N;
           j += 4, c_rowa_ptr += 4, c_rowb_ptr += 4) {
        __m256d c_rowa_vec = _mm256_loadu_pd(c_rowa_ptr);
        __m256d c_rowb_vec = _mm256_loadu_pd(c_rowb_ptr);
        c_rowb_vec = _mm256_add_pd(c_rowb_vec, c_rowa_vec);
        _mm256_storeu_pd(c_rowb_ptr, c_rowb_vec);
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
