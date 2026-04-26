/*
 * Tema 2 ASC
 * 2026 Spring
 */
#include "utils.h"

/*
 * Add your unoptimized implementation here
 */

void fill_with_transpose(int N, const double *A, double *B) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      B[i * N + j] = A[j * N + i];
    }
  }
}

void matrix_mult_upper(int N, const double *A, const double *B, double *C) {
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

void matrix_mult(int N, const double *A, const double *B, double *C) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

// Y = M * X + Y
void mat_vec_mult(int N, const double *M, const double *X, double *Y) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      Y[i] += M[i * N + j] * X[j];
    }
  }
}

double *my_solver(int N, double *A, double *B, double *x) {
  // transpose
  double *aux = malloc(N * N * sizeof(double));
  double *c = malloc(N * N * sizeof(double));

  if (aux == NULL || c == NULL) {
    fprintf(stderr, "Memory allocation failed\n");

    if (aux)
      free(aux);

    if (c)
      free(c);

    return NULL;
  }

  fill_with_transpose(N, A, aux);

  // c = A^T * B
  matrix_mult(N, aux, B, c);

  // transpose c
  fill_with_transpose(N, c, aux);

  double *d = malloc(N * N * sizeof(double));

  if (d == NULL) {
    fprintf(stderr, "Memory allocation failed\n");

    free(aux);
    free(c);

    return NULL;
  }

  // d = c * c^T
  // compute only upper triangular part of d, since d is symmetric
  matrix_mult_upper(N, c, aux, d);

  double *y = calloc(N, sizeof(double));

  if (y == NULL) {
    fprintf(stderr, "Memory allocation failed\n");

    free(aux);
    free(c);
    free(d);

    return NULL;
  }

  const double *c_row_ptr = c;
  for (int rowc = 0; rowc < N; ++rowc, c_row_ptr += N) {

    for (int i = 0; i < N; ++i) {
      y[i] += d[i * N + i] * c_row_ptr[i];

      for (int j = i + 1; j < N; ++j) {
        double d_val = d[i * N + j];
        // since d is symmetric, we have d[j * N + i] = d[i * N + j], we can
        // perform the update for both elements at the same time

        // when multiplying d with sum_rows_c, row i contributes to y[i] and the
        // element on column j gets multiplied by sum_rows_c[j]
        y[i] += d_val * c_row_ptr[j];
        y[j] += d_val * c_row_ptr[i];
      }
    }
  }

  for (int i = 0; i < N; ++i)
    y[i] += x[i];

  free(aux);
  free(c);
  free(d);

  return y;
}
