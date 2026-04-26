/*
 * Tema 2 ASC
 * 2026 Spring
 */
#include "utils.h"
#include <string.h>

/*
 * Add your unoptimized implementation here
 */

void fill_with_transpose(int N, double *A, double *B) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      B[i * N + j] = A[j * N + i];
    }
  }
}

void matrix_mult_upper(int N, double *A, double *B, double *C) {
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      double sum = 0;
      for (int k = 0; k < N; ++k) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

void matrix_mult(int N, double *A, double *B, double *C) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0;
      for (int k = 0; k < N; ++k) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

double *my_solver(int N, double *A, double *B, double *x) {
  // transpose
  double *aux = malloc(N * N * sizeof(double));
  double *c = malloc(N * N * sizeof(double));

  fill_with_transpose(N, A, aux);

  // c = A^T * B
  matrix_mult(N, aux, B, c);

  // transpose c
  fill_with_transpose(N, c, aux);

  double *d = malloc(N * N * sizeof(double));
  // d = c * c^T
  // compute only upper triangular part of d, since d is symmetric
  matrix_mult_upper(N, c, aux, d);

  double *y = calloc(N, sizeof(double));
  double *sum_rows_c = aux;
  memset(sum_rows_c, 0, N * sizeof(double));

  // d * c_0 + d * c_1 + ... + d * c_{N-1} = d * (c_0 + c_1 + ... + c_{N-1})
  // so we compute sum_rows_c = c_0 + c_1 + ... + c_{N-1} and then we do d *
  // sum_rows_c
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      sum_rows_c[j] += c[i * N + j];
    }
  }

  for (int i = 0; i < N; ++i) {
    y[i] += d[i * N + i] * sum_rows_c[i];

    for (int j = i + 1; j < N; ++j) {
      double d_val = d[i * N + j];
      // since d is symmetric, we have d[j * N + i] = d[i * N + j], we can
      // perform the update for both elements at the same time

      // when multiplying d with sum_rows_c, row i contributes to y[i] and the
      // element on column j gets multiplied by sum_rows_c[j]
      y[i] += d_val * sum_rows_c[j];
      y[j] += d_val * sum_rows_c[i];
    }
  }

  for (int i = 0; i < N; ++i)
    y[i] += x[i];

  free(aux);
  free(c);
  free(d);

  return y;
}
