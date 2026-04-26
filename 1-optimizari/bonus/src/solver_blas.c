/*
 * Tema 2 ASC
 * 2024 Spring
 */
#include "utils.h"
#include <cblas.h>

/*
 * Add your BLAS implementation here
 */

double *my_solver(int N, double *A, double *B, double *x) {

  double *data = malloc(2 * N * N * sizeof(double));

  double *c = data;

  // c = A^T * B
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N,
              0.0, c, N);

  double *d = data + N * N;
  // d = c * c^T
  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, N, N, 1.0, c, N, 0.0, d,
              N);

  double *y = calloc(N, sizeof(double));

  double *c_row = c;
  for (int i = 0; i < N; ++i, c_row += N) {
    // y += d * linie(C, i)
    cblas_dsymv(CblasRowMajor, CblasUpper, N, 1.0, d, N, c_row, 1, 1.0, y, 1);
  }

  cblas_daxpy(N, 1.0, x, 1, y, 1);

  free(data);

  return y;
}
