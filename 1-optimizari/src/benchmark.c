#define _GNU_SOURCE

#include <errno.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "argparse.h"
#include "utils.h"

#define RANGE 1

#define MAX(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

#define MIN(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a < _b ? _a : _b;                                                         \
  })

struct benchmark_results {
  double mean;
  double stddev;
  double min;
  double max;
  double p50;
  double p90;
  double p99;
};

static inline long long timespec_to_ns(struct timespec ts) {
  return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

static inline int set_affinity(int affinity) {
  cpu_set_t set;

  CPU_ZERO(&set);
  CPU_SET(affinity, &set);

  return sched_setaffinity(0, sizeof(set), &set);
}

int generate_data(int seed, int M, int N, double **A) {
  int i, j;
  double *aux, sum;

  *A = (double *)calloc(N * N, sizeof(double));
  if (*A == 0) {
    fprintf(stderr, "Failed calloc\n");
    return -1;
  }

  aux = *A;

  srand(seed);

  for (i = 0; i < M; ++i) {
    sum = 0;
    for (j = 0; j < N; ++j) {
      aux[i * N + j] = get_rand_double(RANGE);
      sum += aux[i * N + j];
    }
    for (j = 0; j < N; ++j) {
      aux[i * N + j] /= sum;
    }
  }

  return 0;
}

static int compare_long_long(const void *a, const void *b) {
  long long arg1 = *(const long long *)a;
  long long arg2 = *(const long long *)b;

  if (arg1 < arg2)
    return -1;
  if (arg1 > arg2)
    return 1;
  return 0;
}

static int get_percentile_idx(int runs, double percentile) {
  int idx = (int)ceil(runs * percentile) - 1;
  return MAX(0, MIN(idx, runs - 1));
}

static struct benchmark_results compute_statistics(long long *times_ns,
                                                   int runs) {
  struct benchmark_results results;

  qsort(times_ns, runs, sizeof(long long), compare_long_long);

  results.min = times_ns[0];
  results.max = times_ns[runs - 1];
  results.p50 = times_ns[get_percentile_idx(runs, 0.5)];
  results.p90 = times_ns[get_percentile_idx(runs, 0.9)];
  results.p99 = times_ns[get_percentile_idx(runs, 0.99)];

  // numerically Stable Mean and Variance (Welford's)
  double mean = 0.0;
  double M2 = 0.0;
  for (int i = 0; i < runs; i++) {
    double x = (double)times_ns[i];
    double delta = x - mean;
    mean += delta / (i + 1);
    double delta2 = x - mean;
    M2 += delta * delta2;
  }

  results.mean = mean;
  results.stddev = sqrt(M2 / (runs - 1));

  return results;
}

static struct benchmark_results run_benchmark(int N, int warmup_runs, int runs,
                                              double **A, double **B,
                                              double **C, char print_times) {

  for (int i = 0; i < warmup_runs; ++i) {
    my_solver(N, *A, *B, *C);
  }

  struct timespec start;
  struct timespec end;

  long long *times_ns = malloc(runs * sizeof(long long));
  if (times_ns == NULL) {
    fprintf(stderr, "Failed to allocate memory for times_ns\n");
    return (struct benchmark_results){0};
  }

  if (print_times) {

    for (int i = 0; i < runs; ++i) {
      clock_gettime(CLOCK_MONOTONIC, &start);
      my_solver(N, *A, *B, *C);
      clock_gettime(CLOCK_MONOTONIC, &end);

      long long elapsed_ns = timespec_to_ns(end) - timespec_to_ns(start);
      printf("Run %d: %lld ns\n", i + 1, elapsed_ns);

      times_ns[i] = elapsed_ns;
    }
  } else {

    for (int i = 0; i < runs; ++i) {
      clock_gettime(CLOCK_MONOTONIC, &start);
      my_solver(N, *A, *B, *C);
      clock_gettime(CLOCK_MONOTONIC, &end);

      long long elapsed_ns = timespec_to_ns(end) - timespec_to_ns(start);

      times_ns[i] = elapsed_ns;
    }
  }

  struct benchmark_results bench_results = compute_statistics(times_ns, runs);
  free(times_ns);

  return bench_results;
}

int main(int argc, char *argv[]) {
  int n = 0;
  int seed = 0;
  int warmup_runs = 0;
  int test_runs = 1;
  int affinity = 0;
  char print_times = 0;

  struct argparse_option options[] = {
      OPT_HELP(),
      OPT_INTEGER('n', "size", &n, "size of the matrices"),
      OPT_INTEGER('s', "seed", &seed,
                  "random seed for data generation, default: 0"),
      OPT_INTEGER('w', "warmup", &warmup_runs,
                  "number of warmup runs, default: 0"),
      OPT_INTEGER('t', "test", &test_runs, "number of test runs, default: 1"),
      OPT_INTEGER('a', "affinity", &affinity,
                  "CPU affinity for the benchmark, default: 0"),
      OPT_BOOLEAN('p', "print-times", &print_times,
                  "print individual run times, default: false"),
      OPT_END()};

  struct argparse argparse;
  argparse_init(&argparse, options, NULL, 0);
  argparse_describe(&argparse, "Benchmark for my_solver", NULL);
  argc = argparse_parse(&argparse, argc, (const char **)argv);

  if (n <= 0) {
    fprintf(stderr, "Invalid matrix size: %d\n", n);
    return EXIT_FAILURE;
  }

  if (set_affinity(affinity) != 0) {
    fprintf(stderr, "Failed to set CPU affinity: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  double *A;
  double *B;
  double *C;

  // generate data for A, B and C
  if (generate_data(seed, n, n, &A) < 0) {
    fprintf(stderr, "Failed to generate data for A\n");
    return EXIT_FAILURE;
  }

  if (generate_data(seed, n, n, &B) < 0) {
    fprintf(stderr, "Failed to generate data for B\n");
    free(A);
    return EXIT_FAILURE;
  }

  if (generate_data(seed, 1, n, &C) < 0) {
    fprintf(stderr, "Failed to generate data for C\n");
    free(A);
    free(B);
    return EXIT_FAILURE;
  }

  struct benchmark_results bench_results =
      run_benchmark(n, warmup_runs, test_runs, &A, &B, &C, print_times);

  if (bench_results.mean == 0) {
    fprintf(stderr, "Benchmark failed\n");
  } else {
    printf("N,runs,mean,stddev,min,max,p50,p90,p99\n");
    printf("%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", n, test_runs,
           bench_results.mean, bench_results.stddev, bench_results.min,
           bench_results.max, bench_results.p50, bench_results.p90,
           bench_results.p99);
  }

  free(A);
  free(B);
  free(C);
  return EXIT_SUCCESS;
}
