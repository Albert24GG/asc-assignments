CC=gcc
CFLAGS=-Wall -Werror -O0 -march=native
CFLAGS_BENCH=-Wall -Werror -O0 -march=native -lm
LIBS=-lopenblas

all: compare tema1_blas tema1_neopt tema1_opt_m 

bench: bench_blas bench_neopt bench_opt_m

tema1_blas: solver_blas.c main.c utils.h
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

tema1_neopt: solver_neopt.c main.c utils.h
	$(CC) $(CFLAGS) $^ -o $@

tema1_opt_m: solver_opt.c main.c utils.h
	$(CC) $(CFLAGS) $^ -o $@

compare: compare.c utils.h
	$(CC) $(OPT_CFLAGS) $^ -lm -o $@

clean:
	rm -rf compare tema1_blas tema1_neopt tema1_opt_m
	rm -rf bench_blas bench_neopt bench_opt_m
	rm -rf out1 out2 out3
