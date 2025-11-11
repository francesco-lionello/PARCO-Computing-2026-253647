# ===========================================================
# Makefile for SpMV Project with Benchmarks on Multiple Matrices
# -----------------------------------------------------------
# Usage:
#   make            # build executable spmv
#   make run        # run spmv on default test matrix
#   make bench      # run spmv on ALL matrices
#   make 1138_bus   # run only on 1138_bus.mtx
#   make iris       # run only on iris_dataset_30NN.mtx
#   make clean      # cleanup
# ===========================================================

# Compiler and flags

CC = gcc-9.1.0
CFLAGS = -O3 -Wall -Wextra -march=native -fopenmp
LDFLAGS = -lrt

# -----------------------------------------------------------
# Build spmv
# -----------------------------------------------------------
all: spmv

spmv: spmv.c
	$(CC) $(CFLAGS) -o spmv spmv.c $(LDFLAGS)

# -----------------------------------------------------------
# Run on default matrix
# -----------------------------------------------------------
run: spmv
	./spmv matrix/1138_bus.mtx 20

# -----------------------------------------------------------
# Run on ALL matrices
# -----------------------------------------------------------
bench: spmv
	@for M in 1138_bus.mtx bfwa62.mtx iris_dataset_30NN.mtx mnist_test_norm_10NN.mtx Spielman_k200.mtx; do \
		echo ">>> Running on $$M..."; \
		./spmv matrix/$$M 20; \
		echo ""; \
	done

# -----------------------------------------------------------
# Individual runs
# -----------------------------------------------------------
1138_bus: spmv
	./spmv matrix/1138_bus.mtx 20

bfwa62: spmv
	./spmv matrix/bfwa62.mtx 20

iris: spmv
	./spmv matrix/iris_dataset_30NN.mtx 20

mnist: spmv
	./spmv matrix/mnist_test_norm_10NN.mtx 20

spielman200: spmv
	./spmv matrix/Spielman_k200.mtx 20

# -----------------------------------------------------------
# Cleanup
# -----------------------------------------------------------
clean:
	rm -f results.csv spmv
