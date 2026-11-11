CC      ?= gcc
MPICC   ?= mpicc
CFLAGS  ?= -O3 -std=gnu99 -Wall -Wextra

all: gol_seq gol_hybrid

gol_seq: src/gol_seq.c src/common.h
	$(CC) $(CFLAGS) -o $@ src/gol_seq.c

gol_hybrid: src/gol_hybrid.c src/common.h
	$(MPICC) $(CFLAGS) -fopenmp -o $@ src/gol_hybrid.c

test: all
	python3 tests/test_correctness.py

clean:
	rm -f gol_seq gol_hybrid *.txt
