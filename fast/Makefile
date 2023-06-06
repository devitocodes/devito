NIXOS_CLANG_PREFIX ?= 
CC = clang

# Add flags for C
CFLAGS = -O3 

# MODE decides if we want to compile for gpu or cpu
# might be extended to omp and others as well
MODE ?= cpu
DUMP ?= 1
THREADS ?= 1

ifeq ($(DUMP),0)
CFLAGS += -DNODUMP
endif

# handle gpu flags for clang
ifeq ($(MODE),gpu)
CFLAGS += -lmlir_cuda_runtime
BENCH_OPTS_EXTRA += --gpu
else ifeq ($(MODE),cpu)
else ifeq ($(MODE),openmp)
STENCIL_ENV = OMP_PLACES=threads OMP_NUM_THREADS=$(THREADS)
DEVITO_ENV = DEVITO_LANGUAGE=openmp $(STENCIL_ENV)
endif

MAIN_MLIR_FILE_PIPELINE := "builtin.module(canonicalize, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, $(MEMREF_TO_LLVM_PASS){index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, canonicalize)"

.PHONY: %.bench
ifeq ($(DUMP), 0)
BENCH_OPTS_EXTRA = --no_output_dump
.PHONY: %.stencil.data
endif



# this creates the devito data. afaik devito requires gcc for this on my machine. Probably nixos related :S
# it calls the example script with the bench options, meaning it runs normal devito
%.devito.data %.input.data: CC = gcc
%.devito.data %.input.data:
	$(DEVITO_ENV) python3 fast_benchmarks.py $* $(BENCH_OPTS) $(BENCH_OPTS_EXTRA)

# this calls the example file with the `-xdsl` flag so it dumps the mlir files
%.mlir %.main.o %.interop.o %.kernel.o %.out %.stencil.data: %.input.data
	$(DEVITO_ENV) python3 fast_benchmarks.py $* $(BENCH_OPTS) $(BENCH_OPTS_EXTRA) -xdsl

ifeq ($(DUMP),0)
%.bench: %.stencil.data %.devito.data
	@echo "Done with no dumps, skipping output comparison"
else
%.bench: %.stencil.data %.devito.data
	python3 compare.py --name  $(patsubst %.bench,%,$@) $(BENCH_OPTS)
endif


#python3 compare.py $(BENCH_OPTS) --name $(patsubst %.bench,%,$@)

.PHONY: %.bench clean

# remove all intermediates and results
clean:
	rm -f *.mlir
	rm -f *.o
	rm -f *.data
	rm -f *.out