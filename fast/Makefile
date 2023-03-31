NIXOS_CLANG_PREFIX ?= 
CC = clang

# Add flags for C
CFLAGS = -O3 

# MODE decides if we want to compile for gpu or cpu
# might be extended to omp and others as well
MODE ?= cpu

# handle gpu flags for clang
ifeq ($(MODE),gpu)
CFLAGS += -lmlir_cuda_runtime
endif

CPU_PIPELINE := "builtin.module(canonicalize, loop-invariant-code-motion, convert-scf-to-openmp, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, finalize-memref-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, canonicalize)"
GPU_PIPELINE := "builtin.module(test-math-algebraic-simplification,scf-parallel-loop-tiling{parallel-loop-tile-sizes=1024,1,1}, canonicalize, func.func(gpu-map-parallel-loops), convert-parallel-loops-to-gpu, lower-affine, gpu-kernel-outlining,func.func(gpu-async-region),canonicalize,convert-arith-to-llvm{index-bitwidth=64},finalize-memref-to-llvm{index-bitwidth=64},convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=64},gpu.module(convert-gpu-to-nvvm,reconcile-unrealized-casts,canonicalize,gpu-to-cubin),gpu-to-llvm,canonicalize)"

XDSL_CPU_PIPELINE := stencil-shape-inference,convert-stencil-to-ll-mlir
XDSL_GPU_PIPELINE := stencil-shape-inference,convert-stencil-to-gpu

MAIN_MLIR_FILE_PIPELINE := "builtin.module(canonicalize, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, finalize-memref-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, canonicalize)"

# this creates the devito data. afaik devito requires gcc for this on my machine. Probably nixos related :S
# it calls the example script with the bench options, meaning it runs normal devito
%.devito.data %.input.data: CC = gcc
%.devito.data %.input.data: %.py 
	python3 $< $(BENCH_OPTS)

# this calls the example file with the `-xdsl` flag so it dumps the mlir files
%.mlir %.main.mlir: %.py
	python3 $< $(BENCH_OPTS) -xdsl

# this compiles the specific interop file to read/write the in/out files
# it must be re-compiled for each testcase, as it has the paths it reads/writes as comptime constants
# these are passed in as OUTFILE_NAME and INFILE_NAME
%.interop.o: interop.c
	gcc -O3 -c interop.c -o $@ -DOUTFILE_NAME='"'$(patsubst %.interop.o,%.stencil.data,$@)'"' -DINFILE_NAME='"'$(patsubst %.interop.o,%.input.data,$@)'"'

# this compiles the kernel to gpu code
# here we choose the pipeline to use
%.gpu.o : PIPELINE = $(GPU_PIPELINE) 
%.gpu.o : XDSL_PIPE = $(XDSL_GPU_PIPELINE)
%.gpu.o: %.mlir 
	xdsl-opt $< -t mlir -p $(XDSL_PIPE) | mlir-opt --pass-pipeline=$(PIPELINE) | mlir-translate --mlir-to-llvmir | $(CC) -x ir -c -o $@ -

# this compiles the kernel to cpu
%.cpu.o: PIPELINE = $(CPU_PIPELINE) 
%.cpu.o: XDSL_PIPE = $(XDSL_CPU_PIPELINE)
%.cpu.o: %.mlir 
	xdsl-opt $< -t mlir -p $(XDSL_PIPE) | mlir-opt --pass-pipeline=$(PIPELINE) | mlir-translate --mlir-to-llvmir | $(CC) -x ir -c -o $@ -
	
# this compiles the main file to object
%.main.o: %.main.mlir
	mlir-opt $< --pass-pipeline=$(MAIN_MLIR_FILE_PIPELINE) | tee $<.llvm.mlir | mlir-translate --mlir-to-llvmir | $(CC) -x ir -c -o $@ -

# this assembles all object files
%.out: %.main.o %.$(MODE).o %.interop.o
	$(NIXOS_CLANG_PREFIX) $(CC) $^ -o $@

# this runs the <name>.out file and then compares resuls
%.bench: %.out %.input.data
	time $<
	python3 compare.py $(BENCH_OPTS) --name $(patsubst %.bench,%,$@)

.PHONY: %.bench clean

# remove all intermediates and results
clean:
	rm -f *.mlir
	rm -f *.o
	rm -f *.data
	rm -f *.out


