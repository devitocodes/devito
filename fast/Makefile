NIXOS_CLANG_PREFIX ?= 
CC = clang

CFLAGS = -O3 

MODE ?= cpu

# handle gpu flags for clang
ifeq ($(MODE),gpu)
CFLAGS += -lmlir_cuda_runtime
endif

CPU_PIPELINE := "builtin.module(canonicalize, loop-invariant-code-motion, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, finalize-memref-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, canonicalize)"
GPU_PIPELINE := "builtin.module(test-math-algebraic-simplification,scf-parallel-loop-tiling{parallel-loop-tile-sizes=1024,1,1}, canonicalize, func.func(gpu-map-parallel-loops), convert-parallel-loops-to-gpu, lower-affine, gpu-kernel-outlining,func.func(gpu-async-region),canonicalize,convert-arith-to-llvm{index-bitwidth=64},finalize-memref-to-llvm{index-bitwidth=64},convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=64},gpu.module(convert-gpu-to-nvvm,reconcile-unrealized-casts,canonicalize,gpu-to-cubin),gpu-to-llvm,canonicalize)"

XDSL_CPU_PIPELINE := stencil-shape-inference,convert-stencil-to-ll-mlir
XDSL_GPU_PIPELINE := stencil-shape-inference,convert-stencil-to-gpu

MAIN_MLIR_FILE_PIPELINE := "builtin.module(canonicalize, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, finalize-memref-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, canonicalize)"

%.devito.data %.input.data: CC = gcc
%.devito.data %.input.data: %.py 
	python3 $< $(BENCH_OPTS)

%.mlir %.main.mlir: %.py
	python3 $< $(BENCH_OPTS) -xdsl

%.interop.o: interop.c
	gcc -O3 -c interop.c -o $@ -DOUTFILE_NAME='"'$(patsubst %.interop.o,%.stencil.data,$@)'"' -DINFILE_NAME='"'$(patsubst %.interop.o,%.input.data,$@)'"'

%.gpu.o : PIPELINE = $(GPU_PIPELINE) 
%.gpu.o : XDSL_PIPE = $(XDSL_GPU_PIPELINE)
%.gpu.o: %.mlir 
	xdsl-opt $< -t mlir -p $(XDSL_PIPE) | mlir-opt --pass-pipeline=$(PIPELINE) | mlir-translate --mlir-to-llvmir | $(CC) -x ir -c -o $@ -

%.cpu.o: PIPELINE = $(CPU_PIPELINE) 
%.cpu.o: XDSL_PIPE = $(XDSL_CPU_PIPELINE)
%.cpu.o: %.mlir 
	xdsl-opt $< -t mlir -p $(XDSL_PIPE) | mlir-opt --pass-pipeline=$(PIPELINE) | mlir-translate --mlir-to-llvmir | $(CC) -x ir -c -o $@ -
	
%.main.o: %.main.mlir
	mlir-opt $< --pass-pipeline=$(MAIN_MLIR_FILE_PIPELINE) | tee $<.llvm.mlir | mlir-translate --mlir-to-llvmir | $(CC) -x ir -c -o $@ -

%.out: %.main.o %.$(MODE).o %.interop.o
	$(NIXOS_CLANG_PREFIX) $(CC) $^ -o $@

%.bench: %.out %.input.data
	time $<
	python3 compare.py $(patsubst %.bench,%,$@)

.PHONY: %.bench clean

clean:
	rm -f *.mlir
	rm -f *.o
	rm -f *.data
	rm -f *.out


