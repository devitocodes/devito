from subprocess import check_call

main_filename = "main.mlir"
kernel_filename = "kernel.mlir"
c_filename = "debug.c"
executable_name = "out"
gpu = True

print("Compiling C utils:")
c_object = c_filename+".o"
c_object_cmd = f"clang -c -o {c_object} {c_filename}"
print(c_object_cmd)
check_call(c_object_cmd, shell=True)

print("Compiling MLIR main:")
main_object = main_filename+".o"
main_object_cmd = f'mlir-opt {main_filename} ''--pass-pipeline="builtin.module(canonicalize, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, convert-memref-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, canonicalize)" | mlir-translate --mlir-to-llvmir | 'f'clang -x ir -c -o {main_object} -'
print(main_object_cmd)
check_call(main_object_cmd, shell=True)

print(f'Compiling kernel on {"GPU" if gpu else "CPU"}:')
kernel_object = kernel_filename+".o"
cpu_pipeline = '"builtin.module(                      canonicalize, loop-invariant-code-motion, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, convert-memref-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, canonicalize)"'
gpu_pipeline = '"builtin.module(test-math-algebraic-simplification,scf-parallel-loop-tiling{parallel-loop-tile-sizes=1024,1,1}, canonicalize, func.func(gpu-map-parallel-loops), convert-parallel-loops-to-gpu, lower-affine, gpu-kernel-outlining, canonicalize, convert-arith-to-llvm{index-bitwidth=64},convert-memref-to-llvm{index-bitwidth=64},convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=64},gpu.module(convert-gpu-to-nvvm,reconcile-unrealized-casts,canonicalize,gpu-to-cubin),gpu-to-llvm,canonicalize)"'
pipeline = gpu_pipeline if gpu else cpu_pipeline
xdsl_cpu_pipeline = 'stencil-shape-inference,convert-stencil-to-ll-mlir'
xdsl_gpu_pipeline = 'stencil-shape-inference,convert-stencil-to-gpu'
xdsl_pipeline = xdsl_gpu_pipeline if gpu else xdsl_cpu_pipeline
kernel_object_cmd = f'xdsl-opt {kernel_filename} -t mlir -p {xdsl_pipeline} | 'f'mlir-opt --pass-pipeline={pipeline} | mlir-translate --mlir-to-llvmir | 'f'clang -x ir -c -o {kernel_object} -'
print(kernel_object_cmd)
check_call(kernel_object_cmd, shell=True)

print("Linking executable:")
link_cmd = f'clang {main_object} {kernel_object} {c_object} -lmlir_cuda_runtime -o {executable_name}'
print(link_cmd)
check_call(link_cmd, shell=True)

print("Running executable:")
run_cmd = f'./{executable_name}'
print(run_cmd)
#check_call(run_cmd, shell=True)
