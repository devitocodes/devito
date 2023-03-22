from subprocess import check_call

main_filename = "main.mlir"
kernel_filename = "kernel.mlir"
c_filename = "debug.c"
executable_name = "out"

print("Compiling C utils:")
c_object = c_filename+".o"
c_object_cmd = f"gcc -c -o {c_object} {c_filename}"
print(c_object_cmd)
check_call(c_object_cmd, shell=True)

print("Compiling MLIR main:")
main_object = main_filename+".o"
main_object_cmd = f'mlir-opt {main_filename} ''--pass-pipeline="builtin.module(cse, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, finalize-memref-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, cse)" | mlir-translate --mlir-to-llvmir | 'f'clang -x ir -c -o {main_object} -'
print(main_object_cmd)
check_call(main_object_cmd, shell=True)

print("Compiling kernel:")
kernel_object = kernel_filename+".o"
kernel_object_cmd = f'xdsl-opt {kernel_filename} -t mlir -p convert-stencil-to-ll-mlir | ''mlir-opt --pass-pipeline="builtin.module(cse, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64}, finalize-memref-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, cse)" | mlir-translate --mlir-to-llvmir | 'f'clang -x ir -c -o {kernel_object} -'
print(kernel_object_cmd)
check_call(kernel_object_cmd, shell=True)

print("Linking executable:")
link_cmd = f'steam-run clang {main_object} {kernel_object} {c_object} -o {executable_name}'
print(link_cmd)
check_call(link_cmd, shell=True)

print("Running executable:")
run_cmd = f'./{executable_name}'
print(run_cmd)
#check_call(run_cmd, shell=True)
