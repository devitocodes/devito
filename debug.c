#include <stdio.h>


void print_time(int i) {
    printf("time: %d\n", i);
}
void print_float(float i) {
    printf("element value: %.2f\n", i);
}
void print_idx_3(float i, int t, int x, int y) {
    printf("u[%d][%d][%d] = %.2f\n", t, x, y, i);
}


/*
add these manually in the generated `.iet.mlir` file, at the bottom into the "builtin.module"

llvm.func @print_time(i32) -> () attributes {sym_visibility = "private"}
llvm.func @print_float(f32) -> () attributes {sym_visibility = "private"}
llvm.func @print_idx_3(f32, i32, i32, i32) -> () attributes {sym_visibility = "private"}

Then you can call them using

llvm.call @print_time(%time) : (i32) -> ()
llvm.call @print_float(%val) : (f32) -> ()
llvm.call @print_idx_3(%val, %t, %x, %y) : (f32, i32, i32, i32) -> ()


*/