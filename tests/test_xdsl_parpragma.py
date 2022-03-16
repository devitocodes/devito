from xdsl.dialects.builtin import *
from xdsl.printer import Printer
from devito.ir.ietxdsl import *
from xdsl.dialects.builtin import ModuleOp
from devito.xdslpasses.iet.parpragma import make_simd


def test_example():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([
        Callable.get(
            "kernel", ["u"],
            Block.from_callable([iet.i32], lambda u: [
                Iteration
                .get(["affine", "sequential"], ("time_m", "time_M", "1"),
                     Block.from_callable([
                         iet.i32, iet.i32, iet.i32
                     ], lambda time, t0, t1: [
                         Iteration.
                         get(["affine", "parallel", "skewable"],
                             ("x_m", "x_M", "1"),
                             Block.from_callable([iet.i32], lambda x: [
                                 Iteration.get(
                                     [
                                         "affine",
                                         "parallel", "skewable", "vector-dim"
                                     ], ("y_m", "y_M", "1"),
                                     Block.from_callable([iet.i32], lambda y: [
                                         cst1 := Constant.get(1),
                                         x1 := Addi.get(x, cst1),
                                         y1 := Addi.get(y, cst1),
                                         ut0 := Idx.get(u, t0),
                                         ut0x1 := Idx.get(ut0, x1),
                                         ut0x1y1 := Idx.get(ut0x1, y1),
                                         rhs := Addi.get(ut0x1y1, cst1),
                                         ut1 := Idx.get(u, t1),
                                         ut1x1 := Idx.get(ut1, x1),
                                         lhs := Idx.get(ut1x1, y1),
                                         Assign.build([lhs, rhs])
                                     ]))
                             ]))
                     ]))
            ]))
    ])

    make_simd(ctx, mod)

    printer = Printer()
    printer.print_op(mod)
