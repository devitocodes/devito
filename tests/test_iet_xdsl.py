from xdsl.printer import Printer
from devito.ir.ietxdsl import (MLContext, Builtin, Constant, Addi, CGeneration,
                               IET, Callable, Block, Iteration, Idx, Assign)
from devito.tools import as_tuple
from xdsl.dialects.builtin import ModuleOp

from devito import Grid, TimeFunction, Eq, Operator

# flake8: noqa

def test_expression():
    # Register the devito-specific dialects in xdsl context
    #
    # This initialization should be hidden somewhere in the ir.ietxdsl class
    # and does not need to be user-facing.
    ctx = MLContext()
    Builtin(ctx)
    # iet = IET(ctx)

    # This is a very explicit encoding of the expression tree representing
    # 42 + 3. There are a couple of issues with this encoding:
    #
    # - A list of operations does not necessarily encode a single tree,
    #   but can contain disconnected operations.
    # - Printing this recursively is just by walking over the set of operations
    #   will result in rather length code.
    #
    # We might want to consider a mode where we can transparently move between
    # a sequential and a tree representation of such an ir.
    mod = ModuleOp.from_region_or_ops([
        cst42 := Constant.get(42),
        cst3 := Constant.get(3),
        Addi.get(cst42, cst3),
    ])

    # result_string = """
    #    module() {
    #      %0 : !i32 = iet.constant() ["value" = 42 : !i32]
    #      %1 : !i32 = iet.constant() ["value" = 3 : !i32]
    #      %2 : !i32 = iet.addi(%0 : !i32, %1 : !i32)
    #    }
    #    """
    # I would like to print the module to a string to be able to
    # compare it with the expected result. AFAIU xdsl.util.Printer
    # can only print to stdout, so this is not yet supported.
    printer = Printer()
    printer.print_op(mod)

    cgen = CGeneration()
    cgen.printModule(mod)

    assert cgen.str() == "42 + 3", "Unexpected C output"


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

    printer = Printer()
    printer.print_op(mod)


def test_devito_iet():
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u + 1)
    op = Operator([eq])

    t_limits = as_tuple([str(i) for i in op.body.body[1].body[0].limits])
    t_props = [str(i) for i in op.body.body[1].body[0].properties]

    x_limits = as_tuple([str(i) for i in op.body.body[1].body[0].nodes[0].body[0].body[0].limits])  # noqa
    x_props = [str(i) for i in op.body.body[1].body[0].nodes[0].body[0].body[0].properties]  # noqa

    y_limits = as_tuple([str(i) for i in op.body.body[1].body[0].nodes[0].body[0].body[0].nodes[0].limits])  # noqa
    y_props = [str(i) for i in op.body.body[1].body[0].nodes[0].body[0].body[0].nodes[0].properties]  # noqa

    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([
        Callable.get("kernel", ["u"], Block.from_callable([iet.i32], lambda u: [
            Iteration.get(t_props, t_limits,
                Block.from_callable([iet.i32, iet.i32, iet.i32],
                                    lambda time, t0, t1: [
                    Iteration.get(x_props, x_limits,
                    Block.from_callable([iet.i32], lambda x: [
                                  Iteration.get(y_props, y_limits,
                                  Block.from_callable([iet.i32], lambda y: [
                        cst1    := Constant.get(1),
                        x1      := Addi.get(x, cst1),
                        y1      := Addi.get(y, cst1),
                        ut0     := Idx.get(u, t0),
                        ut0x1   := Idx.get(ut0, x1),
                        ut0x1y1 := Idx.get(ut0x1, y1),
                        rhs     := Addi.get(ut0x1y1, cst1),
                        ut1     := Idx.get(u, t1),
                        ut1x1   := Idx.get(ut1, x1),
                        lhs     := Idx.get(ut1x1, y1),
                        Assign.build([lhs, rhs])
                    ]))
                ]))
            ]))
        ]))
    ])

    printer = Printer()
    printer.print_op(mod)
