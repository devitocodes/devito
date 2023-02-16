from devito import Grid, TimeFunction, Eq, Operator
from devito.tools import as_tuple

from devito.ir.ietxdsl import (MLContext, CGeneration, Powi, IET, Callable,
                               Block, Iteration, Idx, Assign, Initialise,
                               floatingPointLike)

from devito.ir.iet import retrieve_iteration_tree

from xdsl.dialects.builtin import ModuleOp, Builtin, i32
from xdsl.printer import Printer
from xdsl.dialects.arith import Addi, Constant, Subi
from xdsl.dialects.memref import Load
from xdsl.ir import Operation, Block, Region


# flake8: noqa

def test_ops_accessor_II():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)

    block0 = Block.from_ops([a, b, c])
    # Create a region to include a, b, c
    region = Region.from_block_list([block0])

    assert len(region.ops) == 3
    assert len(region.blocks[0].ops) == 3

    # Operation to subtract b from a
    d = Subi.get(a, b)

    assert d.results[0] != c.results[0]

    # Erase operations and block
    region2 = Region()
    region.move_blocks(region2)

    region2.blocks[0].erase_op(a, safe_erase=False)
    region2.blocks[0].erase_op(b, safe_erase=False)
    region2.blocks[0].erase_op(c, safe_erase=False)

    region2.detach_block(block0)
    region2.drop_all_references()

    assert len(region2.blocks) == 0


def test_powi():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([
        cst1 := Constant.from_int_and_width(1, i32),
        ut1 := Powi.get(cst1, cst1),
    ])

    printer = Printer()
    printer.print_op(mod)


def test_Idx():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([
        cst1 := Constant.from_int_and_width(1, i32),
        ut1 := Idx.get(cst1, cst1),
        y1 := Addi.get(cst1, cst1)
    ])

    printer = Printer()
    printer.print_op(mod)

def test_assign():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([
        cst1 := Constant.from_int_and_width(1, i32),
        x1 := Addi.get(cst1, cst1),
        y1 := Addi.get(cst1, cst1),
        ut0 := Idx.get(x1, cst1),
        ut0x1 := Idx.get(ut0, x1),
        ut0x1y1 := Idx.get(ut0x1, y1),
        rhs := Addi.get(ut0x1y1, cst1),
        ut1 := Idx.get(x1, cst1),
        ut1x1 := Idx.get(ut1, x1),
        lhs := Idx.get(ut1x1, y1),
        Assign.build([lhs, rhs])
    ])

    printer = Printer()
    printer.print_op(mod)

def test_Iteration():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([

    Iteration.get(
        [
            "affine",
            "parallel", "skewable", "vector-dim"
        ], ("y_m", "y_M", "1"), "y_loop",
        Block.from_callable([i32], lambda y: [
            cst1 := Constant.from_int_and_width(1, i32),
        ]))


    ])

    printer = Printer()
    printer.print_op(mod)


def test_Initialise():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([
        cst1 := Constant.from_int_and_width(1, i32),
        x1 := Addi.get(cst1, cst1),
        y1 := Addi.get(cst1, cst1),
        ut0 := Idx.get(x1, cst1),
        ut0x1 := Idx.get(ut0, x1),
        ut0x1y1 := Idx.get(ut0x1, y1),
        rhs := Addi.get(ut0x1y1, cst1),
        ut1 := Idx.get(x1, cst1),
        ut1x1 := Idx.get(ut1, x1),
        lhs := Idx.get(ut1x1, y1),
        Assign.build([lhs, rhs])
    ])

    printer = Printer()
    printer.print_op(mod)


def test_blockIteration():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([

        Iteration.get(["affine", "sequential"], ("time_m", "time_M", "1"),"time_loop",
                     Block.from_callable([
                         i32, i32, i32
                     ], lambda time, t0, t1: [
                         Iteration.
                         get(["affine", "parallel", "skewable"],
                             ("x_m", "x_M", "1"),"x_loop",
                             Block.from_callable([i32], lambda x: [
                                 Iteration.get(
                                     [
                                         "affine",
                                         "parallel", "skewable", "vector-dim"
                                     ], ("y_m", "y_M", "1"),"y_loop",
                                     Block.from_callable([i32], lambda y: [
                                         cst1 := Constant.from_int_and_width(1, i32),
                                         x1 := Addi.get(x, cst1),
                                         y1 := Addi.get(y, cst1),
                                         ut0 := Idx.get(x, t0),
                                         ut0x1 := Idx.get(ut0, x1),
                                         ut0x1y1 := Idx.get(ut0x1, y1),
                                         rhs := Addi.get(ut0x1y1, cst1),
                                         ut1 := Idx.get(x, t1),
                                         ut1x1 := Idx.get(ut1, x1),
                                         lhs := Idx.get(ut1x1, y1),
                                         Assign.build([lhs, rhs])
                                     ]))
                             ]))
                     ]))
            ])

    printer = Printer()
    printer.print_op(mod)


def test_callable():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)

    block0 = Block.from_ops([a, b, c])

    mod = ModuleOp.from_region_or_ops([
        Callable.get(
            "kernel", ["u"], ["u"], ["struct dataobj*"], ["restrict"], "int", "",
            block0)
            ])

    printer = Printer()
    printer.print_op(mod)


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
        cst42 := Constant.from_int_and_width(42, i32),
        cst3 := Constant.from_int_and_width(3, i32),
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
            "kernel", ["u"],["u"],["struct dataobj*"], ["restrict"], "int", "",
            Block.from_callable([i32], lambda u: [
                Iteration
                .get(["affine", "sequential"], ("time_m", "time_M", "1"),"time_loop",
                     Block.from_callable([
                         i32, i32, i32
                     ], lambda time, t0, t1: [
                         Iteration.
                         get(["affine", "parallel", "skewable"],
                             ("x_m", "x_M", "1"), "x_loop",
                             Block.from_callable([i32], lambda x: [
                                 Iteration.get(
                                     [
                                         "affine",
                                         "parallel", "skewable", "vector-dim"
                                     ], ("y_m", "y_M", "1"),"y_loop",
                                     Block.from_callable([i32], lambda y: [
                                         cst1 := Constant.from_int_and_width(1, i32),
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

    iters = retrieve_iteration_tree(op.body)
    
    t_limits = as_tuple([str(i) for i in iters[0][0].limits])
    t_props = [str(i) for i in iters[0][0].properties]
 
    x_limits = as_tuple([str(i) for i in iters[0][1].limits])
    x_props = [str(i) for i in iters[0][1].properties]

    y_limits = [str(i) for i in iters[0][2].limits]
    y_props = [str(i) for i in iters[0][2].properties]

    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([
        Callable.get("kernel", ["u"], ["u"],["struct dataobj*"], ["restrict"], "int", "",
                     Block.from_callable([i32], lambda u: [
            Iteration.get(t_props, t_limits, iters[0][0].dim.name,
                Block.from_callable([i32, i32, i32],
                                    lambda time, t0, t1: [
                    Iteration.get(x_props, x_limits, iters[0][1].dim.name,
                    Block.from_callable([i32], lambda x: [
                                  Iteration.get(y_props, y_limits, iters[0][2].dim.name,
                                  Block.from_callable([i32], lambda y: [
                        cst1    := Constant.from_int_and_width(1, i32),
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


def test_mfe():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([
        Callable.get(
            "kernel", ["u"],["u"],["struct dataobj*"], ["restrict"], "int", "",
            Block.from_callable([i32], lambda u: [
                Iteration
                .get(["affine", "sequential"], ("time_m", "time_M", "1"),"time_loop",
                     Block.from_callable([
                         i32, i32, i32
                     ], lambda time, t0, t1: [
                                 Iteration.get(
                                     [
                                         "affine",
                                         "parallel", "skewable", "vector-dim"
                                     ], ("y_m", "y_M", "1"),"y_loop",
                                     Block.from_callable([i32], lambda y: [
                                         cst1 := Constant.from_int_and_width(1, i32),
                                         y1 := Addi.get(y, cst1),
                                         ut0 := Idx.get(u, t0),
                                         rhs := Addi.get(ut0, cst1),
                                         ut1 := Idx.get(u, t1),
                                         ut1x1 := Idx.get(ut1, y1),
                                         lhs := Idx.get(ut1x1, y1),
                                         Assign.build([lhs, rhs])
                                     ]))
                             ]))
                     ]))
    ])

    printer = Printer()
    printer.print_op(mod)


def test_mfe2():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    mod = ModuleOp.from_region_or_ops([
        cst1 := Constant.from_int_and_width(1, i32),
        ut1 := Idx.get(cst1, cst1),
        y1 := Addi.get(cst1, cst1)
    ])

    printer = Printer()
    printer.print_op(mod)