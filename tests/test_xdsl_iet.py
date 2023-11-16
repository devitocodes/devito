from devito import Grid, TimeFunction, Eq, Operator
from devito.tools import as_tuple

from devito.ir.iet import retrieve_iteration_tree

from xdsl.dialects.builtin import ModuleOp, Builtin, i32, f32

from xdsl.printer import Printer
from xdsl.dialects.arith import Addi, Constant, Subi
from xdsl.dialects.experimental.math import FPowIOp
from xdsl.dialects import memref, arith
from xdsl.ir import Operation, Block, Region
import pytest

# flake8: noqa

def test_ops_accessor_II():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi(a, b)

    block0 = Block([a, b, c])
    # Create a region to include a, b, c
    region = Region([block0])

    assert len(region.ops) == 3
    assert len(region.blocks[0].ops) == 3

    # Operation to subtract b from a
    d = Subi(a, b)

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

    mod = ModuleOp([
        cst1 := Constant.from_int_and_width(1, i32),
        ut1 := FPowIOp.get(cst1, cst1),
    ])


@pytest.mark.xfail(reason="Deprecated, will be dropped")
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
                        #ut0     := Idx.get(u, t0),
                        #ut0x1   := Idx.get(ut0, x1),
                        #ut0x1y1 := Idx.get(ut0x1, y1),
                        #rhs     := Addi.get(ut0x1y1, cst1),
                        #ut1     := Idx.get(u, t1),
                        #ut1x1   := Idx.get(ut1, x1),
                        #lhs     := Idx.get(ut1x1, y1),
                        #Assign.build([lhs, rhs])
                    ]))
                ]))
            ]))
        ]))
    ])

    printer = Printer()
    printer.print_op(mod)


@pytest.mark.xfail(reason="Deprecated, will be dropped")
def test_mfe_memref():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    memref_f32_rank2 = memref.MemRefType.from_element_type_and_shape(
        f32, [-1, -1])

    mod = ModuleOp.from_region_or_ops([
        Callable.get(
            "kernel", ["u"],["u"],["struct dataobj*"], ["restrict"], "int", "",
            Block.from_callable([memref_f32_rank2], lambda u: [
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
                                         ut0 := memref.Load.get(u, [t0, y1]),
                                         # ut0 := Idx.get(u, t0),
                                         rhs := Addi.get(ut0, cst1),
                                         memref.Store.get(rhs, u, [t1, y1])
                                         # Assign.build([lhs, rhs])
                                     ]))
                             ]))
                     ]))
    ])

    printer = Printer()
    # import pdb;pdb.set_trace()
    printer.print_op(mod)

def test_mfe2():

    mod = ModuleOp([
        ref := memref.Alloca.get(i32, 0, [3, 3]),
        cst1 := Constant.from_int_and_width(1, i32),
        a1 := memref.Load.get(ref, [cst1, cst1]),
        a2 := arith.Addi(a1, cst1),
        memref.Store.get(a2, ref, [cst1, cst1])
    ])

    printer = Printer()
    printer.print_op(mod)