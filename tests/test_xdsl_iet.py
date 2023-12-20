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
        ut1 := FPowIOp(cst1, cst1),
    ])


def test_mfe2():

    mod = ModuleOp([
        ref := memref.Alloca.get(i32, 0, [3, 3]),
        cst1 := Constant.from_int_and_width(1, i32),
        a1 := memref.Load.get(ref, [cst1, cst1]),
        a2 := arith.Addi(a1, cst1),
        memref.Store.get(a2, ref, [cst1, cst1])
    ])
