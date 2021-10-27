from xdsl.dialects.builtin import *
from xdsl.parser import Printer
from xdsl.util import module
from devito.ir.ietxdsl import *


def test_expression():
    # Register the devito-specific dialects in xdsl context
    #
    # This initialization should be hidden somewhere in the ir.ietxdsl class
    # and does not need to be user-facing.
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

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
    mod = module([
        cst42 := iet.constant(42),
        cst3 := iet.constant(3),
        iet.addi(cst42, cst3),
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
