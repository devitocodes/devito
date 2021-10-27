from xdsl.dialects.builtin import *
from xdsl.parser import Printer
from xdsl.util import module
from devito.ir.ietxdsl import *


def test_expression():
    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

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
