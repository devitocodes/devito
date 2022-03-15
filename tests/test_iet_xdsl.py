from xdsl.dialects.builtin import *
from xdsl.printer import Printer
from devito.ir.ietxdsl import *
from xdsl.dialects.builtin import ModuleOp


def test_expression():
    # Register the devito-specific dialects in xdsl context
    #
    # This initialization should be hidden somewhere in the ir.ietxdsl class
    # and does not need to be user-facing.
    ctx = MLContext()
    Builtin(ctx)
    #iet = IET(ctx)

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


##def test_example():
#    ctx = MLContext()
#    Builtin(ctx)
#    iet = IET(ctx)
#
#    mod = module([
#        iet.callable("kernel", ["u"], block([iet.i32], lambda u: [
#            iet.iteration(["affine", "sequential"], ("time_m", "time_M", "1"),
#                          block([iet.i32, iet.i32, iet.i32], lambda time, t0, t1: [
#                iet.iteration(["affine", "parallel", "skewable"], ("x_m", "x_M", "1"),
#                              block(iet.i32, lambda x: [
#                    iet.iteration(["affine", "parallel", "skewable", "vector-dim"], ("y_m", "y_M", "1"),
#                                  block(iet.i32, lambda y: [
#                        cst1    := iet.constant(1),
#                        x1      := iet.addi(x, cst1),
#                        y1      := iet.addi(y, cst1),
#                        ut0     := iet.idx(u, t0),
#                        ut0x1   := iet.idx(ut0, x1),
#                        ut0x1y1 := iet.idx(ut0x1, y1),
#                        rhs     := iet.addi(ut0x1y1, cst1),
#                        ut1     := iet.idx(u, t1),
#                        ut1x1   := iet.idx(ut1, x1),
#                        lhs     := iet.idx(ut1x1, y1),
#                        iet.assign(lhs, rhs)
#                    ]))
#                ]))
#            ]))
#        ]))
#    ])
#
#    printer = Printer()
#    printer.print_op(mod)
