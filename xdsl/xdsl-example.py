from xdsl.dialects.builtin import *
from xdsl.parser import Printer, Parser
from devito.ir.ietxdsl import *

# TODO setup file check tests
test_prog = """
module() {
  %0 : !i32 = iet.constant() ["value" = 42 : !i32]
  %1 : !i32 = iet.constant() ["value" = 42 : !i32]
  %2 : !i32 = iet.addi(%0 : !i32, %1 : !i32) 
  %3 : !i32 = iet.muli(%0 : !i32, %1 : !i32) 
}
"""


def test_main():
    ctx = MLContext()
    builtin = Builtin(ctx)
    iet = IET(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    printer = Printer()
    printer.print_op(module)
    print()

    print("Done")

test_main()
