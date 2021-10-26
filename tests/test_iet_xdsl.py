import pytest

from ctypes import c_void_p
import cgen
import sympy

from xdsl.dialects.builtin import *
from xdsl.parser import Printer, Parser
from devito.ir.ietxdsl import *


def test_expression():
    test_add = """
        module() {
    %0 : !i32 = iet.constant() ["value" = 42 : !i32]
    %1 : !i32 = iet.constant() ["value" = 43 : !i32]
    %2 : !i32 = iet.addi(%0 : !i32, %1 : !i32)
      }
    """
    ctx = MLContext()
    builtin = Builtin(ctx)
    iet = IET(ctx)

    parser = Parser(ctx, test_add)
    module = parser.parse_op()

    module.verify()
