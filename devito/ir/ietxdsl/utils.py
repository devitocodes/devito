from xdsl.dialects import builtin
from xdsl.ir import SSAValue


def is_int(val: SSAValue):
    return isinstance(val.type, builtin.IntegerType)


def is_float(val: SSAValue):
    return val.type in (builtin.f32, builtin.f64)
