from __future__ import annotations

from xdsl.dialects.builtin import StringAttr
from xdsl.irdl import (irdl_op_definition, result_def, attr_def, Attribute, IRDLOperation)
from xdsl.ir import (OpResult,
                     Dialect)


@irdl_op_definition
class LoadSymbolic(IRDLOperation):
    name = "devito.load_symbolic"
    symbol_name: StringAttr = attr_def(StringAttr)

    result: OpResult = result_def()

    @staticmethod
    def get(name: str, typ: Attribute):
        op = LoadSymbolic.build(
            attributes={'symbol_name': StringAttr(name)},
            result_types=[typ],
        )
        op.result.name_hint = name
        return op


DEVITO_SSA = Dialect([
    LoadSymbolic
], [
])
