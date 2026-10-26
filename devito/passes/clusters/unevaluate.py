from devito.ir import SymbolRegistry
from devito.symbolics import unevaluate as _unevaluate
from devito.types import Temp

__all__ = ['unevaluate']


def unevaluate(clusters):
    # Abstract local temporaries in definition order before choosing operand order
    exprs = [e for c in clusters for e in c.exprs]
    sregistry = SymbolRegistry()
    sregistry.reserve(exprs)
    # Lexical comparison must respect definition order across digit boundaries
    width = len(str(len(exprs)))
    mapper = {e.lhs: e.lhs._rebuild(name=sregistry.make_name(
        prefix=f'_order{n:0{width}}_'))
        for n, e in enumerate(exprs) if isinstance(e.lhs, Temp)}

    processed = []
    for c in clusters:
        exprs = [_unevaluate(e, mapper) for e in c.exprs]
        processed.append(c.rebuild(exprs=exprs))
    return processed
