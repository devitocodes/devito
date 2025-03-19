import cgen as c
import numpy as np
from sympy import Expr, Not, S

from devito.ir.iet import (Call, Conditional, DummyExpr, EntryFunction, Iteration,
                           List, Break, Return, FindNodes, FindSymbols, Transformer,
                           make_callable)
from devito.passes.iet.engine import iet_pass
from devito.symbolics import CondEq, MathFunction
from devito.tools import dtype_to_ctype
from devito.types import Eq, Inc, LocalObject, Symbol

__all__ = ['check_stability', 'error_mapper']


def check_stability(graph, options=None, rcompile=None, sregistry=None, **kwargs):
    """
    Check if the simulation is stable. If not, return to Python as quickly as
    possible with an error code.
    """
    if options['errctl'] != 'max':
        return

    _, wmovs = graph.data_movs

    _check_stability(graph, wmovs=wmovs, rcompile=rcompile, sregistry=sregistry)


@iet_pass
def _check_stability(iet, wmovs=(), rcompile=None, sregistry=None):
    if not isinstance(iet, EntryFunction):
        return iet, {}

    # NOTE: Stability is a domain-specific concept, hence looking for time
    # Iterations and TimeFunctions is acceptable
    efuncs = []
    includes = []
    mapper = {}
    for n in FindNodes(Iteration).visit(iet):
        if not n.dim.is_Time:
            continue

        functions = [f for f in FindSymbols().visit(n)
                     if f.is_TimeFunction and f.time_dim.is_Stepping]

        # We compute the norm of just one TimeFunction, hence we sort for
        # determinism and reproducibility
        candidates = sorted(set(functions) & set(wmovs), key=lambda f: f.name)
        for f in candidates:
            if f in wmovs:
                break
        else:
            continue

        accumulator = Symbol(name='accumulator', dtype=f.dtype)
        eqns = [Eq(accumulator, 0.0),
                Inc(accumulator, f.subs(f.time_dim, 0))]
        irs, byproduct = rcompile(eqns)

        name = sregistry.make_name(prefix='is_finite')
        retval = Return(MathFunction('isfinite', accumulator))
        body = irs.iet.body.body + (retval,)
        efunc = make_callable(name, body, retval='int')

        efuncs.extend([i.root for i in byproduct.funcs])
        efuncs.append(efunc)

        includes.extend(byproduct.includes)

        name = sregistry.make_name(prefix='check')
        check = Symbol(name=name, dtype=np.int32)

        retval = Retval(name='retval')

        errctl = Conditional(CondEq(n.dim % 100, 0), List(body=[
            Call(efunc.name, efunc.parameters, retobj=check),
            Conditional(Not(check), List(body=[
                DummyExpr(retval, error_mapper['Stability']),
                Break()
            ]))
        ]))
        errctl = List(header=c.Comment("Stability check"), body=[errctl])
        mapper[n] = n._rebuild(nodes=n.nodes + (errctl,))

        # One check is enough
        break
    else:
        return iet, {}

    iet = Transformer(mapper).visit(iet)

    # We now must return a suitable error code
    body = iet.body._rebuild(
        body=(DummyExpr(retval, 0, init=True),) + iet.body.body,
        retstmt=Return(retval)
    )
    iet = iet._rebuild(body=body)

    return iet, {'efuncs': efuncs, 'includes': includes}


class Retval(LocalObject, Expr):

    dtype = dtype_to_ctype(np.int32)
    default_initvalue = S.Zero


error_mapper = {
    'Stability': 100,
    'KernelLaunch': 200,
    'KernelLaunchOutOfResources': 201,
    'KernelLaunchClusterConfig': 202,
    'KernelLaunchUnknown': 203,
}
