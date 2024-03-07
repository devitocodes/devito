import cgen as c
from sympy import Not

from devito.finite_differences import Abs
from devito.finite_differences.differentiable import Pow
from devito.ir.iet import (Call, Conditional, EntryFunction, Iteration, List,
                           Return, FindNodes, FindSymbols, Transformer,
                           make_callable)
from devito.passes.iet.engine import iet_pass
from devito.symbolics import CondEq, DefFunction
from devito.tools import dtype_to_cstr
from devito.types import Eq, Inc, Symbol

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

        name = sregistry.make_name(prefix='energy')
        energy = Symbol(name=name, dtype=f.dtype)

        eqns = [Eq(energy, 0.0),
                Inc(energy, Abs(Pow(f.subs(f.time_dim, 0), 2)))]
        irs, byproduct = rcompile(eqns)
        body = irs.iet.body.body + (Return(energy),)

        name = sregistry.make_name(prefix='compute_energy')
        retval = dtype_to_cstr(energy.dtype)
        efunc = make_callable(name, body, retval=retval)

        efuncs.extend([i.root for i in byproduct.funcs])
        efuncs.append(efunc)

        includes.extend(byproduct.includes)

        errctl = Conditional(CondEq(n.dim % 100, 0), List(body=[
            Call(efunc.name, efunc.parameters, retobj=energy),
            Conditional(Not(DefFunction('isfinite', energy)),
                        Return(error_mapper['Stability']))
        ]))
        errctl = List(header=c.Comment("Stability check"), body=[errctl])
        mapper[n] = n._rebuild(nodes=n.nodes + (errctl,))

    iet = Transformer(mapper).visit(iet)

    return iet, {'efuncs': efuncs, 'includes': includes}


error_mapper = {
    'Stability': 100,
    'KernelLaunch': 200,
}
