from collections import OrderedDict

from devito.core.autotuning import autotune
from devito.cgen_utils import printmark
from devito.ir.iet import (Call, List, HaloSpot, MetaCall, FindNodes, Transformer,
                           filter_iterations, retrieve_iteration_tree)
from devito.ir.support import align_accesses
from devito.parameters import configuration
from devito.mpi import copy, sendrecv, update_halo
from devito.operator import OperatorRunnable
from devito.tools import flatten

__all__ = ['Operator']


class OperatorCore(OperatorRunnable):

    def _specialize_exprs(self, expressions):
        # Align data accesses to the computational domain
        key = lambda i: i.is_TensorFunction
        expressions = [align_accesses(e, key=key) for e in expressions]
        return super(OperatorCore, self)._specialize_exprs(expressions)

    def _generate_mpi(self, iet, **kwargs):
        if configuration['mpi'] is False:
            return iet

        halo_spots = FindNodes(HaloSpot).visit(iet)

        # For each MPI-distributed TensorFunction, generate all necessary
        # C-level routines to perform a halo update
        callables = OrderedDict()
        for hs in halo_spots:
            for f, v in hs.fmapper.items():
                callables[f] = [update_halo(f, v.loc_indices)]
                callables[f].append(sendrecv(f, v.loc_indices))
                callables[f].append(copy(f, v.loc_indices))
                callables[f].append(copy(f, v.loc_indices, True))
        callables = flatten(callables.values())

        # Replace HaloSpots with suitable calls performing the halo update
        mapper = {}
        for hs in halo_spots:
            for f, v in hs.fmapper.items():
                stencil = [int(i) for i in hs.mask[f].values()]
                comm = f.grid.distributor._C_comm
                nb = f.grid.distributor._C_neighbours.obj
                loc_indices = list(v.loc_indices.values())
                dsizes = [d.symbolic_size for d in f.dimensions]
                parameters = [f] + stencil + [comm, nb] + loc_indices + dsizes
                call = Call('halo_exchange_%s' % f.name, parameters)
                mapper.setdefault(hs, []).append(call)

        # Sorting is for deterministic code generation. However, in practice,
        # we don't expect `cstructs` to contain more than one element because
        # there should always be one grid per Operator (though we're not really
        # enforcing it)
        cstructs = {f.grid.distributor._C_neighbours.cdef
                    for f in flatten(i.fmapper for i in halo_spots)}
        self._globals.extend(sorted(cstructs, key=lambda i: i.tpname))

        self._includes.append('mpi.h')

        self._func_table.update(OrderedDict([(i.name, MetaCall(i, True))
                                             for i in callables]))

        # Add in the halo update calls
        mapper = {k: List(body=v + list(k.body)) for k, v in mapper.items()}
        iet = Transformer(mapper, nested=True).visit(iet)

        return iet

    def _autotune(self, args, setup):
        if setup is False:
            return args
        elif setup is True:
            level = configuration['autotuning'].level or 'basic'
            args, summary = autotune(self, args, level, configuration['autotuning'].mode)
        elif isinstance(setup, str):
            args, summary = autotune(self, args, setup, configuration['autotuning'].mode)
        elif isinstance(setup, tuple) and len(setup) == 2:
            level, mode = setup
            if level is False:
                return args
            else:
                args, summary = autotune(self, args, level, mode)
        else:
            raise ValueError("Expected bool, str, or 2-tuple, got `%s` instead"
                             % type(setup))

        # Record the tuned values
        self._state.setdefault('autotuning', []).append(summary)

        return args


class OperatorDebug(OperatorCore):
    """
    Decorate the generated code with useful print statements.
    """

    def __init__(self, expressions, **kwargs):
        super(OperatorDebug, self).__init__(expressions, **kwargs)
        self._includes.append('stdio.h')

        # Minimize the trip count of the sequential loops
        iterations = set(flatten(retrieve_iteration_tree(self.body)))
        mapper = {i: i._rebuild(limits=(max(i.offsets) + 2))
                  for i in iterations if i.is_Sequential}
        self.body = Transformer(mapper).visit(self.body)

        # Mark entry/exit points of each non-sequential Iteration tree in the body
        iterations = [filter_iterations(i, lambda i: not i.is_Sequential, 'any')
                      for i in retrieve_iteration_tree(self.body)]
        iterations = [i[0] for i in iterations if i]
        mapper = {t: List(header=printmark('In nest %d' % i), body=t)
                  for i, t in enumerate(iterations)}
        self.body = Transformer(mapper).visit(self.body)


class Operator(object):

    def __new__(cls, *args, **kwargs):
        cls = OperatorDebug if kwargs.pop('debug', False) else OperatorCore
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        return obj
