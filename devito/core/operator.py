from collections import OrderedDict

from devito.core.autotuning import autotune
from devito.ir.iet import Call, List, HaloSpot, MetaCall, FindNodes, Transformer
from devito.ir.support import align_accesses
from devito.parameters import configuration
from devito.mpi import make_halo_exchange_routines
from devito.operator import Operator, is_threaded
from devito.tools import flatten

__all__ = ['OperatorCore']


class OperatorCore(Operator):

    def _specialize_exprs(self, expressions):
        # Align data accesses to the computational domain
        key = lambda i: i.is_DiscreteFunction
        expressions = [align_accesses(e, key=key) for e in expressions]
        return super(OperatorCore, self)._specialize_exprs(expressions)

    def _generate_mpi(self, iet, **kwargs):
        # Drop superfluous HaloSpots
        halo_spots = FindNodes(HaloSpot).visit(iet)
        mapper = {i: None for i in halo_spots if i.is_Redundant}
        iet = Transformer(mapper, nested=True).visit(iet)

        # Nothing else to do if no MPI
        if configuration['mpi'] is False:
            return iet

        halo_spots = FindNodes(HaloSpot).visit(iet)

        callables = OrderedDict()
        mapper = {}
        for hs in halo_spots:
            for f, v in hs.fmapper.items():
                # For each MPI-distributed DiscreteFunction, generate all necessary
                # C-level routines to perform a halo update
                threaded = is_threaded(kwargs.get("dle"))
                routines, extra = make_halo_exchange_routines(f, v.loc_indices, threaded)
                callables[f] = routines

                # Replace HaloSpots with suitable calls performing the halo update
                stencil = [int(i) for i in hs.mask[f].values()]
                comm = f.grid.distributor._obj_comm
                nb = f.grid.distributor._obj_neighborhood
                loc_indices = list(v.loc_indices.values())
                arguments = [f] + stencil + [comm, nb] + loc_indices + extra
                call = Call('halo_exchange_%s' % f.name, arguments)
                mapper.setdefault(hs, []).append(call)

        self._includes.append('mpi.h')

        self._func_table.update(OrderedDict([(i.name, MetaCall(i, True))
                                             for i in flatten(callables.values())]))

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
