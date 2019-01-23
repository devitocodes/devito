from collections import OrderedDict

from devito.core.autotuning import autotune
from devito.ir.iet import List, HaloSpot, MetaCall, FindNodes, Transformer
from devito.ir.support import align_accesses
from devito.parameters import configuration
from devito.mpi import HaloExchangeBuilder
from devito.operator import Operator, is_threaded

__all__ = ['OperatorCore']


class OperatorCore(Operator):

    def _specialize_exprs(self, expressions):
        # Align data accesses to the computational domain
        key = lambda i: i.is_DiscreteFunction
        expressions = [align_accesses(e, key=key) for e in expressions]
        return super(OperatorCore, self)._specialize_exprs(expressions)

    def _generate_mpi(self, iet, **kwargs):
        # Nothing to do if no MPI
        if configuration['mpi'] is False:
            return iet

        # Build halo exchange Callables and Calls
        halo_spots = FindNodes(HaloSpot).visit(iet)
        heb = HaloExchangeBuilder(is_threaded(kwargs.get("dle")), configuration['mpi'])
        callables, calls = heb.make(halo_spots)

        # Update the Operator internal state
        self._includes.append('mpi.h')
        self._func_table.update(OrderedDict([(i.name, MetaCall(i, True))
                                             for i in callables]))

        # Transform the IET by adding in the `haloupdate` Calls
        mapper = {k: List(body=v) for k, v in calls.items()}
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
