from collections import defaultdict, OrderedDict

from devito import Constant, TimeFunction
from devito.types.dimension import SpaceDimension
from devito.symbolics import split_affine
from devito.ops.types import OpsAccess, OpsAccessible
from devito.ops.utils import AccessibleInfo


class OPSNodeFactory(object):
    """
    Generate OPS nodes for building an OPS expression.
    A new OPS argument is created based on the indexed name, and its time dimension.
    Such a pair identifies an unique argument within the OPS kernel. The function
    returns the stored argument associated with this pair, if it was already created.
    """

    def __init__(self):
        self.ops_args = OrderedDict()
        self.ops_args_accesses = defaultdict(list)
        self.ops_params = []

    def new_ops_arg(self, indexed, is_write):
        """
        Create an Indexed node using OPS representation.

        Parameters
        ----------
        indexed : Indexed
            Indexed object using devito representation.

        Returns
        -------
        Indexed
            Indexed node using OPS representation.
        """

        # Build the OPS arg identifier
        time_index = split_affine(indexed.indices[TimeFunction._time_position])
        ops_arg_id = ('%s%s' % (indexed.name, time_index.var)
                      if indexed.function.is_TimeFunction else indexed.name)

        if ops_arg_id not in self.ops_args:
            symbol_to_access = OpsAccessible(
                ops_arg_id,
                dtype=indexed.dtype,
                read_only=not is_write
            )

            accessible_info = AccessibleInfo(
                symbol_to_access,
                time_index.var if indexed.function.is_TimeFunction else None,
                indexed.function.name)

            self.ops_args[ops_arg_id] = accessible_info
            self.ops_params.append(symbol_to_access)
        else:
            symbol_to_access = self.ops_args[ops_arg_id].accessible

        # Get the space indices
        space_indices = [
            split_affine(i).shift for i in indexed.indices
            if isinstance(split_affine(i).var, SpaceDimension)
        ]

        if space_indices not in self.ops_args_accesses[symbol_to_access]:
            self.ops_args_accesses[symbol_to_access].append(space_indices)

        return OpsAccess(symbol_to_access, space_indices)

    def new_ops_gbl(self, c):
        if c in self.ops_args:
            return self.ops_args[c].accessible

        new_c = AccessibleInfo(Constant(name='*%s' % c.name, dtype=c.dtype),
                               None,
                               None)
        self.ops_args[c] = new_c
        self.ops_params.append(new_c.accessible)

        return new_c.accessible
