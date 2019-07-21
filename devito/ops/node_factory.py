from collections import OrderedDict

from devito import TimeFunction
from devito.types.dimension import SpaceDimension
from devito.symbolics import split_affine
from devito.ops.types import OpsAccessible


class OPSNodeFactory(object):
    """
    Generate OPS nodes for building an OPS expression.
    A new OPS argument is created based on the indexed name, and its time dimension.
    Such a pair identifies an unique argument within the OPS kernel. The function
    returns the stored argument associated with this pair, if it was already created.
    """

    def __init__(self):
        self.ops_args = OrderedDict()

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
        ops_arg_id = ('%s%s%s' % (indexed.name, time_index.var, time_index.shift)
                      if indexed.function.is_TimeFunction else indexed.name)

        if ops_arg_id not in self.ops_args:
            symbol_to_access = OpsAccessible(
                ops_arg_id,
                indexed.dtype,
                not is_write
            )
            self.ops_args[ops_arg_id] = symbol_to_access
        else:
            symbol_to_access = self.ops_args[ops_arg_id]

        # Get the space indices
        space_indices = [i for i in indexed.indices if isinstance(
            split_affine(i).var, SpaceDimension)]

        access_indices = [split_affine(i).shift for i in space_indices]

        return symbol_to_access(access_indices)
