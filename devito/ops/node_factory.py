from devito import Dimension, TimeFunction
from devito.ops.utils import namespace
from devito.symbolics import Macro, split_affine
from devito.types import Indexed, Array


class OPSNodeFactory():
    """
    Generate OPS nodes for building an OPS expression.
    A new OPS argument is created based on the indexed name and its time dimension,
    this pair identify a unique argument in the OPS kernel, so  it will return the
    stored argument associated with this pair if it was alredy created.
    """

    def __init__(self):
        self.ops_args = {}

    def new_ops_arg(self, indexed):
        """
        Create an :class:`Indexed` node using OPS representation.

        Parameters
        ----------
        indexed : :class:`Indexed`
            Indexed object using devito representation.

        Returns
        -------
        :class:`Indexed`
            Indexed node using OPS representation.
        """

        # Build the OPS arg identifier
        time_index = split_affine(indexed.indices[TimeFunction._time_position])
        ops_arg_id = '%s%s%s' % (indexed.name, time_index.var, time_index.shift)

        if ops_arg_id not in self.ops_args:
            # Create the indexed object
            ops_arg = Array(name=ops_arg_id,
                            dimensions=[Dimension(name=namespace['ops_acc'])],
                            dtype=indexed.dtype)

            self.ops_args[ops_arg_id] = ops_arg
        else:
            ops_arg = self.ops_args[ops_arg_id]

        # Get the space indices
        space_indices = [e for i, e in enumerate(
            indexed.indices) if i != TimeFunction._time_position]

        # Define the Macro used in OPS arg index
        access_macro = Macro('OPS_ACC%d(%s)' % (len(self.ops_args) - 1,
                                                ','.join(str(split_affine(i).shift)
                                                         for i in space_indices)))

        # Create Indexed object representing the OPS arg access
        new_indexed = Indexed(ops_arg.indexed, access_macro)

        return new_indexed
