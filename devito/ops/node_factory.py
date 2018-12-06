from devito import Dimension, TimeFunction
from devito.ops.utils import namespace
from devito.symbolics import Macro, split_affine
from devito.types import Indexed, Array


class OPSNodeFactory():
    """
    Generate OPS nodes for building an OPS expression.

    Examples
    --------
    >>> a = OPSNodeFactory()
    >>> a.new_symbol('symbol_name')
    symbol_name
    """

    def __init__(self):
        self.ops_args = {}

    def new_ops_arg(self, indexed):
        """
        Create an :class:`Indexed` node using OPS representation.
        If the pair ops argument name and time dimension were alredy created,
        it will return the stored value associated with this pair.

        Parameters
        ----------
        indexed : :class:`Indexed`
            Indexed object using devito representation.

        Returns
        -------
        :class:`Indexed`
            Indexed node using OPS representation.
        """

        # Build the ops argument identifier.
        ops_arg_id = '%s%s' % (indexed.name, indexed.indices[TimeFunction._time_position])

        if ops_arg_id not in self.ops_args:
            # Create the indexed object.
            ops_arg = Array(name=ops_arg_id,
                            dimensions=[Dimension(name=namespace['ops_acc'])],
                            dtype=indexed.dtype)

            self.ops_args[ops_arg_id] = ops_arg
        else:
            ops_arg = self.ops_args[ops_arg_id]

        # Get the space indices.
        space_indices = [e for i, e in enumerate(
            indexed.indices) if i != TimeFunction._time_position]

        # Define the Macro used in this ops argument indice.
        access_macro = Macro('OPS_ACC%d(%s)' % (len(self.ops_args) - 1,
                                                ','.join(str(split_affine(i).shift)
                                                         for i in space_indices)))

        # Create Indexed object representing the ops argument access.
        new_indexed = Indexed(ops_arg.indexed, access_macro)

        return new_indexed
