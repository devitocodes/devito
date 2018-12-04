from devito import Dimension
from devito.symbolics import Macro
from devito.types import Indexed, Array

from devito.ops.utils import namespace


class OPSNodeFactory():
    """
    Generates OPS nodes for building an OPS expression.

    Examples
    --------
    >>> a = OPSNodeFactory()
    >>> a.new_symbol('symbol_name')
    symbol_name
    """

    def __init__(self):
        self.ops_grids = {}

    def new_grid(self, indexed):
        """
        Creates an :class:`Indexed` node using OPS representation.
        If the pair grid name and time dimension were alredy created, it will return
        the stored value associated with this pair.

        Notes
        -----
        Its expected the first item in the list `indexed.indices` to be the time
        dimension, and then the spaces dimensions.

        Parameters
        ----------
        indexed : :class:`Indexed`
            Indexed object using devito representation.

        Returns
        -------
        :class:`Indexed`
            Indexed node using OPS representation.
        """

        def getDimensionsDisplacement(dimension):
            if dimension.is_Symbol:
                if dimension.is_Dimension:
                    return 0
                return dimension
            elif dimension.is_Integer:
                return dimension
            else:
                lhs, rhs = dimension.as_two_terms()
                return getDimensionsDisplacement(lhs) + getDimensionsDisplacement(rhs)

        # Builds the grid identifier.
        grid_id = '%s%s' % (indexed.name, indexed.indices[0])

        if grid_id not in self.ops_grids:
            # Creates the indexed object.
            grid = Array(name=grid_id,
                         dimensions=[Dimension(name=namespace['ops_acc'])],
                         dtype=indexed.dtype)

            self.ops_grids[grid_id] = grid
        else:
            grid = self.ops_grids[grid_id]

        # Defines the Macro used in this grid indice.
        access_macro = Macro('OPS_ACC%d(%s)' %
                             (len(self.ops_grids) - 1,
                              ','.join(str(getDimensionsDisplacement(i))
                                       for i in indexed.indices[1:])))

        # Creates Indexed object representing the grid access.
        new_indexed = Indexed(grid.indexed, access_macro)

        return new_indexed
