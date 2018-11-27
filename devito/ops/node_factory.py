from sympy import Eq, Add, Mul, Pow, Integer, Float, Rational
from devito import Dimension
from devito.symbolics import Macro
from devito.types import Symbol, Indexed, Array

from devito.ops.utils import namespace


class OpsNodeFactory():
    """
    Generates ops nodes for building an OPS expression.

    Examples
    --------
    >>> a = OpsNodeFactory()
    >>> b = a.new_symbol('symbol_name')
    """

    def __init__(self):
        self.ops_grids = {}

    def new_symbol(self, name):
        """
        Creates a new sympy :class:`Symbol` object with the given name.
        Its used to define/use variables.

        Parameters
        ----------
        name : str
            Name of the symbol to be created.

        Returns
        -------
        :class:`Symbol` 
            Object :class:`Symbol` with the given name.
        """
        return Symbol(name=name)

    def new_int_node(self, number):
        """
        Creates a new sympy :class:`Integer` object.

        Parameters
        ----------
        number : int
            Number to initialize an Integer node.

        Returns
        -------
        :class:`Integer`
            Object of :class:`Integer`.
        """
        return Integer(number)

    def new_float_node(self, number):
        """
        Creates a new sympy :class:`float` object.

        Parameters
        ----------
        number : float
            Number to initialize an Float node.

        Returns
        -------
        :class:`Float`
            Object of :class:`Float`.
        """
        return Float(number)

    def new_rational_node(self, num, den):
        """
        Creates a new sympy :class:`Rational` object to represent 
        a mathematical rational.

        Parameters
        ----------
        num : float
            Rational numerator.
        den : float
            Rational denominator.

        Returns
        -------
        :class:`Rational`
            Object :class:`Rational`. 
        """
        return Rational(num, den)

    def new_add_node(self, lhs, rhs):
        """
        Represents a new mathematical sum node operation.

        Parameters
        ----------
        lhs 
            The left hand side of the sum operation.
        rhs : [type]
            The right hand side of the sum operation.

        Returns
        -------
        :class:`Add`
            Object of :class:`Add`.
        """
        return Add(lhs, rhs)

    def new_mul_node(self, lhs, rhs):
        """
        Creates a new mathematical multiplication node operation.

        Parameters
        ----------
        lhs
            The left hand side of the multiplication.
        rhs : [type]
            The right hand side of the multiplication.

        Returns
        -------
        :class:`Mul`
            Object of :class:`Mul`.
        """
        return Mul(lhs, rhs)

    def new_divide_node(self, num, den):
        """
        Creates a new division node operation.

        Notes
        -----
        The division is represented by the multiplicating the numerator with 
        denominator's inverse.

        Parameters
        ----------
        num
            The division numerator.
        den : [type]
            The division denominator.

        Returns
        -------
        :class:`Mul`
            Object of :class:`Mul` representing the division.
        """
        return Mul(num, Pow(den, Integer(-1)))

    def new_grid(self, devito_indexed, dimensions):
        """
        Creates an :class:`Indexed` object using OPS representation.
        If the pair grid name and time dimension was alredy created, it will return
        the stored value associated with this pair.

        Parameters
        ----------
        devito_indexed : :class:`Indexed`
            Indexed object using devito representation.

        dimensions : list
            List of the dimensions alredy translated to OPS.

        Returns
        -------
        :class:`Indexed`
            Indexed object using OPS representation.
        """

        def getDimensionsDisplacement(dimensions):
            disp = []

            for dim in dimensions:
                if dim.is_Add:
                    lhs, = dim.as_two_terms()
                    disp.append(str(lhs))
                else:
                    disp.append('0')

            return disp

        # Gets the dimension displacement.
        disp = getDimensionsDisplacement(dimensions[1:])

        # Builds the grid identifier.
        grid_id = '%s%s' % (devito_indexed.name, dimensions[0])

        if grid_id not in self.ops_grids:
            # Creates the indexed object.
            # Always will have one dimension that will be generated using the macro
            # OPS_ACC but it will not represent the real dimension used, because we
            # can have a 3 dimension problem and only one Dimension object.
            grid = Array(name=grid_id,
                         dimensions=[Dimension(name=namespace['ops_acc'])],
                         dtype=devito_indexed.dtype)
            
            self.ops_grids[grid_id] = grid
        else:
            grid = self.ops_grids[grid_id]

        # Defines the Macro used in this grid indice.
        access_macro = Macro('OPS_ACC%d(%s)' % (len(self.ops_grids), ','.join(disp)))

        # Creates Indexed object representing the grid access.
        indexed = Indexed(grid.indexed, access_macro)

        return indexed

    def new_equation_node(self, *args):
        """
            Creates a new sympy equation with the provided arguments.

            :param *args: arguments to construct the new equation.
        """
        return Eq(*args)
