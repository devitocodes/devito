from sympy import Eq, Add, Mul, Pow, Integer, Float, Rational
from devito.types import Symbol

from devito.ops.utils import namespace


class OPSNodeFactory():
    """
        Class responsible to generate ops expression for building the OPS ast.
    """

    def __init__(self):
        self.ops_access = {}

    def new_symbol(self, name):
        """
            Creates a new sympy Symbol object with the given name.

            :param name: Name of the symbol to be created.
        """
        return Symbol(name=name)

    def new_int_node(self, number):
        """
            Creates a new sympy integer object.

            :param number: integer number.
        """
        # Should I test for integer?
        return Integer(number)

    def new_float_node(self, number):
        """
            Creates a new sympy float object.

            :param number: float number.
        """
        # Should I test for float?
        return Float(number)

    def new_rational_node(self, num, den):
        """
            Creates a new sympy rational object.

            :param num: Rational numerator.
            :param den: Rational denominator.
        """
        return Rational(num, den)

    def new_add_node(self, lhs, rhs):
        """
            Creates a new sympy Add object.

            :param lhs: Left hand side of the sum.
            :param rhs: Right hand side of the sum.
        """
        return Add(lhs, rhs)

    def new_mul_node(self, lhs, rhs):
        """
            Creates a new sympy Mul object.

            :param lhs: Left hand side of the multiplication.
            :param rhs: Right hand side of the multiplication.
        """
        return Mul(lhs, rhs)

    def new_divide_node(self, num, den):
        """
            Creates a new sympy Div object.

            :param num: The division numerator.
            :param den: The division denominator.
        """
        return Mul(num, Pow(den, Integer(-1)))

    def new_grid(self, name, dimensions):
        """
            Creates a new grid access given a  variable name and its dimensions.
            If the pair grid name and time dimension was alredy created, it will return
            the stored value associated with this pair.

            :param name: grid name.
            :param dimensions: time and space dimensions to access the grid. Its expected
                               the first parameter be the time dimension.
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

        disp = getDimensionsDisplacement(dimensions[1:])

        grid_id = '%s%s' % (name, dimensions[0])

        # If the grid was alredy created, then use the same Access,
        # but we should look for different displacements.
        if grid_id in self.ops_access:
            ops_acc = self.ops_access[grid_id]
            symbol = Symbol(name='%s[%s(%s)]' %
                            (grid_id, ops_acc, ','.join(disp)))
        else:
            symbol = Symbol(name='%s[%s%s(%s)]' %
                            (grid_id, namespace['ops_acc'],
                             str(len(self.ops_access)), ','.join(disp)))
            self.ops_access[grid_id] = '%s%s' % (namespace['ops_acc'],
                                                 str(len(self.ops_access)))

        return (symbol, grid_id)

    def new_equation_node(self, *args):
        """
            Creates a new sympy equation with the provided arguments.

            :param *args: arguments to construct the new equation.
        """
        return Eq(*args)
