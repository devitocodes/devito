import sympy
import pytest

import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, SubDomain, Dimension,
                    ConditionalDimension)
from devito.ir.iet.visitors import Specializer

# Test that specializer replaces symbols as expected

# Create a couple of arbitrary operators
# Reference bounds, subdomains, spacings, constants, conditionaldimensions with symbolic
# factor
# Create a couple of different substitution sets

# Check that all the instances in the kernel are replaced
# Check that all the instances in the parameters are removed

# Check that sanity check catches attempts to specialize non-scalar types
# Check that trying to specialize symbols not in the Operator parameters results
# in an error being thrown

# Check that sizes and strides get specialized when using `linearize=True`


class TestSpecializer:
    """Tests for the Specializer transformer"""

    @pytest.mark.parametrize('pre_gen', [True, False])
    @pytest.mark.parametrize('expand', [True, False])
    def test_bounds(self, pre_gen, expand):
        """Test specialization of dimension bounds"""
        grid = Grid(shape=(11, 11))

        ((x_m, x_M), (y_m, y_M)) = [d.symbolic_extrema for d in grid.dimensions]
        time_m = grid.time_dim.symbolic_min
        minima = (x_m, y_m, time_m)
        maxima = (x_M, y_M)

        def check_op(mapper, operator):
            for k, v in mapper.items():
                assert k not in operator.parameters
                assert k.name not in str(operator.ccode)
                # Check that the loop bounds are modified correctly
                if k in minima:
                    assert f"{k.name.split('_')[0]} = {v}" in str(operator.ccode)
                elif k in maxima:
                    assert f"{k.name.split('_')[0]} <= {v}" in str(operator.ccode)

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = TimeFunction(name='h', grid=grid)

        eq0 = Eq(f, f + 1)
        eq1 = Eq(g, f.dx)
        eq2 = Eq(h.forward, (g + x_m).dy)
        eq3 = Eq(f, x_M)

        # Check behaviour with expansion since we have a replaced symbol inside a
        # derivative
        if expand:
            kwargs = {'opt': ('advanced', {'expand': True})}
        else:
            kwargs = {'opt': ('advanced', {'expand': False})}

        op = Operator([eq0, eq1, eq2, eq3], **kwargs)

        if pre_gen:
            # Generate C code for the unspecialized Operator - the result should be
            # the same regardless, but it ensures that the old generated code is
            # purged and replaced in the specialized Operator
            _ = op.ccode

        mapper0 = {x_m: sympy.S.Zero}
        mapper1 = {x_M: sympy.Integer(20), y_m: sympy.S.Zero}
        mapper2 = {**mapper0, **mapper1}
        mapper3 = {y_M: sympy.Integer(10), time_m: sympy.Integer(5)}

        mappers = (mapper0, mapper1, mapper2, mapper3)
        ops = tuple(Specializer(m).visit(op) for m in mappers)

        for m, o in zip(mappers, ops):
            check_op(m, o)

    def test_subdomain(self):
        """Test that SubDomain thicknesses can be specialized"""

        def check_op(mapper, operator):
            for k in mapper.keys():
                assert k not in operator.parameters
                assert k.name not in str(operator.ccode)

        class SD(SubDomain):
            name = 'sd'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 1), y: ('right', 2)}

        grid = Grid(shape=(11, 11))
        sd = SD(grid=grid)

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=sd)

        eqs = [Eq(f, f+1, subdomain=sd),
               Eq(g, g+1, subdomain=sd)]

        op = Operator(eqs)

        subdims = [d for d in op.dimensions if d.is_Sub]
        ((xltkn, xrtkn), (_, yrtkn)) = [d.thickness for d in subdims]

        mapper0 = {xltkn: sympy.S.Zero}
        mapper1 = {xrtkn: sympy.Integer(2), yrtkn: sympy.S.Zero}
        mapper2 = {**mapper0, **mapper1}

        mappers = (mapper0, mapper1, mapper2)
        ops = tuple(Specializer(m).visit(op) for m in mappers)

        for m, o in zip(mappers, ops):
            check_op(m, o)

    def test_factor(self):
        """Test that ConditionalDimensions can have their symbolic factors specialized"""
        size = 16
        factor = 4
        i = Dimension(name='i')
        ci = ConditionalDimension(name='ci', parent=i, factor=factor)

        g = Function(name='g', shape=(size,), dimensions=(i,))
        f = Function(name='f', shape=(int(size/factor),), dimensions=(ci,))

        op0 = Operator([Eq(f, g)])

        mapper = {ci.symbolic_factor: sympy.Integer(factor)}

        op1 = Specializer(mapper).visit(op0)

        assert ci.symbolic_factor not in op1.parameters
        assert ci.symbolic_factor.name not in str(op1.ccode)
        assert "if ((i)%(4) == 0)" in str(op1.ccode)

    def test_spacing(self):
        """Test that grid spacings can be specialized"""
        grid = Grid(shape=(11,))
        f = Function(name='f', grid=grid)

        op0 = Operator(Eq(f, f.dx))

        mapper = {grid.dimensions[0].spacing: sympy.Float(grid.spacing[0])}
        op1 = Specializer(mapper).visit(op0)

        assert grid.dimensions[0].spacing not in op1.parameters
        assert grid.dimensions[0].spacing.name not in str(op1.ccode)
        assert "/1.0e-1F;" in str(op1.ccode)

    def test_sizes(self):
        """Test that strides generated for linearization can be specialized"""
        grid = Grid(shape=(11, 11))

        f = TimeFunction(name='f', grid=grid, space_order=2)

        op0 = Operator(Eq(f.forward, f.dx2),
                       opt=('advanced', {'expand': True, 'linearize': True}))

        mapper = {f.symbolic_shape[1]: sympy.Integer(11),
                  f.symbolic_shape[2]: sympy.Integer(11)}

        op1 = Specializer(mapper).visit(op0)

        assert "const int x_fsz0 = 11;" in str(op1.ccode)
        assert "const int y_fsz0 = 11;" in str(op1.ccode)

    # TODO: Should strides get specialized? If so, how?

    def test_apply_basic(self):
        """
        Test that a trivial operator runs and returns the same results when
        specialized.
        """
        grid = Grid(shape=(11, 11))
        ((x_m, x_M), (y_m, y_M)) = [d.symbolic_extrema for d in grid.dimensions]
        f = Function(name='f', grid=grid, dtype=np.int32)

        op0 = Operator(Eq(f, f+1))

        mapper = {x_m: sympy.Integer(2), x_M: sympy.Integer(7),
                  y_m: sympy.Integer(3), y_M: sympy.Integer(8)}

        op1 = Specializer(mapper).visit(op0)

        assert op1.cfunction is not op0.cfunction

        op1.apply()

        check = np.array(f.data[:])
        f.data[:] = 0

        op0.apply(x_m=2, x_M=7, y_m=3, y_M=8)

        assert np.all(check == f.data)


# class TestApply:
#     """Tests for specialization of operators at apply time"""

#     def test_basic(self):
#         grid = Grid(shape=(11, 11))

#         f = TimeFunction(name='f', grid=grid, space_order=2)

#         op = Operator(Eq(f.forward, f + 1))

#         op.apply(time_M=10, specialize=('x_m', 'x_M'))
