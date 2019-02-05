from devito.ir.iet.nodes import ArrayCast, Call, Callable, Expression, List
from devito.ir.iet.scheduler import iet_insert_C_decls
from devito.ir.iet.utils import derive_parameters
from devito.ir.iet.visitors import FindSymbols, FindNodes
from devito.tools import as_tuple

__all__ = ['make_efunc']


class ElementalFunction(Callable):

    """
    A Callable performing a computation over an abstract convex iteration space.

    A Call to an ElementalFunction will "instantiate" such iteration space by
    supplying bounds and step increment for each Dimension listed in
    ``dynamic_parameters``.
    """

    def __init__(self, name, body, retval, parameters=None, prefix=('static', 'inline'),
                 dynamic_parameters=None):
        super(ElementalFunction, self).__init__(name, body, retval, parameters, prefix)

        self._mapper = {}
        for i in as_tuple(dynamic_parameters):
            if i.is_Dimension:
                self._mapper[i] = (parameters.index(i.symbolic_min),
                                   parameters.index(i.symbolic_max))
            else:
                self._mapper[i] = (parameters.index(i),)

    def make_call(self, dynamic_parameters_mapper=None):
        dynamic_parameters_mapper = dynamic_parameters_mapper or {}
        arguments = list(self.parameters)
        for k, v in dynamic_parameters_mapper.items():
            # Sanity check
            if k not in self._mapper:
                raise ValueError("`k` is not a dynamic parameter" % k)
            if len(self._mapper[k]) != len(v):
                raise ValueError("Expected %d values for dynamic parameter `%s`, given %d"
                                 % (len(self._mapper[k]), k, len(v)))
            # Create the argument list
            for i, j in zip(self._mapper[k], v):
                arguments[i] = j
        return Call(self.name, tuple(arguments))


def make_efunc(name, iet, dynamic_parameters=None, retval='void', prefix='static'):
    """
    Create an ElementalFunction from (a sequence of) perfectly nested Iterations.
    """
    # Arrays are by definition (vector) temporaries, so if they are written
    # within `iet`, they can also be declared and allocated within the `efunc`
    items = FindSymbols().visit(iet)
    local = [i.write for i in FindNodes(Expression).visit(iet) if i.write.is_Array]
    external = [i for i in items if i.is_Tensor and i not in local]

    # Insert array casts
    casts = [ArrayCast(i) for i in external]
    iet = List(body=casts + [iet])

    # Insert declarations
    iet = iet_insert_C_decls(iet, external)

    # The Callable parameters
    params = [i for i in derive_parameters(iet) if i not in local]

    return ElementalFunction(name, iet, retval, params, prefix, dynamic_parameters)
