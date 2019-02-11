from anytree import NodeMixin, PreOrderIter, RenderTree, ContStyle
from cached_property import cached_property

from devito.ir.iet.nodes import ArrayCast, Call, Callable, Expression, List
from devito.ir.iet.scheduler import iet_insert_C_decls
from devito.ir.iet.utils import derive_parameters
from devito.ir.iet.visitors import FindSymbols, FindNodes
from devito.tools import as_tuple

__all__ = ['ElementalFunction', 'ElementalCall', 'EFuncNode', 'make_efunc']


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

    @cached_property
    def dynamic_defaults(self):
        return {k: tuple(self.parameters[i] for i in v) for k, v in self._mapper.items()}

    def make_call(self, dynamic_args_mapper=None, incr=False):
        return ElementalCall(self.name, list(self.parameters), dict(self._mapper),
                             dynamic_args_mapper, incr)


class ElementalCall(Call):

    def __init__(self, name, arguments=None, mapper=None, dynamic_args_mapper=None,
                 incr=False):
        self._mapper = mapper or {}

        arguments = list(as_tuple(arguments))
        dynamic_args_mapper = dynamic_args_mapper or {}
        for k, v in dynamic_args_mapper.items():
            # Sanity check
            if k not in self._mapper:
                raise ValueError("`k` is not a dynamic parameter" % k)
            if len(self._mapper[k]) != len(v):
                raise ValueError("Expected %d values for dynamic parameter `%s`, given %d"
                                 % (len(self._mapper[k]), k, len(v)))
            # Create the argument list
            for i, j in zip(self._mapper[k], v):
                arguments[i] = j if incr is False else (arguments[i] + j)

        super(ElementalCall, self).__init__(name, arguments)

    @cached_property
    def dynamic_defaults(self):
        return {k: tuple(self.arguments[i] for i in v) for k, v in self._mapper.items()}


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
    parameters = [i for i in derive_parameters(iet) if i not in local]

    return ElementalFunction(name, iet, retval, parameters, prefix, dynamic_parameters)


class EFuncNode(NodeMixin):

    """A simple utility class to keep track of ElementalFunctions and Call sites."""

    def __init__(self, iet, parent=None, name=None):
        self.iet = iet
        if name is not None:
            self.name = name
        else:
            assert iet.is_Callable
            self.name = iet.name
        self.parent = parent

        self._rendered_name = "<%s>" % self.name

    def __repr__(self):
        return RenderTree(self, style=ContStyle()).by_attr('_rendered_name')

    def visit(self):
        for i in PreOrderIter(self):
            yield i
