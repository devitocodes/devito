from cached_property import cached_property

from devito.ir.iet.nodes import Call, Callable, Expression
from devito.ir.iet.utils import derive_parameters
from devito.ir.iet.visitors import FindNodes
from devito.tools import as_tuple

__all__ = ['ElementalFunction', 'ElementalCall', 'make_efunc']


class ElementalFunction(Callable):

    """
    A Callable performing a computation over an abstract convex iteration space.

    A Call to an ElementalFunction will "instantiate" such iteration space by
    supplying bounds and step increment for each Dimension listed in
    ``dynamic_parameters``.
    """

    is_ElementalFunction = True

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
            tv = as_tuple(v)

            # Sanity check
            if k not in self._mapper:
                raise ValueError("`k` is not a dynamic parameter" % k)
            if len(self._mapper[k]) != len(tv):
                raise ValueError("Expected %d values for dynamic parameter `%s`, given %d"
                                 % (len(self._mapper[k]), k, len(tv)))
            # Create the argument list
            for i, j in zip(self._mapper[k], tv):
                arguments[i] = j if incr is False else (arguments[i] + j)

        super(ElementalCall, self).__init__(name, arguments)

    def _rebuild(self, *args, dynamic_args_mapper=None, incr=False, **kwargs):
        # This guarantees that `ec._rebuild(arguments=ec.arguments) == ec`
        return super(ElementalCall, self)._rebuild(
            *args, dynamic_args_mapper=dynamic_args_mapper, incr=incr, **kwargs
        )

    @cached_property
    def dynamic_defaults(self):
        return {k: tuple(self.arguments[i] for i in v) for k, v in self._mapper.items()}


def make_efunc(name, iet, dynamic_parameters=None, retval='void', prefix='static'):
    """
    Create an ElementalFunction from (a sequence of) perfectly nested Iterations.
    """
    # Arrays are by definition (vector) temporaries, so if they are written-only
    # within `iet`, they can also be declared and allocated within the `efunc`
    exprs = FindNodes(Expression).visit(iet)
    write_arrays = {i.write for i in exprs if i.write.is_Array}
    rw_arrays = write_arrays.intersection(set().union(*[i.reads for i in exprs]))

    # The Callable parameters
    parameters = [i for i in derive_parameters(iet) if i not in rw_arrays]

    return ElementalFunction(name, iet, retval, parameters, prefix, dynamic_parameters)
