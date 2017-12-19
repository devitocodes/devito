import abc

import numpy as np
from collections import OrderedDict
from functools import reduce
from itertools import chain

from devito.exceptions import InvalidArgument
from devito.tools import filter_ordered, flatten, GenericVisitor
from devito.function import CompositeFunction, SymbolicFunction
from devito.dimension import Dimension
from devito.ir.support.stencil import Stencil


""" This module contains a set of classes and functions to deal with runtime arguments
to Operators. It represents the arguments and their relationships as a DAG (N, E) where
every node (N) is represented by an object of :class:`Parameter` and every edge is an
object of class :class:`Dependency`.

The various class hierarchies are explained here:
Parameter: Any node of the dependency graph has to necessarily be of this type. The node
           may or may not represent an actual runtime argument passed to the kernel.
Argument: Subclass of Parameter. This represents a node in the dependency graph directly
          corresponding to a runtime argument passed to the kernel.

             Parameter
                 |
        -------------------
DimensionParameter        |
                          |
                       Argument
                          |
                          |
                --------------------
                |         |        |
        ScalarArgument    |        |
                    TensorArgument |
                                PtrArgument

ArgumentEngine: The main external API for this module. It encapsulates all the argument
                derivation and verification logic.

ArgumentVisitor: Visits objects of :class:`function.Basic` types to return appropriate
                 objects from the above parameter-argument hierarchy.

ValueVisitor: Used by the argument derivation method to derive the value of each parameter
              based on the dependency tree.

Dependency: Edges of the dependency graph are represented by objects of this class.

UnevaluatedDependency: A "future" object representing an argument derivation that is yet
                       to happen.
"""


class Parameter(object):
    """ Abstract base class for any object that represents a node in the dependency
        graph. It may or may not represent a runtime argument.

    :param name: Name of the parameter
    :param dependencies: A list of :class:`Dependency` objects that represent all the
                         incoming edges into this node of the graph.
    """
    is_DimensionParameter = False
    is_Argument = False
    is_ScalarArgument = False
    is_TensorArgument = False
    is_PtrArgument = False

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, dependencies):
        self.name = name
        self.dependencies = dependencies

    @property
    def gets_value_from(self):
        return [x for x in self.dependencies if x.is_Value]

    @property
    def verified_by(self):
        return [x for x in self.dependencies if x.is_Verify]

    def __repr__(self):
        return "%s, Depends: %s" % (self.name, str(self.dependencies))


class DimensionParameter(Parameter):
    """ Parameter object (node in the dependency graph) that represents a
        :class:`Dimension`. A dimension object plays an important role in value derivation
        and verification but does not represent a runtime argument itself (since it
        provides multiple ScalarArguments).
    """
    is_DimensionParameter = True

    def __init__(self, provider, dependencies):
        super(DimensionParameter, self).__init__(provider.name, dependencies)
        self.provider = provider


class Argument(Parameter):

    """ Base class for any object that represents a run time argument for
        generated kernels. It is necessarily a node in the dependency graph.
    """
    is_Argument = True

    def __init__(self, name, dependencies, dtype=np.int32):
        super(Argument, self).__init__(name, dependencies)
        self.dtype = dtype


class ScalarArgument(Argument):

    """ Class representing scalar arguments that a kernel might expect.
        Most commonly used to pass dimension sizes
    """
    is_ScalarArgument = True


class TensorArgument(Argument):

    """ Class representing tensor arguments that a kernel might expect.
        Most commonly used to pass numpy-like multi-dimensional arrays.
    """
    is_TensorArgument = True

    def __init__(self, provider, dependencies=None):
        if dependencies is None:
            dependencies = []
        super(TensorArgument, self).__init__(provider.name,
                                             dependencies + [ValueDependency(provider)],
                                             provider.dtype)
        self.provider = provider


class PtrArgument(Argument):

    """ Class representing arbitrary arguments that a kernel might expect.
        These are passed as void pointers and then promptly casted to their
        actual type.
    """

    is_PtrArgument = True

    def __init__(self, provider):
        super(PtrArgument, self).__init__(provider.name,
                                          [ValueDependency(provider)], provider.dtype)


class ArgumentEngine(object):
    """ Class that encapsulates the argument derivation and verification subsystem
    """
    def __init__(self, stencils, parameters, dle_arguments):
        self.parameters = parameters
        self.dle_arguments = dle_arguments
        argument_list = self._build_arguments_list(parameters)
        self.arguments = filter_ordered([x for x in argument_list if x.is_Argument],
                                        key=lambda x: x.name)
        self.dimension_params = [x for x in argument_list if x.is_DimensionParameter]
        self.offsets = {d.end_name: v for d, v in retrieve_offsets(stencils).items()}

    def handle(self, **kwargs):
        """ The main method by which the :class:`Operator` interacts with this class.
            The arguments passed into Operator.apply() all end up in kwargs here.
        """

        user_autotune = kwargs.pop('autotune', False)

        kwargs = self._offset_adjust(kwargs)

        kwargs = self._extract_children_of_composites(kwargs)

        values = self._derive_values(kwargs)

        # The following is only being done to update the autotune flag. The actual value
        # derivation for the dle arguments has moved inside the above _derive_values
        # method.
        # TODO: Refactor so this is not required
        dim_sizes = dict([(d.name, runtime_dim_extent(d, values))
                          for d in self.dimensions])
        dle_arguments, dle_autotune = self._dle_arguments(dim_sizes)

        assert(self._verify(values))

        arguments = OrderedDict([(k.name, v) for k, v in values.items()])

        return arguments, user_autotune and dle_autotune

    def _offset_adjust(self, kwargs):
        for k, v in kwargs.items():
            if k in self.offsets:
                kwargs[k] = v + self.offsets[k]
        return kwargs

    def _build_arguments_list(self, parameters):
        # Pass through SymbolicFunction
        symbolic_functions = [x for x in parameters if isinstance(x, SymbolicFunction)]
        dim_dep_mapper = OrderedDict()  # Mapper for dependencies between dimensions
        tensor_arguments = []
        for f in symbolic_functions:
            argument = ArgumentVisitor().visit(f)
            tensor_arguments.append(argument)
            for i, d in enumerate(f.indices):
                v = dim_dep_mapper.setdefault(d, [])
                v.append(ValueDependency(argument, param=i))

        for arg in self.dle_arguments:
            v = dim_dep_mapper.setdefault(arg.argument, [])
            v.append(ValueDependency(derive_dle_arg_value, param=arg))

        # Record dependencies in Dimensions
        dim_param_mapper = OrderedDict([(k, DimensionParameter(k, v)) for k, v in
                                        dim_dep_mapper.items()])

        # Dimensions that are in parameters but not directly referenced in the expressions
        more = dict([(x, DimensionParameter(x, [])) for x in self.dimensions if x not in
                     dim_dep_mapper])
        dim_param_mapper.update(more)

        for dim in [x for x in self.dimensions if x.is_Stepping]:
            v = dim_param_mapper[dim.parent].dependencies
            v.append(ValueDependency(dim_param_mapper[dim]))

        dimension_parameters = list(dim_param_mapper.values())

        scalar_arguments = flatten([ArgumentVisitor().visit(x)
                                    for x in dimension_parameters])

        other_arguments = [ArgumentVisitor().visit(x) for x in parameters
                           if x not in tensor_arguments + scalar_arguments]

        other_arguments = [x for x in other_arguments if x is not None]

        return tensor_arguments + dimension_parameters + scalar_arguments +\
            other_arguments

    def _extract_children_of_composites(self, kwargs):
        new_params = {}
        # If we've been passed CompositeFunction objects as kwargs,
        # they might have children that need to be substituted as well.
        for k, v in kwargs.items():
            if isinstance(v, CompositeFunction):
                orig_param_l = [i for i in self.parameters if i.name == k]
                # If I have been passed a parameter, I must have seen it before
                if len(orig_param_l) == 0:
                    raise InvalidArgument("Parameter %s does not exist in expressions " +
                                          "passed to this Operator" % k)
                # We've made sure the list isn't empty. Names should be unique so it
                # should have exactly one entry
                assert(len(orig_param_l) == 1)
                orig_param = orig_param_l[0]
                # Pull out the children and add them to kwargs
                for orig_child, new_child in zip(orig_param.children, v.children):
                    new_params[orig_child.name] = new_child
        kwargs.update(new_params)
        return kwargs

    def _dle_arguments(self, dim_sizes):
        # Add user-provided block sizes, if any
        dle_arguments = OrderedDict()
        autotune = True
        for i in self.dle_arguments:
            dim_size = dim_sizes.get(i.original_dim.name, None)
            if dim_size is None:
                raise InvalidArgument('Unable to derive size of dimension %s from '
                                      'defaults. Please provide an explicit '
                                      'value.' % i.original_dim.name)
            if i.value:
                try:
                    dle_arguments[i.argument.name] = i.value(dim_size)
                except TypeError:
                    dle_arguments[i.argument.name] = i.value
                    autotune = False
            else:
                dle_arguments[i.argument.name] = dim_size
        return dle_arguments, autotune

    def _derive_values(self, kwargs):
        """ Populate values for all the arguments. The values provided in kwargs will
            be used wherever provided. The values for the rest of the arguments will be
            derived from the ones provided. The default values for the tensors come from
            the data property of the symbols used in the Operator.
        """
        values = OrderedDict()

        values = OrderedDict([(i, get_value(i, kwargs.pop(i.name, None), values))
                              for i in self.arguments])

        dimension_values = OrderedDict([(i, kwargs.pop(i.name, None))
                                        for i in self.dimension_params])

        # Make sure we've used all arguments passed
        if len(kwargs) > 0:
            raise InvalidArgument("Unknown arguments passed: " + ", ".join(kwargs.keys()))

        # Derive values for other arguments
        for i in self.arguments:
            if values[i] is None:
                known_values = OrderedDict(chain(values.items(),
                                                 dimension_values.items()))
                provided_values = [get_value(i, x, known_values)
                                   for x in i.gets_value_from]
                assert(len(provided_values) > 0)
                values[i] = reduce(max, provided_values)

        # Second pass to evaluate any Unevaluated dependencies from the first pass
        for k, v in values.items():
            if isinstance(v, UnevaluatedDependency):
                values[k] = v.evaluate(values)
        return values

    def _verify(self, values):
        verify = True
        for i in values:
            verify = verify and all(verify(i, x) for x in i.verified_by)
        return verify

    @property
    def dimensions(self):
        return [x for x in self.parameters if isinstance(x, Dimension)]


class ArgumentVisitor(GenericVisitor):
    """ Visits types to return their runtime arguments
    """
    def visit_SymbolicFunction(self, o):
        return TensorArgument(o)

    def visit_Argument(self, o):
        return o

    def visit_DimensionParameter(self, o):
        dependency = ValueDependency(o)
        size = ScalarArgument(o.provider.size_name, [dependency])
        start = ScalarArgument(o.provider.start_name, [dependency])
        end = ScalarArgument(o.provider.end_name, [dependency])
        return [size, start, end]

    def visit_Object(self, o):
        return PtrArgument(o)

    def visit_Array(self, o):
        return TensorArgument(o)

    def visit_Scalar(self, o):
        arg = ScalarArgument(o.name, o, dtype=o.dtype)
        arg.provider = o
        return arg

    def visit_Constant(self, o):
        # TODO: Add option for delayed query of default value
        arg = ScalarArgument(o.name, [ValueDependency(o)], dtype=o.dtype)
        arg.provider = o
        return arg


class ValueVisitor(GenericVisitor):
    """Visits types to derive their value
    """
    def __init__(self, consumer, known_values):
        self.consumer = consumer
        self.known_values = dict([(k, v) for k, v in known_values.items()
                                  if v is not None])
        super(ValueVisitor, self).__init__()

    def visit_Function(self, o, param=None):
        assert(isinstance(self.consumer, TensorArgument))
        try:
            return o._data_buffer
        except AttributeError:
            return o.data

    def visit_Constant(self, o, param=None):
        return o.data

    def visit_Dependency(self, o):
        return self.visit(o.obj, o.param)

    def visit_function(self, o, param=None):
        return o(self.consumer, self.known_values, param)

    def visit_Object(self, o, param=None):
        if callable(o.value):
            return o.value()
        else:
            return o.value

    def visit_object(self, o, param=None):
        return o

    def visit_DimensionParameter(self, o, param=None):
        """ Called when some Argument's value is being derived and the dependency tree
            was followed till a :class:`DimensionParameter` object. The Argument for
            which we are deriving the value is self.consumer
        """

        # We are being asked to provide a default value for dim_start
        if self.consumer.name == o.provider.start_name:
            return 0

        if o in self.known_values and not isinstance(self.known_values[o],
                                                     UnevaluatedDependency):
            provided_values = [self.known_values[o]]
        else:
            provided_values = [get_value(o, x, self.known_values)
                               for x in o.gets_value_from]

        if len(provided_values) > 0:
            if not all(x is not None for x in provided_values):
                unknown_args = [x.obj for x in o.gets_value_from
                                if get_value(o, x, self.known_values) is None]

                def late_evaluate_dim_size(consumer, known_values, partial_values):
                    known = []
                    try:
                        new_values = [known_values[x] for x in unknown_args]
                        known = [x for x in new_values if x is not None]
                    except KeyError:
                        pass
                    return reduce(max, partial_values + known)

                return UnevaluatedDependency(o, late_evaluate_dim_size,
                                             [x for x in provided_values
                                              if x is not None])
            value = reduce(max, provided_values)
        else:
            value = None
        return value

    def visit_TensorArgument(self, o, param):
        assert(isinstance(self.consumer, DimensionParameter))
        return self.known_values[o].shape[param]


class Dependency(object):
    """ Object that represents an edge between two nodes on a dependency graph
        Dependencies are directional, i.e. A -> B != B -> A
        A dependency can either by of type :class:`ValueDependency` in which case this
        dependency will be followed for value derivation or it can be of type
        :class:`VerifyDependency` in which case this dependency will be followed for
        verification. Both types of dependencies may exist between a pair of nodes.
        However, only a single dependency of a type may exist between any pair of nodes.
        :param obj: Dependencies have a source node and a target node. The source node
                    will store the dependency object on itself. The obj referred to here
                    is the target node of the dependency.
        :param param: An optional parameter that might be required to evaluate the
                      relationship being defined by this Dependency. e.g. when a Dimension
                      derives its value from a SymbolicFunction's shape, this param
                      carries the index of a dimension in the SymbolicFunction's shape.
    """
    is_Value = False
    is_Verify = False

    def __init__(self, obj, param=None):
        self.obj = obj
        self.param = param

    def __repr__(self):
        return "(%s : %s)" % (str(type(self)), str(self.obj))


class ValueDependency(Dependency):
    """ A Dependency that specifies that one argument gets its value from another """
    is_Value = True


class VerifyDependency(Dependency):
    """ A Dependency that specifies that one argument verifies the value of another """
    is_Verify = True


class UnevaluatedDependency(object):
    def __init__(self, consumer, evaluator, extra_param=None):
        self.consumer = consumer
        self.evaluator = evaluator
        self.extra_param = extra_param

    def evaluate(self, known_values):
        return self.evaluator(self.consumer, known_values, self.extra_param)


def runtime_arguments(parameters):
    return flatten([ArgumentVisitor().visit(p) for p in parameters])


def log_args(arguments):
    arg_str = []
    for k, v in arguments.items():
        if hasattr(v, 'shape'):
            arg_str.append('(%s, shape=%s, L2 Norm=%d, type=%s)' %
                           (k, str(v.shape), np.linalg.norm(v.view()), type(v)))
        else:
            arg_str.append('(%s, value=%s, type=%s)' % (k, str(v), type(v)))
    print("Passing Arguments: " + ", ".join(arg_str))


def find_argument_by_name(name, haystack):
    filtered = [v for k, v in haystack.items() if k.name == name]
    assert(len(filtered) < 2)
    if len(filtered) == 1:
        return filtered[0]
    else:
        return None


def runtime_dim_extent(dimension, values):
    try:
        return find_argument_by_name(dimension.end_name, values) -\
            find_argument_by_name(dimension.start_name, values)
    except (KeyError, TypeError):
        return None


def retrieve_offsets(stencils):
    """
    Return a mapper from :class:`Dimension`s to the min/max integer offsets
    within ``stencils``.
    """
    offs = Stencil.union(*stencils)
    mapper = {d: v for d, v in offs.diameter.items()}
    mapper.update({d.parent: v for d, v in mapper.items() if d.is_Stepping})
    return mapper


def derive_dle_arg_value(blocked_dim, known_values, dle_argument):
    dim_size = runtime_dim_extent(dle_argument.original_dim, known_values)
    if dim_size is None:
        return UnevaluatedDependency(blocked_dim, derive_dle_arg_value, dle_argument)
    value = None
    if dle_argument.value:
        try:
            value = dle_argument.value(dim_size)
        except TypeError:
            value = dle_argument.value
    else:
        value = dim_size
    return value


def get_value(consumer, provider, known_values):
    return ValueVisitor(consumer, known_values).visit(provider)
