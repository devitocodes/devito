"""The Iteration/Expression Tree (IET) hierarchy."""

import abc
import ctypes
import inspect
from functools import cached_property
from collections import OrderedDict, namedtuple
from collections.abc import Iterable

import cgen as c
from sympy import IndexedBase, sympify

from devito.data import FULL
from devito.ir.cgen import ccode
from devito.ir.equations import DummyEq, OpInc, OpMin, OpMax
from devito.ir.support import (INBOUND, SEQUENTIAL, PARALLEL, PARALLEL_IF_ATOMIC,
                               PARALLEL_IF_PVT, VECTORIZED, AFFINE, Property,
                               Forward, WithLock, PrefetchUpdate, detect_io)
from devito.symbolics import ListInitializer, CallFromPointer
from devito.tools import (Signer, as_tuple, filter_ordered, filter_sorted, flatten,
                          ctypes_to_cstr)
from devito.types.basic import (AbstractFunction, AbstractSymbol, Basic, Indexed,
                                Symbol)
from devito.types.object import AbstractObject, LocalObject

__all__ = ['Node', 'MultiTraversable', 'Block', 'Expression', 'Callable',
           'Call', 'ExprStmt', 'Conditional', 'Iteration', 'List', 'Section',
           'TimedList', 'Prodder', 'MetaCall', 'PointerCast', 'HaloSpot',
           'Definition', 'ExpressionBundle', 'AugmentedExpression', 'Break',
           'Increment', 'Return', 'While', 'ListMajor', 'ParallelIteration',
           'ParallelBlock', 'Dereference', 'Lambda', 'SyncSpot', 'Pragma',
           'DummyExpr', 'BlankLine', 'ParallelTree', 'BusyWait', 'UsingNamespace',
           'Using', 'CallableBody', 'Transfer', 'EmptyList']

# First-class IET nodes


class Node(Signer):

    __metaclass__ = abc.ABCMeta

    is_Block = False
    is_Iteration = False
    is_While = False
    is_Expression = False
    is_Callable = False
    is_CallableBody = False
    is_Conditional = False
    is_ElementalFunction = False
    is_Call = False
    is_List = False
    is_Definition = False
    is_PointerCast = False
    is_Dereference = False
    is_Section = False
    is_HaloSpot = False
    is_ExpressionBundle = False
    is_SyncSpot = False

    _traversable = []
    """
    :attr:`_traversable`. The traversable fields of the Node; that is, fields
    walked over by a Visitor. All arguments in __init__ whose name
    appears in this list are treated as traversable fields.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        argnames, _, _, defaultvalues, _, _, _ = inspect.getfullargspec(cls.__init__)
        try:
            defaults = dict(zip(argnames[-len(defaultvalues):], defaultvalues))
        except TypeError:
            # No default kwarg values
            defaults = {}
        obj._args = {k: v for k, v in zip(argnames[1:], args)}
        obj._args.update(kwargs.items())
        obj._args.update({k: defaults.get(k) for k in argnames[1:] if k not in obj._args})
        return obj

    def _rebuild(self, *args, **kwargs):
        """Reconstruct ``self``."""
        handle = self._args.copy()  # Original constructor arguments
        argnames = [i for i in self._traversable if i not in kwargs]
        handle.update(OrderedDict([(k, v) for k, v in zip(argnames, args)]))
        handle.update(kwargs)
        return type(self)(**handle)

    @cached_property
    def ccode(self):
        """
        Generate C code.

        This is a shorthand for

            .. code-block:: python

              from devito.ir.iet import CGen
              CGen().visit(self)
        """
        from devito.ir.iet.visitors import CGen
        return CGen().visit(self)

    @property
    def view(self):
        """A high-level representation of the IET rooted in `self`."""
        from devito.ir.iet.visitors import printAST
        return printAST(self)

    @property
    def view_cir(self):
        from devito.ir.iet.visitors import CGen
        from devito.passes.iet.languages.CIR import CIRPrinter
        return str(CGen(printer=CIRPrinter).visit(self))

    @property
    def children(self):
        """Return the traversable children."""
        return tuple(getattr(self, i) for i in self._traversable)

    @property
    def args(self):
        """Arguments used to construct the Node."""
        return self._args.copy()

    @property
    def args_frozen(self):
        """Arguments used to construct the Node that cannot be traversed."""
        return {k: v for k, v in self.args.items() if k not in self._traversable}

    def __str__(self):
        return str(self.ccode)

    def __repr__(self):
        return self.__class__.__name__

    @property
    def functions(self):
        """All AbstractFunctions and AbstractObjects used by this node."""
        return ()

    @property
    def expr_symbols(self):
        """All symbols appearing within this node."""
        return ()

    @property
    def defines(self):
        """All Basic objects defined by this node."""
        return ()

    @property
    def writes(self):
        """All Basic objects modified by this node."""
        return ()

    def _signature_items(self):
        return (self.view_cir,)


class ExprStmt:

    """
    A mixin for Nodes that represent C expression statements, which are expressions
    followed by a semicolon. For example, the lines:

        * i = 0;
        * j = a[i] + 8;
        * int a = 3;
        * foo(b)

    are all expression statements.

    Notes
    -----
    An ExprStmt does *not* have children Nodes.
    """

    pass


class MultiTraversable(Node):

    """
    An abstract base class for Nodes comprising more than one traversable children.
    """

    pass


class List(Node):

    """A sequence of Nodes."""

    is_List = True

    _traversable = ['body']

    def __init__(self, header=None, body=None, footer=None):
        body = as_tuple(body)
        if len(body) == 1 and all(type(i) is List for i in [self, body[0]]):
            # De-nest Lists
            #
            # Note: to avoid disgusting metaclass voodoo (due to
            # https://stackoverflow.com/questions/56514586/\
            #     arguments-of-new-and-init-for-metaclasses)
            # we change the internal state here in __init__
            # rather than in __new__
            self._args['header'] = self.header = as_tuple(header) + body[0].header
            self._args['body'] = self.body = body[0].body
            self._args['footer'] = self.footer = as_tuple(footer) + body[0].footer
        else:
            self.header = as_tuple(header)
            self.body = as_tuple(body)
            self.footer = as_tuple(footer)

    def __repr__(self):
        return "<%s (%d, %d, %d)>" % (self.__class__.__name__, len(self.header),
                                      len(self.body), len(self.footer))


class EmptyList(List):
    """A plain List node without a body, such as a header/footer only"""
    _traversable = []


class Block(List):

    """A sequence of Nodes, wrapped in a block {...}."""

    is_Block = True

    def __init__(self, header=None, body=None, footer=None):
        self.header = as_tuple(header)
        self.body = as_tuple(body)
        self.footer = as_tuple(footer)


class Call(ExprStmt, Node):

    """
    A function call.

    Parameters
    ----------
    name : str or CallFromPointer
        The called function.
    arguments : list of Basic, optional
        The objects in input to the function call.
    retobj : Symbol or Indexed, optional
        The object the return value of the Call is assigned to.
    is_indirect : bool, optional
        If True, the object represents an indirect function call. The emitted
        code will be `name, arg1, ..., argN` rather than `name(arg1, ..., argN)`.
        Defaults to False.
    cast : bool, optional
        If True, the Call return value is explicitly casted to the `retobj` type.
        Defaults to False.
    writes : list, optional
        The AbstractFunctions that will be written to by the called function.
        Explicitly tagging these AbstractFunctions is useful in the case of
        external calls, that is whenever the compiler would be unable to
        retrieve that information by analysis of the IET graph.
    templates : list of Basic, optional
        The template arguments of the Call.
    """

    is_Call = True

    def __init__(self, name, arguments=None, retobj=None, is_indirect=False,
                 cast=False, writes=None, templates=None):
        if isinstance(name, CallFromPointer):
            self.base = name.base
        else:
            self.base = None
        self.name = str(name)
        self.arguments = as_tuple(arguments)
        self.retobj = retobj
        self.is_indirect = is_indirect
        self.cast = cast
        self._writes = as_tuple(writes)
        self.templates = as_tuple(templates)

    def __repr__(self):
        ret = "" if self.retobj is None else "%s = " % self.retobj
        return "%sCall::\n\t%s(...)" % (ret, self.name)

    def _rebuild(self, *args, **kwargs):
        if args:
            # Not elegant, but basically it handles the fact that a Call might
            # have nested Calls/Lambdas among its `arguments`, and these might
            # change, and we are in such a case *if and only if* we have `args`
            assert len(args) == len(self.children)
            mapper = dict(zip(self.children, args))
            kwargs['arguments'] = [mapper.get(i, i) for i in self.arguments]
        return super()._rebuild(**kwargs)

    @property
    def children(self):
        return tuple(i for i in self.arguments if isinstance(i, (Call, Lambda)))

    @cached_property
    def functions(self):
        retval = []
        for i in self.arguments:
            if isinstance(i, (AbstractFunction, Indexed, IndexedBase, AbstractObject)):
                retval.append(i.function)
            elif isinstance(i, Call):
                retval.extend(i.functions)
            else:
                try:
                    v = i.free_symbols
                except AttributeError:
                    continue
                for s in v:
                    try:
                        # `try-except` necessary for e.g. Macro
                        if isinstance(s.function, (AbstractFunction, AbstractObject)):
                            retval.append(s.function)
                    except AttributeError:
                        continue

        if self.base is not None:
            retval.append(self.base.function)

        if self.retobj is not None:
            retval.append(self.retobj.function)

        return tuple(filter_ordered(retval))

    @cached_property
    def expr_symbols(self):
        retval = []
        for i in self.arguments:
            if isinstance(i, AbstractFunction):
                continue
            elif isinstance(i, (Indexed, IndexedBase, AbstractObject, Symbol)):
                retval.extend(i.free_symbols)
            elif isinstance(i, Call):
                retval.extend(i.expr_symbols)
            else:
                try:
                    retval.extend(i.free_symbols)
                except AttributeError:
                    pass

        if self.base is not None:
            retval.append(self.base)

        if isinstance(self.retobj, Indexed):
            retval.extend(self.retobj.free_symbols)
        elif self.retobj is not None:
            retval.append(self.retobj)

        return tuple(filter_ordered(retval))

    @property
    def defines(self):
        ret = ()
        if isinstance(self.retobj, Basic):
            ret += (self.retobj,)
        return ret

    @property
    def writes(self):
        return self._writes


class Expression(ExprStmt, Node):

    """
    A node encapsulating a ClusterizedEq.

    Parameters
    ----------
    expr : ClusterizedEq
        The encapsulated expression.
    pragmas : cgen.Pragma or list of cgen.Pragma, optional
        A bag of pragmas attached to this Expression.
    init : bool, optional
        True if an initialization, False otherwise (default).
    operation : Operation, optional
        Special operation performed by the Expression (e.g., a reduction).
    """

    is_Expression = True

    def __init__(self, expr, pragmas=None, init=False, operation=None):
        self.expr = expr
        self.pragmas = as_tuple(pragmas)
        self.init = init
        self.operation = operation

    def __repr__(self):
        return "<%s::%s=%s>" % (self.__class__.__name__,
                                type(self.write),
                                ','.join('%s' % type(f) for f in self.functions))

    @property
    def dtype(self):
        return self.expr.dtype

    @property
    def output(self):
        """The Symbol/Indexed this Expression writes to."""
        return self.expr.lhs

    @cached_property
    def reads(self):
        """The Functions read by the Expression."""
        return detect_io(self.expr, relax=True)[0]

    @cached_property
    def write(self):
        """The Function written by the Expression."""
        return self.expr.lhs.base.function

    @cached_property
    def dimensions(self):
        retval = flatten(i.indices for i in self.functions if i.is_Indexed)
        return tuple(filter_ordered(retval))

    @property
    def is_scalar(self):
        """True if the LHS is a scalar, False otherwise."""
        return isinstance(self.expr.lhs, (AbstractSymbol, IndexedBase, LocalObject))

    @property
    def is_tensor(self):
        """True if the LHS is an array entry, False otherwise."""
        return self.expr.lhs.is_Indexed

    @property
    def is_reduction(self):
        """True if the RHS performs a reduction operation, False otherwise."""
        return self.operation in (OpInc, OpMin, OpMax)

    @property
    def is_initializable(self):
        """
        True if it can be an initializing assignment, False otherwise.
        """
        return (((self.is_scalar and not self.is_reduction) or
                 (self.is_tensor and isinstance(self.expr.rhs, ListInitializer))))

    @property
    def defines(self):
        return (self.output.base,) if self.is_initializable else ()

    @property
    def expr_symbols(self):
        return tuple(self.expr.free_symbols)

    @cached_property
    def functions(self):
        functions = list(self.reads)
        if self.write is not None:
            functions.append(self.write)
        return tuple(filter_ordered(functions))

    @property
    def writes(self):
        return (self.write,)


class AugmentedExpression(Expression):

    """A node representing an augmented assignment, such as +=, -=, &=, ...."""

    def __init__(self, expr, pragmas=None, operation=None):
        super().__init__(expr, pragmas=pragmas, operation=operation)

    @property
    def is_initializable(self):
        return False

    @property
    def op(self):
        try:
            return self.operation.name
        except AttributeError:
            # Not an ir.Operation
            assert not self.is_reduction
            return self.operation


class Increment(AugmentedExpression):

    """Shortcut for ``AugmentedExpression(expr, '+'), since it's so widely used."""

    def __init__(self, expr, pragmas=None):
        super().__init__(expr, pragmas=pragmas, operation=OpInc)


class Iteration(Node):

    """
    Implement a for-loop over nodes.

    Parameters
    ----------
    nodes : Node or list of Node
        The for-loop body.
    dimension : Dimension
        The Dimension over which the for-loop iterates.
    limits : expr-like or 3-tuple
        If an expression, it represents the for-loop max point; in this case, the
        min point is 0 and the step increment is unitary. If a 3-tuple, the
        format is ``(min point, max point, stepping)``.
    direction: IterationDirection, optional
        The for-loop direction. Accepted:
        - ``Forward``: i += stepping (defaults)
        - ``Backward``: i -= stepping
    properties : Property or list of Property, optional
        Iteration decorators, denoting properties such as parallelism.
    pragmas : cgen.Pragma or list of cgen.Pragma, optional
        A bag of pragmas attached to this Iteration.
    uindices : DerivedDimension or list of DerivedDimension, optional
        An uindex is an additional iteration variable defined by the for-loop. The
        for-loop bounds are independent of all ``uindices`` (hence the name uindex,
        or "unbounded index"). An uindex must have ``dimension`` as its parent.
    """

    is_Iteration = True

    _traversable = ['nodes']

    def __init__(self, nodes, dimension, limits, direction=None, properties=None,
                 pragmas=None, uindices=None):
        self.nodes = as_tuple(nodes)
        self.dim = dimension
        self.index = self.dim.name
        self.direction = direction or Forward

        # Generate loop limits
        if isinstance(limits, Iterable):
            assert(len(limits) == 3)
            self.limits = tuple(limits)
        elif self.dim.is_Incr:
            self.limits = (self.dim.symbolic_min, limits, self.dim.step)
        else:
            self.limits = (0, limits, 1)

        # Track this Iteration's properties, pragmas and unbounded indices
        properties = as_tuple(properties)
        assert (i in Property._KNOWN for i in properties)
        self.properties = as_tuple(filter_sorted(properties))
        self.pragmas = as_tuple(pragmas)
        self.uindices = as_tuple(uindices)
        assert all(i.is_Derived and self.dim in i._defines for i in self.uindices)

    def __repr__(self):
        properties = ""
        if self.properties:
            properties = [str(i) for i in self.properties]
            properties = "WithProperties[%s]::" % ",".join(properties)
        index = self.index
        if self.uindices:
            index += '[%s]' % ','.join(i.name for i in self.uindices)
        return "<%sIteration %s; %s>" % (properties, index, self.limits)

    @property
    def is_Affine(self):
        return AFFINE in self.properties

    @property
    def is_Sequential(self):
        return SEQUENTIAL in self.properties

    @property
    def is_Parallel(self):
        return PARALLEL in self.properties

    @property
    def is_ParallelAtomic(self):
        return PARALLEL_IF_ATOMIC in self.properties

    @property
    def is_ParallelPrivate(self):
        return PARALLEL_IF_PVT in self.properties

    @property
    def is_ParallelNoAtomic(self):
        return any([self.is_Parallel, self.is_ParallelPrivate])

    @property
    def is_ParallelRelaxed(self):
        return any([self.is_Parallel, self.is_ParallelAtomic, self.is_ParallelPrivate])

    @property
    def is_Vectorized(self):
        return VECTORIZED in self.properties

    @property
    def is_Inbound(self):
        return INBOUND in self.properties

    @property
    def symbolic_bounds(self):
        """A 2-tuple representing the symbolic bounds [min, max] of the Iteration."""
        return (sympify(self.limits[0]), sympify(self.limits[1]))

    @property
    def symbolic_size(self):
        """The symbolic size of the Iteration."""
        return self.symbolic_bounds[1] - self.symbolic_bounds[0] + 1

    @property
    def symbolic_min(self):
        """The symbolic min of the Iteration."""
        return self.symbolic_bounds[0]

    @property
    def symbolic_max(self):
        """The symbolic max of the Iteration."""
        return self.symbolic_bounds[1]

    def bounds(self, _min=None, _max=None):
        """
        The bounds [min, max] of the Iteration, as numbers if min/max are supplied,
        as symbols otherwise.
        """
        _min = _min if _min is not None else self.limits[0]
        _max = _max if _max is not None else self.limits[1]

        return (_min, _max)

    @property
    def start(self):
        """The start value."""
        if self.direction is Forward:
            return self.dim.symbolic_min
        else:
            return self.dim.symbolic_max

    @property
    def step(self):
        """The step value."""
        return self.limits[2]

    def size(self, _min=None, _max=None):
        """The size of the iteration space if _min/_max are supplied, None otherwise."""
        _min, _max = self.bounds(_min, _max)
        return _max - _min + 1

    @property
    def dimensions(self):
        """All Dimensions appearing in the Iteration header."""
        return tuple(self.dim._defines) + self.uindices

    @cached_property
    def expr_symbols(self):
        return tuple(self.symbolic_min.free_symbols) \
            + tuple(self.symbolic_max.free_symbols) \
            + self.uindices \
            + tuple(flatten(i.symbolic_min.free_symbols for i in self.uindices)) \
            + tuple(flatten(i.symbolic_incr.free_symbols for i in self.uindices))

    @property
    def defines(self):
        return self.dimensions


class DoIf(Node):

    """
    An abstract Node to express computation subjected to a condition.
    """

    def __init__(self, condition):
        self.condition = condition

    @property
    def functions(self):
        ret = []
        for i in self.condition.free_symbols:
            try:
                ret.append(i.function)
            except AttributeError:
                pass
        return tuple(ret)

    @property
    def expr_symbols(self):
        return tuple(self.condition.free_symbols)


class While(DoIf):

    """
    Implement a while-loop.

    Parameters
    ----------
    condition : sympy.Function or sympy.Relation or bool
        The while-loop exit condition.
    body : Node or list of Node, optional
        The whie-loop body.
    """

    is_While = True

    _traversable = ['body']

    def __init__(self, condition, body=None):
        super().__init__(condition)
        self.body = as_tuple(body)

    def __repr__(self):
        return "<While %s; %d>" % (self.condition, len(self.body))


class Callable(Node):

    """
    A callable function.

    Parameters
    ----------
    name : str
        The name of the callable.
    body : Node or list of Node
        The Callable body.
    retval : str
        The return type of the Callable.
    parameters : list of Basic, optional
        The objects in input to the Callable.
    prefix : list of str, optional
        Qualifiers to prepend to the Callable signature. None by default.
    templates : list of Basic, optional
        The template parameters of the Callable.
    attributes : list of str, optional
        Additional attributes to append to the Callable signature. An
        attributes is one or more keywords that appear in between the
        return type and the function name. None by default.
    """

    is_Callable = True

    _traversable = ['body']

    def __init__(self, name, body, retval, parameters=None, prefix=None,
                 templates=None, attributes=None):
        self.name = name
        if not isinstance(body, CallableBody):
            self.body = CallableBody(body)
        else:
            self.body = body
        self.retval = retval
        self.prefix = as_tuple(prefix)
        self.parameters = as_tuple(parameters)
        self.templates = as_tuple(templates)
        self.attributes = as_tuple(attributes)

    def __repr__(self):
        param_types = [ctypes_to_cstr(i._C_ctype) for i in self.parameters]
        return "%s[%s]<%s; %s>" % (self.__class__.__name__, self.name, self.retval,
                                   ",".join(param_types))

    @property
    def all_parameters(self):
        return self.parameters + self.templates

    @property
    def functions(self):
        return tuple(i.function for i in self.all_parameters
                     if isinstance(i.function, (AbstractFunction, AbstractObject)))

    @property
    def defines(self):
        return self.all_parameters


class CallableBody(MultiTraversable):

    """
    The immediate child of a Callable.

    Parameters
    ----------
    body : Node or list of Node
        The actual body.
    unpacks : list of Nodes, optional
        Statements unpacking data from composite types.
    init : Node, optional
        A piece of IET to perform some initialization relevant for `body`
        (e.g., to initialize the target language runtime).
    standalones : list of Definitions, optional
        Object definitions for `body`. Instantiating these objects does not
        require passing any arguments to their constructors, so these
        Definitions can be scheduled safely right after `init`. They may or may
        not be required by some of the subsequent nodes (e.g., `allocs`,
        `maps`).
    allocs : list of Nodes, optional
        Data definitions and allocations for `body`.
    stacks : list of Definitions, optional
        Definitions for the stack-scoped objects appearing in `body`.
    casts : list of PointerCasts, optional
        Sequence of PointerCasts required by the `body`.
    bundles : list of Nodes, optional
        Data bundling for `body`. Used to initialize data subjected to layout
        transformation (w.r.t. how it arrives from Python), such as vector types.
    maps : Transfer or list of Transfer, optional
        Data maps for `body` (a data map may e.g. trigger a data transfer from
        host to device).
    strides : list of Nodes, optional
        Statements defining symbols used to access linearized arrays.
    objs : list of Definitions, optional
        Object definitions for `body`. Instantiating these objects may or may
        not require some of the symbols defined in the previous nodes (e.g.,
        `allocs`, `maps`).
    unmaps : Transfer or list of Transfer, optional
        Data unmaps for `body`.
    unbundles : list of Nodes, optional
        Data unbundling for `body`.
    frees : list of Calls, optional
        Data deallocations for `body`.
    errors : list of Nodes, optional
        Error handling for `body`.
    retstmt : Node, optional
        The return statement for `body`.
    """

    is_CallableBody = True

    _traversable = ['unpacks', 'init', 'standalones', 'allocs', 'stacks',
                    'casts', 'bundles', 'maps', 'strides', 'objs', 'body',
                    'unmaps', 'unbundles', 'frees', 'errors', 'retstmt']

    def __init__(self, body, init=(), standalones=(), unpacks=(), strides=(),
                 allocs=(), stacks=(), casts=(), bundles=(), objs=(), maps=(),
                 unmaps=(), unbundles=(), frees=(), errors=(), retstmt=()):
        # Sanity check
        assert not isinstance(body, CallableBody), "CallableBody's cannot be nested"

        self.body = as_tuple(body)

        self.unpacks = as_tuple(unpacks)
        self.init = as_tuple(init)
        self.standalones = as_tuple(standalones)
        self.allocs = as_tuple(allocs)
        self.stacks = as_tuple(stacks)
        self.casts = as_tuple(casts)
        self.strides = as_tuple(strides)
        self.bundles = as_tuple(bundles)
        self.maps = as_tuple(maps)
        self.objs = as_tuple(objs)
        self.unmaps = as_tuple(unmaps)
        self.unbundles = as_tuple(unbundles)
        self.frees = as_tuple(frees)
        self.errors = as_tuple(errors)
        self.retstmt = as_tuple(retstmt)

    def __repr__(self):
        return ("<CallableBody <unpacks=%d, allocs=%d, casts=%d, maps=%d, "
                "objs=%d> <unmaps=%d, frees=%d>>" %
                (len(self.unpacks), len(self.allocs), len(self.casts),
                 len(self.maps), len(self.objs), len(self.unmaps),
                 len(self.frees)))


class Conditional(DoIf):

    """
    A node to express if-then-else blocks.

    Parameters
    ----------
    condition : expr-like
        The if condition.
    then_body : Node or list of Node
        The then body.
    else_body : Node or list of Node
        The else body.
    """

    is_Conditional = True

    _traversable = ['then_body', 'else_body']

    def __init__(self, condition, then_body, else_body=None):
        super().__init__(condition)
        self.then_body = as_tuple(then_body)
        self.else_body = as_tuple(else_body)

    def __repr__(self):
        if self.else_body:
            return "<[%s] ? [%s] : [%s]>" %\
                (ccode(self.condition), repr(self.then_body), repr(self.else_body))
        else:
            return "<[%s] ? [%s]" % (ccode(self.condition), repr(self.then_body))


# Second level IET nodes

class TimedList(List):

    """
    Wrap a Node with C-level timers.

    Parameters
    ----------
    timer : Timer
        The Timer used by the TimedList.
    lname : str
        A unique name for the timed code block.
    body : Node or list of Node
        The TimedList body.
    """

    def __init__(self, timer, lname, body):
        self._name = lname
        self._timer = timer

        super().__init__(header=c.Line('START(%s)' % lname),
                         body=body,
                         footer=c.Line('STOP(%s,%s)' % (lname, timer.name)))

    @classmethod
    def _start_timer_header(cls):
        return ('START(S)', ('struct timeval start_ ## S , end_ ## S ; '
                             'gettimeofday(&start_ ## S , NULL);'))

    @classmethod
    def _stop_timer_header(cls):
        return ('STOP(S,T)', ('gettimeofday(&end_ ## S, NULL); T->S += (double)'
                              '(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)'
                              '(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;'))

    @property
    def name(self):
        return self._name

    @property
    def timer(self):
        return self._timer

    @property
    def functions(self):
        return (self.timer,)


class Definition(ExprStmt, Node):

    """
    A node encapsulating a variable definition.
    """

    is_Definition = True

    def __init__(self, function):
        self.function = function

    def __repr__(self):
        return "<Def(%s)>" % self.function

    @property
    def functions(self):
        ret = [self.function]
        for i in self.expr_symbols:
            f = i.function
            if f.is_AbstractFunction or f.is_AbstractObject:
                ret.append(i.function)
        return tuple(ret)

    @property
    def defines(self):
        if self.function._mem_stack:
            return (self.function.indexed,)
        else:
            return (self.function,)

    @property
    def expr_symbols(self):
        f = self.function
        if f.is_LocalObject:
            ret = set(flatten(i.free_symbols for i in f.cargs))
            try:
                ret.update(f.initvalue.free_symbols)
            except AttributeError:
                pass
            return tuple(ret)
        elif f.is_Array and f.initvalue is not None:
            # These are just a handful of values so it's OK to iterate them over
            ret = set()
            for i in f.initvalue:
                try:
                    ret.update(i.free_symbols)
                except AttributeError:
                    pass
            return tuple(ret)
        else:
            return ()


class PointerCast(ExprStmt, Node):

    """
    A node encapsulating a cast of a raw void pointer to a non-void,
    potentially multi-dimensional array.
    """

    is_PointerCast = True

    def __init__(self, function, obj=None, alignment=True, flat=None):
        self.function = function
        self.obj = obj
        self.alignment = alignment
        self.flat = flat

    def __repr__(self):
        return "<PointerCast(%s)>" % self.function

    @property
    def castshape(self):
        """
        The shape used in the left-hand side and right-hand side of the PointerCast.
        """
        if self.function.is_ArrayBasic:
            return self.function.symbolic_shape[1:]
        else:
            return tuple(self.function._C_get_field(FULL, d).size
                         for d in self.function.dimensions[1:])

    @property
    def functions(self):
        return (self.function,)

    @property
    def expr_symbols(self):
        f = self.function
        if not self.flat and f.is_ArrayBasic:
            return tuple(flatten(s.free_symbols for s in f.symbolic_shape[1:]))
        else:
            return ()

    @property
    def defines(self):
        if isinstance(self.obj, IndexedBase):
            return (self.obj,)
        else:
            return (self.function.indexed,)


class Dereference(ExprStmt, Node):

    """
    A node encapsulating a dereference from a `pointer` to a `pointee`.
    The following cases are supported:

        * `pointer` is a PointerArray or TempFunction, and `pointee` is an Array.
        * `pointer` is an ArrayObject representing a pointer to a C struct, and
          `pointee` is a field in `pointer`.
        * `pointer` is a Symbol with its _C_ctype deriving from ct._Pointer, and
          `pointee` is a Symbol representing the dereferenced value.
    """

    is_Dereference = True

    def __init__(self, pointee, pointer, flat=None):
        self.pointee = pointee
        self.pointer = pointer
        self.flat = flat

    def __repr__(self):
        return "<Dereference(%s,%s)>" % (self.pointee, self.pointer)

    @property
    def functions(self):
        return (self.pointee, self.pointer)

    @property
    def expr_symbols(self):
        ret = []
        if self.pointer.is_Symbol:
            assert issubclass(self.pointer._C_ctype, ctypes._Pointer), \
                   "Scalar dereference must have a pointer ctype"
            ret.extend([self.pointer._C_symbol, self.pointee._C_symbol])
        elif self.pointer.is_PointerArray or self.pointer.is_TempFunction:
            ret.extend([self.pointer.indexed, self.pointee.indexed])
            ret.extend(flatten(i.free_symbols
                               for i in self.pointee.symbolic_shape[1:]))
            ret.extend(self.pointer.free_symbols)
        else:
            ret.extend([self.pointer.indexed, self.pointee._C_symbol])
        return tuple(filter_ordered(ret))

    @property
    def defines(self):
        if self.pointer.is_PointerArray or \
           self.pointer.is_TempFunction or \
           self.pointee._mem_stack:
            return (self.pointee.indexed, self.pointee)
        else:
            return (self.pointee,)


class Lambda(Node):

    """
    A callable C++ lambda function. Several syntaxes are possible; here we
    implement one of the common ones:

        [captures](parameters){body} SPECIAL [[attributes]]

    For more info about C++ lambda functions:

        https://en.cppreference.com/w/cpp/language/lambda

    Parameters
    ----------
    body : Node or list of Node
        The lambda function body.
    captures : list of str or expr-like, optional
        The captures of the lambda function.
    parameters : list of Basic or expr-like, optional
        The objects in input to the lambda function.
    special : list of Basic, optional
        Placeholder for custom lambdas, to add in e.g. macros.
    attributes : list of str, optional
        The attributes of the lambda function.
    """

    _traversable = ['body']

    def __init__(self, body, captures=None, parameters=None, special=None,
                 attributes=None):
        self.body = as_tuple(body)
        self.captures = as_tuple(captures)
        self.parameters = as_tuple(parameters)
        self.special = as_tuple(special)
        self.attributes = as_tuple(attributes)

    def __repr__(self):
        return "Lambda[%s](%s)" % (self.captures, self.parameters)

    @property
    def functions(self):
        return tuple(i.function for i in self.parameters
                     if isinstance(i.function, AbstractFunction))

    @cached_property
    def expr_symbols(self):
        return tuple(self.parameters)

    @property
    def defines(self):
        return tuple(self.parameters)


class Section(List):

    """
    A sequence of nodes.

    Functionally, a Section is identical to a List; that is,
    they generate the same code (i.e., their ``body``). However, a Section should
    be used to define sub-trees that, for some reasons, have a relevance within
    the IET (e.g., groups of statements that logically represent the same
    computation unit).
    """

    is_Section = True

    def __init__(self, name, body=None, is_subsection=False):
        super().__init__(body=body)
        self.name = name
        self.is_subsection = is_subsection

    def __repr__(self):
        return "<Section (%s)>" % self.name

    @property
    def roots(self):
        return self.body


class ExpressionBundle(List):

    """
    A sequence of Expressions.
    """

    is_ExpressionBundle = True

    def __init__(self, ispace, ops, traffic, body=None):
        super().__init__(body=body)
        self.ispace = ispace
        self.ops = ops
        self.traffic = traffic

    def __repr__(self):
        return "<ExpressionBundle (%d)>" % len(self.exprs)

    @property
    def exprs(self):
        return self.body

    @property
    def size(self):
        return self.ispace.size


class Prodder(Call):

    """
    A Call promoting asynchronous progress, to minimize latency.

    Example use cases:

        * To trigger asynchronous progress in the case of distributed-memory
          parallelism.
        * Software prefetching.
    """

    def __init__(self, name, arguments=None, single_thread=False, periodic=False):
        super().__init__(name, arguments)

        # Prodder properties
        self._single_thread = single_thread
        self._periodic = periodic

    @property
    def single_thread(self):
        return self._single_thread

    @property
    def periodic(self):
        return self._periodic


class Using(Node):

    """
    A C++ using directive.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "<Using(%s)>" % self.name


class UsingNamespace(Node):

    """
    A C++ using namespace directive.
    """

    def __init__(self, namespace):
        self.namespace = namespace

    def __repr__(self):
        return "<UsingNamespace(%s)>" % self.namespace


class Pragma(Node):

    """
    One or more pragmas floating in the IET.
    """

    def __init__(self, pragma, arguments=None):
        super().__init__()

        if not isinstance(pragma, str):
            raise TypeError("Pragma name must be a string, not %s" % type(pragma))

        self.pragma = pragma
        self.arguments = as_tuple(arguments)

    def __repr__(self):
        return '<Pragma>'

    @cached_property
    def _generate(self):
        # Subclasses may override this property to customize the pragma generation
        return self.pragma % self.arguments


class Transfer:

    """
    An interface for Nodes that represent host-device data transfers.
    """

    @property
    def function(self):
        raise NotImplementedError

    @property
    def imask(self):
        raise NotImplementedError


class ParallelIteration(Iteration):

    """
    Implement a parallel for-loop.
    """

    pass


class ParallelTree(List):

    """
    This class is to group together a parallel for-loop with some setup
    statements, for example:

        .. code-block:: C

          int chunk_size = ...
          #pragma parallel for ... schedule(..., chunk_size)
          for (int i = ...)
          {
            ...
          }
    """

    _traversable = ['prefix', 'body']

    def __init__(self, prefix, body, nthreads=None):
        # Normalize and sanity-check input
        body = as_tuple(body)
        assert len(body) == 1 and body[0].is_Iteration

        self.prefix = as_tuple(prefix)
        self.nthreads = nthreads

        super().__init__(body=body)

    def __getattr__(self, name):
        if 'body' in self.__dict__:
            # During unpickling, `__setattr__` calls `__getattr__(..., 'body')`,
            # which would cause infinite recursion if we didn't check whether
            # 'body' is present or not
            return getattr(self.body[0], name)
        raise AttributeError

    @property
    def functions(self):
        return as_tuple(self.nthreads)

    @property
    def expr_symbols(self):
        return as_tuple(self.nthreads)

    @property
    def root(self):
        return self.body[0]


class ParallelBlock(Block):

    """
    A sequence of Nodes, wrapped in a parallel block {...}.
    """

    def __init__(self, body, private=None):
        # Normalize and sanity-check input. A bit ugly, but it makes everything
        # much simpler to manage and reconstruct
        body = as_tuple(body)
        assert len(body) == 1
        body = body[0]
        assert body.is_List
        if isinstance(body, ParallelTree):
            partree = body
        elif body.is_List:
            assert len(body.body) == 1 and isinstance(body.body[0], ParallelTree)
            assert len(body.footer) == 0
            partree = body.body[0]
            partree = partree._rebuild(prefix=(List(header=body.header,
                                                    body=partree.prefix)))

        header = self._make_header(partree.nthreads, private)
        super().__init__(header=header, body=partree)

    @classmethod
    def _make_header(cls, nthreads, private=None):
        return None

    @property
    def partree(self):
        return self.body[0]

    @property
    def root(self):
        return self.partree.root

    @property
    def nthreads(self):
        return self.partree.nthreads

    @property
    def collapsed(self):
        return self.partree.collapsed


class BusyWait(While):

    """
    A while-loop implementing a busy waiting.
    """

    pass


class SyncSpot(List):

    """
    A node representing one or more synchronization operations, e.g., WaitLock,
    withLock, etc.
    """

    is_SyncSpot = True

    def __init__(self, sync_ops, body=None):
        super().__init__(body=body)
        self.sync_ops = sync_ops

    def __repr__(self):
        return "<SyncSpot (%s)>" % ",".join(str(i) for i in self.sync_ops)

    @property
    def is_async_op(self):
        """
        True if the SyncSpot contains an asynchronous operation, False otherwise.
        If False, the SyncSpot may for example represent a wait on a lock.
        """
        return any(isinstance(s, (WithLock, PrefetchUpdate))
                   for s in self.sync_ops)

    @property
    def functions(self):
        ret = [(s.lock, s.function, s.target) for s in self.sync_ops]
        ret = tuple(filter_ordered(f for f in flatten(ret) if f is not None))
        return ret


class CBlankLine(List):

    def __init__(self, **kwargs):
        super().__init__(header=c.Line())

    def __repr__(self):
        return ""


class Break(Node):
    pass


class Return(Node):

    def __init__(self, value=None):
        self.value = value


class ListMajor(List):
    pass


def DummyExpr(*args, init=False):
    return Expression(DummyEq(*args), init=init)


BlankLine = CBlankLine()


# Nodes required for distributed-memory halo exchange


class HaloSpot(Node):

    """
    A halo exchange operation (e.g., send, recv, wait, ...) required to
    correctly execute the subtree in the case of distributed-memory parallelism.
    """

    is_HaloSpot = True

    _traversable = ['body']

    def __init__(self, body, halo_scheme):
        super().__init__()

        if isinstance(body, Node):
            self._body = body
        elif isinstance(body, (list, tuple)) and len(body) == 1:
            self._body = body[0]
        elif body is None:
            self._body = List()
        else:
            raise ValueError("`body` is expected to be a single Node")

        self._halo_scheme = halo_scheme

    def __repr__(self):
        functions = "(%s)" % ",".join(i.name for i in self.functions)
        return "<%s%s>" % (self.__class__.__name__, functions)

    @property
    def halo_scheme(self):
        return self._halo_scheme

    @property
    def fmapper(self):
        return self.halo_scheme.fmapper

    @property
    def omapper(self):
        return self.halo_scheme.omapper

    @property
    def dimensions(self):
        return self.halo_scheme.dimensions

    @property
    def arguments(self):
        return self.halo_scheme.arguments

    @property
    def is_void(self):
        return self.halo_scheme.is_void

    @property
    def body(self):
        return self._body

    @property
    def functions(self):
        return tuple(self.fmapper)


# Utility classes


MetaCall = namedtuple('MetaCall', 'root local')
"""
Metadata for Callables. ``root`` is a pointer to the callable
Iteration/Expression tree. ``local`` is a boolean indicating whether the
definition of the callable is known or not.
"""
