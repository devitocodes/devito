from collections import Counter, OrderedDict, namedtuple
from functools import partial, wraps

from devito.ir.iet import (Call, FindNodes, FindSymbols, MetaCall, Transformer,
                           EntryFunction, ThreadCallable, Uxreplace,
                           derive_parameters)
from devito.tools import DAG, as_tuple, filter_ordered, timed_pass
from devito.types import CompositeObject, Lock
from devito.types.args import ArgProvider
from devito.types.dense import AliasFunction

__all__ = ['Graph', 'iet_pass', 'Jitting']


class Graph(object):

    """
    A special DAG representing call graphs.

    The nodes of the graph are IET Callables; an edge from node `a` to node `b`
    indicates that `b` calls `a`.

    The `apply` method may be used to visit the Graph and apply a transformer `T`
    to all nodes. This may change the state of the Graph: node `a` gets replaced
    by `a' = T(a)`; new nodes (Callables), and therefore new edges, may be added.

    The `visit` method collects info about the nodes in the Graph.
    """

    def __init__(self, iet):
        self.efuncs = OrderedDict([(iet.name, iet)])

        self.includes = []
        self.headers = []

    @property
    def root(self):
        return self.efuncs[list(self.efuncs).pop(0)]

    @property
    def funcs(self):
        return tuple(MetaCall(v, True) for v in self.efuncs.values())[1:]

    def apply(self, func, **kwargs):
        """
        Apply `func` to all nodes in the Graph. This changes the state of the Graph.
        """
        dag = create_call_graph(self.root.name, self.efuncs)

        # Apply `func`
        for i in dag.topological_sort():
            efunc, metadata = func(self.efuncs[i], **kwargs)

            self.includes.extend(as_tuple(metadata.get('includes')))
            self.headers.extend(as_tuple(metadata.get('headers')))

            # Update jit-compiler if necessary
            try:
                jitting = metadata['jitting']
                self.includes.extend(jitting.includes)

                compiler = kwargs['compiler']
                compiler.add_include_dirs(jitting.include_dirs)
                compiler.add_libraries(jitting.libs)
                compiler.add_library_dirs(jitting.lib_dirs)
            except KeyError:
                pass

            if efunc is self.efuncs[i]:
                continue

            # Minimize code size by abstracting semantically identical efuncs
            efunc, efuncs = reuse_efuncs(efunc, metadata.get('efuncs', []))

            self.efuncs[i] = efunc
            self.efuncs.update(OrderedDict([(i.name, i) for i in efuncs]))

            # Update the parameters / arguments lists since `func` may have
            # introduced or removed objects
            self.efuncs = update_args(efunc, self.efuncs, dag)

        # Uniqueness
        self.includes = filter_ordered(self.includes)
        self.headers = filter_ordered(self.headers, key=str)

    def visit(self, func, **kwargs):
        """
        Apply `func` to all nodes in the Graph. `func` gathers info about the
        state of each node. The gathered info is returned to the called as a mapper
        from nodes to info. Unlike `apply`, `visit` does not change the state
        of the Graph.
        """
        dag = create_call_graph(self.root.name, self.efuncs)
        toposort = dag.topological_sort()

        mapper = OrderedDict([(i, func(self.efuncs[i], **kwargs)) for i in toposort])

        return mapper


def iet_pass(func):
    if isinstance(func, tuple):
        assert len(func) == 2 and func[0] is iet_visit
        call = lambda graph: graph.visit
        func = func[1]
    else:
        call = lambda graph: graph.apply

    @wraps(func)
    def wrapper(*args, **kwargs):
        if timed_pass.is_enabled():
            maybe_timed = timed_pass
        else:
            maybe_timed = lambda func, name: func
        try:
            # Pure function case
            graph, = args
            return maybe_timed(call(graph), func.__name__)(func, **kwargs)
        except ValueError:
            # Instance method case
            self, graph = args
            return maybe_timed(call(graph), func.__name__)(partial(func, self), **kwargs)
    return wrapper


def iet_visit(func):
    return iet_pass((iet_visit, func))


Jitting = namedtuple('Jitting', 'includes include_dirs libs lib_dirs')


def create_call_graph(root, efuncs):
    """
    Create a Call graph -- a Direct Acyclic Graph with edges from callees
    to callers.
    """
    dag = DAG(nodes=[root])
    queue = [root]

    while queue:
        caller = queue.pop(0)
        callees = FindNodes(Call).visit(efuncs[caller])

        for callee in filter_ordered([i.name for i in callees]):
            if callee in efuncs:  # Exclude foreign Calls, e.g., MPI calls
                try:
                    dag.add_node(callee)
                    queue.append(callee)
                except KeyError:
                    # `callee` already in `dag`
                    pass
                dag.add_edge(callee, caller)

    # Sanity check
    assert dag.size == len(efuncs)

    return dag


def reuse_efuncs(root, efuncs):
    """
    Generalise `efuncs` so that syntactically identical Callables may be dropped,
    thus maximizing code reuse.

    For example, given two Callables

        foo0(u(x)) : u(x)**2
        foo1(v(x)) : v(x)**2

    Reduce them to one single Callable

        foo0(a(x)) : a(x)**2

    The call sites in `root` are transformed accordingly.
    """
    # Topological sorting ensures that nested Calls are abstract first.
    # For example, given `[foo0(u(x)): bar0(u), foo1(u(x)): bar1(u)]`,
    # assuming that `bar0` and `bar1` are compatible, we first process the
    # `bar`'s to obtain `[foo0(u(x)): bar0(u), foo1(u(x)): bar0(u)]`,
    # and finally `foo0(u(x)): bar0(u)`
    efuncs = {i.name: i for i in efuncs}
    efuncs[root.name] = root
    dag = create_call_graph(root.name, efuncs)

    mapper = {}
    for i in dag.topological_sort():
        if i == root.name:
            continue

        efunc = efuncs[i]
        afunc = abstract_efunc(efunc)

        key = afunc._signature()

        try:
            # If we manage to succesfully map `efunc` to a previously abstracted
            # `afunc`, we need to update the call sites to use the new Call name
            afunc, mapped = mapper[key]
            mapped.append(efunc)

            for n in dag.downstream(i):
                subs = {c: c._rebuild(name=afunc.name)
                        for c in FindNodes(Call).visit(efuncs[n])
                        if c.name == efuncs[i].name}
                efuncs[n] = Transformer(subs).visit(efuncs[n])

        except KeyError:
            afunc = afunc._rebuild(name=efunc.name)
            mapper[key] = (afunc, [efunc])

    root = efuncs.pop(root.name)
    processed = [afunc if len(efuncs) > 1 else efuncs.pop()
                 for afunc, efuncs in mapper.values()]

    return root, processed


def abstract_efunc(efunc):
    """
    Abstract `efunc` applying a set of rules:

        * The `efunc` names becomes "foo".
        * Any concrete AbstractFunction gets replaced with a "more abstract" object:
            - DiscreteFunctions become AliasFunctions with name "f0", "f1", ...
            - Arrays remain Arrays but are renamed as "a0", "a1", ...
        * Objects remain Objects but are renamed as "o0", "o1", ...
    """
    mapper = {}
    counter = Counter()
    for i in FindSymbols().visit(efunc):
        if i.is_DiscreteFunction:
            base = 'f'

            kwargs = {k: getattr(i, k, None) for k in i.__rkwargs__}
            kwargs.pop('initializer')
            kwargs['name'] = '%s%d' % (base, counter[base])
            v = AliasFunction(**kwargs)

            mapper.update({
                i: v,
                i.indexed: v.indexed,
                i._C_symbol: v._C_symbol,
            })

        elif i.is_Array:
            # Use special names for special objects
            if isinstance(i, Lock):
                base = 'lock'
            else:
                base = 'a'

            kwargs = {k: getattr(i, k, None) for k in i.__rkwargs__}
            kwargs['name'] = '%s%d' % (base, counter[base])
            v = type(i).__base__(**kwargs)

            mapper.update({
                i: v,
                i.indexed: v.indexed,
                i._C_symbol: v._C_symbol,
            })

        elif isinstance(i, CompositeObject):
            base = 'o'

            args = [getattr(i, k, None) for k in i.__rargs__]
            args[0] = '%s%d' % (base, counter[base])
            kwargs = {k: getattr(i, k, None) for k in i.__rkwargs__}
            v = i.func(*args, **kwargs)

            mapper[i] = v

        else:
            base = i.name

            v = i

        counter[base] += 1

    efunc = Uxreplace(mapper).visit(efunc)
    efunc = efunc._rebuild(name='foo')

    return efunc


def update_args(root, efuncs, dag):
    """
    Re-derive the parameters of `root` and apply the changes in cascade through
    the `efuncs`.

    For example, given the Callable `root`:

        root(x, y) : x + z

    The re-derivation detects that the formal parameter `y` is now unused
    within `root.body`, while a free symbol `z` appears in an expression. Thus,
    `root` is reconstructed as:

        root(x, z) : x + z

    Now assume among `efuncs` there's a Callable `foo` that calls `root`:

        foo(...) : z = 5; root(x, y)

    Then `foo` will be reconstructed as:

        foo(...) : z = 5; root(x, z)

    If instead `foo` were as below:

        foo(...) : root(x, y)

    That is, no definition for `z` available, then it would be reconstructed as:

        foo(..., z) : root(x, z)
    """
    if isinstance(root, ThreadCallable):
        return efuncs

    # The parameters/arguments lists may have changed since a pass may have:
    # 1) introduced a new symbol
    new_args = derive_parameters(root)
    new_args = [a for a in new_args if not a._mem_internal_eager]

    # 2) defined a symbol for which no definition was available yet (e.g.
    # via a malloc, or a Dereference)
    defines = FindSymbols('defines').visit(root.body)
    drop_args = [a for a in root.parameters if a in defines]

    if not (new_args or drop_args):
        return efuncs

    def _filter(v, efunc=None):
        processed = list(v)
        for a in new_args:
            if a in processed:
                # A child efunc trying to add a symbol alredy added by a
                # sibling efunc
                continue

            if isinstance(efunc, EntryFunction) and not isinstance(a, ArgProvider):
                # Temporaries (e.g., Arrays) *cannot* be arguments of an
                # EntryFunction. So if we end up here, `a` remains for
                # now undefined inside `efunc`
                continue

            processed.append(a)

        processed = [a for a in processed if a not in drop_args]

        return processed

    efuncs = OrderedDict(efuncs)

    # Update to use the new signature
    parameters = _filter(root.parameters, root)
    efuncs[root.name] = root._rebuild(parameters=parameters)

    # Update all call sites to use the new signature
    for n in dag.downstream(root.name):
        mapper = {c: c._rebuild(arguments=_filter(c.arguments))
                  for c in FindNodes(Call).visit(efuncs[n])
                  if c.name == root.name}
        efuncs[n] = Transformer(mapper).visit(efuncs[n])

    return efuncs
