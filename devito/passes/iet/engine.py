from collections import defaultdict
from functools import partial, singledispatch, wraps

import numpy as np
from sympy import Mul

from devito.ir.iet import (
    Call, ExprStmt, Expression, Iteration, SyncSpot, AsyncCallable, FindNodes,
    FindSymbols, MapNodes, MetaCall, Transformer, EntryFunction, ThreadCallable,
    Uxreplace, derive_parameters
)
from devito.ir.support import SymbolRegistry
from devito.mpi.distributed import MPINeighborhood
from devito.mpi.routines import Gather, Scatter, HaloUpdate, HaloWait, MPIMsg
from devito.passes import needs_transfer
from devito.symbolics import (FieldFromComposite, FieldFromPointer, IndexedPointer,
                              search)
from devito.tools import DAG, as_tuple, filter_ordered, sorted_priority, timed_pass
from devito.types import (
    Array, Bundle, ComponentAccess, CompositeObject, Lock, IncrDimension,
    ModuloDimension, Indirection, Pointer, SharedData, ThreadArray, Symbol, Temp,
    NPThreads, NThreadsBase, Wildcard
)
from devito.types.args import ArgProvider
from devito.types.dense import DiscreteFunction
from devito.types.dimension import AbstractIncrDimension, BlockDimension

__all__ = ['Graph', 'iet_pass', 'iet_visit']


class Byproduct:

    """
    A Byproduct is a mutable collection of metadata produced by one or more
    compiler passes.

    This metadata may be used internally or by the caller itself, typically
    for code generation purposes.
    """

    def __init__(self, efuncs=None, includes=None, headers=None, namespaces=None,
                 globs=None):
        self.efuncs = efuncs or {}
        self.includes = includes or []
        self.headers = headers or []
        self.namespaces = namespaces or []
        self.globals = globs or []

    @property
    def funcs(self):
        return tuple(MetaCall(v, True) for v in self.efuncs.values()
                     if not isinstance(v, EntryFunction))


class Graph(Byproduct):

    """
    DAG representation of a call graph.

    The nodes of the DAG are IET Callables.
    An edge from node `a` to node `b` indicates that `b` calls `a`.

    The `apply` method visits the Graph and applies a transformation `T`
    to all nodes. This may change the state of the Graph:

        * Node `a` gets replaced by `a' = T(a)`.
        * New nodes (Callables) may be added.
          * Consequently, edges are added.
        * New global objects may be introduced:
          * Global symbols, header files, ...

    The `visit` method collects info about the nodes in the Graph.
    """

    def __init__(self, iet, options=None, sregistry=None, **kwargs):
        self.sregistry = sregistry

        super().__init__({iet.name: iet})

        # All written user-level objects
        writes = FindSymbols('writes').visit(iet)
        self.writes_input = frozenset(f for f in writes if f.is_Input)

        # All symbols requiring host-device data transfers when running
        # on device
        self.data_movs = rmovs, wmovs = set(), set()
        gpu_fit = (options or {}).get('gpu-fit', ())
        for i in FindNodes(ExprStmt).visit(iet):
            wmovs.update({w for w in i.writes
                          if needs_transfer(w, gpu_fit)})
        for i in FindNodes(ExprStmt).visit(iet):
            rmovs.update({f for f in i.functions
                          if needs_transfer(f, gpu_fit) and f not in wmovs})

    @property
    def root(self):
        return self.efuncs[list(self.efuncs).pop(0)]

    @property
    def sync_mapper(self):
        """
        A mapper {Iteration -> SyncSpot} describing the Iterations, if any,
        living an asynchronous region, across all Callables in the Graph.
        """
        dag = create_call_graph(self.root.name, self.efuncs)

        mapper = MapNodes(SyncSpot, (Iteration, Call)).visit(self.root)

        found = {}
        for k, v in mapper.items():
            if not k.is_async_op:
                continue

            while v:
                i = v.pop(0)

                if i.is_Iteration:
                    found[i] = k
                    continue

                for j in dag.all_predecessors(i.name):
                    try:
                        v.extend(FindNodes(Iteration).visit(self.efuncs[j]))
                    except KeyError:
                        # `j` is a foreign Callable
                        pass

        return found

    def apply(self, func, **kwargs):
        """
        Apply `func` to all nodes in the Graph. This changes the state of the Graph.
        """
        dag = create_call_graph(self.root.name, self.efuncs)

        # Apply `func`
        efuncs = dict(self.efuncs)
        for i in dag.topological_sort():
            efunc, metadata = func(efuncs[i], **kwargs)

            self.includes.extend(as_tuple(metadata.get('includes')))
            self.headers.extend(as_tuple(metadata.get('headers')))
            self.namespaces.extend(as_tuple(metadata.get('namespaces')))
            self.globals.extend(as_tuple(metadata.get('globals')))

            # Update jit-compiler if necessary
            try:
                compiler = kwargs['compiler']
                compiler.add_include_dirs(as_tuple(metadata.get('include_dirs')))
                compiler.add_libraries(as_tuple(metadata.get('libs')))
                compiler.add_library_dirs(as_tuple(metadata.get('lib_dirs')),
                                          rpath=metadata.get('rpath', False))
            except KeyError:
                pass

            if efunc is efuncs[i]:
                continue

            new_efuncs = metadata.get('efuncs', [])

            efuncs[i] = efunc
            efuncs.update(dict([(i.name, i) for i in new_efuncs]))

            # Update the parameters / arguments lists since `func` may have
            # introduced or removed objects
            efuncs = update_args(efunc, efuncs, dag)

        # Minimize code size
        if len(efuncs) > len(self.efuncs):
            efuncs = reuse_compounds(efuncs, self.sregistry)
            efuncs = abstract_component_accesses(efuncs)
            efuncs = reuse_efuncs(self.root, efuncs, self.sregistry)

        self.efuncs = efuncs

        # Uniqueness
        self.includes = filter_ordered(self.includes)
        self.headers = filter_ordered(self.headers, key=str)
        self.namespaces = filter_ordered(self.namespaces, key=str)
        self.globals = filter_ordered(self.globals)

    def visit(self, func, **kwargs):
        """
        Apply `func` to all nodes in the Graph. `func` gathers info about the
        state of each node. The gathered info is returned to the called as a mapper
        from nodes to info. Unlike `apply`, `visit` does not change the state
        of the Graph.
        """
        dag = create_call_graph(self.root.name, self.efuncs)
        toposort = dag.topological_sort()

        mapper = dict([(i, func(self.efuncs[i], **kwargs)) for i in toposort])

        return mapper

    def filter(self, key):
        """
        Return a Byproduct containing only the Callables in the Graph
        for which `key` evaluates to True. The resulting object cannot be
        further modified by an IET pass.
        """
        return Byproduct(
            efuncs={i: v for i, v in self.efuncs.items() if key(v)},
            includes=as_tuple(self.includes),
            headers=as_tuple(self.headers),
            namespaces=as_tuple(self.namespaces),
            globs=as_tuple(self.globals)
        )


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
            # If the pass has been disabled, skip it
            if not kwargs['options'][func.__name__]:
                return
        except KeyError:
            pass
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


def reuse_compounds(efuncs, sregistry=None):
    """
    Generalise `efuncs` so that groups of semantically identical compound types
    are replaced with a unique compound type, thus maximizing code reuse.

    For example, given two C structs originating from e.g. a CompositeObject

        struct foo {int a, char v}
        struct bar {int g, char e}

    Reduce them to:

        struct foo {int a, char v}

    Which requires replacing all references to `bar` with the new `foo`. Note that
    in this case the transformed `foo` is also syntactically identical to the
    input `foo`, but this isn't necessarily the case.
    """
    # Build {key -> [objs]} where [objs] can be, or are already, mapped to the
    # same abstract compound type
    candidates = defaultdict(list)
    for efunc in efuncs.values():
        for i in FindSymbols().visit(efunc):
            a = abstract_compound(i)
            if a is not None:
                candidates[a._C_ctype].append(i)

    subs = {}
    for v in candidates.values():
        # Do they actually need to be remapped to a common type or do they
        # already share the same type?
        ctypes = {i._C_ctype for i in v}
        if len(ctypes) == 1:
            continue

        base = abstract_compound(v[0], sregistry)

        subs.update({i: base._rebuild(name=i.name) for i in v})

    # Replace all occurrences in the form of FieldFrom{Composite,Pointer}
    mapper = {}
    for i0, i1 in subs.items():
        b0, b1 = i0.indexed, i1.indexed

        mapper.update({i0: i1, b0: b1})

        for f0, f1 in zip(i0.fields, i1.fields):
            for cls in (FieldFromComposite, FieldFromPointer):
                if f0.is_AbstractFunction:
                    mapper[cls(f0._C_symbol, b0)] = cls(f1._C_symbol, b1)
                else:
                    mapper[cls(f0.base, b0)] = cls(f1.base, b1)

    if mapper:
        efuncs = {i: Uxreplace(mapper).visit(efunc) for i, efunc in efuncs.items()}

    return efuncs


@singledispatch
def abstract_compound(i, sregistry=None):
    """
    Singledispatch-based implementation of type abstraction.
    """
    return


@abstract_compound.register(SharedData)
def _(i, sregistry=None):
    try:
        pname = sregistry.make_name(prefix='tsd')
    except AttributeError:
        pname = 'tsd'

    m = abstract_objects(i.fields)
    cfields = tuple(m.get(i, i) for i in i.cfields)
    ncfields = tuple(m.get(i, i) for i in i.ncfields)

    return i._rebuild(pname=pname, cfields=cfields, ncfields=ncfields, function=None)


def abstract_component_accesses(efuncs):
    """
    Generalise `efuncs` by replacing ComponentAccesses with pointer arithmetic
    where possible and useful.

    This pass is only applied to selected Callables:

        1) HaloUpdate/Gather
        2) HaloWait/Scatter
    """
    processed = dict(efuncs)

    for k, efunc in efuncs.items():
        if not isinstance(efunc, (HaloUpdate, HaloWait)):
            continue

        # Retrieve expected objects. If this fails, it means it's a structure
        # we don't recognize (e.g., an unsupported MPI scheme?), so we just
        # give up and move on
        try:
            call_copy_buffer, = FindNodes((Gather, Scatter)).visit(efunc)
            copy_buffer = efuncs[call_copy_buffer.name]
            f, = [i for i in copy_buffer.parameters if i.is_Bundle]
            msg, = [i for i in efunc.parameters if isinstance(i, MPIMsg)]
            dim, = FindSymbols('dimensions').visit(efunc)
        except ValueError:
            continue

        exprs = FindNodes(Expression).visit(copy_buffer)
        compaccs = set().union(*[search(i.expr, ComponentAccess) for i in exprs])
        if not compaccs or len(compaccs) == f.ncomp:
            continue

        # Sorted for deterministic codegen
        compaccs = sorted(compaccs, key=lambda i: i.index)

        # Access the same entry as `ca` but via pointer arithmetic instead
        arity_param = Symbol(name='arity', dtype=np.uint32, is_const=True)
        compoff_params = [Symbol(name=f'c{i}', dtype=np.uint32, is_const=True)
                          for i in range(len(compaccs))]

        f_flatten = f.func(name='flat_data', components=f.c0)

        subs = {}
        for ca, o in zip(compaccs, compoff_params):
            indices = [Mul(arity_param, i, evaluate=False) for i in ca.indices]
            indices[-1] += o
            subs[ca] = f_flatten.indexed[indices]

        # Transform the `copy_buffer` to use a flatten representation of the Bundle
        copy_buffer1 = Uxreplace(subs).visit(copy_buffer)
        parameters = [*copy_buffer1.parameters, arity_param, *compoff_params]
        parameters[parameters.index(f)] = f_flatten
        copy_buffer1 = copy_buffer1._rebuild(parameters=parameters)

        # Update the `efunc` to use the new `copy_buffer1` version
        arity_arg = FieldFromPointer(f._C_field_arity, f._C_symbol)
        compoff_args = [IndexedPointer(
            FieldFromComposite(msg._C_field_components, IndexedPointer(msg, dim)), i
        ) for i in range(len(compoff_params))]

        arguments = [*call_copy_buffer.arguments, arity_arg, *compoff_args]
        call_copy_buffer1 = call_copy_buffer._rebuild(arguments=arguments)
        efunc1 = Transformer({call_copy_buffer: call_copy_buffer1}).visit(efunc)

        processed.update({k: efunc1, copy_buffer.name: copy_buffer1})

    return processed


def reuse_efuncs(root, efuncs, sregistry=None):
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
    dag = create_call_graph(root.name, efuncs)

    mapper = {}
    for i in dag.topological_sort():
        if i == root.name:
            continue

        efunc = efuncs[i]

        # Avoid premature lowering of AsyncCalls -- it would obscenely
        # complicate the logic of the `pthredify` pass
        if isinstance(efunc, AsyncCallable):
            mapper[len(mapper)] = (efunc, [efunc])
            continue

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

    processed = [afunc if len(v) > 1 else v.pop() for afunc, v in mapper.values()]

    retval = {root.name: efuncs[root.name]}
    retval.update({i.name: i for i in processed})

    return retval


def abstract_efunc(efunc):
    """
    Abstract `efunc` applying a set of rules:

        * The `efunc` names becomes "foo".
        * Symbolic objects get replaced with "more abstract" objects:
            - DiscreteFunctions are renamed as "f0", "f1", ...
            - Arrays are renamed as "a0", "a1", ...
            - Objects are renamed as "o0", "o1", ...
    """
    functions = FindSymbols('basics|symbolics|dimensions').visit(efunc)

    mapper = abstract_objects(functions)

    efunc = Uxreplace(mapper).visit(efunc)
    efunc = efunc._rebuild(name='foo')

    return efunc


def abstract_objects(objects0, sregistry=None):
    """
    Proxy for `abstract_object`.
    """
    # Expose hidden objects for complete reconstruction
    objects = []
    for i in objects0:
        if i.is_Bundle:
            objects.extend(i.components)
        objects.append(i)

    # Precedence rules make it possible to reconstruct objects that depend on
    # higher priority objects
    keys = [Bundle, Array, DiscreteFunction, AbstractIncrDimension, BlockDimension]
    priority = {k: i for i, k in enumerate(keys, start=1)}
    objects = sorted_priority(objects, priority)

    # Build abstraction mappings
    mapper = {}
    sregistry = sregistry or SymbolRegistry()
    for i in objects:
        abstract_object(i, mapper, sregistry)

    return mapper


@singledispatch
def abstract_object(i, mapper, sregistry):
    """
    Singledispatch-based implementation of object abstraction.
    """
    return


@abstract_object.register(DiscreteFunction)
def _(i, mapper, sregistry):
    name = sregistry.make_name(prefix='f')

    v = i._rebuild(name=name, initializer=None, alias=True)

    mapper.update({
        i: v,
        i.indexed: v.indexed,
        i.dmap: v.dmap,
        i._C_symbol: v._C_symbol,
    })


@abstract_object.register(Array)
def _(i, mapper, sregistry):
    if isinstance(i, Lock):
        name = sregistry.make_name(prefix='lock')
    else:
        name = sregistry.make_name(prefix='a')

    v = i._rebuild(name=name, alias=True)

    mapper.update({
        i: v,
        i.indexed: v.indexed,
        i._C_symbol: v._C_symbol,
    })
    if i.dmap is not None:
        mapper[i.dmap] = v.dmap


@abstract_object.register(Bundle)
def _(i, mapper, sregistry):
    name = sregistry.make_name(prefix='a')
    components = [mapper[f] for f in i.components]

    v = i._rebuild(name=name, components=components, alias=True)

    mapper.update({
        i: v,
        i.indexed: v.indexed,
        i._C_symbol: v._C_symbol,
    })
    if i.dmap is not None:
        mapper[i.dmap] = v.dmap


@abstract_object.register(CompositeObject)
def _(i, mapper, sregistry):
    name = sregistry.make_name(prefix='o')

    v = i._rebuild(name)

    mapper[i] = v


@abstract_object.register(ThreadArray)
def _(i, mapper, sregistry):
    if isinstance(i, SharedData):
        name = sregistry.make_name(prefix='sdata')
    else:
        name = sregistry.make_name(prefix='threads')

    v = i._rebuild(name=name)

    mapper.update({
        i: v,
        i.indexed: v.indexed,
        i._C_symbol: v._C_symbol,
    })


@abstract_object.register(MPINeighborhood)
def _(i, mapper, sregistry):
    mapper[i] = i._rebuild()


@abstract_object.register(BlockDimension)
def _(i, mapper, sregistry):
    if i._depth != 2:
        return

    p = i.parent
    pp = i.parent.parent

    name0 = pp.name
    base = sregistry.make_name(prefix=name0)
    name1 = sregistry.make_name(prefix='%s_blk' % base)

    bd = i.parent._rebuild(name1, pp)
    d = i._rebuild(name0, bd, i._min.subs(p, bd), i._max.subs(p, bd))

    mapper.update({
        i: d,
        i.parent: bd
    })


@abstract_object.register(IncrDimension)
@abstract_object.register(ModuloDimension)
def _(i, mapper, sregistry):
    try:
        p = mapper[i.parent]
    except KeyError:
        return

    v = i._rebuild(i.name, p)

    mapper[i] = v


@abstract_object.register(Indirection)
def _(i, mapper, sregistry):
    mapper[i] = i._rebuild(name=sregistry.make_name(prefix='ind'))


@abstract_object.register(Temp)
@abstract_object.register(Wildcard)
def _(i, mapper, sregistry):
    mapper[i] = i._rebuild(name=sregistry.make_name(prefix='r'))


@abstract_object.register(Pointer)
def _(i, mapper, sregistry):
    mapper[i] = i._rebuild(name=sregistry.make_name(prefix='ptr'))


@abstract_object.register(NPThreads)
def _(i, mapper, sregistry):
    mapper[i] = i._rebuild(name=sregistry.make_name(prefix='npthreads'))


@abstract_object.register(NThreadsBase)
def _(i, mapper, sregistry):
    mapper[i] = i._rebuild(name=sregistry.make_name(prefix='nthreads'))


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
    new_params = derive_parameters(root)

    # 2) defined a symbol for which no definition was available yet (e.g.
    # via a malloc, or a Dereference)
    defines = FindSymbols('defines').visit(root.body)
    drop_params = [a for a in root.parameters if a in defines]

    # 3) removed a symbol that was previously necessary (e.g., `x_size` after
    # linearization)
    symbols = FindSymbols('basics').visit(root.body)
    drop_params.extend(a for a in root.parameters
                       if (a.is_Symbol or a.is_LocalObject) and a not in symbols)

    # 4) removed a function that was previously necessary
    functions = FindSymbols('symbolics').visit(root.body)
    drop_params.extend(a for a in root.parameters
                       if a.is_AbstractFunction and a not in functions)

    # Must record the index, not the param itself, since a param may be
    # bound to whatever arg, possibly a generic SymPy expr
    drop_params = [root.parameters.index(a) for a in drop_params]

    if not (new_params or drop_params):
        return efuncs

    # Create the new parameters and arguments lists

    def _filter(v, efunc=None):
        processed = [a for i, a in enumerate(v) if i not in drop_params]

        for a in new_params:
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

        return processed

    efuncs = dict(efuncs)
    efuncs[root.name] = root._rebuild(parameters=_filter(root.parameters, root))

    # Update all call sites to use the new signature
    for n in dag.downstream(root.name):
        mapper = {c: c._rebuild(arguments=_filter(c.arguments))
                  for c in FindNodes(Call).visit(efuncs[n])
                  if c.name == root.name}
        efuncs[n] = Transformer(mapper).visit(efuncs[n])

    return efuncs
