from collections import OrderedDict, namedtuple
from functools import partial, wraps

from devito.ir.iet import (Call, FindNodes, FindSymbols, MetaCall, Transformer,
                           DeviceFunction, EntryFunction, SharedDataInitFunction,
                           ThreadFunction, Uxreplace, derive_parameters)
from devito.tools import DAG, as_tuple, filter_ordered, timed_pass
from devito.types.args import ArgProvider

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
        self.rname = iet.name

        # Internal "known" functions
        self.efuncs = OrderedDict([(self.rname, iet)])

        # Foreign functions
        self.ffuncs = []

        self.includes = []
        self.headers = []

    @property
    def root(self):
        return self.efuncs[self.rname]

    @property
    def funcs(self):
        retval = [MetaCall(v, True) for k, v in self.efuncs.items() if k != self.rname]
        retval.extend([MetaCall(i, False) for i in self.ffuncs])
        return tuple(retval)

    def _create_call_graph(self):
        dag = DAG(nodes=[self.rname])
        queue = [self.rname]
        while queue:
            caller = queue.pop(0)
            callees = FindNodes(Call).visit(self.efuncs[caller])
            for callee in filter_ordered([i.name for i in callees]):
                try:
                    n = self.efuncs[callee]
                except KeyError:
                    # E.g., foreign Calls such as MPI calls
                    continue

                if isinstance(n, ThreadFunction):
                    # ThreadFunctions aren't called directly via a Call
                    dag.add_edge(callee, n.idata_function.name, force_add=True)
                    queue.append(callee)
                    continue

                try:
                    dag.add_node(callee)
                    queue.append(callee)
                except KeyError:
                    # `callee` already in `dag`
                    pass
                dag.add_edge(callee, caller)

        # Sanity check
        assert dag.size == len(self.efuncs)

        return dag

    def apply(self, func, **kwargs):
        """
        Apply `func` to all nodes in the Graph. This changes the state of the Graph.
        """
        dag = self._create_call_graph()

        # Apply `func`
        for i in dag.topological_sort():
            efunc, metadata = func(self.efuncs[i], **kwargs)

            # Track all objects introduced by `func`
            self.includes.extend(as_tuple(metadata.get('includes')))
            self.headers.extend(as_tuple(metadata.get('headers')))
            self.ffuncs.extend(as_tuple(metadata.get('ffuncs', [])))
            self.efuncs.update(OrderedDict([(i.name, i)
                                            for i in metadata.get('efuncs', [])]))

            # Update compiler if necessary
            try:
                jitting = metadata['jitting']
                self.includes.extend(jitting.includes)

                compiler = kwargs['compiler']
                compiler.add_include_dirs(jitting.include_dirs)
                compiler.add_libraries(jitting.libs)
                compiler.add_library_dirs(jitting.lib_dirs)
            except KeyError:
                pass

            # ThreadFunctions are a special beast. They don't need an
            # explicit reconstruction of the parameters list (this is
            # carried out by the __init__ directly)
            if isinstance(efunc, ThreadFunction):
                self.efuncs[i] = efunc
                continue

            if isinstance(efunc, SharedDataInitFunction):
                # SharedDataInitFunctions are reconstructed automatically by
                # ThreadFunctions, so they're also treated specially
                efunc = self.efuncs[efunc.caller].idata_function
            else:
                # The still undefined symbols that a pass may have introduced
                new_params = derive_parameters(efunc)
                new_params = [a for a in new_params if not a._mem_internal_eager]
                if isinstance(efunc, EntryFunction):
                    new_params = [a for a in new_params if isinstance(a, ArgProvider)]
                if isinstance(efunc, DeviceFunction):
                    new_params = [a for a in new_params if not a.is_AbstractFunction]

                # The parameters that have obtained a definition inside the Callable
                defines = FindSymbols('defines').visit(efunc.body)
                drop_params = [a for a in efunc.parameters if a in defines]

                # Update the `efunc` parameters list
                parameters = [a for a in efunc.parameters if a not in drop_params]
                parameters.extend(new_params)
                efunc = efunc._rebuild(parameters=parameters)

            # No IET-level transformations, no-op
            if efunc is self.efuncs[i]:
                continue

            # Stash old and new signature
            old_params = self.efuncs[i].parameters
            new_params = efunc.parameters

            self.efuncs[i] = efunc
            if old_params == new_params:
                continue

            # Update all the Calls to `efunc`
            for n in dag.downstream(i):
                try:
                    v = self.efuncs[n]
                except KeyError:
                    continue

                mapper = {}
                for c in FindNodes(Call).visit(v):
                    if c.name != efunc.name:
                        continue

                    assert len(old_params) == len(c.arguments)
                    binding = dict(zip(old_params, c.arguments))
                    weakmap = {i.name: i for i in binding}

                    arguments = []
                    for a in new_params:
                        # If `old is new`, reuse same argument
                        if a in binding:
                            arguments.append(binding[a])
                            continue

                        # If `old is not new`, it may still be, logically, the
                        # same parameter (e.g., same name), so a substitution
                        # of all bound symbols will be performed
                        try:
                            a0 = weakmap[a.name]
                            mapper.update(make_symbols_map(a0, a))
                            arguments.append(binding[a0])
                        except (KeyError, ValueError):
                            # Totally new parameter, we just add it
                            arguments.append(a)

                    mapper[c] = c._rebuild(arguments=arguments)

                # Replace Calls
                v = Transformer(mapper).visit(v)
                # Replace old->new symbols throughout the IET
                v = Uxreplace(mapper).visit(v)

                self.efuncs[n] = v

        # Uniqueness
        self.includes = filter_ordered(self.includes)
        self.headers = filter_ordered(self.headers, key=str)
        self.ffuncs = filter_ordered(self.ffuncs)

        # Apply `func` to the external functions
        for i in range(len(self.ffuncs)):
            self.ffuncs[i], _ = func(self.ffuncs[i], **kwargs)

    def visit(self, func, **kwargs):
        """
        Apply `func` to all nodes in the Graph. `func` gathers info about the
        state of each node. The gathered info is returned to the called as a mapper
        from nodes to info. Unlike `apply`, `visit` does not change the state
        of the Graph.
        """
        dag = self._create_call_graph()
        toposort = dag.topological_sort()

        mapper = OrderedDict([(i, func(self.efuncs[i], **kwargs)) for i in toposort])

        return mapper


# API to define compiler passes


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


# Misc


def make_symbols_map(a0, a1):
    if len(a0.bound_symbols) != len(a1.bound_symbols):
        raise ValueError

    bs0 = sorted(a0.bound_symbols, key=lambda i: i.name)
    bs1 = sorted(a1.bound_symbols, key=lambda i: i.name)
    mapper = dict(zip(bs0, bs1))

    if any(k.name != v.name for k, v in mapper.items()):
        raise ValueError

    mapper[a0.function] = a1.function

    return mapper


Jitting = namedtuple('Jitting', 'includes include_dirs libs lib_dirs')
