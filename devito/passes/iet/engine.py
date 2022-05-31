from collections import OrderedDict, namedtuple
from functools import partial, wraps

from devito.ir.iet import (Call, FindNodes, FindSymbols, MetaCall, Transformer,
                           ThreadFunction, derive_parameters)
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
        # Internal "known" functions
        self.efuncs = OrderedDict([('root', iet)])

        # Foreign functions
        self.ffuncs = []

        self.includes = []
        self.headers = []

    @property
    def root(self):
        return self.efuncs['root']

    @property
    def funcs(self):
        retval = [MetaCall(v, True) for k, v in self.efuncs.items() if k != 'root']
        retval.extend([MetaCall(i, False) for i in self.ffuncs])
        return tuple(retval)

    def _create_call_graph(self):
        dag = DAG(nodes=['root'])
        queue = ['root']
        while queue:
            caller = queue.pop(0)
            callees = FindNodes(Call).visit(self.efuncs[caller])
            for callee in filter_ordered([i.name for i in callees]):
                if callee in self.efuncs:  # Exclude foreign Calls, e.g., MPI calls
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

            if efunc is self.efuncs[i]:
                continue
            self.efuncs[i] = efunc

            if isinstance(efunc, ThreadFunction):
                continue

            # The parameters/arguments lists may have changed since a pass may have:
            # 1) introduced a new symbol
            new_args = derive_parameters(efunc)
            new_args = [a for a in new_args if not a._mem_internal_eager]

            # 2) defined a symbol for which no definition was available yet (e.g.
            # via a malloc, or a Dereference)
            defines = FindSymbols('defines').visit(efunc.body)
            drop_args = [a for a in efunc.parameters if a in defines]

            if not (new_args or drop_args):
                continue

            def _filter(v, ef=None):
                processed = list(v)
                for a in new_args:
                    if a in processed:
                        # A child efunc trying to add a symbol alredy added by a
                        # sibling efunc
                        continue

                    if ef is self.root and not isinstance(a, ArgProvider):
                        # Temporaries (e.g., Arrays) *cannot* be args in `root`.
                        # So if we end up here, `a` keeps being undefined
                        # inside it, and we rely on a later pass to define it
                        continue

                    processed.append(a)

                processed = [a for a in processed if a not in drop_args]

                return processed

            # Update to use the new signature
            parameters = _filter(efunc.parameters, efunc)
            self.efuncs[i] = efunc._rebuild(parameters=parameters)

            # Update all call sites to use the new signature
            for n in dag.downstream(i):
                efunc = self.efuncs[n]

                mapper = {c: c._rebuild(arguments=_filter(c.arguments))
                          for c in FindNodes(Call).visit(efunc)
                          if c.name == self.efuncs[i].name}
                efunc = Transformer(mapper).visit(efunc)
                self.efuncs[n] = efunc

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
