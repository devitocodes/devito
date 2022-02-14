from collections import OrderedDict, namedtuple
from functools import partial, wraps

from sympy.tensor.indexed import IndexException

from devito.ir.iet import Call, FindNodes, MetaCall, Transformer
from devito.tools import DAG, as_tuple, filter_ordered, timed_pass

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
            self.efuncs[i], metadata = func(self.efuncs[i], **kwargs)

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

            # If there's a change to the `args` and the `iet` is an efunc, then
            # we must update the call sites as well, as the arguments dropped down
            # to the efunc have just increased
            args = as_tuple(metadata.get('args'))
            if not args:
                continue

            def filter_args(v, efunc=None):
                processed = list(v)
                for _a in args:
                    try:
                        # Should the arg actually be dropped?
                        a, drop = _a
                        if drop:
                            if a in processed:
                                processed.remove(a)
                            continue
                    except (TypeError, ValueError, IndexException):
                        a = _a

                    if a in processed:
                        # A child efunc trying to add a symbol alredy added by a
                        # sibling efunc
                        continue

                    if efunc is self.root and not (a.is_Input or a.is_Object):
                        # Temporaries (ie, Symbol, Arrays) *cannot* be args in `root`
                        continue

                    processed.append(a)

                return processed

            stack = [i] + dag.all_downstreams(i)
            for n in stack:
                efunc = self.efuncs[n]

                mapper = {}
                for c in FindNodes(Call).visit(efunc):
                    if c.name not in stack:
                        continue
                    mapper[c] = c._rebuild(arguments=filter_args(c.arguments))

                parameters = filter_args(efunc.parameters, efunc)
                efunc = Transformer(mapper).visit(efunc)
                efunc = efunc._rebuild(parameters=parameters)

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
