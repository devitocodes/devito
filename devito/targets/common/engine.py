from collections import OrderedDict
from functools import wraps
from time import time

from devito.ir.iet import Call, FindNodes, MetaCall, Transformer
from devito.tools import DAG, as_tuple, filter_ordered

__all__ = ['Graph', 'pass']


class Graph(object):

    """
    A special DAG representing call graphs.

    The nodes of the graph are IET Callables; an edge from node `a` to node `b`
    indicates that `b` calls `a`.

    The `apply` method may be used to visit the Graph and apply a transformer `T`
    to all nodes. This may change the state of the Graph: node `a` gets replaced
    by `a' = T(a)`; new nodes (Callables), and therefore new edges, may be added.
    """

    def __init__(self, iet):
        self.efuncs = OrderedDict([('root', iet)])
        self.ffuncs = []

        self.dimensions = []
        self.includes = []
        self.headers = []

        # Track performance of each pass
        self.timings = OrderedDict()

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
        Apply ``func`` to all nodes in the Graph. This changes the state of the Graph.
        """
        dag = self._create_call_graph()

        # Apply `func`
        for i in dag.topological_sort():
            self.efuncs[i], metadata = func(self.efuncs[i], **kwargs)

            # Track any new Dimensions introduced by `func`
            self.dimensions.extend(list(metadata.get('dimensions', [])))

            # Track any new #include and #define required by `func`
            self.includes.extend(list(metadata.get('includes', [])))
            self.includes = filter_ordered(self.includes)
            self.headers.extend(list(metadata.get('headers', [])))
            self.headers = filter_ordered(self.headers)

            # Tracky any new external function
            self.ffuncs.extend(list(metadata.get('ffuncs', [])))
            self.ffuncs = filter_ordered(self.ffuncs)

            # Track any new ElementalFunctions
            self.efuncs.update(OrderedDict([(i.name, i)
                                            for i in metadata.get('efuncs', [])]))

            # If there's a change to the `args` and the `iet` is an efunc, then
            # we must update the call sites as well, as the arguments dropped down
            # to the efunc have just increased
            args = as_tuple(metadata.get('args'))
            if args:
                # `extif` avoids redundant updates to the parameters list, due
                # to multiple children wanting to add the same input argument
                extif = lambda v: list(v) + [e for e in args if e not in v]
                stack = [i] + dag.all_downstreams(i)
                for n in stack:
                    efunc = self.efuncs[n]
                    calls = [c for c in FindNodes(Call).visit(efunc) if c.name in stack]
                    mapper = {c: c._rebuild(arguments=extif(c.arguments)) for c in calls}
                    efunc = Transformer(mapper).visit(efunc)
                    if efunc.is_Callable:
                        efunc = efunc._rebuild(parameters=extif(efunc.parameters))
                    self.efuncs[n] = efunc

        # Apply `func` to the external functions
        for i in range(len(self.ffuncs)):
            self.ffuncs[i], _ = func(self.ffuncs[i], **kwargs)


def pass(func):
    @wraps(func)
    def wrapper(graph, **kwargs):
        tic = time()
        graph.apply(func, **kwargs)
        toc = time()
        graph.timings[func.__name__] = toc - tic
    return wrapper
