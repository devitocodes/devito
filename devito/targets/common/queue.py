from collections import OrderedDict
from functools import wraps
from time import time

from devito.ir.iet import Call, FindNodes, MetaCall, Transformer
from devito.tools import DAG, as_tuple, filter_ordered

__all__ = ['State', 'dle_pass']


class State(object):

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


def process(func, state, **kwargs):
    """
    Apply ``func`` to the IETs in ``state.efuncs``, and update ``state`` accordingly.
    """
    # Create a Call graph. `func` will be applied to each node in the Call graph.
    # `func` might change an `efunc` signature; the Call graph will be used to
    # propagate such change through the `efunc` callers
    dag = DAG(nodes=['root'])
    queue = ['root']
    while queue:
        caller = queue.pop(0)
        callees = FindNodes(Call).visit(state.efuncs[caller])
        for callee in filter_ordered([i.name for i in callees]):
            if callee in state.efuncs:  # Exclude foreign Calls, e.g., MPI calls
                try:
                    dag.add_node(callee)
                    queue.append(callee)
                except KeyError:
                    # `callee` already in `dag`
                    pass
                dag.add_edge(callee, caller)
    assert dag.size == len(state.efuncs)

    # Apply `func`
    for i in dag.topological_sort():
        state.efuncs[i], metadata = func(state.efuncs[i], **kwargs)

        # Track any new Dimensions introduced by `func`
        state.dimensions.extend(list(metadata.get('dimensions', [])))

        # Track any new #include and #define required by `func`
        state.includes.extend(list(metadata.get('includes', [])))
        state.includes = filter_ordered(state.includes)
        state.headers.extend(list(metadata.get('headers', [])))
        state.headers = filter_ordered(state.headers)

        # Tracky any new external function
        state.ffuncs.extend(list(metadata.get('ffuncs', [])))
        state.ffuncs = filter_ordered(state.ffuncs)

        # Track any new ElementalFunctions
        state.efuncs.update(OrderedDict([(i.name, i)
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
                efunc = state.efuncs[n]
                calls = [c for c in FindNodes(Call).visit(efunc) if c.name in stack]
                mapper = {c: c._rebuild(arguments=extif(c.arguments)) for c in calls}
                efunc = Transformer(mapper).visit(efunc)
                if efunc.is_Callable:
                    efunc = efunc._rebuild(parameters=extif(efunc.parameters))
                state.efuncs[n] = efunc

    # Apply `func` to the external functions
    for i in range(len(state.ffuncs)):
        state.ffuncs[i], _ = func(state.ffuncs[i], **kwargs)


def dle_pass(func):
    @wraps(func)
    def wrapper(state, **kwargs):
        tic = time()
        process(func, state, **kwargs)
        toc = time()
        state.timings[func.__name__] = toc - tic
    return wrapper
