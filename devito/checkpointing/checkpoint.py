from pyrevolve import Checkpoint, Operator
from devito import TimeFunction
from devito.tools import flatten


class CheckpointOperator(Operator):
    """Devito's concrete implementation of the ABC pyrevolve.Operator. This class wraps
       devito.Operator so it conforms to the pyRevolve API. pyRevolve will call apply
       with arguments t_start and t_end. Devito calls these arguments time_m and time_M
       so the following dict is used to perform the translations between different names.

       Parameters
       ----------
       op : Operator
            devito.Operator object that this object will wrap.
       args : dict
            If devito.Operator.apply() expects any arguments, they can be provided
            here to be cached. Any calls to CheckpointOperator.apply() will
            automatically include these cached arguments in the call to the
            underlying devito.Operator.apply().
    """
    t_arg_names = {'t_start': 'time_m', 't_end': 'time_M'}

    def __init__(self, op, **kwargs):
        self.op = op
        self.args = self.op._prepare_arguments(**kwargs)
        self.start_offset = self.args[self.t_arg_names['t_start']]

    def _prepare_args(self, t_start, t_end):
        args = self.args.copy()
        args[self.t_arg_names['t_start']] = t_start + self.start_offset
        args[self.t_arg_names['t_end']] = t_end - 1 + self.start_offset
        return args

    def apply(self, t_start, t_end):
        """ If the devito operator requires some extra arguments in the call to apply
            they can be stored in the args property of this object so pyRevolve calls
            pyRevolve.Operator.apply() without caring about these extra arguments while
            this method passes them on correctly to devito.Operator
        """
        # Build the arguments list to invoke the kernel function
        args = self._prepare_args(t_start, t_end)
        # Invoke kernel function with args
        arg_values = [args[p.name] for p in self.op.parameters]
        self.op.cfunction(*arg_values)


class DevitoCheckpoint(Checkpoint):
    """Devito's concrete implementation of the Checkpoint abstract base class provided by
       pyRevolve. Holds a list of symbol objects that hold data.
    """
    def __init__(self, objects):
        """Intialise a checkpoint object. Upon initialisation, a checkpoint
        stores only a reference to the objects that are passed into it."""
        assert(all(isinstance(o, TimeFunction) for o in objects))
        dtypes = set([o.dtype for o in objects])
        assert(len(dtypes) == 1)
        self._dtype = dtypes.pop()
        self.objects = objects

    @property
    def dtype(self):
        return self._dtype

    def get_data(self, timestep):
        data = flatten([get_symbol_data(s, timestep) for s in self.objects])
        return data

    def get_data_location(self, timestep):
        return self.get_data(timestep)

    @property
    def size(self):
        """The memory consumption of the data contained in a checkpoint."""
        return sum([int((o.size_allocated/(o.time_size))*o.time_order)
                   for o in self.objects])

    def save(*args):
        raise RuntimeError("Invalid method called. Did you check your version" +
                           " of pyrevolve?")

    def load(*args):
        raise RuntimeError("Invalid method called. Did you check your version" +
                           " of pyrevolve?")


def get_symbol_data(symbol, timestep):
    timestep += symbol.time_order - 1
    ptrs = []
    for i in range(symbol.time_order):
        # Use `._data`, instead of `.data`, as `.data` is a view of the DOMAIN
        # data region which is non-contiguous in memory. The performance hit from
        # dealing with non-contiguous memory is so big (introduces >1 copy), it's
        # better to checkpoint unneccesarry stuff to get a contiguous chunk of memory.
        ptr = symbol._data[timestep - i, :, :]
        ptrs.append(ptr)
    return ptrs
