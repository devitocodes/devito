from pyrevolve import Checkpoint, Operator
from devito import TimeFunction


class CheckpointOperator(Operator):
    """Devito's concrete implementation of the ABC pyrevolve.Operator. This class wraps
       devito.Operator so it conforms to the pyRevolve API. pyRevolve will call apply
       with arguments t_start and t_end. Devito calls these arguments t_s and t_e so
       the following dict is used to perform the translations between different names.
       :param op: The devito.Operator object that this object will wrap
       :param args: If devito.Operator.apply() expects any arguments, they can be provided
                    here to be cached. Any calls to CheckpointOperator.apply() will
                    automatically include these cached arguments in the call to the
                    underlying devito.Operator.apply().
    """
    t_arg_names = {'t_start': 'time_m', 't_end': 'time_M'}

    def __init__(self, op, **kwargs):
        self.op = op
        self.args = kwargs
        op_default_args = self.op.prepare_arguments()
        self.start_offset = op_default_args[self.t_arg_names['t_start']]

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
        args = self._prepare_args(t_start, t_end)
        self.op.apply(**args)


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

    def save(self, ptr):
        """Overwrite live-data in this Checkpoint object with data found at
        the ptr location."""
        i_ptr_lo = 0
        i_ptr_hi = 0
        for o in self.objects:
            i_ptr_hi = i_ptr_hi + o.size
            ptr[i_ptr_lo:i_ptr_hi] = o.data.flatten()[:]
            i_ptr_lo = i_ptr_hi

    def load(self, ptr):
        """Copy live-data from this Checkpoint object into the memory given by
        the ptr."""
        i_ptr_lo = 0
        i_ptr_hi = 0
        for o in self.objects:
            i_ptr_hi = i_ptr_hi + o.size
            o.data[:] = ptr[i_ptr_lo:i_ptr_hi].reshape(o.shape)
            i_ptr_lo = i_ptr_hi

    @property
    def size(self):
        """The memory consumption of the data contained in a checkpoint."""
        return sum([o.size for o in self.objects])
