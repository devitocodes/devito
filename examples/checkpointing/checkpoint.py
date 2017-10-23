from pyrevolve import Checkpoint, Operator
from devito import TimeFunction


class DevitoOperator(Operator):
    """Class to wrap devito operators so they conform to the pyRevolve API
    """
    def __init__(self, op, args, argnames):
        self.op = op
        self.args = args
        self.argnames = argnames

    def apply(self, t_start, t_end):
        """ If the devito operator requires some extra arguments in the call to apply
            they can be stored in the args property of this object so pyRevolve calls
            pyRevolve.Operator.apply() without caring about these extra arguments while
            this method passes them on correctly to devito.Operator
        """
        args = self.args.copy()
        args[self.argnames['t_start']] = t_start
        args[self.argnames['t_end']] = t_end
        self.op.apply(**args)


class DevitoCheckpoint(Checkpoint):
    """Devito's concrete implementation of the Checkpoint abstract base class provided by
       pyRevolve. Holds a list of symbol objects that hold data.
    """
    def __init__(self, symbols):
        """Intialise a checkpoint object. Upon initialisation, a checkpoint
        stores only a reference to the symbols that are passed into it.
        The symbols must be passed as a mapping symbolname->symbolobject."""
        assert(all(isinstance(s, TimeFunction) for s in symbols))
        self._dtype = symbols[0].dtype
        self.symbols = symbols
        self.revolver = None

    @property
    def dtype(self):
        return self._dtype

    def save(self, ptr):
        """Overwrite live-data in this Checkpoint object with data found at
        the ptr location."""
        i_ptr_lo = 0
        i_ptr_hi = 0
        for s in self.symbols:
            i_ptr_hi = i_ptr_hi + s.size
            ptr[i_ptr_lo:i_ptr_hi] = s.data.flatten()[:]
            i_ptr_lo = i_ptr_hi

    def load(self, ptr):
        """Copy live-data from this Checkpoint object into the memory given by
        the ptr."""
        i_ptr_lo = 0
        i_ptr_hi = 0
        for s in self.symbols:
            i_ptr_hi = i_ptr_hi + s.size
            s.data[:] = ptr[i_ptr_lo:i_ptr_hi].reshape(s.shape)
            i_ptr_lo = i_ptr_hi

    @property
    def size(self):
        """The memory consumption of the data contained in a checkpoint."""
        return sum([s.size for s in self.symbols])
