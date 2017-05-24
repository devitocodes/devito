from ctypes import Structure, byref, c_double
from cgen import Struct, Value


class Profiler(object):

    """
    A Profiler is used to manage profiling information for Devito generated C code.
    """

    varname = "timings"
    typename = "profiler"

    def __init__(self):
        self._timers = []
        self._C_timings = None

    def add(self, name):
        """
        Add a C-level timer.
        """
        self._timers.append(name)

    @property
    def ctype(self):
        """
        Returns a :class:`cgen.Struct` relative to the profiler.
        """
        return Struct(Profiler.typename,
                      [Value('double', n) for n in self._timers])

    def setup(self):
        """
        Allocate and return a pointer to the timers C-level Struct, which includes
        all timers added to ``self`` through ``self.add(...)``.
        """
        cls = type("Timings", (Structure,),
                   {"_fields_": [(n, c_double) for n in self._timers]})
        self._C_timings = cls()
        return byref(self._C_timings)

    @property
    def timings(self):
        """
        Return the timings, up to microseconds, as a dictionary.
        """
        if self._C_timings is None:
            raise RuntimeError("Cannot extract timings with non-finalized Profiler.")
        return {field: max(getattr(self._C_timings, field), 10**-6)
                for field, _ in self._C_timings._fields_}
