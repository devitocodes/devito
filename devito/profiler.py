from ctypes import Structure, byref, c_double

import numpy as np

from devito.cgen_wrapper import Block, Statement, Struct, Value


class Profiler(object):
    """The Profiler class is used to manage profiling information for Devito
    generated C code.

    :param openmp: True if OpenMP is on.
    """
    TIME = 0
    t_name = "timings"

    def __init__(self, openmp=False, dtype=np.float32):
        self.openmp = openmp
        self.float_size = np.dtype(dtype).itemsize
        self.profiled = []
        self.t_fields = []

        self._C_timings = None

    def add_profiling(self, code, name, omp_flag=None, to_ignore=None):
        """Function to add profiling code to the given :class:`cgen.Block`.

        :param code: A list of :class:`cgen.Generable` with the code to be
                      profiled.
        :param name: The name of the field that will contain the timings.
        :param byte_size: The size in bytes of the values used in the code.
                          Defaults to 4.
        :param omp_flag: OpenMP flag to add before profiling operations,
                         if needed.
        :param to_ignore: List of strings containing the labels of
                          symbols used as loop variables

        :returns: A list of :class:`cgen.Generable` with the added profiling
                  code.
        """
        if code == []:
            return []

        self.profiled.append(name)
        omp_flag = omp_flag or []

        self.t_fields.append((name, c_double))

        init = [
            Statement("struct timeval start_%s, end_%s" % (name, name))
        ] + omp_flag + [Statement("gettimeofday(&start_%s, NULL)" % name)]

        end = omp_flag + [
            Block([
                Statement("gettimeofday(&end_%s, NULL)" % name),
                Statement(("%(sn)s->%(n)s += " +
                           "(double)(end_%(n)s.tv_sec-start_%(n)s.tv_sec)+" +
                           "(double)(end_%(n)s.tv_usec-start_%(n)s.tv_usec)" +
                           "/1000000") %
                          {"sn": self.t_name, "n": name}),
            ])
        ]

        return init + code + end

    def get_class(self, choice):
        """Returns a :class:`ctypes.Structure` subclass defining our structure
        for Ois or Timings

        :param choice: Profiler.OIS or Profiler.TIMINGS

        :returns: A class definition
        """
        assert choice in [Profiler.TIME]

        name = "Timings"
        fields = self.t_fields

        return type(name, (Structure,), {"_fields_": fields})

    def as_cgen_struct(self, choice):
        """Returns the :class:`cgen.Struct` relative to the profiler

        :returns: The struct
        """
        assert choice in [Profiler.TIME]

        fields = []
        s_name = None

        s_name = "profiler"
        for name, _ in self.t_fields:
            fields.append(Value("double", name))

        return Struct(s_name, fields)

    def as_ctypes_pointer(self, choice):
        """Returns a pointer to the ctypes structure for the chosen
        metric

        :param choice: The structure needed

        :returns: The pointer
        """
        assert choice in [Profiler.TIME]

        struct = self.get_class(choice)()
        self._C_timings = struct

        return byref(struct)

    @property
    def timings(self):
        """
        Return the timings, up to microseconds, as a python dictionary.
        """
        if self._C_timings:
            return {field: max(getattr(self._C_timings, field), 10**-6)
                    for field, _ in self._C_timings._fields_}
        else:
            return {}
