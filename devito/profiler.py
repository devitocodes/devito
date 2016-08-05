from ctypes import Structure, byref, c_double

from cgen_wrapper import Block, Statement, Struct, Value


def _get_timings_class(fields):
    """Returns a :class:`ctypes.Structure` subclass defining our structure
    """

    return type("Timings", (Structure,), {"_fields_": fields})


class Profiler(Structure):
    """The Profiler class is used to manage profiling information for Devito
    generated C code.
    """

    def __init__(self):
        self.name = "timings"
        self.fields = []

    def add_profiling(self, block, name, omp_flag=None):
        """Function to add profiling code to the given :class:`cgen.Block`.

        :param block: A list of :class:`cgen.Generable` with the code to be
                      profiled.
        :param name: The name of the field that will contain the timings.
        :param omp_flag: OpenMP flag to add before profiling operations,
                         if needed.
        :returns: A list of :class:`cgen.Generable` with the added profiling
                  code.
        """
        self.fields.append(name)

        omp_flag = omp_flag or []

        init = [
            Statement("struct timeval start_%s, end_%s" % (name, name))
        ] + omp_flag + [Statement("gettimeofday(&start_%s, NULL)" % name)]

        end = omp_flag + [
            Block([
                Statement("gettimeofday(&end_%s, NULL)" % name),
                Statement(("%(sn)s->%(n)s += (end_%(n)s.tv_sec - start_%(n)s.tv_sec) * " +
                           "1000000u + (end_%(n)s.tv_usec - start_%(n)s.tv_usec)") %
                          {"sn": self.name, "n": name}),
            ])
        ]

        return init + block + end

    @property
    def as_cgen_struct(self):
        """Returns the :class:`cgen.Struct` relative to the profiler

        :returns: The struct
        """
        fields = []

        for name in self.fields:
            fields.append(Value("double", name))

        return Struct("profiler", fields)

    @property
    def as_ctypes_pointer(self):
        """Returns a pointer to the ctypes structure

        :returns: The pointer
        """
        fields = []

        for name in self.fields:
            fields.append((name, c_double))

        timings_class = _get_timings_class(fields)

        self.timings = timings_class()

        return byref(self.timings)

    @property
    def as_dictionary(self):
        """Returns the timings as a python dictionary

        :returns: A dictionary containing the timings
        """
        return dict((field, getattr(self.timings, field)) for field, _ in self.timings._fields_)
