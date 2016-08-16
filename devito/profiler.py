from ctypes import Structure, byref, c_double, c_longlong

from cgen_wrapper import Block, Statement, Struct, Value
from devito.logger import error


class Profiler(object):
    """The Profiler class is used to manage profiling information for Devito
    generated C code.
    """
    TIMINGS = 1
    OIS = 2

    def __init__(self):
        self.t_fields = []
        self.o_fields = []
        self.timings = None
        self.ois = None
        self.gflops = None
        self.oi_defaults = {}

    def add_profiling(self, code, name, omp_flag=None):
        """Function to add profiling code to the given :class:`cgen.Block`.

        :param code: A list of :class:`cgen.Generable` with the code to be
                      profiled.
        :param name: The name of the field that will contain the timings.
        :param omp_flag: OpenMP flag to add before profiling operations,
                         if needed.
        :returns: A list of :class:`cgen.Generable` with the added profiling
                  code.
        """
        self.t_fields.append((name, c_double))
        self.o_fields.append((name, c_longlong))
        self.oi_defaults[name] = self.get_oi(name, code)

        omp_flag = omp_flag or []

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
                          {"sn": self.name, "n": name}),
            ])
        ]

        return init + code + end

    def get_oi(self, code):
        """Calculates the total operation intensity of the code provided.
        If needed, injects the code with operations for runtime calculation.
        """
        # TODO: Add the fixed OIs to self.ois_defaults. Inject code to calculate
        #       OIs in loops
        pass

    @property
    def get_class(self, choice):
        """Returns a :class:`ctypes.Structure` subclass defining our structure
        for Ois or Timings

        :param choice: Profiler.OIS or Profiler.TIMINGS

        :returns: A class definition
        """
        if choice == Profiler.OIS:
            name = "Ois"
            fields = self.t_fields
        elif choice == Profiler.TIMINGS:
            name = "Timings"
            fields = self.o_fields
        else:
            error("Wrong choice")

        return type(name, (Structure,), {"_fields_": fields})

    @property
    def as_cgen_struct(self, choice):
        """Returns the :class:`cgen.Struct` relative to the profiler

        :returns: The struct
        """
        fields = []

        if choice == Profiler.TIMINGS:
            for name, _ in self.t_fields:
                fields.append(Value("double", name))
        elif choice == Profiler.OIS:
            for name, _ in self.o_fields:
                fields.append(Value("long long", name))
        else:
            error("Wrong choice")

        return Struct("profiler", fields)

    @property
    def timings_as_ctypes_pointer(self):
        """Returns a pointer to the ctypes structure for the timings

        :returns: The pointer
        """
        self.timings = self.get_class(Profiler.TIMINGS)

        return byref(self.timings)

    @property
    def ois_as_ctypes_pointer(self):
        """Returns a pointer to the ctypes structure for the OIs

        :returns: The pointer
        """
        self.ois = self.get_class(Profiler.OIS)

        return byref(self.ois)

    @property
    def timings_as_dictionary(self):
        """Returns the timings as a python dictionary

        :returns: A dictionary containing the timings
        """
        if not self.timings:
            return {}

        return dict((field, getattr(self.timings, field)) for field, _ in self.timings._fields_)

    @property
    def ois_as_dictionary(self):
        """Returns the operation intensities as a dictionary

        :returns: A dictionary containing the OIs
        """
        if not self.ois:
            return {}

        return dict((field, getattr(self.ois, field)) for field, _ in self.ois._fields_)

    @property
    def get_gflops(self):
        """Returns the GFLOPS of the profiled codes

        :return: A dictionary containing the GFLOPS
        """
        if self.gflops:
            return self.gflops

        self.gflops = {}

        for name, oi in self.ois_as_dictionary.items():
            self.gflops[name] = ((self.oi_defaults[name] + oi) /
                                 (self.timings_as_dictionary[name] * 10**9))

        return self.gflops
