from ctypes import Structure, byref, c_double, c_longlong

from cgen_wrapper import Assign, Block, For, Statement, Struct, Value
from devito.logger import error


class Profiler(object):
    """The Profiler class is used to manage profiling information for Devito
    generated C code.
    """
    TIME = 1
    FLOP = 2

    def __init__(self):
        self.t_name = "timings"
        self.o_name = "oi"
        self.f_name = "flops"
        self.t_fields = []
        self.f_fields = []
        self.oi = {}
        self._timings = None
        self._flops = None
        self.flops_defaults = {}

    def add_profiling(self, code, name, byte_size=4, omp_flag=None):
        """Function to add profiling code to the given :class:`cgen.Block`.

        :param code: A list of :class:`cgen.Generable` with the code to be
                      profiled.
        :param name: The name of the field that will contain the timings.
        :param byte_size: The size in bytes of the values used in the code.
                          Defaults to 4.
        :param omp_flag: OpenMP flag to add before profiling operations,
                         if needed.
        :returns: A list of :class:`cgen.Generable` with the added profiling
                  code.
        """
        self.t_fields.append((name, c_double))
        self.f_fields.append((name, c_longlong))
        self.get_oi_and_flops(name, code, byte_size)

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
                          {"sn": self.t_name, "n": name}),
            ])
        ]

        return init + code + end

    def get_oi_and_flops(self, name, code, size):
        """Calculates the total operation intensity of the code provided.
        If needed, lets the C code calculate it.

        :param name: The name of the field to be populated
        :param code: The code to be profiled.
        """
        if name not in self.oi:
            self.oi[name] = 0
        if name not in self.flops_defaults:
            self.flops_defaults[name] = 0

        for elem in code:
            if isinstance(elem, Assign):
                assign_oi, assign_flops = self._get_assign_oi(elem, size)
                self.oi[name] += assign_oi
                self.flops_defaults[name] += assign_flops
            elif isinstance(elem, For):
                self._get_for_oi_and_flops(name, elem, size)
            elif isinstance(elem, Block):
                block_oi, block_flops = self._get_block_oi_and_flops(name, elem, size)
                self.oi[name] += block_oi
                self.flops_defaults[name] += block_flops

    def _get_for_oi_and_flops(self, name, loop, size):
        loop_oi = 0
        loop_flops = 0

        if isinstance(loop.body, Assign):
            loop_oi, loop_flops = self._get_assign_oi_and_flops(loop.body, size)
        elif isinstance(loop.body, Block):
            loop_oi, loop_flops = self._get_block_oi_and_flops(name, loop.body, size)
        elif isinstance(loop.body, For):
            self._get_for_oi_and_flops(name, loop.body, size)

        self.oi[name] += loop_oi
        flops_calc_stmt = Statement("%s->%s += %f" % (self.f_name, name, loop_flops))

        loop.body = Block([flops_calc_stmt, loop.body])

    def _get_block_oi_and_flops(self, name, block, size):
        block_oi = float(0)
        block_flops = 0
        for elem in block.contents:
            if isinstance(elem, Assign):
                assign_oi, assign_flops = self._get_assign_oi_and_flops(elem, size)
                block_oi += assign_oi
                block_flops += assign_flops
            elif isinstance(elem, Block):
                n_block_oi, n_block_flops = self._get_block_oi_and_flops(name, elem, size)
                block_oi += n_block_oi
                block_flops += n_block_flops
            elif isinstance(elem, For):
                self._get_for_oi_and_flops(name, elem, size)

        return block_oi, block_flops

    def _get_assign_oi_and_flops(self, assign, size):
        loads = {}
        flops = 0
        cur_load = ""

        idx = 0
        brackets = 0
        while idx < len(assign.rvalue):
            char = assign.rvalue[idx]
            if len(cur_load) == 0:
                if char in "+-*/" and assign.rvalue[idx - 1] is not 'e':
                    flops += 1
                if char.isalpha() and not assign.rvalue[idx - 1].isdigit():
                    cur_load += char
                idx += 1
            else:
                if char is '[':
                    cur_load += char
                    brackets += 1
                    idx += 1
                elif char is ' ' and brackets == 0:
                    cur_load += char
                    loads[cur_load] = True
                    cur_load = ""
                    idx += 1
                elif char is ']' and idx == len(assign.rvalue) - 1:
                    cur_load += char
                    loads[cur_load] = True
                    idx += 1
                elif cur_load[-1] is ']' and char is not '[':
                    loads[cur_load] = True
                    cur_load = ""
                    brackets = 0
                else:
                    cur_load += char
                    idx += 1

        oi = float(flops) / float(size * (len(loads) + 1))

        return oi, flops

    def get_class(self, choice):
        """Returns a :class:`ctypes.Structure` subclass defining our structure
        for Ois or Timings

        :param choice: Profiler.OIS or Profiler.TIMINGS

        :returns: A class definition
        """
        if choice == Profiler.TIME:
            name = "Timings"
            fields = self.t_fields
        elif choice == Profiler.FLOP:
            name = "Flops"
            fields = self.f_fields
        else:
            error("Wrong choice")

        return type(name, (Structure,), {"_fields_": fields})

    def as_cgen_struct(self, choice):
        """Returns the :class:`cgen.Struct` relative to the profiler

        :returns: The struct
        """
        fields = []
        s_name = None

        if choice == Profiler.TIME:
            s_name = "profiler"
            for name, _ in self.t_fields:
                fields.append(Value("double", name))
        elif choice == Profiler.FLOP:
            s_name = "flops"
            for name, _ in self.f_fields:
                fields.append(Value("long long", name))
        else:
            error("Wrong choice")

        return Struct(s_name, fields)

    def as_ctypes_pointer(self, choice):
        """Returns a pointer to the ctypes structure for the chosen
        metric

        :param choice: The structure needed

        :returns: The pointer
        """
        struct = self.get_class(choice)()

        if choice == Profiler.TIME:
            self._timings = struct
        elif choice == Profiler.FLOP:
            self._flops = struct

        return byref(struct)

    @property
    def timings(self):
        """Returns the timings as a python dictionary

        :returns: A dictionary containing the timings
        """
        if not self._timings:
            return {}

        return dict((field, getattr(self._timings, field))
                    for field, _ in self._timings._fields_)

    @property
    def gflops(self):
        """Returns the GFLOPS as a dictionary

        :returns: A dictionary containing the calculated GFLOPS
        """
        if not self._flops:
            return {}

        return dict((field,
                     float(
                         getattr(self._flops, field) + self.flops_defaults[field]
                     )/self.timings[field]/10**9)
                    for field, _ in self._flops._fields_)
