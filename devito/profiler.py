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
        if code == []:
            return []

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

        loads = {"stores": 0}

        for elem in code:
            if isinstance(elem, Assign):
                assign_flops = self._get_assign_flops(elem, loads)
                self.oi[name] += assign_flops
                self.flops_defaults[name] += assign_flops
            elif isinstance(elem, For):
                for_flops = self._get_for_flops(name, elem, loads)
                self.oi[name] += for_flops
            elif isinstance(elem, Block):
                block_oi, block_flops = self._get_block_oi_and_flops(name, elem, loads)
                self.oi[name] += block_oi
                self.flops_defaults[name] += block_flops

        self.oi[name] = float(self.oi[name]) / (size*(len(loads) + loads["stores"] - 1))

    def _get_for_flops(self, name, loop, loads):
        loop_flops = 0
        loop_oi_f = 0

        if isinstance(loop.body, Assign):
            loop_flops = self._get_assign_flops(loop.body, loads)
            loop_oi_f = loop_flops
        elif isinstance(loop.body, Block):
            loop_oi_f, loop_flops = self._get_block_oi_and_flops(name, loop.body, loads)
        elif isinstance(loop.body, For):
            loop_oi_f = self._get_for_flops(name, loop.body, loads)

        if loop_flops == 0:
            return loop_oi_f

        flops_calc_stmt = Statement("%s->%s += %f" % (self.f_name, name, loop_flops))
        loop.body = Block([flops_calc_stmt, loop.body])

        return loop_oi_f

    def _get_block_oi_and_flops(self, name, block, loads):
        block_flops = 0
        block_oi = 0

        for elem in block.contents:
            if isinstance(elem, Assign):
                a_flops = self._get_assign_flops(elem, loads)
                block_flops += a_flops
                block_oi += a_flops
            elif isinstance(elem, Block):
                nblock_oi, nblock_flops = self._get_block_oi_and_flops(name, elem, loads)
                block_oi += nblock_oi
                block_flops += nblock_flops
            elif isinstance(elem, For):
                block_oi += self._get_for_flops(name, elem, loads)

        return block_oi, block_flops

    def _get_assign_flops(self, assign, loads):
        flops = 0
        cur_load = ""
        loads["stores"] += 1

        idx = 0
        brackets = 0
        # removing casting statements and function calls to floor that can confuse the parser
        string = (assign.lvalue + " " + assign.rvalue).replace("float", '').replace("int", '').replace("floor", '')

        while idx < len(string):
            char = string[idx]
            if len(cur_load) == 0:
                if char == '[':
                    brackets += 1
                elif char == ']':
                    brackets -= 1
                elif char in "+-*/" and string[idx - 1] is not 'e' and not string[idx + 1].isdigit() and brackets == 0:
                    flops += 1
                elif char.isalpha() and not string[idx - 1].isdigit() and char not in "it":
                    cur_load += char
                idx += 1
            else:
                if char is '[' or char is ']':
                    loads[cur_load] = True
                    cur_load = ""
                    brackets += 1 if char is '[' else -1
                    idx += 1
                elif char is ' ' and brackets == 0:
                    loads[cur_load] = True
                    cur_load = ""
                    idx += 1
                else:
                    cur_load += char
                    idx += 1

        if len(cur_load) > 0:
            loads[cur_load] = True

        return flops

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
