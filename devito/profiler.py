import re

from collections import defaultdict
from ctypes import Structure, byref, c_double, c_longlong

from cgen_wrapper import Assign, Block, For, Statement, Struct, Value
from devito.logger import error


class Profiler(object):
    """The Profiler class is used to manage profiling information for Devito
    generated C code.

    :param openmp: True if OpenMP is on.
    """
    TIME = 1
    FLOP = 2
    t_name = "timings"
    f_name = "flops"
    loop_temp_prefix = "temp_"

    def __init__(self, openmp=False):
        self.openmp = openmp
        self.profiled = []
        self.t_fields = []
        self.f_fields = []
        self.oi = defaultdict(int)
        self.oi_low = defaultdict(int)
        # _C_ fields are ctypes structs used for code generation
        self._C_timings = None
        self._C_flops = None
        self.flops_defaults = defaultdict(int)
        self.total_load_count = defaultdict(int)

    def get_loop_reduction(self, op, variables):
        """Function to generate reduction pragma, used for profiling under
        OpenMP settings.

        :param op: The reduction operator.
        :param varaiables: A list of string, each is the name of a variable
                           to be reduced.
        :returns: String representing the reduction pragma
        """
        if len(variables) <= 0:
            return ""
        return " reduction(%s:%s%s%s)" % (op, self.loop_temp_prefix, variables[0],
                                          "".join([(", %s%s" % (self.loop_temp_prefix, v))
                                                  for v in variables[1:]]))

    def get_loop_temp_var_decl(self, value, variables):
        """Function to generate the declariation and initialisation of the
        temporary variables used for reduction for loop profiling.

        :param value: String represeting the initial value of temporary
                      varaibles.
        :param variables: A list of string, each is the name of a variable to
                          be declaried
        :returns: String representing the declariation statement
        """
        if len(variables) <= 0:
            return ""
        return "long long %s%s = %s%s" % (self.loop_temp_prefix, variables[0], value,
                                          "".join([(", %s%s = %s"
                                                  % (self.loop_temp_prefix, v, value))
                                                  for v in variables[1:]]))

    def get_loop_flop_update(self, variables):
        """Function to generate a list of statement representing the amount
        of flops update per iteration of the loop.

        :param variables: A list of string, each is the suffix of the
                          temperary variables used for update
        :returns: A list of string representing statements used as updating
                  statements
        """
        return ["%s->%s+=%s%s" % (self.f_name, v, self.loop_temp_prefix, v)
                for v in variables]

    def add_profiling(self, code, name, byte_size=4, omp_flag=None, to_ignore=None):
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
        to_ignore = to_ignore or []
        omp_flag = omp_flag or []

        self.t_fields.append((name, c_double))
        self.f_fields.append((name, c_longlong))

        self.get_oi_and_flops(name, code, byte_size, to_ignore)

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

    def get_oi_and_flops(self, name, code, size, to_ignore):
        """Calculates the total operation intensity of the code provided.
        If needed, lets the C code calculate it.

        :param name: The name of the field to be populated
        :param code: The code to be profiled.
        """
        loads = defaultdict(int)

        for elem in code:
            if isinstance(elem, Assign):
                assign_flops = self._get_assign_flops(elem, loads, to_ignore)
                self.oi[name] += assign_flops
                self.flops_defaults[name] += assign_flops
            elif isinstance(elem, For):
                for_flops = self._get_for_flops(name, elem, loads, to_ignore)
                self.oi[name] += for_flops
            elif isinstance(elem, Block):
                block_oi, block_flops = self._get_block_oi_and_flops(
                    name, elem, loads, to_ignore)
                self.oi[name] += block_oi
                self.flops_defaults[name] += block_flops
            else:
                # no op
                pass

        load_val_sum = sum(loads.values())
        self.oi[name] = float(self.oi[name]) / (size*(len(loads) + loads["stores"] - 1))
        self.oi_low[name] = float(self.oi[name]) / (size*load_val_sum)
        self.total_load_count[name] = load_val_sum - loads["stores"]

    def _get_for_flops(self, name, loop, loads, to_ignore):
        loop_flops = 0
        loop_oi_f = 0

        if isinstance(loop.body, Assign):
            loop_flops = self._get_assign_flops(loop.body, loads, to_ignore)
            loop_oi_f = loop_flops
        elif isinstance(loop.body, Block):
            loop_oi_f, loop_flops = self._get_block_oi_and_flops(
                name, loop.body, loads, to_ignore)
        elif isinstance(loop.body, For):
            loop_oi_f = self._get_for_flops(name, loop.body, loads, to_ignore)
        else:
            # no op
            pass

        if loop_flops == 0:
            return loop_oi_f

        flops_calc_stmt = Statement(
            "%s%s += %f" % (self.loop_temp_prefix, name, loop_flops))\
            if self.openmp else\
            Statement("%s->%s += %f" % (self.f_name, name, loop_flops))
        loop.body = Block([flops_calc_stmt, loop.body])

        return loop_oi_f

    def _get_block_oi_and_flops(self, name, block, loads, to_ignore):
        block_flops = 0
        block_oi = 0

        for elem in block.contents:
            if isinstance(elem, Assign):
                a_flops = self._get_assign_flops(elem, loads, to_ignore)
                block_flops += a_flops
                block_oi += a_flops
            elif isinstance(elem, Block):
                nblock_oi, nblock_flops = self._get_block_oi_and_flops(
                    name, elem, loads, to_ignore)
                block_oi += nblock_oi
                block_flops += nblock_flops
            elif isinstance(elem, For):
                block_oi += self._get_for_flops(name, elem, loads, to_ignore)
            else:
                # no op
                pass

        return block_oi, block_flops

    def _get_assign_flops(self, assign, loads, to_ignore):
        flops = 0
        loads["stores"] += 1

        # removing casting statements and function calls to floor
        # that can confuse the parser
        string = assign.lvalue + " " + assign.rvalue

        to_ignore = [
            "int",
            "float",
            "double",
            "F",
            "e",
            "fabsf",
            "powf",
            "floor",
            "ceil",
            "temp",
            "i",
            "t",
            "p",  # This one shouldn't be here.
                  # It should be passed in by an Iteration object.
                  # Added only because tti_example uses it.
        ] + to_ignore

        # Matches all variable names
        # Variable names can contain:
        # - uppercase and lowercase letters
        # - underscores
        # - numbers (at the end)
        # eg: src_coord, temp123, u
        symbols = re.findall(r"[a-z_A-Z]+(?:\d?)+", string)

        for symbol in symbols:
            if filter(lambda x: x.isalpha(), symbol) not in to_ignore:
                loads[symbol] += 1

        brackets = 0
        for idx in range(len(string)):
            c = string[idx]

            # We skip index operations. The third check works because in the
            # generated code constants always precede variables in operations
            # and is needed because Sympy prints fractions like this: 1.0F/4.0F
            if brackets == 0 and c in "*/-+" and not string[idx+1].isdigit():
                flops += 1
            elif c == "[":
                brackets += 1
            elif c == "]":
                brackets -= 1

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
            self._C_timings = struct
        elif choice == Profiler.FLOP:
            self._C_flops = struct

        return byref(struct)

    @property
    def timings(self):
        """Returns the timings as a python dictionary

        :returns: A dictionary containing the timings
        """
        if not self._C_timings:
            return {}

        return dict((field, getattr(self._C_timings, field))
                    for field, _ in self._C_timings._fields_)

    @property
    def gflops(self):
        """Returns the GFLOPS as a dictionary

        :returns: A dictionary containing the calculated GFLOPS
        """
        if not self._C_flops:
            return {}

        return dict((field,
                     (float(
                         getattr(self._C_flops, field) + self.flops_defaults[field]
                     )/self.timings[field]/10**9)
                     if self.timings[field] > 0.0 else float("nan"))
                    for field, _ in self._C_flops._fields_)
