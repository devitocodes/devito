import re
from collections import defaultdict
from copy import copy
from ctypes import Structure, byref, c_double, c_longlong

from cgen_wrapper import Assign, Block, For, Pragma, Statement, Struct, Value
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

    def __init__(self, openmp=False):
        self.openmp = openmp
        self.profiled = []
        self.t_fields = []
        self.f_fields = []
        self.temps = defaultdict(set)
        self.oi = defaultdict(int)
        self.oi_low = defaultdict(int)
        # _C_ fields are ctypes structs used for code generation
        self._C_timings = None
        self._C_flops = None
        self.flops_defaults = defaultdict(int)
        self.total_load_count = defaultdict(int)
        self._var_count = 0

    def get_loop_reduction(self, op, variable):
        """Function to generate reduction pragma, used for profiling under
        OpenMP settings.

        :param op: The reduction operator
        :param variable: The name of the variable to be reduced

        :returns: String representing the reduction clause
        """
        return " reduction(%s:%s)" % (op, variable)

    def get_loop_temp_var_decl(self, name):
        """Function to generate the declariation and initialisation of the
        temporary variables used for reduction for loop profiling.

        :param value: String represeting the initial value of the temporary
                      variables.
        :param variables: The names of the variables to be declaried

        :returns: String representing the declariation statement
        """
        assignments = ", ".join([("%s = 0" % v)
                                 for v in self.temps[name]])

        return Statement("long long %s" % assignments)

    def get_loop_flop_update(self, name):
        """Function to generate a statement that collects all the values from temps
        in the main flops variable

        :param name: Name of the variable to update

        :returns: A :class:`cgen.Statement` representing the sum of all the temps
        """
        temps_sum = "+".join(self.temps[name])

        return Statement("%s->%s=%s" % (self.f_name, name, temps_sum))

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

    def add_reduction(self, pragma, name):
        if "simd" in pragma.value or "for schedule" in pragma.value:
            pragma.value += self.get_loop_reduction(
                "+", name
            )

    def get_oi_and_flops(self, name, code, size, to_ignore):
        """Calculates the total operation intensity of the code provided.
        If needed, lets the C code calculate it.

        :param name: The name of the field to be populated
        :param code: The code to be profiled.
        """
        loads = defaultdict(int)
        pragmas = []

        for elem in code:
            if isinstance(elem, Assign):
                assign_flops = self._get_assign_flops(elem, loads, to_ignore)
                self.oi[name] += assign_flops
                self.flops_defaults[name] += assign_flops
            elif isinstance(elem, For):
                for_flops = self._get_for_flops(
                    name, elem, loads, to_ignore, pragmas)
                pragmas = []
                self.oi[name] += for_flops
            elif isinstance(elem, Block):
                block_oi, block_flops = self._get_block_oi_and_flops(
                    name, elem, loads, to_ignore, pragmas)
                pragmas = []
                self.oi[name] += block_oi
                self.flops_defaults[name] += block_flops
            elif isinstance(elem, Pragma) and self.openmp:
                pragmas.append(elem)
            else:
                # no op
                pass

        load_val_sum = sum(loads.values())
        self.oi[name] = float(self.oi[name]) / (size*(len(loads) + loads["stores"] - 1))
        self.oi_low[name] = float(self.oi[name]) / (size*load_val_sum)
        self.total_load_count[name] = load_val_sum - loads["stores"]

    def _get_for_flops(self, name, loop, loads, to_ignore, pragmas):
        loop_flops = 0
        loop_oi_f = 0

        if isinstance(loop.body, Assign):
            loop_flops = self._get_assign_flops(loop.body, loads, to_ignore)
            loop_oi_f = loop_flops
        elif isinstance(loop.body, Block):
            loop_oi_f, loop_flops = self._get_block_oi_and_flops(
                name, loop.body, loads, to_ignore, pragmas)
        elif isinstance(loop.body, For):
            loop_oi_f = self._get_for_flops(
                name, loop.body, loads, to_ignore, pragmas)
        else:
            # no op
            pass

        if loop_flops == 0:
            return loop_oi_f

        new_temp = "%s%d" % (name, self._var_count)
        self._var_count += 1
        self.temps[name].add(new_temp)

        for pragma in pragmas:
            self.add_reduction(pragma, new_temp)

        flops_calc_stmt = Statement(
            "%s += %f" % (new_temp, loop_flops))\
            if self.openmp else\
            Statement("%s->%s += %f" % (self.f_name, name, loop_flops))
        loop.body = Block([flops_calc_stmt, loop.body])

        return loop_oi_f

    def _get_block_oi_and_flops(self, name, block, loads, to_ignore, pragmas):
        block_flops = 0
        block_oi = 0

        inner_pragmas = copy(pragmas)

        for elem in block.contents:
            if isinstance(elem, Assign):
                a_flops = self._get_assign_flops(elem, loads, to_ignore)
                block_flops += a_flops
                block_oi += a_flops
            elif isinstance(elem, Block):
                nblock_oi, nblock_flops = self._get_block_oi_and_flops(
                    name, elem, loads, to_ignore, inner_pragmas)
                inner_pragmas = copy(pragmas)
                block_oi += nblock_oi
                block_flops += nblock_flops
            elif isinstance(elem, For):
                block_oi += self._get_for_flops(
                    name, elem, loads, to_ignore, inner_pragmas)
                inner_pragmas = copy(pragmas)
            elif isinstance(elem, Pragma) and self.openmp:
                inner_pragmas.append(elem)
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
