import re
from collections import defaultdict
from copy import copy
from ctypes import Structure, byref, c_double, c_longlong

import numpy as np

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

    def __init__(self, openmp=False, dtype=np.float32):
        self.openmp = openmp
        self.float_size = np.dtype(dtype).itemsize
        self.profiled = []
        self.t_fields = []
        self.f_fields = []

        # _C_ fields are ctypes structs used for code generation
        self._C_timings = None
        self._C_flops = None

        self.num_flops = {}

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
        self.f_fields.append((name, c_longlong))

        self.num_flops[name] = FlopsCounter(code, name, self.openmp,
                                            self.float_size, to_ignore or []).run()

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
        """GFlops per loop iteration, keyed by code section."""
        return self.num_flops


class FlopsCounter(object):

    """Compute the operational intensity of a stencil."""

    def __init__(self, code, name, openmp, float_size, to_ignore):
        self.code = code
        self.name = name
        self.openmp = openmp
        self.float_size = float_size

        self.to_ignore = [
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
        self.seen = set()

    def run(self):
        """
        Calculates the total operation intensity of the code provided.
        If needed, lets the C code calculate it.
        """
        num_flops = 0

        for elem in self.code:
            if isinstance(elem, Assign):
                num_flops += self._handle_assign(elem)
            elif isinstance(elem, For):
                num_flops += self._handle_for(elem)
            elif isinstance(elem, Block):
                num_flops += self._handle_block(elem)[0]
            else:
                # no op
                pass

        return num_flops

    def _handle_for(self, loop):
        loop_flops = 0
        loop_oi_f = 0

        if isinstance(loop.body, Assign):
            loop_flops = self._handle_assign(loop.body)
            loop_oi_f = loop_flops
        elif isinstance(loop.body, Block):
            loop_oi_f, loop_flops = self._handle_block(loop.body)
        elif isinstance(loop.body, For):
            loop_oi_f = self._handle_for(loop.body)
        else:
            # no op
            pass

        old_body = loop.body

        while isinstance(old_body, Block) and isinstance(old_body.contents[0], Statement):
            old_body = old_body.contents[1]

        if loop_flops == 0:
            if old_body in self.seen:
                return 0

            self.seen.add(old_body)

            return loop_oi_f

        if old_body in self.seen:
            return 0

        self.seen.add(old_body)
        return loop_oi_f

    def _handle_block(self, block):
        block_flops = 0
        block_oi = 0

        for elem in block.contents:
            if isinstance(elem, Assign):
                a_flops = self._handle_assign(elem)
                block_flops += a_flops
                block_oi += a_flops
            elif isinstance(elem, Block):
                nblock_oi, nblock_flops = self._handle_block(elem)
                block_oi += nblock_oi
                block_flops += nblock_flops
            elif isinstance(elem, For):
                block_oi += self._handle_for(elem)
            else:
                # no op
                pass

        return block_oi, block_flops

    def _handle_assign(self, assign):
        flops = 0

        # removing casting statements and function calls to floor
        # that can confuse the parser
        string = assign.lvalue + " " + assign.rvalue

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
