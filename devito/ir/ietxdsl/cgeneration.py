import io
from devito.ir.ietxdsl.operations import SSAValue, Callable, BlockArgument, Addi, \
    Modi, StructDecl, Statement, Iteration, Assign, PointerCast, Idx, Initialise, \
    List, Constant, Dict

SSAValueNames: Dict[SSAValue, str] = {}


class CGeneration:

    def __init__(self):
        self.output = io.StringIO()
        self.indentation = 0
        self.iterator_names = {}

    def str(self):
        s = self.output.getvalue()
        self.output.close()
        return s

    def indent(self):
        self.indentation += 2

    def dedent(self):
        self.indentation -= 2

    def print(self, *args, **kwargs):
        indent = True

        if 'indent' in kwargs.keys():
            if not kwargs['indent']:
                indent = False
            kwargs.pop('indent')

        if indent:
            print(" " * self.indentation, file=self.output, end='')

        print(*args, file=self.output, **kwargs)

    # To translate code such as:
    #
    #   cst42 := iet.constant(42)
    #   cst3 := iet.constant(3)
    #   iet.addi(cst42, cst3)
    #
    # into a single-line expression such as:
    #
    #   42 + 3
    #
    # we look at the very last operation in the module and then walk iand
    # recursively print the following tree expressed by the def-use chain of
    # these operations.
    def printModule(self, module):
        # Get the last operation in the module
        self.printOperation(module.ops[-1])

    def printCallable(self, callable_op: Callable):
        arglist = callable_op.body.blocks[0].args
        i = 0
        self.print("int Kernel(", end='', indent=False)
        for arg in arglist:
            SSAValueNames[arg] = callable_op.parameters.data[i].data
            i = i + 1
        i = 0
        num_params = len(list(callable_op.types.data))
        # TODO: fix this workaround
        # need separate loop because only header parameters have types
        for op_type in callable_op.types.data:
            self.print(op_type.data, end=' ', indent=False)
            self.print(callable_op.header_parameters.data[i].data,
                       end='',
                       indent=False)
            if i < (num_params - 1):
                self.print(",", end='', indent=False)
            i = i + 1
        self.print("){")
        self.indent()
        for each_op in callable_op.body.ops:
            self.printOperation(each_op)
        self.print("return 0;")
        self.dedent()
        self.print("}")
        pass

    def printIteration(self, iteration_op: Iteration):
        ssa_val = iteration_op.body.blocks[0].args[0]
        iterator = "i_" + str(len(self.iterator_names))
        SSAValueNames[ssa_val] = iterator
        self.iterator_names[
            iteration_op.regions[0].blocks[0].args[0]] = iterator
        lower_bound = iteration_op.limits.data[0].data
        upper_bound = iteration_op.limits.data[1].data
        increment = iteration_op.limits.data[2].data
        self.print(f"for (int {iterator} = {lower_bound}; ", end='')
        self.print(f"{iterator} <= {upper_bound}; ", end='', indent=False)
        self.print(f"{iterator} += {increment}) ", indent=False)
        self.print("{")
        self.indent()
        self.printOperation(iteration_op.body.ops)
        self.dedent()
        self.print("}")
        pass

    def printResult(self, result):
        if isinstance(result, BlockArgument):
            name = SSAValueNames[result]
            self.print(name, indent=False, end="")
            return
        if isinstance(result, SSAValue):
            self.printOperation(result.op)

    def printOperation(self, operation):
        if isinstance(operation, BlockArgument):
            self.print("uuu", indent=False, end="")
            return
        if (isinstance(operation, List)):
            for op in operation:
                if isinstance(op, Constant) or isinstance(
                        op, Addi) or isinstance(op, Idx) or isinstance(
                            op, Modi):
                    continue
                self.printOperation(op)
            return

        if (isinstance(operation, Constant)):
            self.print(operation.value.parameters[0].data,
                       indent=False,
                       end='')
            return

        if (isinstance(operation, Addi)):
            self.printResult(operation.input1)
            self.print(" + ", end='', indent=False)
            self.printResult(operation.input2)
            return

        if (isinstance(operation, Modi)):
            self.print("(", end="", indent=False)
            self.printResult(operation.input1)
            self.print(")", end="", indent=False)
            self.print(" % ", end='', indent=False)
            self.print("(", end="", indent=False)
            self.printResult(operation.input2)
            self.print(")", end="", indent=False)
            return

        if (isinstance(operation, Callable)):
            self.printCallable(operation)
            return

        if (isinstance(operation, Iteration)):
            self.printIteration(operation)
            return

        if (isinstance(operation, Assign)):
            self.print("", end="")
            self.printResult(operation.lhs)
            self.print(" = ", indent=False, end="")
            self.printResult(operation.rhs)
            self.print("", indent=False)
            return

        if (isinstance(operation, Initialise)):
            type = operation.results[0].typ.width.name
            self.print(type, indent=True, end=" ")

            assignee = operation.id.data
            self.print(assignee, indent=False, end="")
            ssa_val = operation.lhs
            SSAValueNames[ssa_val] = assignee

            self.print(" = ", indent=False, end="")
            self.printResult(operation.rhs)
            self.print("", indent=False)
            return

        if (isinstance(operation, Idx)):
            self.printResult(operation.array)
            self.print("[", indent=False, end="")
            self.printResult(operation.index)
            self.print("]", indent=False, end="")
            return

        if (isinstance(operation, PointerCast)):
            self.print(operation.statement.data)
            return

        if (isinstance(operation, Statement)):
            self.print(operation.statement.data)
            return

        if (isinstance(operation, StructDecl)):
            self.print("struct", indent=False, end=" ")
            self.print(operation.id.data)
            self.print("{")
            for field in operation.fields.data:
                self.print(field.data)
            self.print("};")
            return

        self.print(f"// Operation {operation.name} not supported inprinter")
