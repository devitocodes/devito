import io
from devito.ir.ietxdsl.operations import *


class CGeneration:
    def __init__(self):
        self.output = io.StringIO()

    def str(self):
        s = self.output.getvalue()
        self.output.close()
        return s

    def print(self, *args, **kwargs):
        print(*args, file=self.output, **kwargs)

    def printModule(self, module):
        # Get the last operation in the module
        self.printOperation(module.ops[-1])

    def printOperation(self, operation):
        if (isinstance(operation, Constant)):
            self.print(operation.value.parameters[0].data, end='')
            return

        if (isinstance(operation, Addi)):
            self.printOperation(operation.operands[0].op)
            self.print(" + ", end='')
            self.printOperation(operation.operands[1].op)
            return

        assert False, "Trying to generate C for an unsupported operation"
