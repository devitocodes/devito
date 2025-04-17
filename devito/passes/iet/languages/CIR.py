from .C import CPrinter


class CIRPrinter(CPrinter):

    """
    A smarter printer for IETs undergoing lowering.
    """

    def _print_ComponentAccess(self, expr):
        return f"{self._print(expr.base)}.<{expr.sindex},{expr.function.ncomp}>"
