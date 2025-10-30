from .C import CPrinter


class CIRPrinter(CPrinter):

    """
    A smarter printer for IETs undergoing lowering.
    """

    def _print_ComponentAccess(self, expr):
        try:
            ncomp = expr.function.ncomp
        except AttributeError:
            # E.g., the argument is a plain Symbol
            ncomp = 'N'
        return f"{self._print(expr.base)}.<{expr.sindex},{ncomp}>"
