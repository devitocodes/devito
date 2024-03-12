class DevitoError(Exception):
    pass


class CompilationError(DevitoError):
    pass


class InvalidArgument(DevitoError):
    pass


class InvalidOperator(DevitoError):
    pass


class ExecutionError(DevitoError):
    pass


class VisitorException(DevitoError):
    pass
