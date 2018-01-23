class DevitoError(Exception):
    pass


class CompilationError(DevitoError):
    pass


class InvalidArgument(DevitoError):
    pass


class InvalidOperator(DevitoError):
    pass


class VisitorException(DevitoError):
    pass


class DSEException(DevitoError):
    pass


class DLEException(DevitoError):
    pass
