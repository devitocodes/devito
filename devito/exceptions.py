class DevitoError(Exception):
    """
    Base class for all Devito-related exceptions.
    """


class CompilationError(DevitoError):
    """
    Raised by the JIT compiler when the generated code cannot be compiled,
    typically due to a syntax error.

    These errors typically stem by one of the following:

    * A flaw in the user-provided equations;
    * An issue with the user-provided compiler options, not compatible
      with the given equations and/or backend;
    * A bug or a limitation in the Devito compiler itself.
    """


class InvalidArgument(ValueError, DevitoError):
    """
    Raised by the runtime system when an `op.apply(...)` argument, either a
    default argument or a user-provided one ("override"), is not valid.

    These are typically user-level errors, such as passing an incorrect
    type of argument, or passing an argument with an incorrect value.
    """


class InvalidOperator(DevitoError):
    """
    Raised by the runtime system when an `Operator` cannot be constructed.

    This generally occurs when an invalid combination of arguments is supplied to
    `Operator(...)` (e.g., a GPU-only optimization option is provided, while the
    Operator is being generated for the CPU).
    """


class ExecutionError(DevitoError):
    """
    Raised after `op.apply(...)` if a runtime error occurred during the execution
    of the Operator is detected.

    The nature of these errors can be various, for example:

    * Unstable numerical behavior (e.g., NaNs);
    * Out-of-bound accesses to arrays, which in turn can be caused by:
        * Incorrect user-provided equations (e.g., abuse of the "indexed notation");
        * A buggy optimization pass;
    * Running out of resources:
        * Memory (e.g., too many temporaries in the generated code);
        * Device shared memory or registers (e.g., too many threads per block);
    * etc.
    """
