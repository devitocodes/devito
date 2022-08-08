import difflib

__all__ = ['unidiff_output', 'print_kernel']


def unidiff_output(expected, actual):
    """
    Return a string containing the unified diff of two multiline strings.
    """
    expected = expected.splitlines(1)
    actual = actual.splitlines(1)

    diff = difflib.unified_diff(expected, actual)

    return ''.join(diff)


def print_kernel(op):
    """
    Print the core part of an Operator used in this notebook.
    This is less verbose than printing the whole Operator.
    """
    print(op.body.body[-1])
