import warnings

from devito.logger import warning as log_warning


class DevitoWarning(Warning):
    pass


def warn(message, category=None, stacklevel=2, source=None):
    """
    `devito.warn` follows the Python call signature for `warning.warn`:
    https://docs.python.org/3/library/warnings.html#warnings.warn

    Parameters
    ----------
    message: str or Warning
        Message to display
    category: None or Warning
        Leave as None to get a `DevitoWarning`
    stacklevel: int
        Set a custom stack level
    source: None or object
        the destroyed object which emitted a `ResourceWarning`
    """
    warning_type = None
    if isinstance(message, Warning):
        warning_type = message.__class__.__name__
    elif category is not None:
        warning_type = category.__name__

    if warning_type is not None:
        message = f'from {warning_type}: {str(message)}'

    log_warning(message)
    warnings.warn(message, category=DevitoWarning, stacklevel=stacklevel, source=source)
