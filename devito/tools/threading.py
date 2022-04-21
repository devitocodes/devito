import threading

__all__ = ['sympy_mutex', 'safe_dict_copy']


sympy_mutex = threading.RLock()


def safe_dict_copy(v):
    """
    Thread-safe copy of a dict.

    Being implemented as a retry loop around the copy(), this function is
    indicated for situations in which concurrent dict updates are unlikely,
    otherwise it might eventually cause performance degradations (in which
    case, lock-based solutions would be preferable).

    Notes
    -----
    See https://bugs.python.org/issue40327
    """
    while True:
        try:
            return v.copy()
        except RuntimeError:
            pass
