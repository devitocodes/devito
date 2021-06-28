import threading

__all__ = ['sympy_mutex']


sympy_mutex = threading.RLock()
