import numpy as np
from collections import Iterable, Mapping
from functools import reduce
from multidict import MultiDict
from devito.logger import error


class ArgumentMap(MultiDict):
    """
    Specialised :class:`MultiDict` object that maps a single key to a
    list of potential values and provides a reduction method for
    retrieval.
    """

    def update(self, values):
        """
        Update internal mapping with standard dictionary semantics.
        """
        if isinstance(values, Mapping):
            self.extend(values)
        elif isinstance(values, Iterable) and not isinstance(values, str):
            for v in values:
                self.extend(v)
        else:
            self.extend(values)

    def unique(self, key):
        """
        Returns a unique value for a given key, if such a value
        exists, and raises a ``ValueError`` if it does not.

        :param key: Key for which to retrieve a unique value
        """
        candidates = self.getall(key)

        def compare_to_first(v):
            first = candidates[0]
            if isinstance(first, np.ndarray) or isinstance(v, np.ndarray):
                return (first == v).all()
            else:
                return first == v

        if len(candidates) == 1:
            return candidates[0]
        elif all(map(compare_to_first, candidates)):
            return candidates[0]
        else:
            error("Unable to find unique value for key %s, candidates: %s" %
                  (key, candidates))
            raise ValueError('Inconsistent values for argument reduction')

    def reduce(self, key, op=None):
        """
        Returns a reduction of all candidate values for a given key.

        :param key: Key for which to retrieve candidate values
        :param op: Operator for reduction among candidate values.
                   If not provided, a unique value will be returned,
                   or a ``ValueError`` raised if no unique value exists.
        """
        if op is None:
            # Return a unique value if it exists
            return self.unique(key)
        else:
            return reduce(op, self.getall(key))

    def reduce_all(self):
        """
        Returns a dictionary with reduced/unique values for all keys.
        """
        return {k: self.reduce(key=k) for k in self}
