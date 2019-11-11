from itertools import groupby

__all__ = ['Queue']


class Queue(object):

    """
    A special queue to process objects in nested IterationSpaces based on
    a divide-and-conquer algorithm.

    Notes
    -----
    Subclasses must override :meth:`callback`, which may get executed either
    before (fdta -- first divide then apply) or after (fatd -- first apply
    then divide) the divide phase of the algorithm.
    """

    def callback(self, *args):
        raise NotImplementedError

    def process(self, elements):
        return self._process_fdta(elements, 1)

    def _process_fdta(self, elements, level, prefix=None):
        """
        fdta -> First Divide Then Apply
        """
        prefix = prefix or []

        # Divide part
        processed = []
        for pfx, g in groupby(elements, key=lambda i: i.itintervals[:level]):
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Recursion
                processed.extend(self._process_fdta(list(g), level + 1, pfx))

        # Apply callback
        processed = self.callback(processed, prefix)

        return processed

    def _process_fatd(self, elements, level, prefix=None):
        """
        fatd -> First Apply Then Divide
        """
        prefix = prefix or []

        # Divide part
        processed = []
        for pfx, g in groupby(elements, key=lambda i: i.itintervals[:level]):
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Apply callback
                _elements = self.callback(list(g), prefix)
                # Recursion
                processed.extend(self._process_fatd(_elements, level + 1, pfx))

        return processed
