from itertools import groupby

__all__ = ['Queue']


class Queue(object):

    """
    A special queue to process Clusters based on a divide-and-conquer algorithm.

    Notes
    -----
    Subclasses must override :meth:`callback`, which may get executed either
    before (fdta -- first divide then apply) or after (fatd -- first apply
    then divide) the divide phase of the algorithm.
    """

    def callback(self, *args):
        raise NotImplementedError

    def process(self, clusters):
        return self._process_fdta(clusters, 1)

    def _make_key(self, element, level):
        itintervals = element.itintervals[:level]

        subkey = self._make_key_hook(element, level)

        return (itintervals,) + subkey

    def _make_key_hook(self, element, level):
        return ()

    def _process_fdta(self, clusters, level, prefix=None):
        """
        fdta -> First Divide Then Apply
        """
        prefix = prefix or []

        # Divide part
        processed = []
        for k, g in groupby(clusters, key=lambda i: self._make_key(i, level)):
            pfx = k[0]
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Recursion
                processed.extend(self._process_fdta(list(g), level + 1, pfx))

        # Apply callback
        processed = self.callback(processed, prefix)

        return processed

    def _process_fatd(self, clusters, level):
        """
        fatd -> First Apply Then Divide
        """
        # Divide part
        processed = []
        for k, g in groupby(clusters, key=lambda i: self._make_key(i, level)):
            pfx = k[0]
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Apply callback
                _clusters = self.callback(list(g), pfx)
                # Recursion
                processed.extend(self._process_fatd(_clusters, level + 1))

        return processed
