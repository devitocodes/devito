from __future__ import absolute_import

from devito.ir.clusters import ClusterGroup, groupby
from devito.dse.backends import (BasicRewriter, AdvancedRewriter, SpeculativeRewriter,
                                 AggressiveRewriter, CustomRewriter)
from devito.exceptions import DSEException
from devito.logger import dse_warning
from devito.parameters import configuration

__all__ = ['rewrite']


modes = {
    'basic': BasicRewriter,
    'advanced': AdvancedRewriter,
    'speculative': SpeculativeRewriter,
    'aggressive': AggressiveRewriter
}
"""The DSE transformation modes."""

configuration.add('dse', 'advanced', list(modes))


def rewrite(clusters, mode='advanced'):
    """
    Transform N :class:`Cluster` objects of SymPy expressions into M
    :class:`Cluster` objects of SymPy expressions with reduced
    operation count, with M >= N.

    :param clusters: The clusters to be transformed.
    :param mode: drive the expression transformation

    The ``mode`` parameter recognises the following values: ::

         * 'noop': Do nothing.
         * 'basic': Apply common sub-expressions elimination.
         * 'advanced': Apply all transformations that will reduce the
                       operation count w/ minimum increase to the memory pressure,
                       namely 'basic', factorization, CSRE for time-invariants only.
         * 'speculative': Like 'advanced', but apply CSRE also to time-varying
                          sub-expressions, which might further increase the memory
                          pressure.
         * 'aggressive': Like 'speculative', but apply CSRE to any non-trivial
                         sub-expression (i.e., anything that is at least in a
                         sum-of-products form). This may substantially increase
                         the memory pressure.
    """
    if not (mode is None or isinstance(mode, str)):
        raise ValueError("Parameter 'mode' should be a string, not %s." % type(mode))

    if mode is None or mode == 'noop':
        return clusters

    processed = ClusterGroup()
    for cluster in clusters:
        if cluster.is_dense:
            if mode in modes:
                processed.extend(modes[mode]().run(cluster))
            else:
                try:
                    processed.extend(CustomRewriter().run(cluster))
                except DSEException:
                    dse_warning("Unknown rewrite mode(s) %s" % mode)
                    processed.append(cluster)
        else:
            # Downgrade sparse clusters to basic rewrite mode since it's
            # pointless to expose loop-redundancies when the iteration space
            # only consists of a few points
            processed.extend(BasicRewriter(False).run(cluster))

    return groupby(processed).finalize()
