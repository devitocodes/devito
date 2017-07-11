from devito.dse.backends import AdvancedRewriter, dse_pass
from devito.dse.inspection import estimate_cost
from devito.dse.manipulation import xreplace_constrained
from devito.dse.queries import iq_timevarying

from devito.interfaces import ScalarFunction


class SpeculativeRewriter(AdvancedRewriter):

    def _pipeline(self, state):
        self._extract_time_varying(state)
        self._extract_time_invariants(state)
        self._eliminate_inter_stencil_redundancies(state)
        self._eliminate_intra_stencil_redundancies(state)
        self._factorize(state)

    @dse_pass
    def _extract_time_varying(self, cluster, **kwargs):
        """
        Extract time-varying subexpressions, and assign them to temporaries.
        Time varying subexpressions arise for example when approximating
        derivatives through finite differences.
        """

        template = self.conventions['time-dependent'] + "%d"
        make = lambda i: ScalarFunction(name=template % i).indexify()

        rule = iq_timevarying(cluster.trace)

        cm = lambda i: estimate_cost(i) > 0

        processed, _ = xreplace_constrained(cluster.exprs, make, rule, cm)

        return cluster.reschedule(processed)


class AggressiveRewriter(SpeculativeRewriter):

    def _pipeline(self, state):
        self._extract_time_varying(state)
        self._extract_time_invariants(state, with_cse=False)


class CustomRewriter(AggressiveRewriter):

    def _pipeline(self, state):
        raise NotImplementedError
