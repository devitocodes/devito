from devito.dse.backends import AdvancedRewriter, dse_pass
from devito.dse.inspection import estimate_cost
from devito.dse.manipulation import xreplace_constrained
from devito.dse.queries import iq_timevarying, q_leaf, q_sum_of_product, q_terminalop

from devito.interfaces import ScalarFunction


class SpeculativeRewriter(AdvancedRewriter):

    def _pipeline(self, state):
        self._extract_time_varying(state)
        self._extract_time_invariants(state)
        self._eliminate_inter_stencil_redundancies(state)
        self._eliminate_intra_stencil_redundancies(state)
        self._factorize(state)

    @dse_pass
    def _extract_time_varying(self, cluster, template, **kwargs):
        """
        Extract time-varying subexpressions, and assign them to temporaries.
        Time varying subexpressions arise for example when approximating
        derivatives through finite differences.
        """

        make = lambda i: ScalarFunction(name=template(i)).indexify()
        rule = iq_timevarying(cluster.trace)
        cm = lambda i: estimate_cost(i) > 0
        processed, _ = xreplace_constrained(cluster.exprs, make, rule, cm)

        return cluster.reschedule(processed)


class AggressiveRewriter(SpeculativeRewriter):

    def _pipeline(self, state):
        self._extract_time_varying(state)
        self._extract_time_invariants(state, with_cse=False)

        # Iteratively apply CSRE until no further redundancies are spotted
        i = 0
        while state.has_changed:
            self._eliminate_inter_stencil_redundancies(state, start=i)
            self._extract_sum_of_products(state, start=i)
            i += 1

        self._eliminate_intra_stencil_redundancies(state)
        self._factorize(state)

    @dse_pass
    def _extract_sum_of_products(self, cluster, template, **kwargs):
        """
        Extract sub-expressions in sum-of-product form, and assign them to temporaries.
        """
        targets = [i for i in cluster.exprs if i.is_tensor]
        untouched = [i for i in cluster.exprs if i not in targets]

        make = lambda i: ScalarFunction(name=template(i)).indexify()
        rule = q_sum_of_product
        cm = lambda e: not (q_leaf(e) or q_terminalop(e))
        processed, _ = xreplace_constrained(targets, make, rule, cm)

        return cluster.reschedule(untouched + processed)


class CustomRewriter(AggressiveRewriter):

    def _pipeline(self, state):
        raise NotImplementedError
