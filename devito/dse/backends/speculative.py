from devito.dse.backends import AdvancedRewriter, dse_pass
from devito.symbolics import (estimate_cost, xreplace_constrained,
                              iq_timevarying, q_leaf, q_sum_of_product, q_terminalop)
from devito.types import Scalar


class SpeculativeRewriter(AdvancedRewriter):

    def _pipeline(self, state):
        self._extract_time_varying(state)
        self._extract_time_invariants(state, costmodel=lambda e: e.is_Function)
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

        make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
        rule = iq_timevarying(cluster.trace)
        costmodel = lambda i: estimate_cost(i) > 0
        processed, _ = xreplace_constrained(cluster.exprs, make, rule, costmodel)

        return cluster.rebuild(processed)


class AggressiveRewriter(SpeculativeRewriter):

    def _pipeline(self, state):
        # Three CIRE phases, progressively searching for less structure
        self._extract_time_varying(state)
        self._extract_time_invariants(state, with_cse=False,
                                      costmodel=lambda e: e.is_Function)
        self._eliminate_inter_stencil_redundancies(state)

        self._extract_sum_of_products(state)
        self._eliminate_inter_stencil_redundancies(state)
        self._extract_sum_of_products(state)

        self._factorize(state)
        self._eliminate_intra_stencil_redundancies(state)

    @dse_pass
    def _extract_sum_of_products(self, cluster, template, **kwargs):
        """
        Extract sub-expressions in sum-of-product form, and assign them to temporaries.
        """
        make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
        rule = q_sum_of_product
        costmodel = lambda e: not (q_leaf(e) or q_terminalop(e))
        processed, _ = xreplace_constrained(cluster.exprs, make, rule, costmodel)

        return cluster.rebuild(processed)


class CustomRewriter(AggressiveRewriter):

    def _pipeline(self, state):
        raise NotImplementedError
