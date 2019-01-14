from collections import OrderedDict

import numpy as np
from sympy import cos, sin

from devito.dse.backends import AbstractRewriter, dse_pass
from devito.dse.manipulation import common_subexprs_elimination
from devito.symbolics import (Eq, bhaskara_cos, bhaskara_sin, retrieve_indexed,
                              q_affine, q_scalar)
from devito.types import Scalar


class BasicRewriter(AbstractRewriter):

    def _pipeline(self, state):
        self._eliminate_intra_stencil_redundancies(state)
        self._extract_nonaffine_indices(state)
        self._extract_increments(state)

    @dse_pass
    def _extract_nonaffine_indices(self, cluster, template, **kwargs):
        """
        Extract non-affine array indices, and assign them to temporaries.
        """
        make = lambda: Scalar(name=template(), dtype=np.int32).indexify()

        mapper = OrderedDict()
        for e in cluster.exprs:
            # Note: using mode='all' and then checking for presence in the mapper
            # (a few lines below), rather retrieving unique indexeds only (a set),
            # is the key to deterministic code generation
            for indexed in retrieve_indexed(e, mode='all'):
                for i, d in zip(indexed.indices, indexed.function.indices):
                    if q_affine(i, d) or q_scalar(i):
                        continue
                    elif i not in mapper:
                        mapper[i] = make()

        processed = [Eq(v, k) for k, v in mapper.items()]
        processed.extend([e.xreplace(mapper) for e in cluster.exprs])

        return cluster.rebuild(processed)

    @dse_pass
    def _extract_increments(self, cluster, template, **kwargs):
        """
        Extract the RHS of non-local tensor expressions performing an associative
        and commutative increment, and assign them to temporaries.
        """
        processed = []
        for e in cluster.exprs:
            if e.is_Increment and e.lhs.function.is_Input:
                handle = Scalar(name=template(), dtype=e.dtype).indexify()
                if e.rhs.is_Symbol:
                    extracted = e.rhs
                else:
                    extracted = e.rhs.func(*[i for i in e.rhs.args if i != e.lhs])
                processed.extend([Eq(handle, extracted), e.func(e.lhs, handle)])
            else:
                processed.append(e)

        return cluster.rebuild(processed)

    @dse_pass
    def _eliminate_intra_stencil_redundancies(self, cluster, template, **kwargs):
        """
        Perform common subexpression elimination, bypassing the tensor expressions
        extracted in previous passes.
        """

        skip = [e for e in cluster.exprs if e.lhs.base.function.is_Array]
        candidates = [e for e in cluster.exprs if e not in skip]

        make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()

        processed = common_subexprs_elimination(candidates, make)

        return cluster.rebuild(skip + processed)

    @dse_pass
    def _optimize_trigonometry(self, cluster, **kwargs):
        """
        Rebuild ``exprs`` replacing trigonometric functions with Bhaskara
        polynomials.
        """

        processed = []
        for expr in cluster.exprs:
            handle = expr.replace(sin, bhaskara_sin)
            handle = handle.replace(cos, bhaskara_cos)
            processed.append(handle)

        return cluster.rebuild(processed)
