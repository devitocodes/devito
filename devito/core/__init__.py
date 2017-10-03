"""
The ``core`` Devito backend is simply a "shadow" of the ``base`` backend,
common to all other backends. The ``core`` backend (and therefore the ``base``
backend as well) are used to run Devito on standard CPU architectures.
"""

# The following used by backends.backendSelector
from devito.function import Constant, Function, TimeFunction, SparseFunction  # noqa
from devito.core.operator import Operator  # noqa
