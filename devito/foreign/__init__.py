"""
The ``foreign`` Devito backend is meant to be used by codes that don't
run Python natively. This backend is only capable of generating and compiling
kernels; however, kernels must be executed explicitly from outside Devito.
Further, with the ``foreign`` backed, Devito doesn't allocate any data.
"""

# The following used by backends.backendSelector
from devito.interfaces import DenseData, TimeData  # noqa
from devito.pointdata import PointData  # noqa
from devito.foreign.operator import Operator  # noqa
