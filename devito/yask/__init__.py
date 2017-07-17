"""
The ``yask`` Devito backend uses the YASK stencil optimizer to generate,
JIT-compile, and run kernels.
"""

# The following used by backends.backendSelector
from devito.pointdata import PointData  # noqa
from devito.yask.interfaces import DenseData, TimeData  # noqa
from devito.yask.operator import Operator  # noqa
