from collections import OrderedDict
from itertools import product

from devito.archinfo import Platform
from devito.exceptions import InvalidOperator
from devito.tools import Singleton

__all__ = ['operator_registry', 'operator_selector']


class OperatorRegistry(OrderedDict, metaclass=Singleton):

    """
    A registry for Operators:

        (platform, mode, language) -> operator

    where:

        * `platform` is an object of type Platform, that is the architecture
          the code is generated for.
        * `mode` is the optimization level (e.g., `advanced`).
        * `language` is the generated code language (default is C+OpenMP+MPI,
          but in the future it could also be OpenACC or CUDA.
        * `operator` is an object of type Operator.
    """

    _modes = ('noop', 'advanced')
    _languages = ('C',)
    _accepted = _modes + tuple(product(_modes, _languages))

    def add(self, operator, platform, mode, language='C'):
        assert issubclass(platform, Platform)
        assert mode in OperatorRegistry._modes or mode == 'custom'

        self[(platform, mode, language)] = operator

    def fetch(self, platform=None, mode=None, language='C', **kwargs):
        """
        Retrieve an Operator for the given `<platform, mode, language>`.
        """
        if mode not in OperatorRegistry._modes:
            # DLE given as an arbitrary sequence of passes
            mode = 'custom'

        if language not in OperatorRegistry._languages:
            raise ValueError("Unknown language `%s`" % language)

        for cls in platform.__class__.mro():
            for (p, m, l), kls in self.items():
                if issubclass(p, cls) and m == mode and l == language:
                    return kls

        raise InvalidOperator("Cannot compile an Operator for `%s`" % str((p, m, l)))


operator_registry = OperatorRegistry()
"""To be populated by the individual backends."""


operator_selector = operator_registry.fetch
