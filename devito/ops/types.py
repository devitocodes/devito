import devito.types.basic as basic

__all__ = ['String']


class String(basic.Basic):

    is_StringLiteral = True

    def __init__(self, value):
        self.value = value

    @property
    def _C_name(self):
        return '"%s"' % self.value
