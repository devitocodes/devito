import devito.types.basic as basic

__all__ = ['Array']


class Array(basic.Array):

    from_OPS = True
    is_Symbol = True

    @property
    def _C_name(self):
        return self.name
