import devito.types.basic as basic

__all__ = ['Array']


class Array(basic.Array):

    from_OPS = True
    is_Symbol = True

    def __init__(self, is_Write, *args, **kwargs):
        self.is_Write = is_Write
        super().__init__(args, kwargs)

    @property
    def _C_name(self):
        return self.name

    @property
    def _C_typename(self):
        if self.is_Write:
            return super()._C_typename

        return "const %s" % super()._C_typename
