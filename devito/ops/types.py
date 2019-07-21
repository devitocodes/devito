import devito.types.basic as basic


class OpsAccessible(basic.AbstractObject):

    def __init__(self, symbol, access_indexes, read_only):
        self.symbol = symbol
        self.access_indexes = [str(idx) for idx in access_indexes]
        self.read_only = read_only
        super().__init__(self._C_name, symbol.dtype)

    @property
    def _C_name(self):
        return '%s(%s)' % (self.symbol.name, ",".join(self.access_indexes))

    @property
    def _C_typename(self):
        return '%sACC<%s>' % ('const ' if self.read_only else '', self.symbol._C_typename)

    @property
    def _C_typedata(self):
        return 'ACC<%s>' % self.symbol._C_typename
