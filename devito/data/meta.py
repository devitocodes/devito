from devito.tools import Tag

__all__ = ['DOMAIN', 'CORE', 'OWNED', 'HALO', 'NOPAD', 'FULL',
           'LEFT', 'RIGHT', 'CENTER']


class DataRegion(Tag):

    def __str__(self):
        return self.name

    __repr__ = __str__


HALO = DataRegion('halo', 0)
CORE = DataRegion('core', 1)  # within DOMAIN
OWNED = DataRegion('owned', 2)  # within DOMAIN
DOMAIN = DataRegion('domain', 3)  # == CORE + OWNED
NOPAD = DataRegion('nopad', 4)  # == DOMAIN+HALO
FULL = DataRegion('full', 5)  # == DOMAIN+HALO+PADDING


class DataSide(Tag):

    def __init__(self, name, val, flipto=None):
        super(DataSide, self).__init__(name, val)
        self.flipto = flipto
        if flipto is not None:
            flipto.flipto = self

    def flip(self):
        if self.flipto is not None:
            return self.flipto
        else:
            return self

    def __str__(self):
        return self.name

    __repr__ = __str__


LEFT = DataSide('left', -1)
CENTER = DataSide('center', 0)
RIGHT = DataSide('right', 1, LEFT)
