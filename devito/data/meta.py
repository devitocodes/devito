from devito.tools import Tag

__all__ = ['DOMAIN', 'OWNED', 'HALO', 'NOPAD', 'FULL',
           'LEFT', 'RIGHT', 'CENTER']


class DataRegion(Tag):
    pass


DOMAIN = DataRegion('domain')
OWNED = DataRegion('owned')  # within DOMAIN
HALO = DataRegion('halo')
NOPAD = DataRegion('nopad')  # == DOMAIN+HALO
FULL = DataRegion('full')  # == DOMAIN+HALO+PADDING


class DataSide(Tag):

    def __str__(self):
        return self.name

    __repr__ = __str__


LEFT = DataSide('left', -1)
CENTER = DataSide('center', 0)
RIGHT = DataSide('right', 1)
