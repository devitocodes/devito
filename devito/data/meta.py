from devito.tools import Tag

__all__ = ['DOMAIN', 'OWNED', 'HALO', 'NOPAD', 'LEFT', 'RIGHT', 'CENTER']


class DataRegion(Tag):
    pass


DOMAIN = DataRegion('domain')
OWNED = DataRegion('owned')  # within DOMAIN
HALO = DataRegion('halo')
NOPAD = DataRegion('nopad')  # == DOMAIN+HALO


class DataSide(Tag):
    pass


LEFT = DataSide('left')
RIGHT = DataSide('right')
CENTER = DataSide('center')
