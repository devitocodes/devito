from devito.tools import Tag

__all__ = ['DOMAIN', 'OWNED', 'HALO', 'LEFT', 'RIGHT', 'CENTER']


class DataRegion(Tag):
    pass


DOMAIN = DataRegion('domain')
OWNED = DataRegion('owned')
HALO = DataRegion('halo')


class DataSide(Tag):
    pass


LEFT = DataSide('left')
RIGHT = DataSide('right')
CENTER = DataSide('center')
