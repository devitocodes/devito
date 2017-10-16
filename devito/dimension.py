from sympy import Symbol
from cached_property import cached_property

__all__ = ['Dimension', 'x', 'y', 'z', 't', 'p', 'd', 'time']


class Dimension(Symbol):

    is_Buffered = False
    is_Lowered = False
    is_Fixed = False
    is_Space = False

    """Index object that represents a problem dimension and thus
    defines a potential iteration space.

    :param name: Name of the dimension symbol.
    :param reverse: Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """
    
    _argument_names = ["size", "start", "end"]
    
    def __new__(cls, name, **kwargs):
        newobj = Symbol.__new__(cls, name)
        newobj.reverse = kwargs.get('reverse', False)
        newobj.spacing = kwargs.get('spacing', Symbol('h_%s' % name))
        return newobj

    def __str__(self):
        return self.name

    @property
    def symbolic_size(self):
        """The symbolic size of this dimension."""
        return Symbol(self.size_name)

    @property
    def symbolic_start(self):
        return Symbol(self.start_name)
    
    @property
    def symbolic_end(self):
        return Symbol(self.end_name)

    @property
    def size(self):
        return None

    @property
    def limits(self):
        return (self.symbolic_start, self.symbolic_end, 1)

    @property
    def symbolic_extent(self):
        """Return the extent of the loop over this dimension.
        Would be the same as size if using default values """
        return (self.symbolic_end - self.symbolic_start)

    @property
    def size_name(self):
        return "%s_size" % self.name

    @property
    def start_name(self):
        return "%s_s" % self.name

    @property
    def end_name(self):
        return "%s_e" % self.name


class SpaceDimension(Dimension):

    is_Space = True

    """
    Dimension symbol to represent a space dimension that defines the
    extent of physical grid. :class:`SpaceDimensions` create dedicated
    shortcut notations for spatial derivatives on :class:`Function`
    symbols.

    :param name: Name of the dimension symbol.
    :param reverse: Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """


class BufferedDimension(Dimension):

    is_Buffered = True

    """
    Dimension symbol that implies modulo buffered iteration.

    :param parent: Parent dimension over which to loop in modulo fashion.
    """

    def __new__(cls, name, parent, **kwargs):
        newobj = Symbol.__new__(cls, name)
        assert isinstance(parent, Dimension)
        newobj.parent = parent
        newobj.modulo = kwargs.get('modulo', 2)
        return newobj

    @property
    def reverse(self):
        return self.parent.reverse

    @property
    def spacing(self):
        return self.parent.spacing


class LoweredDimension(Dimension):

    is_Lowered = True

    """
    Dimension symbol representing modulo iteration created when resolving a
    :class:`BufferedDimension`.

    :param buffered: BufferedDimension from which this Dimension originated.
    :param offset: Offset value used in the modulo iteration.
    """

    def __new__(cls, name, buffered, offset, **kwargs):
        newobj = Symbol.__new__(cls, name)
        assert isinstance(buffered, BufferedDimension)
        newobj.buffered = buffered
        newobj.offset = offset
        return newobj

    @property
    def origin(self):
        return self.buffered + self.offset

    @property
    def size(self):
        return self.buffered.size

    @property
    def reverse(self):
        return self.buffered.reverse


# Default dimensions for time
time = Dimension('time', spacing=Symbol('s'))
t = BufferedDimension('t', parent=time)

# Default dimensions for space
x = SpaceDimension('x', spacing=Symbol('h_x'))
y = SpaceDimension('y', spacing=Symbol('h_y'))
z = SpaceDimension('z', spacing=Symbol('h_z'))

d = Dimension('d')
p = Dimension('p')
