from devito.functions.dense import Function, DiscretizedFunction

__all__ = ["VectorFunction"]


class VectorFunction(object):
    """A :class:`DiscretizedFunction` providing operations to express
    finite-difference approximation. A ``VectorFunction`` encapsulates
    vector valued space-varying data (one value per dimension i.e u_x, u_y, u_z in 3D)
    for time-varying data, use :class:`VectorTimeFunction`.

    Parameters
    ----------
    name : str
        Name of the symbol.
    grid : Grid, optional
        Carries shape, dimensions, and dtype of the Function. When grid is not
        provided, shape and dimensions must be given. For MPI execution, a
        Grid is compulsory.
    space_order : int or 3-tuple of ints, optional
        Discretisation order for space derivatives. Defaults to 1. ``space_order`` also
        impacts the number of points available around a generic point of interest.  By
        default, ``space_order`` points are available on both sides of a generic point of
        interest, including those nearby the grid boundary. Sometimes, fewer points
        suffice; in other scenarios, more points are necessary. In such cases, instead of
        an integer, one can pass a 3-tuple ``(o, lp, rp)`` indicating the discretization
        order (``o``) as well as the number of points on the left (``lp``) and right
        (``rp``) sides of a generic point of interest.
    shape : tuple of ints, optional
        Shape of the domain region in grid points. Only necessary if ``grid`` isn't given.
    dimensions : tuple of Dimension, optional
        Dimensions associated with the object. Only necessary if ``grid`` isn't given.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.
    staggered : Dimension or tuple of Dimension or Stagger, optional
        Define how the Function is staggered.
    padding : int or tuple of ints, optional
        Allocate extra grid points to maximize data access alignment. When a tuple
        of ints, one int per Dimension should be provided.
    initializer : callable or any object exposing the buffer interface, optional
        Data initializer. If a callable is provided, data is allocated lazily.
    allocator : MemoryAllocator, optional
        Controller for memory allocation. To be used, for example, when one wants
        to take advantage of the memory hierarchy in a NUMA architecture. Refer to
        `default_allocator.__doc__` for more information.


    Notes
    -----

    The parameters must always be given as keyword arguments, since SymPy
    uses ``*args`` to (re-)create the dimension arguments of the symbolic object.
    """

    is_VectorFunction = True

    _property_names = ['name', 'space_order', 'components']

    def __init__(self, *args, **kwargs):
        # if not self._cached():
        self._name = kwargs.get("name")
        space_order = kwargs.get("space_order", 1)
        # Get gird
        grid = kwargs.get("grid")
        # Each component is a Function
        components = []
        for d in grid.dimensions:
            components += [Function(name=self._name + '_' + d.name,
                                    grid=grid, space_order=space_order)]
        self._components = tuple(components)
        print("hello")

    def __repr__(self):
        return str(tuple(i.__repr__() for i in self._components))

    @property
    def name(self):
        return self._name

    @property
    def components(self):
        return self._components

    def __getattr__(self, name):
        """
        Return tuple of attribute for each function
        .. note::

            This method acts as a fallback for __getattribute__
        """
        if name == "name":
            return self._name
        elif name == "components":
            return self._components
        else:
            return tuple(i.__getattr__(name) for i in self._components)
