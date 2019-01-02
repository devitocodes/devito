from devito.functions.dense import Function, DiscretizedFunction

__all__ = ["VectorFunction"]


class VectorFunction(DiscretizedFunction):
    """A :class:`GridedFunction` providing operations to express
    finite-difference approximation. A ``VectorFunction`` encapsulates
    vector valued space-varying data (one value per dimension i.e u_x, u_y, u_z in 3D)
    for time-varying data, use :class:`VectorTimeFunction`.

    :param name: Name of the symbol
    :param grid: :class:`Grid` object from which to infer the data shape
                 and :class:`Dimension` indices.
    :param space_order: Discretisation order for space derivatives. By default,
                        ``space_order`` points are available on both sides of
                        a generic point of interest, including those on the grid
                        border. Sometimes, fewer points may be necessary; in
                        other cases, depending on the PDE being approximated,
                        more points may be necessary. In such cases, one
                        can pass a 3-tuple ``(o, lp, rp)`` instead of a single
                        integer representing the discretization order. Here,
                        ``o`` is the discretization order, while ``lp`` and ``rp``
                        indicate how many points are expected on left (``lp``)
                        and right (``rp``) of a point of interest.
    :param shape: (Optional) shape of the domain region in grid points.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param staggered: (Optional) a :class:`Dimension`, or a tuple of :class:`Dimension`s,
                      or a :class:`Stagger`, defining how the function is staggered.
                      For example:
                      * ``staggered=x`` entails discretization on x edges,
                      * ``staggered=y`` entails discretization on y edges,
                      * ``staggered=(x, y)`` entails discretization on xy facets,
                      * ``staggered=NODE`` entails discretization on node,
                      * ``staggerd=CELL`` entails discretization on cell.
    :param padding: (Optional) allocate extra grid points at a space dimension
                    boundary. These may be used for data alignment. Defaults to 0.
                    In alternative to an integer, a tuple, indicating the padding
                    in each dimension, may be passed; in this case, an error is
                    raised if such tuple has fewer entries then the number of space
                    dimensions.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.
    :param allocator: (Optional) an object of type :class:`MemoryAllocator` to
                      specify where to allocate the function data when running
                      on a NUMA architecture. Refer to ``default_allocator()``'s
                      __doc__ for more information about possible allocators.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses ``*args`` to (re-)create the dimension arguments of the
        symbolic function.

    .. note::

       If the parameter ``grid`` is provided, the values for ``shape``,
       ``dimensions`` and ``dtype`` will be derived from it.

    .. note::

       :class:`Function` objects are assumed to be constant in time
       and therefore do not support time derivatives. Use
       :class:`TimeFunction` for time-varying grid data.
    """

    is_VectorFunction = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.name = kwargs.get("name")
            self.space_order = kwargs.get("space_order", 1)
            # Get gird
            grid = kwargs.get("grid")
            # Each component is a Function
            components = []
            for d in grid.dimensions:
                components += [Function(name=self.name+'_'+d.name, grid=self.grid,
                                        space_order=self.space_order)]
            self.components = tuple(components)

    def __repr__(self):
        return tuple(i.__repr__() for i in self.components)

    def __getattr__(self, name):
        """
        Return tuple of attribute for each function
        .. note::

            This method acts as a fallback for __getattribute__
        """
        return tuple(i.getattr(name) for i in self.components)
