from devito.types.dimension import Thickness


class _SubDimBound(Thickness):
    def __init_finalize__(self, *args, **kwargs):
        self._subdim = kwargs.pop('subdim')
        self._dtype = self._subdim.dtype
        super().__init_finalize__(*args, **kwargs)

    @property
    def subdim(self):
        return self._subdim


class _SpaceDimBound(Thickness):
    def __init_finalize__(self, *args, **kwargs):
        self._space_dim = kwargs.pop('space_dim')
        self._dtype = self._space_dim.dtype
        super().__init_finalize__(*args, **kwargs)

    @property
    def space_dim(self):
        return self._space_dim


class SubDimMax(_SubDimBound):
    """
    Local index of a SubDimension's right global boundary, which may not correspond
    to a locally owned point. Not used for indexing into data.
    """
    def _arg_values(self, grid=None, **kwargs):
        dist = grid.distributor
        grtkn = kwargs.get(self.subdim.rtkn.name, self.subdim.rtkn.value)
        decomp = dist.decomposition[self.subdim.parent]
        val = decomp.index_glb_to_loc_unsafe(decomp.glb_max - grtkn)
        return {self.name: int(val)}


class SubDimMin(_SubDimBound):
    """
    Local index of a SubDimension's left global boundary, which may not correspond
    to a locally owned point. Not used for indexing into data.
    """
    def _arg_values(self, grid=None, **kwargs):
        dist = grid.distributor
        gltkn = kwargs.get(self.subdim.ltkn.name, self.subdim.ltkn.value)
        decomp = dist.decomposition[self.subdim.parent]
        val = decomp.index_glb_to_loc_unsafe(decomp.glb_min + gltkn)
        return {self.name: int(val)}


class SpaceDimMax(_SpaceDimBound):
    """
    Local index of a SpaceDimension's right global boundary, which may not correspond
    to a locally owned point. Not used for indexing into data.
    """
    def _arg_values(self, grid=None, **kwargs):
        dist = grid.distributor
        decomp = dist.decomposition[self.space_dim]
        val = decomp.index_glb_to_loc_unsafe(decomp.glb_max)
        return {self.name: int(val)}


class SpaceDimMin(_SpaceDimBound):
    """
    Local index of a SpaceDimension's left global boundary, which may not correspond
    to a locally owned point. Not used for indexing into data.
    """
    def _arg_values(self, grid=None, **kwargs):
        dist = grid.distributor
        decomp = dist.decomposition[self.space_dim]
        val = decomp.index_glb_to_loc_unsafe(decomp.glb_min)
        return {self.name: int(val)}
