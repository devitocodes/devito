from devito.types.dimension import Thickness



class SubDimMax(Thickness):
    """
    """

    def __init_finalize__(self, *args, **kwargs):
        self._subdim = kwargs.pop('subdim')
        self._dtype = self._subdim.dtype

        super().__init_finalize__(*args, **kwargs)

    @property
    def subdim(self):
        return self._subdim


    def _arg_values(self, grid=None, **kwargs):

        dist = grid.distributor

        # global rtkn
        grtkn = kwargs.get(self.subdim.rtkn.name, self.subdim.rtkn.value)
        # print(g_x_M)
        # decomposition info
        decomp = dist.decomposition[self.subdim.parent]
        g_x_M = decomp.glb_max
        # print(g_x_M)
        val = decomp.index_glb_to_loc_unsafe(g_x_M - grtkn)
        # print(val)


        return {self.name: int(val)}
    


class SubDimMin(Thickness):
    """
    """

    def __init_finalize__(self, *args, **kwargs):
        self._subdim = kwargs.pop('subdim')
        self._dtype = self._subdim.dtype

        super().__init_finalize__(*args, **kwargs)

    @property
    def subdim(self):
        return self._subdim

    def _arg_values(self, grid=None, **kwargs):

        dist = grid.distributor

        # global ltkn
        gltkn = kwargs.get(self.subdim.ltkn.name, self.subdim.ltkn.value)

        # decomposition info
        decomp = dist.decomposition[self.subdim.parent]
        g_x_m = decomp.glb_min
        val = decomp.index_glb_to_loc_unsafe(g_x_m + gltkn)

        return {self.name: int(val)}
    


class SpaceDimMax(Thickness):
    """
    """

    def __init_finalize__(self, *args, **kwargs):
        self._space_dim = kwargs.pop('space_dim')
        self._dtype = self._space_dim.dtype

        super().__init_finalize__(*args, **kwargs)

    @property
    def space_dim(self):
        return self._space_dim


    def _arg_values(self, grid=None, **kwargs):

        dist = grid.distributor

        # global rtkn
        # grtkn = kwargs.get(self.subdim.rtkn.name, self.subdim.rtkn.value)
        # print(g_x_M)
        # decomposition info
        decomp = dist.decomposition[self.space_dim]
        # obvs not just x etc..
        g_x_M = decomp.glb_max
        # print(g_x_M)
        val = decomp.index_glb_to_loc_unsafe(g_x_M)
        # print(val)


        return {self.name: int(val)}
    


class SpaceDimMin(Thickness):
    """
    """

    def __init_finalize__(self, *args, **kwargs):
        self._space_dim = kwargs.pop('space_dim')
        self._dtype = self._space_dim.dtype

        super().__init_finalize__(*args, **kwargs)

    @property
    def space_dim(self):
        return self._space_dim


    def _arg_values(self, grid=None, **kwargs):

        dist = grid.distributor


        decomp = dist.decomposition[self.space_dim]
        # obvs not just x etc..
        g_x_m = decomp.glb_min
        # print(g_x_M)
        val = decomp.index_glb_to_loc_unsafe(g_x_m)
        # print(val)


        return {self.name: int(val)}
    



