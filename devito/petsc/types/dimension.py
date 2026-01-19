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
    
    def _arg_defaults(self, alias=None):
        key = alias or self
        # from IPython import embed; embed()
        return {key.name: self.data}

    def _arg_values(self, grid=None, **kwargs):

        # global rtkn
        grtkn = kwargs.get(self.name, self.value)

        g_x_M = grid.distributor.decomposition[self.subdim.parent].glb_max


        return {self.name: g_x_M}
