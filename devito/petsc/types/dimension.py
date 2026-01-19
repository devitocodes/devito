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

        # # global rtkn
        # grtkn = kwargs.get(self.subdim.rtkn.name, self.subdim.rtkn.value)

        # g_x_M = grid.distributor.decomposition[self.subdim.parent].glb_max
        # val = grid.distributor.decomposition[self.subdim.parent].index_glb_to_loc(g_x_M - grtkn)

        # return {self.name: int(val)}
    

        dist = grid.distributor
        rank = dist.myrank
        comm = dist.comm

        # global rtkn
        grtkn = kwargs.get(self.subdim.rtkn.name, self.subdim.rtkn.value)

        # decomposition info
        decomp = dist.decomposition[self.subdim.parent]
        g_x_M = decomp.glb_max
        val = decomp.index_glb_to_loc_unsafe(g_x_M - grtkn)


        print(
            f"[Rank {rank}] "
            f"grtkn={grtkn}, "
            f"g_x_M={g_x_M}, "
            f"glb_idx={g_x_M - grtkn}, "
            f"loc_val={val}",
            flush=True
        )

        if val is None:
            return {}

        return {self.name: int(val)}

