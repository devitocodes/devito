import pytest
import numpy as np

from devito import Grid, Function
from devito.data import loc_data_idx


class TestDataParallel(object):

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('gslice', [
        (slice(None, None, -1), slice(None, None, -1), 0),
        (slice(None, None, -1), slice(None, None, -1), slice(0, 1, 1)),
        (slice(None, None, -1), 0, slice(None, None, -1)),
        (slice(None, None, -1), slice(0, 1, 1), slice(None, None, -1)),
        (0, slice(None, None, -1), slice(None, None, -1)),
        (slice(0, 1, 1), slice(None, None, -1), slice(None, None, -1))])
    def test_inversions(self, gslice):
        """ Test index flipping along different axes."""
        nx = 8
        ny = 8
        nz = 8
        shape0 = (nx, ny)
        shape1 = (nx, ny, nz)
        grid = Grid(shape=(8, 8, 8))
        f = Function(name='f', grid=grid)
        dat = np.arange(64).reshape(shape0)
        vdat = np.zeros(shape1)

        f.data[:, :, 0] = dat
        res = f.data[gslice]

        # construct solution to test res against
        vdat[:, :, 0] = dat
        tdat = vdat[gslice]
        lslice = loc_data_idx(f.data._index_glb_to_loc(gslice))
        sl = []
        Null = slice(-1, -2, None)
        for s, gs, d in zip(lslice, gslice, f._decomposition):
            if type(s) is slice and s == Null:
                sl.append(s)
            elif type(gs) is not slice:
                continue
            else:
                try:
                    start = d.index_loc_to_glb(s.start)
                    stop = d.index_loc_to_glb(s.stop-1)+1
                    step = s.step
                    sl.append(slice(start, stop, step))
                except TypeError:
                    sl.append(Null)

        if len(sl) == len(tdat.shape):
            assert np.all(np.array(res) == tdat[tuple(sl)])
            assert res.shape == tdat[tuple(sl)].shape
        else:
            assert np.all(np.array(res) == vdat[tuple(sl)])
            assert res.shape == vdat[tuple(sl)].shape
