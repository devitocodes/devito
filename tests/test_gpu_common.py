from devito.archinfo import get_gpu_info

from conftest import skipif


class TestGPUInfo(object):

    @skipif('nodevice')
    def test_get_gpu_info(self):
        info = get_gpu_info()
        assert 'tesla' in info['architecture'].lower()
