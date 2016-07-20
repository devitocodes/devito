import math
import time
import cpuinfo
from scipy.optimize import minimize


def get_optimal_block_size(shape, load_c):
    """Gets optimal block size. Based on architecture.

    Args:
        shape (tuple|list): shape of kernel
        load_c (int): load count

    Returns:
        int: most optimal size  for the block
    """

    cache_s = int(cpuinfo.get_cpu_info()['l2_cache_size'].split(' ')[0])  # cache size
    core_c = cpuinfo.get_cpu_info()['count']  # number of cores

    # assuming no prefetching, square block will give the most cache reuse.
    # We then take the cache/core divide by the size of the inner most dimension in which we do not block.
    #  This gives us the X*Y block space, of which we take the square root to get the size of our blocks.
    # ((C size / cores) / (4 * length inner most * kernel loads)
    optimal_b_size = math.sqrt(((1000 * cache_s) / core_c) / (4 * shape[len(shape) - 1] * load_c))
    return int(round(optimal_b_size))  # rounds to the nearest integer


class AtController(object):

    def __init__(self, function, args, initial_guess):
        self.function = function
        self.args = args
        self.initial_guess = initial_guess

    def brute_force(self, shape, minimum, maximum):
        print "Starting auto tuning using brute force"
        times = []
        for x in range(minimum, maximum):
            for y in range(minimum, maximum):
                if len(shape) > 2:
                    for z in range(minimum, maximum):
                        blocks = [x, y, z]
                        times.append((blocks, self._time_it(blocks)))
                else:
                    blocks = [x, y]
                    times.append((blocks, self._time_it(blocks)))

        return sorted(times, key=lambda element: element[1])

    def minimize(self, method='powell'):
        x0 = self.initial_guess
        res = minimize(self._time_it, x0, method=method, options={'disp': True})
        print res.x

        return [int(round(element)) for element in res.x]

    def _time_it(self, block_sizes):
        start = time.clock()

        for block in block_sizes:  # appends the block size parameters and runs a function
            self.args += [int(block)]
        self.function(*self.args)

        return time.clock() - start  # returns elapsed time
