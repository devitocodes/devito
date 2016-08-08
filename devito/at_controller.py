import math
import random
import time
from collections import Iterable
from os import mkdir, path

import cpuinfo

import logger

# global vars
final_report_name = "final_report.txt"
default_at_dir = path.join(path.dirname(path.realpath(__file__)), "At Report")


def get_at_block_size(f_name, time_order, spc_border, shape, block_dims, at_report_dir=None):
    """
    Gets block size from auto tuning report
    :param f_name: str - function name. Used for naming
    :param time_order: int - time order of kernel
    :param spc_border: int - space border of kernel
    :param shape: list - shape of the data buffer
    :param block_dims: list/int - indicating which dims are blocked.
    :param at_report_dir: string - indicating path to auto tuning report directory.
                          If not set default at report directory is used
    :raises ValueError: if auto tuning report is not found
    """
    global final_report_name, default_at_dir

    report_dir = at_report_dir if at_report_dir is not None else default_at_dir
    final_report_path = path.join(report_dir, final_report_name)

    if not path.isfile(final_report_path):
        raise ValueError("Auto tuning report at %s not found" % final_report_path)

    # model description string
    model_descr_str = "%s %d %d %s %s" % (f_name, time_order, spc_border,
                                          str(shape).replace(" ", ''), str(block_dims).replace(" ", ''))

    with open(final_report_path, 'r') as f:
        for line in f.readlines():

            if model_descr_str in line:

                blocks_str = line.split(' ')[5]
                block_split = blocks_str[1:len(blocks_str) - 2].split(',')

                return [int(block) if block != "None" else None for block in block_split]

    return None  # returns none if no matching block size was found


def get_optimal_block_size(cache_blocking, shape, load_c):
    """
    Gets optimal block size based on architecture
    :param cache_blocking: Flags indicating which dims to block
    :param shape: list - shape of the data buffer
    :param load_c: int - load count
    :return: list of optimal block sizes
    """

    cache_s = int(cpuinfo.get_cpu_info()['l2_cache_size'].split(' ')[0])  # cache size
    core_c = cpuinfo.get_cpu_info()['count']  # number of cores

    # assuming no prefetching, square block will give the most cache reuse.
    # We then take the cache/core divide by the size of the inner most dimension in which we do not block.
    #  This gives us the X*Y block space, of which we take the square root to get the size of our blocks.
    # ((C size / cores) / (4 * length inner most * kernel loads)
    optimal_b_size = math.sqrt(((1000 * cache_s) / core_c) / (4 * shape[len(shape) - 1] * load_c))
    optimal_b_size = int(round(optimal_b_size))  # rounds to the nearest integer

    if isinstance(cache_blocking, Iterable):
        # set which dims not to block and return a list
        return [optimal_b_size if item else None for item in cache_blocking]
    else:
        # if a cache block is not a list. Cast optimal_b_size to the list same len as spacial domain
        return [optimal_b_size] * len(shape)


class AutoTuner(object):

    def __init__(self, operator, at_report_dir=None):
        """
        Object responsible for auto tuning block sizes.
        :param operator: Operator object.
        :param at_report_dir: string - indicating path to auto tuning report directory.
                              If not set default at report directory is used
        :raises ValueError: if operator is not of Operator type
        """
        global final_report_name, default_at_dir

        from operator import Operator

        if not isinstance(operator, Operator):
            raise ValueError("Auto tuner requires Operator object to be passed as an argument")

        self.op = operator
        self.report_dir = at_report_dir if at_report_dir is not None else default_at_dir
        self.final_report_path = path.join(self.report_dir, final_report_name)

        if not path.isdir(self.report_dir):  # Creates report dir if does not exist
            mkdir(self.report_dir)

    def auto_tune_blocks(self, minimum=5, maximum=20):
        """
        Auto tunes block sizes. Times all block size combinations withing given range and writes it into report
        :param minimum: int (optional) - minimum value for auto tuning range. Default 5
        :param maximum: int (optional) - maximum value for auto tuning range. Default 20
        :raises ValueError: if  minimum is >= maximum
        :raises EnvironmentError: If cache blocking is not set inside the Operator object
        """
        if minimum >= maximum:
            raise ValueError("Invalid parameters. Minimum tune range has to be less than maximum")

        if not self.op.propagator.cache_blocking:
            raise EnvironmentError("Invalid parameters. Cache_blocking is set to False")

        logger.info("Starting auto tuning of block sizes using brute force")

        self.op.auto_tune = True
        self.op.propagator.auto_tune = True

        f, args = self.op.apply(auto_tune=True)

        if maximum - minimum > 2:  # No point to estimate if diff is below 2.
            self._estimate_b_run_time(f, args, minimum, maximum)

        times = []  # list where times and block sizes will be kept
        block_list = []  # used to make sure we do not test the same block sizes
        org_block_sizes = self.op.propagator.block_sizes
        blocks = list(self.op.propagator.block_sizes)

        for x in range(minimum, maximum):
            blocks[0] = x if org_block_sizes[0] else None

            if len(blocks) > 1:
                for y in range(minimum, maximum):
                    blocks[1] = y if org_block_sizes[1] else None

                    if len(blocks) > 2:
                        for z in range(minimum, maximum):
                            blocks[2] = z if org_block_sizes[2] else None
                            block_list = block_list + [list(blocks)] if blocks not in block_list else block_list
                    else:
                        block_list = block_list + [list(blocks)] if blocks not in block_list else block_list
            else:
                block_list = block_list + [list(blocks)] if blocks not in block_list else block_list

        # runs function for each block_size
        for block in block_list:
            times.append((block, self._time_blocks(f, args, block)))

        logger.info("Auto tuning using brute force complete")

        times = sorted(times, key=lambda element: element[1])  # sorts the list of tuples based on time
        self._write_b_report(times)  # writes the report

    def _estimate_b_run_time(self, f, args, minimum, maximum):
        """
        Estimates run time for auto tuning
        :param f: compiled function that we are auto tuning
        :param args: arguments for that function
        :param minimum: int - minimum tune range
        :param maximum: int - maximum tune range
        """
        timing_run = 0  # estimating run time
        for i in range(0, 5):
            logger.info('Estimating auto-tuning runtime...sample %d/5' % (i + 1))

            blocks = []
            for block in self.op.propagator.block_sizes:
                blocks = blocks + [random.randrange(minimum, maximum)] if block else blocks + [None]

            timing_run += self._time_blocks(f, args, blocks)

        logger.info("Estimated runtime: %f minutes." % float(
            timing_run / 5 * ((maximum - minimum) * (maximum - minimum)) / 600))

    def _time_blocks(self, f, args, block_sizes):
        """
        Runs and times the function with given block size
        :param f: compiled function that we are auto tuning
        :param args: arguments for that function
        :param block_sizes: list|int - block sizes to be passed as arguments to the function
        :return: time - how long it took to execute with given block sizes
        """
        start = time.clock()

        for block in block_sizes:  # appends the block size
            if block is not None:
                args += [int(block)]

        f(*args)  # run function

        return time.clock() - start  # returns elapsed time

    def _write_b_report(self, times):
        """
        Writes auto tuning report for block sizes
        :param times: sorted list - times with block sizes
        :raises IOError: if fails to write report
        """
        try:
            full_report_text = ["%s %f\n" % (str(block), timee) for block, timee in times]

            # Cache blocking dimensions
            cb_dims = str(self.op.propagator.cache_blocking).replace(" ", '')
            shape = str(self.op.shape).replace(" ", '')

            # Writes all auto tuning information into full report
            with open(path.join(self.report_dir, "%s_time_o_%s_spc_bo_%s_shp_%s_b_%s_report.txt" %
                      (self.op.getName(), self.op.time_order, self.op.spc_border, shape, cb_dims)), 'w') as f:
                f.writelines(full_report_text)

            # string that describes the model
            model_descr_str = "%s %d %d %s %s" % (self.op.getName(), self.op.time_order,
                                                  self.op.spc_border, shape, cb_dims)
            str_to_write = "%s %s\n" % (model_descr_str, str(times[0][0]).replace(" ", ''))

            if not path.isfile(self.final_report_path):  # initialises report file if it does not exist
                with open(self.final_report_path, 'w') as final_report:
                    final_report.write("f name, time o, space bo ,shape, block dims, best block size\n")
                    final_report.write(str_to_write)  # writes the string
                return
            else:
                with open(self.final_report_path, 'r') as final_report:  # reads all the contents
                    lines = final_report.readlines()

                # checks whether entry already exist and updates it. Otherwise appends to the end of the file
                entry_found = False

                for i in range(1, len(lines)):
                    if model_descr_str in lines[i]:
                        lines[i] = str_to_write
                        entry_found = True
                        break

            if not entry_found:  # if entry not found append string to the end of file
                lines.append(str_to_write)

            with open(self.final_report_path, 'w') as final_report:  # writes all the contents
                final_report.writelines(lines)

        except IOError as e:
            logger.error("Failed to write auto tuning report because %s" % e.message)
