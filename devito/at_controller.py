import os
import math
import time
import random
import cpuinfo
import logger

# env var for auto tuning report directory path
_ENV_VAR_REPORT_PATH = "AT_REPORT_DIR"

# global var
final_report_name = "final_report.txt"


def get_at_block_size(f_name, shape, time_order, spc_border, block_size):
    """
    Gets block size from auto tuning report
    :param f_name: str - function name. Used for naming
    :param shape: list - shape of the data buffer
    :param time_order: int - time order of kernel
    :param spc_border: int - space border of kernel
    :param block_size: list<int> - block sizes. Used to see which dims were blocked and for ref in report
    """
    global final_report_name, _ENV_VAR_REPORT_PATH
    report_dir = os.getenv(_ENV_VAR_REPORT_PATH, os.path.join(os.path.dirname(os.path.realpath(__file__)), "At Report"))

    final_report_path = os.path.join(report_dir, final_report_name)

    if not os.path.isfile(final_report_path):
        return None  # returns none if report for best block sizes does not exist

    with open(final_report_path, 'r') as f:
        for line in f.readlines():

            if "time" in line or "spc" in line:  # ignores the first line of the report file
                continue

            split = line.split(' ')                     # finds the line we are looking for
            if split[0] == f_name and int(split[1]) == time_order and int(split[2]) == spc_border and \
               split[3] == str(shape).replace(" ", '') and split[4] == str(block_size).replace(" ", ''):

                # gets the block sizes as string
                block_split = split[5][1:len(split[5]) - 1].split(',')

                return [int(block) if block is not None else None for block in block_split]

    return None  # returns none if no matching time/space order was found


def get_optimal_block_size(shape, load_c):
    """
    Gets optimal block size based on architecture
    :param shape: list - shape of the data buffer
    :param load_c: int - load count
    :return: int - optimal block size
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

    def __init__(self, function, args, f_name, shape, time_order, spc_border, block_size):
        """
        Object responsible for auto tuning block sizes.
        :param function: compiled function that is auto tuned
        :param args: arguments for function
        :param f_name: str - function name. Used for naming
        :param shape: list - shape of the data buffer
        :param time_order: int - time order of kernel
        :param spc_border: int - space border of kernel
        :param block_size: list<int> - block sizes. Used to see which dims are blocked
        """

        global final_report_name, _ENV_VAR_REPORT_PATH
        report_dir = os.getenv(_ENV_VAR_REPORT_PATH,
                               os.path.join(os.path.dirname(os.path.realpath(__file__)), "At Report"))

        self.function = function
        self.args = args
        self.f_name = f_name
        self.shape = shape
        self.time_order = time_order
        self.spc_border = spc_border
        self.block_sizes = block_size
        self.report_dir = report_dir

        if not os.path.isdir(self.report_dir):  # Creates report dir if does not exist
            os.mkdir(self.report_dir)

    def brute_force(self, minimum=5, maximum=20):
        """
        Auto tunes using brute force. Times all block size combinations withing given range and writes it into report
        :param minimum: int (optional) - minimum value for auto tuning range. Default 5
        :param maximum: int (optional) - maximum value for auto tuning range. Default 20
        :raises ValueError: minimum is >= maximum
        """
        if minimum >= maximum:
            raise ValueError("Invalid parameters. Minimum tune range has to be less than maximum")
        logger.log("Starting auto tuning using brute force")

        if maximum - minimum > 2:  # No point to estimate if diff is below 2.
            self._estimate_run_time(minimum, maximum)

        times = []  # list where times and block sizes will be kept
        iterated_blocks = []  # used to make sure we do not test the same block sizes
        blocks = self.block_sizes
        for x in range(minimum, maximum):
            blocks[0] = x if self.block_sizes[0] else blocks[0]

            if len(blocks) > 1:
                for y in range(minimum, maximum):
                    blocks[1] = y if self.block_sizes[1] else blocks[1]

                    if len(self.shape) > 2:
                        for z in range(minimum, maximum):
                            blocks[2] = z if self.block_sizes[2] else blocks[2]

                            if blocks not in iterated_blocks:
                                times.append((blocks, self._time_it(blocks)))
                                iterated_blocks.append(blocks)
                    else:
                        if blocks not in iterated_blocks:
                            times.append((blocks, self._time_it(blocks)))
                            iterated_blocks.append(blocks)
            else:
                if blocks not in iterated_blocks:
                    times.append((blocks, self._time_it(blocks)))
                    iterated_blocks.append(blocks)

        logger.log("Auto tuning using brute force complete")

        times = sorted(times, key=lambda element: element[1])  # sorts the list of tuples based on time
        self._write_report(times)  # writes the report

    def _estimate_run_time(self, minimum, maximum):
        """
        Estimates run time for auto tuning
        :param minimum: int - minimum tune range
        :param maximum: int - maximum tune range
        """
        timing_run = 0  # estimating run time
        for i in range(0, 5):
            logger.log('Estimating auto-tuning runtime...sample %d/5' % (i + 1))

            blocks = []
            for block in self.block_sizes:
                blocks = blocks.append(random.randrange(minimum, maximum)) if block else blocks.append(None)

            test = self._time_it(blocks)
            timing_run += test

        logger.log("Estimated runtime: %f minutes." % float(
            timing_run / 5 * ((maximum - minimum) * (maximum - minimum)) / 600))

    def _time_it(self, block_sizes):
        """
        Runs and times the function with given block size
        :param block_sizes: list|int - block sizes to be passed as arguments to the function
        :return: time - how long it took to execute with given block sizes
        """
        start = time.clock()

        for block in block_sizes:  # appends the block size parameters and runs a function
            if block is not None:
                self.args += [int(block)]

        self.function(*self.args)

        return time.clock() - start  # returns elapsed time

    def _write_report(self, times):
        """
        Writes auto tuning report
        :param times: sorted list - times with block sizes
        :raises IOError: if fails to write report
        """
        global final_report_name
        try:
            full_report_text = ["%s %f\n" % (str(block), timee) for block, timee in times]

            # Writes all auto tuning information into full report
            with open(os.path.join(self.report_dir, "%s_time_o_%s_spc_bo_%s_report.txt" %
                      (self.f_name, self.time_order, self.spc_border)), 'w') as f:
                f.writelines(full_report_text)

            final_report_path = os.path.join(self.report_dir, final_report_name)
            str_to_write = "%s %d %d %s %s %s\n" % (self.f_name, self.time_order, self.spc_border,
                                                    str(self.shape).replace(" ", ''),
                                                    str(self.block_sizes).replace(" ", ''),
                                                    str(times[0][0]).replace(" ", ''))

            if not os.path.isfile(final_report_path):  # initialises report file if it does not exist
                with open(final_report_path, 'w') as final_report:
                    final_report.write("f name,time o,space bo,shape,original_block,best block size\n")
                    final_report.write(str_to_write)  # writes the string
                return
            else:
                with open(final_report_path, 'r') as final_report:  # reads all the contents
                    lines = final_report.readlines()

                # checks whether entry already exist and updates it. Otherwise appends to the end of the file
                entry_found = False
                str_to_check = "%s %d %d %s %s" % (self.f_name, self.time_order, self.spc_border,
                                                   str(self.shape).replace(" ", ''),
                                                   str(self.block_sizes).replace(" ", ''))
                for i in range(1, len(lines)):
                    if str_to_check in lines[i]:
                        lines[i] = str_to_write
                        entry_found = True
                        break

            if not entry_found:  # if entry not found append string to the end of file
                lines.append(str_to_write)

            with open(final_report_path, 'w') as final_report:  # writes all the contents
                final_report.writelines(lines)

        except Exception as e:
            print "Failed to extract write auto tuning report because %s" % e.message
