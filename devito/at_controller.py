import os
import math
import time
import random
import cpuinfo
from scipy.optimize import minimize

# env var for auto tuning report directory path
_ENV_VAR_REPORT_PATH = "AT_REPORT_DIR"

# global var
final_report_name = "final_report.txt"


def get_at_block_size(f_name, time_order, space_border):
    """
     :param f_name: function name. Used for naming reference
    :param time_order: int - time order of kernel
    :param space_border: int - space border of kernel
    :return list<int> of block sizes or None if report file was not found:
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
            if split[0] == f_name and int(split[1]) == time_order and int(split[2]) == space_border:

                # gets the block size
                block_size = line[line.find("[") + 1:line.find("]")].replace(" ", '').split(',')
                return [int(element) for element in block_size]

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

    # Constants for auto tuning range
    at_low_range = 5
    at_high_range = 15

    def __init__(self, function, args, f_name, shape, time_order, spc_border, cb_inner_most=False):
        """
        Object responsible for auto tuning block sizes. Note this object assumes that cache blocking is in 3d
        :param function: compiled function that is auto tuned
        :param args: arguments for function
        :param f_name: str- function name. Used for naming
        :param shape: list -  of the data buffer
        :param time_order: int - time order of kernel
        :param spc_border: int - space border of kernel
        :param cb_inner_most: bool - flag indicating whether cache blocking on inner most dim
        :raises ValueError: if shape of kernel is not 3D
        """

        global final_report_name, _ENV_VAR_REPORT_PATH
        report_dir = os.getenv(_ENV_VAR_REPORT_PATH,
                               os.path.join(os.path.dirname(os.path.realpath(__file__)), "At Report"))

        if len(shape) != 3:
            raise ValueError("Invalid shape of the kernel. Auto tuning works only in 3D")

        self.function = function
        self.args = args
        self.f_name = f_name
        self.shape = shape
        self.time_order = time_order
        self.spc_border = spc_border
        self.cb_inner_most = cb_inner_most
        self.report_dir = report_dir

        if not os.path.isdir(self.report_dir):  # Creates report dir if does not exist
            os.mkdir(self.report_dir)

    def brute_force(self, minimum=at_low_range, maximum=at_high_range):
        """
        Auto tunes using brute force. Times all block size combinations withing given range and writes it into report
        :param minimum: int (optional) - minimum value for auto tuning range. Default 5
        :param maximum: int (optional) - maximum value for auto tuning range. Default 20
        :raises ValueError: minimum is >= maximum
        """
        if minimum >= maximum:
            raise ValueError("Invalid parameters. Minimum tune range has to be less than maximum")
        print "Starting auto tuning using brute force, for time order %d, space border %d." % (self.time_order,
                                                                                               self.spc_border)

        if minimum - maximum > 2:  # No point to estimate if diff is below 2.
            timing_run = 0   # estimating run time
            for i in range(0, 5):
                print 'Estimating auto-tuning runtime...sample %d/5' % (i + 1)
                blocks = [random.randrange(minimum, maximum), random.randrange(minimum, maximum)]
                # append another block var if cache blocking inner most dim
                blocks = blocks.append(random.randrange(minimum, maximum)) if self.cb_inner_most else blocks
                test = self._time_it(blocks)
                timing_run += test

            print "Estimated runtime: %f minutes." % float(
                timing_run / 5 * ((maximum - minimum) * (maximum - minimum)) / 600)

        times = []  # list where times and block sizes will be kept
        for x in range(minimum, maximum):  # loops through all block sizes in given range
            for y in range(minimum, maximum):
                if len(self.shape) > 2 and self.cb_inner_most:
                    for z in range(minimum, maximum):
                        blocks = [x, y, z]
                        times.append((blocks, self._time_it(blocks)))
                else:
                    blocks = [x, y]
                    times.append((blocks, self._time_it(blocks)))

        print "Auto tuning using brute force complete"

        times = sorted(times, key=lambda element: element[1])  # sorts the list of tuples based on time
        self._write_report(times)  # writes the report

    # Need to decide if we are using this
    def minimize(self, initial_guess, method='powell'):
        """
        Auto tunes by solving minimization problem.
        :param initial_guess: initial guess of best block size. The better it is the more accurate the result
        :param method: method of minimization. Default 'powell'
        """
        offset = 0 if self.cb_inner_most else 1
        x0 = [initial_guess] * (len(self.shape) - offset)
        res = minimize(self._time_it, x0, method=method, options={'disp': True})

        result = [int(round(element)) for element in res.x]
        print result

    def _time_it(self, block_sizes):
        """
        Runs and times the function with given block size
        :param block_sizes: list|int - block sizes to be passed as arguments to the function
        :return: time - how long it took to execute with given block sizes
        """
        start = time.clock()

        for block in block_sizes:  # appends the block size parameters and runs a function
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
            str_to_write = "%s %d %d %s\n" % (self.f_name, self.time_order, self.spc_border, str(times[0][0]))

            if not os.path.isfile(final_report_path):  # initialises report file if it does not exist
                with open(final_report_path, 'w') as final_report:
                    final_report.write("f name, time o,space bo, best block size\n")
                    final_report.write(str_to_write)  # writes the string
                return
            else:
                with open(final_report_path, 'r') as final_report:  # reads all the contents
                    lines = final_report.readlines()

                # checks whether entry already exist and updates it. Otherwise appends to the end of the file
                entry_found = False
                str_to_check = "%s %d %d " % (self.f_name, self.time_order, self.spc_border)
                for i in range(1, len(lines)):
                    if str_to_check in lines[i]:  # remove the newline from beginning of the string
                        lines[i] = str_to_write
                        entry_found = True
                        break

            if not entry_found:  # if entry not found append string to the end of file
                lines.append(str_to_write)

            with open(final_report_path, 'w') as final_report:  # writes all the contents
                final_report.writelines(lines)

        except Exception as e:
            print "Failed to extract write auto tuning report because %s" % e.message