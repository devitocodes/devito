from __future__ import absolute_import

from os import mkdir, path
from operator import itemgetter

import numpy as np

from devito.logger import info_at, error
from devito.operator import Operator


class AutoTuner(object):

    def __init__(self, op, blocked_dims=None, at_report_dir=None):
        """Object responsible for auto tuning block sizes.

        :param op: Operator object.
        :param blocked_dims: list indicating which dims are blocked and we want to tune
                             Default all except inner most.
        :param at_report_dir: string - indicating path to auto tuning report directory.
                              If not set current directory is used
        :param blocked_dims: list indicating which dimensions we want to auto tune
        :raises ValueError: if operator is not of Operator type
        """

        if not isinstance(op, Operator):
            raise ValueError("AT requires Operator object to be passed as an argument")

        self.op = op
        self.nt_full = self.op.nt

        default_blocked_dims = ([True] * (len(self.op.shape) - 1))
        default_blocked_dims.append(None)  # By default don't autotune innermost dim
        self.blocked_dims = blocked_dims or default_blocked_dims

        default_at_dir = path.join(path.dirname(path.realpath(__file__)), "At Report")
        self.report_dir = at_report_dir or default_at_dir
        self.final_report_path = path.join(self.report_dir, "final_report.txt")
        self.model_desc_template = "%s %d %d %s %s"

        if not path.isdir(self.report_dir):  # Creates report dir if does not exist
            mkdir(self.report_dir)

    @property
    def block_size(self):
        """ Gets block size from auto tuning report

        :returns: auto tuned block size
        :raises ValueError: if auto tuning report not found
        :raises EnvironmentError: if matching model for auto tuned block size not found
        """
        if not path.isfile(self.final_report_path):
            raise ValueError("AT report at %s not found" % self.final_report_path)

        # model description string
        model_desc_str = self.model_desc_template % (self.op.getName(),
                                                     self.op.time_order,
                                                     self.op.spc_border,
                                                     str(self.op.shape).replace(" ", ''),
                                                     str(self.blocked_dims).replace(" ",
                                                                                    ''))
        with open(self.final_report_path, 'r') as f:
            for line in f.readlines():

                if model_desc_str in line:
                    blocks_str = line.split(' ')[5]
                    block_split = blocks_str[1:len(blocks_str) - 2].split(',')
                    block_size = [int(block) if block != "None" else None
                                  for block in block_split]

                    info_at("Picked: %s" % block_size)
                    return block_size

        raise EnvironmentError("Matching model with auto tuned block size not found.")

    def set_report_dir(self, report_path):
        """Sets report directory path

        :param report_path: path to report directory"""
        self.report_dir = report_path
        self.final_report_path = path.join(self.report_dir, "final_report.txt")

    def auto_tune_blocks(self, minimum=5, maximum=20):
        """Auto tunes block sizes. Times all block size combinations withing given range
           and writes it into report

        :param minimum: int (optional) - minimum value for auto tuning range. Default 5
        :param maximum: int (optional) - maximum value for auto tuning range. Default 20
        :raises ValueError: if  minimum is >= maximum
        """
        if minimum >= maximum:
            raise ValueError("Invalid parameters. Min tune range has to be less than Max")
        # setting time step to 3 as we don't need to iterate more than that
        # for auto tuning purposes
        at_nt = 3
        self.op.propagator.nt = at_nt
        self.op.propagator.profile = True

        info_at("Start. Mode: brute force")

        block_list = set()  # used to make sure we do not test the same block sizes
        mask = [i if i else None for i in self.blocked_dims]
        block = [None for i in mask]

        for x in range(minimum, maximum):
            block[0] = mask[0] and x

            if len(block) > 1:
                for y in range(minimum, maximum):
                    block[1] = mask[1] and y

                    if len(block) > 2:
                        for z in range(minimum, maximum):
                            block[2] = mask[2] and z
                            block_list.add((tuple(block)))
                    else:
                        block_list.add(tuple(block))
            else:
                block_list.add(tuple(block))

        # filter off some of the block sizes, heuristically
        block_list = sorted(self._filter(block_list))

        # populate output arrays with random values, to make sure that loads
        # and stores are performed
        for param in self.op.output_params:
            param._data = np.random.rand(*param.data.shape).astype(self.op.dtype)

        info_at("Number of block sizes that will be attempted: %d" % len(block_list))

        # runs function for each block_size
        times = []
        self.op.propagator.cache_blocking = list(block_list[0])
        for block in block_list:
            self.op.propagator.block_sizes = list(block)
            times.append((block, self.get_execution_time()))

        # sorts the list of tuples based on time
        times = sorted(times, key=itemgetter(1))

        info_at("Finish.")
        info_at("Estimated runtime for %s and %d time steps: %f hours" %
                (self.op.getName(), self.nt_full,
                 self.nt_full * times[0][1] / (at_nt * 3600)))

        self._write_block_report(times)  # writes the report

    def _filter(self, block_list):
        """
        Filter off some block sizes to speed up autotuning. The current
        heuristic is: ::

            * block sizes that are not a multiple of 2 are ditched;
            * only square blocks are retained.

        If no block sizes match the heuristic, then the original list is returned.
        """
        filtered_list = [b for b in block_list if all(i % 2 == 0 for i in b if i)]
        filtered_list = [b for b in filtered_list if all(i == b[0] for i in b if i)]
        return filtered_list or block_list

    def get_execution_time(self):
        """Runs and times the function

        :return: time - how long it took to execute
        """
        self.op.propagator.run(self.op.get_args())
        return self.op.propagator.timings['kernel']

    def _write_block_report(self, times):
        """Writes auto tuning report for block sizes

        :param times: sorted list - times with block sizes
        :raises IOError: if fails to write report
        """
        try:
            full_report_text = ["%s %f\n" % (str(block), timee) for block, timee in times]

            # Cache blocking dimensions
            cb_dims = str(self.blocked_dims).replace(" ", '')
            shape = str(self.op.shape).replace(" ", '')

            # Writes all auto tuning information into full report
            with open(path.join(self.report_dir,
                                "%s_time_o_%s_spc_bo_%s_shp_%s_b_%s_report.txt" %
                      (self.op.getName(), self.op.time_order,
                       self.op.spc_border, shape, cb_dims)), 'w') as f:
                f.writelines(full_report_text)

            # string that describes the model
            model_desc_str = self.model_desc_template % (self.op.getName(),
                                                         self.op.time_order,
                                                         self.op.spc_border,
                                                         shape, cb_dims)
            str_to_write = "%s %s\n" % (model_desc_str, str(times[0][0]).replace(" ", ''))

            # initialises report file if it does not exist
            if not path.isfile(self.final_report_path):
                with open(self.final_report_path, 'w') as final_report:
                    final_report.write("f name, time o, space bo ,"
                                       "shape, block dims, best block size\n")
                    final_report.write(str_to_write)  # writes the string
                return
            else:
                # reads all the contents, checks whether entry already exist and updates.
                # Otherwise appends to the end of the file
                with open(self.final_report_path, 'r') as final_report:
                    lines = final_report.readlines()
                entry_found = False

                for i in range(1, len(lines)):
                    if model_desc_str in lines[i]:
                        lines[i] = str_to_write
                        entry_found = True
                        break

            if not entry_found:  # if entry not found append string to the end of file
                lines.append(str_to_write)
            # writes all the contents
            with open(self.final_report_path, 'w') as final_report:
                final_report.writelines(lines)

        except IOError as e:
            error("Failed to write auto tuning report because %s" % e.message)
