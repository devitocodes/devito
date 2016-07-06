import os
import stat
import math
import shutil
import cpuinfo
import subprocess

# global vars.
# report folder will be created in current dir. Change if necessary
reports_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "auto-tune-reports")

# where best block sizes for each kernel will be kept
final_report_path = os.path.join(reports_folder_path, "final_report.txt")


def get_at_block_size(time_order, space_order):
    """Gets the best block sizes for given kernel if they are in the final_report_path

    Args:
        time_order (int): time order of kernel
        space_order (int): space order of kernel

    Returns:
        list: best block size. Starting from outer most dimension
        None: if report does not exist or does not contain required block sizes
    """
    global final_report_path

    if not os.path.isfile(final_report_path):
        return None  # returns none if report for best block sizes does not exist

    with open(final_report_path, 'r') as f:
        for line in f.readlines():

            if "time" in line or "space" in line:  # ignores the first line of the report file
                continue

            split = line.split(' ')
            if int(split[0]) == time_order and int(split[1]) == space_order:  # finds the one we are looking for

                # Splits, converts all block sizes into int and returns
                return [int(element) for element in split[2].split(',')]

    return None  # returns none if no matching time/space order was found


def get_optimal_block_size(shape, time_order, space_order):
    """Gets optimal block size. Based on architecture Currently works only on linux

    Args:
        shape (tuple|list): shape of kernel
        time_order (int): time order of kernel
        space_order (int): space order of kernel

    Returns:
        int: most optimal size  for the block
    """

    # list [time_order][space_order] hardcoded number of loads from the roof line at: (change if necessary)
    # https://docs.google.com/spreadsheets/d/1OmvsTftH3uCfYZj-1Lb-5Ji7edrURf0UzzywYaw0-FY/edit?ts=5745964f#gid=0
    number_of_loads = [[11, 17, 23, 29, 35, 41, 47, 53], [19, 25, 31, 37, 43, 49, 55]]  # works only till space order 16

    cache_s = int(cpuinfo.get_cpu_info()['l2_cache_size'].split(' ')[0])  # cache size
    core_c = cpuinfo.get_cpu_info()['count']  # number of cores
    loads_c = number_of_loads[time_order / 2][space_order / 2]  # number of loads

    # ((C size / cores) / (4 * length inner most * kernel loads)
    optimal_b_size = math.sqrt(((1000 * cache_s) / core_c) / (4 * shape[len(shape) - 1] * loads_c))
    return int(round(optimal_b_size))  # rounds to the nearest integer


class AtController(object):
    """Class responsible for controlling auto tuning process, copying relevant files and collecting data"""

    def __init__(self, isat_dir_path):
        """Initialises at controller. Creates at working dir. Creates all isat-files

        Args:
            isat_dir_path (str): full path to isat installation directory

        Raises:
            ValueError: if isat installation directory does not exist
        """
        if not os.path.exists(isat_dir_path):
            raise ValueError("isat installation directory does not exits")

        self.time_order = None
        self.space_order = None

        self._isat_py_path = os.path.join(isat_dir_path, "Source/Python/isat.py")  # path to isat script that does at
        self._at_work_dir = os.path.join(isat_dir_path, "Auto-tuning")  # working dir for at

        at_file_dir = os.path.join(self._at_work_dir, "Isat-files")  # dir for keeping files necessary for at

        # all these files are necessary for auto tuning process and they have to be in same dir as tuned file
        # do not change these names
        self._isat_build_f = "isat-build"
        self._isat_clean_f = "isat-clean"
        self._isat_test_f = "isat-test"
        self._make_f = "makefile"

        self._build_f_path = os.path.join(at_file_dir, self._isat_build_f)
        self._clean_f_path = os.path.join(at_file_dir, self._isat_clean_f)
        self._test_f_path = os.path.join(at_file_dir, self._isat_test_f)
        self._make_f_path = os.path.join(at_file_dir, self._make_f)

        self._at_parent_folder = ""  # saves the location of current at parent folder
        self._at_src_dir = "src"  # source folder where file which we'll auto tune will be kept
        self._at_dst_dir = "dst"  # destination folder where at result will be kept

        # for simplicity renaming all the tuned files to this name
        # their parent dir will contain the time and space order in cpp file kernel
        self._at_f_name = "tune_me"
        shell_ref_str = "#!/bin/sh"

        if not os.path.exists(self._at_work_dir):  # creates main working directory
            os.mkdir(self._at_work_dir)

        # Uncomment if contents of isat files changes
        # shutil.rmtree(at_file_dir)

        if not os.path.exists(at_file_dir):  # creates directory where all necessary files for at will be kept
            os.mkdir(at_file_dir)

            with open(self._build_f_path, 'w') as f:  # creates isat-build
                f.writelines([shell_ref_str, "\nmake build"])
            self._set_x_permission(self._build_f_path)

            with open(self._clean_f_path, 'w') as f:  # create isat-clean
                f.writelines([shell_ref_str, "\nmake clean"])
            self._set_x_permission(self._clean_f_path)

            with open(self._test_f_path, 'w') as f:  # create isat-test
                f.writelines([shell_ref_str, "\n./%s.exe" % self._at_f_name])
            self._set_x_permission(self._test_f_path)

            with open(self._make_f_path, 'w') as f:  # creates make-file
                # copied make file contents from one of examples in isat folder. If there's a need feel free to change
                make_file_contents = ["CXX = icc", "\nCXX_OPTS = -O3 -qopenmp",  # compiler and its options
                                      "\nSRCS = %s.cpp" % self._at_f_name, "\nEXE = %s.exe" % self._at_f_name,
                                      "\n$(EXE): $(SRCS)", "\n\t$(CXX) $(CXX_OPTS) $< -o $@",
                                      "\nclean:", "\n\t-rm *.exe", "\nbuild: $(EXE)"]

                f.writelines(make_file_contents)
                self._set_x_permission(self._make_f_path)

    def auto_tune(self, file_path, time_order, space_order):
        """Does the auto tuning for selected file and collects relevant data

        Args:
            file_path (str): full path to the file that we want to auto tune
            time_order (int): time order of kernel.
            space_order (int): space order of kernel.
        """

        try:
            print "Starting Auto tuning for %s" % file_path
            self.time_order = time_order
            self.space_order = space_order

            # these functions have to run in this order
            self._prep_file_for_at(file_path)
            self._run_at()
            self._extract_report_info()
            print "Auto tuning  complete for %s" % file_path
        except ValueError:
            print "Auto tuning for %s failed" % file_path

        self._at_parent_folder = ""  # reset parent folder name

    def _prep_file_for_at(self, file_path):
        """Copies file to work dir and copies all at isat-files there

        Args:
            file_path (str): full path to the file that we want to auto tune

        Raises:
            ValueError: if file at file_path is not found
        """

        if not os.path.isfile(file_path):
            print "%s not found"
            raise ValueError()

        self._at_parent_folder = os.path.join(self._at_work_dir, "time_o_%d_space_o_%d" %
                                              (self.time_order, self.space_order))
        src_dir_path = os.path.join(self._at_parent_folder, self._at_src_dir)

        # prep the parent folder
        if not os.path.isdir(self._at_parent_folder):
            os.mkdir(self._at_parent_folder)
        else:
            self._clean_folder(self._at_parent_folder)

        # create src and dst folders which will be used for at
        os.mkdir(src_dir_path)

        # copy file which will be at to src dir
        shutil.copyfile(file_path, os.path.join(src_dir_path, "%s.cpp" % self._at_f_name))

        # copy required isat files
        shutil.copy2(self._build_f_path, os.path.join(src_dir_path, self._isat_build_f))
        shutil.copy2(self._clean_f_path, os.path.join(src_dir_path, self._isat_clean_f))
        shutil.copy2(self._test_f_path, os.path.join(src_dir_path, self._isat_test_f))
        shutil.copy2(self._make_f_path, os.path.join(src_dir_path, self._make_f))

    def _run_at(self):
        """Runs isat.py which does the auto tuning

        Raises:
            ValueError: If isat.py return code != 0
        """

        log_file_path = os.path.join(self._at_parent_folder, "log.txt")  # Writes STDOUT in here
        log = open(log_file_path, 'w')

        # auto tuning command. -i for source of at -o for destination of at result
        cmd = "%s -i %s -o %s" % (self._isat_py_path, os.path.join(self._at_parent_folder, self._at_src_dir, ),
                                  os.path.join(self._at_parent_folder, self._at_dst_dir))
        process = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                                   stdout=log, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        process.stdin.write('y')  # isat asks whether we want to continue
        process.communicate()[0]

        log.close()

        if process.returncode != 0:
            print("Isat.py return code != 0. Check %s for errors" % log_file_path)
            raise ValueError()

    def _extract_report_info(self):
        """Copies report over to auto-tune-reports dir and extracts best block size from it

        Raises:
            ValueError: If auto tuning report is not found
        """

        global reports_folder_path

        # prep report dir if does not exist
        if not os.path.isdir(reports_folder_path):
            os.mkdir(reports_folder_path)

        at_report_path = os.path.join(os.path.join(self._at_parent_folder, self._at_dst_dir), "isat-report.txt")

        if not os.path.isfile(at_report_path):  # checks if report file exist
            print "%s not found" % at_report_path
            raise ValueError()

        # extracts the best block size
        self._extract_best_block_size(at_report_path)

        # note if running on windows change into back slash
        split = self._at_parent_folder.split('/')
        report_name_ref = split[len(split) - 1]  # get at parent folder name which is used for naming reports

        # copy file for ref
        shutil.copyfile(at_report_path, os.path.join(reports_folder_path, "%s-at-report.txt" % report_name_ref))

    def _extract_best_block_size(self, at_report_path):
        """Extracts the best best block size from at report and writes it to  the final report file

        Args:
            at_report_path (str): full path to a at report

        Raises:
            ValueError: If failed to extract block size. Can't open report files, can't write to them. etc
        """
        global final_report_path
        try:
            # Gets the required info from  at report
            with open(at_report_path, 'r') as at_report:
                # if all these keywords are in the line then next line is the one we want
                keyword_lst = ["Rank", "Value", "Time(secs)"]

                lines = at_report.readlines()
                for i in range(0, len(lines)):  # loops through all the lines trying to find the best
                    if all(keyword in lines[i] for keyword in keyword_lst):
                        #                                   finds text between parenthesis
                        best_block = lines[i + 1][lines[i + 1].find("(") + 1:lines[i + 1].find(")")]
                        best_block = best_block.replace(" ", '')

                        # writes the best block size for dimensions starting from outer most
                        str_to_write = "\n%d %d %s" % (self.time_order, self.space_order, best_block)
                        break

            if not os.path.isfile(final_report_path):  # initialises report file if it does not exist
                with open(final_report_path, 'w') as final_report:
                    final_report.write("time o,space o,best block size")
                    final_report.write(str_to_write)  # writes the string
            else:
                with open(final_report_path, 'r') as final_report:  # reads all the contents
                    lines = final_report.readlines()

                # checks whether entry already exist and updates it. Otherwise appends to the end of the file
                entry_found = False
                str_to_check = "%d %d " % (self.time_order, self.space_order)
                for i in range(1, len(lines)):
                    if str_to_check in lines[i]:  # remove the newline from beginning of the string
                        lines[i] = str_to_write.replace('\n', '')  # replace line if matching found
                        entry_found = True
                        break

                if not entry_found:  # if entry not found append string to the end of file
                    lines.append(str_to_write)

                with open(final_report_path, 'w') as final_report:  # writes all the contents
                    final_report.writelines(lines)

        except Exception:
            print "Failed to extract best block size from %s" % at_report_path
            raise ValueError()

    def _clean_folder(self, folder_path):
        """Helper method. Deletes all files and folders in the specified directory

        Args:
            folder_path(str): full path to the folder where we want to delete everything (use with care)
        """

        if not os.path.isdir(folder_path):  # returns if folder does not exist
            return

        try:
            for the_file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, the_file)

                if os.path.isfile(file_path):  # removes all files
                    os.unlink(file_path)
                elif os.path.isdir(file_path):  # removes all dirs
                    shutil.rmtree(file_path)

        except Exception as e:
            print "Failed to clean %s\n%s" % (folder_path, e)

    def _set_x_permission(self, file_path):
        """Helper method. Sets os executable permission for a given file
        file_path(str): full path to the file that we want to set x permission
        """
        if not os.path.isfile(file_path):
            return

        st = os.stat(file_path)
        os.chmod(file_path, st.st_mode | stat.S_IEXEC)
