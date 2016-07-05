import os
import re
import stat
import math
import shutil
import subprocess
import multiprocessing

# global vars.
# report folder will be created in current dir. Change if necessary
reports_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "auto-tune-reports")

# where best block sizes for each kernel will be kept
final_report_path = os.path.join(reports_folder_path, "final_report.txt")


# Gets the best block sizes for given kernel if they are in the final_report_path
# param - int time order - time order of kernel.
# param - int space_order - space order of kernel.
# returns - list - best block size. Starting from outer most dimension
# returns - None - if report does not exist or does not contain required block sizes
def get_best_best_block_size(time_order, space_order):
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


# Gets optimal block size. Currently works only on linux
# param - tuple|list shape - shape of kernel
# param - int time order - time order of kernel
# param - int space_order - space order of kernel
# returns - int - most optimal size  for the block
def get_optimal_block_size(shape, time_order, space_order):
    # list [time_order][space_order] hardcoded number of loads from the roof line at (change if necessary):
    # https://docs.google.com/spreadsheets/d/1OmvsTftH3uCfYZj-1Lb-5Ji7edrURf0UzzywYaw0-FY/edit?ts=5745964f#gid=0
    number_of_loads = [[11, 17, 23, 29, 35, 41, 47, 53], [19, 25, 31, 37, 43, 49, 55]]

    # find the cache size in KB
    process = subprocess.Popen("cat /proc/cpuinfo", shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    out = process.communicate()[0]
    for line in out.split('\n'):
        if "cache size" in line:
            cache_s = int(re.findall('\d+', line)[0])
            break

    core_c = multiprocessing.cpu_count()  # number of cores
    loads_c = number_of_loads[time_order / 2][space_order / 2]  # number of loads

    # ((C size / cores) / (4 * length inner most * kernel loads)
    optimal_b_size = math.sqrt(((1000 * cache_s) / core_c) / (4 * shape[len(shape) - 1] * loads_c))
    return int(round(optimal_b_size))  # rounds to the nearest integer


# Class responsible for controlling auto tuning process, copying relevant files and collecting data
class AtController(object):

    def __init__(self):
        self.time_order = None
        self.space_order = None

        at_base_path = "%s/isat" % os.getenv("HOME")  # path to base directory of ISAT auto tuning tool

        self._isat_py_path = os.path.join(at_base_path, "Source/Python/isat.py")  # path to isat script that does at
        self._at_work_dir = os.path.join(at_base_path, "Auto-tuning")  # working dir for at

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

        if not os.path.exists(at_base_path):
            raise BaseException("isat installation directory does not exits")

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

    # Does the auto tuning for selected file
    # param - string file_path - full path to the file that we want to auto tune
    # param - int time order - time order of kernel. Used for naming ref
    # param - int space_order - space order of kernel. Used for naming ref
    def auto_tune(self, file_path, time_order, space_order):
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

    # Copies file to work dir and copies all at isat-files there
    # param - string file_path - full path to the file that we want to auto tune
    def _prep_file_for_at(self, file_path):
        if not os.path.isfile(file_path):  # if necessary change to throw the exception
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

    # runs isat.py which does the auto tuning
    def _run_at(self):
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

    # collects  at report file after completion and copies it to report dir
    # returns - string - copied report path
    def _extract_report_info(self):
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

    # extracts the best best block size from at report and appends it to final report file
    # param - string at_report_path - full path to a at report
    # param - string final_report_path - full path to a final report
    def _extract_best_block_size(self, at_report_path):
        global final_report_path
        try:
            if not os.path.isfile(final_report_path):  # initialises report file if it does not exist
                final_report = open(final_report_path, 'w')
                final_report.write("time o,space o,best block size")
            else:
                final_report = open(self.final_report_path, 'a')

            with open(at_report_path, 'r') as at_report:
                # if all these keywords are in the line then next line is the one we want
                keyword_lst = ["Rank", "Value", "Time(secs)"]

                lines = at_report.readlines()
                for i in range(0, len(lines)): # loops through all the lines trying to find the best
                    if all(keyword in lines[i] for keyword in keyword_lst):
                        #                                   finds text between parenthesis
                        best_block = lines[i + 1][lines[i + 1].find("(") + 1:lines[i + 1].find(")")]
                        best_block = best_block.replace(" ", '')

                        # writes the best block size for dimensions starting from outer most
                        final_report.write("\n%d %d %s" % (self.time_order, self.space_order, best_block))
                        break

            final_report.close()

        except Exception:
            print "Failed to extract best block size from %s" % at_report_path
            raise ValueError()

    # Helper method. Deletes all files and folders in the specified directory
    # param - string folder_path - full path to the folder where we want to delete everything (use with care)
    def _clean_folder(self, folder_path):
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

    # Helper method. Sets os executable permission for a given file
    # param - string file_path - full path to the file that we want to auto tune
    def _set_x_permission(self, file_path):
        st = os.stat(file_path)
        os.chmod(file_path, st.st_mode | stat.S_IEXEC)
