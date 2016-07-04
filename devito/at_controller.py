import os
import stat
import shutil
import subprocess


# Class responsible for controlling auto tuning process, copying relevant files and collecting data
class AtController(object):

    def __init__(self):
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

        self._at_parent_folder = ""  # saves the location of current at parent folder folder
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
        shutil.rmtree(self.at_file_dir)

        if not os.path.exists(self.at_file_dir):  # creates directory where all necessary files for at will be kept
            os.mkdir(self.at_file_dir)

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
            # these functions have to run in this order
            self._prep_file_for_at(file_path, time_order, space_order)
            self._run_at()
            self._collect_at_data()
        except ValueError:
            print "Auto tuning for %s failed" % file_path

        self._at_parent_folder = ""  # reset parent folder name

    # Copies file to work dir and copies all at isat-files there
    # param - string file_path - full path to the file that we want to auto tune
    # param - int time order - time order of kernel. Used for naming ref
    # param - int space_order - space order of kernel. Used for naming ref
    def _prep_file_for_at(self, file_path, time_order, space_order):
        if not os.path.isfile(file_path):  # if necessary change to throw the exception
            print "%s not found"
            raise ValueError()

        self._at_parent_folder = os.path.join(self._at_work_dir, "time_o_%d_space_o_%d" % (time_order, space_order))
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
        process.stdin.write('y')
        process.communicate()[0]

        log.close()

        if process.returncode != 0:
            print("Isat.py return code != 0. Check %s for errors" % log_file_path)
            raise ValueError()

    # collects  at report file after completion and copies it to report dir
    def _collect_at_data(self):
        # TODO potentially extend to only collect the best at result and put it in one file for all time/space orders
        # report folder will be created in current dir. Change if necessary
        reports_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "auto-tune-reports")

        # prep report dir
        if not os.path.isdir(reports_folder_path):
            os.mkdir(reports_folder_path)

        report_path = os.path.join(os.path.join(self._at_parent_folder, self._at_dst_dir), "isat-report.txt")

        if not os.path.isfile(report_path):  # checks if report file exist
            print "%s not found" % report_path
            raise ValueError()

        # note if running on windows change into back slash
        split = self._at_parent_folder.split('/')
        report_name_ref = split[len(split) - 1]  # get at parent folder name which is used for naming reports

        shutil.copyfile(report_path, os.path.join(reports_folder_path, "%s-at-report.txt" % report_name_ref))

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




lol = AtController()

pathas = os.path.join(lol.at_base_path, "Optimizing/src/testRef2.cpp")
lol.auto_tune(pathas, 3, 4)






