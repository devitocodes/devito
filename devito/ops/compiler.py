from devito.parameters import configuration


__all__ = ['CompilerOPS']


class CompilerOPS(configuration['compiler'].__class__):
    def __init__(self, *args, **kwargs):
        super(CompilerOPS, self).__init__(*args, **kwargs)

    def jit_compile(self, soname, code, hcode):
        super(CompilerOPS, self).jit_compile(soname, code)
        target = str(self.get_jit_dir().joinpath(soname))
        h_file = "%s.h" % (target)
        kernel_file = open(h_file, "w")
        kernel_file.write(hcode)
