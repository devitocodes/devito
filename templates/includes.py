import cgen


def common_include():
    libraries = ['cassert', 'cstdlib', 'cmath', 'iostream',
                 'fstream', 'vector', 'cstdio', 'string']
    statements = [cgen.Include(s) for s in libraries]
    return statements
