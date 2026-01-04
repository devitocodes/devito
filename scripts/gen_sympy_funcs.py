from sympy.functions import __all__ as funcs

print("import sympy\n")
print("from devito.finite_differences.differentiable\
 import (DifferentiableOp, diffify)\n\n")

for fn in funcs:
    try:
        strc = f"""class {fn}(DifferentiableOp, sympy.{fn}):
    __sympy_class__ = sympy.{fn}
    __new__ = DifferentiableOp.__new__\n\n"""
        exec(strc)
        print(strc)
    except:
        # Some are not classes such as sqrt
        print(f"""def {fn}(x):
    diffify(sympy.{fn}(x))\n\n""")
