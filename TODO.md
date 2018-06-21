- Generation of parametric loop bounds instead of fixed integers.
  Reproducer: py.test test_dimension.py -s -r f -vv -k test_bcs
  Look at the generated code: several for-loops express bounds as
  "x_M + 4" or something along these lines. That 4 should be turned
  into a symbol, such as "x_M_upper", and be provided from Python
  to the Kernel function, just like any other parameter
