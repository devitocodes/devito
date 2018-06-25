Possible work items:

* Fix numpy 1.15

* Generation of parametric loop bounds instead of fixed integers.  Reproducer:
  py.test test_dimension.py -s -r f -vv -k test_bcs Look at the generated code:
  several for-loops express bounds as "x_M + 4" or something along these lines.
  That 4 should be turned into a symbol, such as "x_M_upper", and be provided
  from Python to the Kernel function, just like any other parameter

* A mixin class for [Sparse,PrecomputedSparse]Time[Function] classes in
  function.py. There is some redundant code to be shared.

* DLE-arguments refactor. ``devito.Operator`` currently has a ``TODO`` in the
  ``prepare_arguments`` method. The DLE arguments are the "tunable" arguments,
  ie those whose value can be either imposed by the Devito compiler (giving
  them a fixed value based on some sort of heuristic) or determined through a
  search -- what we call "autotuning", or simply AT. The autotuner can run an
  Operator for a few iterations for a number of times, each time assigning the
  tunable arguments a different value. You can see it at work if you for
  example run ``DEVITO_LOGGING=DEBUG python
  examples/seimisc/acoustic/acoustic_example.py`` -- the variables/symbols
  ``x_block_size`` and ``y_block_size`` will be the target of the AT. Clearly,
  the goal of the AT is to determine those values of ``x_block_size`` and
  ``y_block_size`` that MAXIMIZE the runtime performance of the Operator (ie.,
  those values which MINIMIZE its completion time).  The goal of this exercise
  is to get rid of that ``TODO``; basically, this piece of code:

  ```
  for arg in self._dle_args:
    dim = arg.argument
    osize = (1 + arg.original_dim.symbolic_end
             - arg.original_dim.symbolic_start).subs(args)

    if dim.symbolic_size in self.parameters:
        if arg.value is None:
            args[dim.symbolic_size.name] = osize
        elif isinstance(arg.value, int):
            args[dim.symbolic_size.name] = arg.value
        else:
            args[dim.symbolic_size.name] = arg.value(osize)
  ```
  should be dropped, and the following should be extended:
  ```
  if kwargs.pop('autotune', False):
      args = self._autotune(args)
  else:
      <do something>
  ```
  in the ``do something`` part, we should replicate the logic of the dropped
  code, but w/o using ``_dle_args`` that I want to remove. We should somehow
  use ``self.dimensions/self.parameters``. Likewise,
  ``devito.core.autotuning.py`` will have to be tweaked, as ``self._dle_args``
  is disappearing. Possible roadmap:

  * The DLE's ``Arg`` and ``BlockingArg`` classes should disapper.
  * In ``dle.backends.advanced``, ``loop_blocking`` should be modified to avoid
    the creation of ``BlockingArg``. The same information that ``BlockingArg``
    provides, should rather be provided through a special ``Dimension``,
    perhaps the existing ``IncrDimension``. In particular ... (see next point)
  * The following piece of code should be changed:
    ```
    # Build Iteration over blocks
    dim = blocked.setdefault(i, Dimension(name=name))
    bsize = dim.symbolic_size
    ...
    inter_blocks.append(inter_block)
    ```
    ``Dimension`` should/could be turned into the ``IncrDimension`` I mentioned
    above; the ``Iteration`` now shouldn't need ad-hoc ``limits``, as the loop
    bounds information is carried through the ``IncrDimension``.
  * In fact, we need a bit more than just an ``IncrDimension``. We need a type,
    or a mechanism, to specify that this dimension is actually tunable; we
    could let ``Dimension``s accept a further parameter, ``tunable``, which
    defaults to False, but here we would set it to True
  * Now wherever we had ``_dle_args``, we can rather search and get the
    ``._is_tunable`` ``Dimension``s...
