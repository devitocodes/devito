__all__ = ['Evaluable']


class Evaluable(object):

    """
    A mixin class for types inherited from SymPy that may carry nested
    unevaluated arguments.

    This mixin is used to implement lazy evaluation of expressions.
    """

    @classmethod
    def _evaluate_maybe_nested(cls, maybe_evaluable):
        if isinstance(maybe_evaluable, Evaluable):
            return maybe_evaluable.evaluate
        try:
            # Not an Evaluable, but some Evaluables may still be hidden within `args`
            if maybe_evaluable.args:
                evaluated = [Evaluable._evaluate_maybe_nested(i)
                             for i in maybe_evaluable.args]
                return maybe_evaluable.func(*evaluated)
            else:
                return maybe_evaluable
        except AttributeError:
            # No `args` to be visited
            return maybe_evaluable

    @property
    def args(self):
        return ()

    @property
    def func(self):
        return self.__class__

    def _evaluate_args(self):
        return [Evaluable._evaluate_maybe_nested(i) for i in self.args]

    @property
    def evaluate(self):
        """Return a new object from the evaluation of ``self``."""
        args = self._evaluate_args()
        return self.func(*args)
