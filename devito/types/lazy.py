from cached_property import cached_property

from devito.tools import Reconstructable

__all__ = ['Evaluable']


class Evaluable(Reconstructable):

    """
    A mixin class for types inherited from SymPy that may carry nested
    unevaluated arguments.

    This mixin is used to implement lazy evaluation of expressions.
    """

    @classmethod
    def _evaluate_maybe_nested(cls, maybe_evaluable, **kwargs):
        if isinstance(maybe_evaluable, Evaluable):
            return maybe_evaluable._evaluate(**kwargs)
        try:
            # Not an Evaluable, but some Evaluables may still be hidden within `args`
            if maybe_evaluable.args:
                args = [Evaluable._evaluate_maybe_nested(i, **kwargs)
                        for i in maybe_evaluable.args]
                evaluate = not all(i is j for i, j in zip(args, maybe_evaluable.args))
                try:
                    return maybe_evaluable.func(*args, evaluate=evaluate)
                except TypeError:
                    # Not all objects support the `evaluate` kwarg
                    return maybe_evaluable.func(*args)
            else:
                return maybe_evaluable
        except AttributeError:
            # No `args` to be visited
            return maybe_evaluable

    @property
    def args(self):
        return ()

    def _evaluate_args(self, **kwargs):
        return [Evaluable._evaluate_maybe_nested(i, **kwargs) for i in self.args]

    def _evaluate(self, **kwargs):
        """
        Carry out the bulk of `evaluate`.

        Notes
        -----
        Subclasses should override this helper method, not the public
        property `evaluate`.
        """
        args = self._evaluate_args(**kwargs)
        evaluate = not all(i is j for i, j in zip(args, self.args))
        return self.func(*args, evaluate=evaluate)

    @cached_property
    def evaluate(self):
        """
        Return a new object from the evaluation of ``self``.
        """
        return self._evaluate()
