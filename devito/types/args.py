import abc

__all__ = ['ArgProvider']


class ArgProvider(object):

    """
    A mixin class for types that can provide runtime values for dynamically
    executed (JIT-compiled) code.
    """

    @abc.abstractproperty
    def _arg_names(self):
        raise NotImplementedError('%s does not provide any default argument names' %
                                  self.__class__)

    @abc.abstractmethod
    def _arg_defaults(self):
        """
        A map of default argument values defined by this type.
        """
        raise NotImplementedError('%s does not provide any default arguments' %
                                  self.__class__)

    @abc.abstractmethod
    def _arg_values(self, **kwargs):
        """
        A map of argument values after evaluating user input.

        Parameters
        ----------
        **kwargs
            User-provided argument overrides.
        """
        raise NotImplementedError('%s does not provide argument value derivation' %
                                  self.__class__)

    def _arg_check(self, *args, **kwargs):
        """
        Raises
        ------
        InvalidArgument
            If an argument value is illegal.
        """
        # By default, this is a no-op
        return

    def _arg_apply(self, *args, **kwargs):
        """
        Postprocess arguments upon returning from dynamically executed code. May be
        called if self's state needs to be updated.
        """
        # By default, this is a no-op
        return

    def _arg_finalize(self, args, **kwargs):
        """
        Finalize the arguments produced by self, eventually turning them into ctypes.
        """
        # By default, this is a no-op
        return {}
