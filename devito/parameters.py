"""The parameters dictionary contains global parameter settings."""

__all__ = ['Parameters', 'parameters']

# Be EXTREMELY careful when writing to a Parameters dictionary
# Read here for reference: http://wiki.c2.com/?GlobalVariablesAreBad
# https://softwareengineering.stackexchange.com/questions/148108/why-is-global-state-so-evil
# If any issues related to global state arise, the following class should
# be made immutable. It shall only be written to at application startup
# and never modified. 

class Parameters(dict):
    """ A dictionary-like class to hold global configuration parameters for devito
        On top of a normal dict, this provides the option to provide callback functions
        so that any interested module can be informed when the configuration changes. 
    """
    def __init__(self, name=None, **kwargs):
        self._name = name
        self.update_functions = None

        for key, value in iteritems(kwargs):
            self[key] = value

    def __setitem__(self, key, value):
        super(Parameters, self).__setitem__(key, value)

        # If a Parameters dictionary is being added as a child,
        # ask it to tell us when it is updated
        
        if isinstance(value, Parameters):
            child_update = lambda x: self._updated(*x)
            value.update_functions.push(child_update) 

        # Tell everyone we've been updated
        self._updated(key, value)

    def _updated(self, key, value):
        """ Call any provided update functions so everyone knows we've been updated
        """
        for f in self.update_functions:
            f(key, value)

parameters = Parameters()
parameters["log_level"] = 'info'
