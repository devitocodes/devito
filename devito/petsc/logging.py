from collections import namedtuple, OrderedDict
from devito.petsc.iet.logging import PetscInfo


class PetscSummary(OrderedDict):
    """
    A summary of PETSc statistics collected for all solver runs
    associated with a single operator during execution.
    """
    PetscKey = namedtuple('PetscKey', 'name options_prefix')

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        petscinfos = [i for i in params if isinstance(i, PetscInfo)]
        self.petscinfos = petscinfos

        # Gather all unique PETSc function names across all PetscInfo objects
        functions = set()
        for i in petscinfos:
            functions.update(i.function_mapper.keys())

        self._property_name_map = {}
        # Dynamically create a property on this class for each PETSc function
        self._add_properties(functions)
        self._functions = functions

        # Initialize the summary by adding PETSc information
        # from each PetscInfo object (each corresponding to
        # an individual PETScSolve)
        for i in petscinfos:
            self.add_info(i)

    def add_info(self, petscinfo):
        """
        For a given PetscInfo object, create a key
        and entry and add it to the PetscSummary.
        """
        key = self.PetscKey(*petscinfo.summary_key)
        entry = self.petsc_entry(petscinfo)
        self[key] = entry

    def petsc_entry(self, petscinfo):
        """
        Create a named tuple entry for the given PetscInfo object,
        containing the values for each PETSc function call.
        """
        calls = petscinfo.function_mapper.keys()
        values = tuple(getattr(petscinfo, c) for c in calls)
        PetscEntry = namedtuple('PetscEntry', calls)
        return PetscEntry(*values)

    def _add_properties(self, functions):
        """
        For each function name in `functions` (e.g., 'KSPGetIterationNumber'),
        dynamically add a property to the class with the same name.

        Each property returns an OrderedDict that maps each PetscKey to the
        result of looking up that function on the corresponding PetscEntry,
        if the function exists on that entry.
        """
        def make_property(function):
            def getter(self):
                return OrderedDict(
                    (k, getattr(v, function))
                    for k, v in self.items()
                    # Only include entries that have the function
                    if hasattr(v, function)
                )
            return property(getter)

        for f in functions:
            # Inject the new property onto the class itself
            setattr(self.__class__, f, make_property(f))
            self._property_name_map[f.lower()] = f

    def get_entry(self, name, options_prefix=None):
        """
        Retrieve a single PetscEntry for a given name
        and options_prefix.
        """
        key = self.PetscKey(name, options_prefix)
        if key not in self:
            raise ValueError(
                f"No PETSc information for:"
                f" name='{name}'"
                f" options_prefix='{options_prefix}'"
            )
        return self[key]

    def __getitem__(self, key):
        if isinstance(key, str):
            # Allow case insensitive key access
            original = self._property_name_map.get(key.lower())
            if original:
                return getattr(self, original)
            raise KeyError(f"No PETSc function named '{key}'")
        elif isinstance(key, tuple) and len(key) == 2:
            # Allow tuple keys (name, options_prefix)
            key = self.PetscKey(*key)
            return super().__getitem__(key)
        else:
            return super().__getitem__(key)

    def __getattr__(self, name):
        """
        Allow case insensitive attribute access.
        """
        original = self._property_name_map.get(name.lower())
        if original:
            return getattr(self, original)
        raise AttributeError(f"No attribute named '{name}'")
