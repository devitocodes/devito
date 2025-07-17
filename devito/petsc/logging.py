from collections import namedtuple, OrderedDict
from dataclasses import dataclass

from devito.types import CompositeObject

from devito.petsc.types import PetscInt
from devito.petsc.utils import petsc_type_mappings


class PetscEntry:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in self.kwargs.items():
            setattr(self, k, v)
        self._properties = {k.lower(): v for k, v in kwargs.items()}

    def __getitem__(self, key):
        return self._properties[key.lower()]

    def __repr__(self):
        return f"PetscEntry({', '.join(f'{k}={v}' for k, v in self.kwargs.items())})"


class PetscSummary(dict):
    """
    A summary of PETSc statistics collected for all solver runs
    associated with a single operator during execution.
    """
    PetscKey = namedtuple('PetscKey', 'name options_prefix')

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.petscinfos = [i for i in params if isinstance(i, PetscInfo)]

        # Gather all unique PETSc function names across all PetscInfo objects
        self._functions = list(dict.fromkeys(
            petsc_return_variable_dict[key].name
            for struct in self.petscinfos
            for key in struct.function_list
        ))
        self._property_name_map = {}
        # Dynamically create a property on this class for each PETSc function
        self._add_properties()

        # Initialize the summary by adding PETSc information from each PetscInfo
        # object (each corresponding to an individual PETScSolve)
        for i in self.petscinfos:
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
        funcs = self._functions
        values = tuple(getattr(petscinfo, c) for c in funcs)
        return PetscEntry(**{k: v for k, v in zip(funcs, values)})

    def _add_properties(self):
        """
        For each function name in `self._functions` (e.g., 'KSPGetIterationNumber'),
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

        for f in self._functions:
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


class PetscInfo(CompositeObject):

    __rargs__ = ('name', 'pname', 'logobjs', 'sobjs', 'section_mapper',
                 'inject_solve', 'function_list')

    def __init__(self, name, pname, logobjs, sobjs, section_mapper,
                 inject_solve, function_list):

        self.logobjs = logobjs
        self.sobjs = sobjs
        self.section_mapper = section_mapper
        self.inject_solve = inject_solve
        self.function_list = function_list

        mapper = {v: k for k, v in petsc_type_mappings.items()}
        fields = [(str(i), mapper[str(i._C_ctype)]) for i in logobjs.values()]
        super().__init__(name, pname, fields)

    @property
    def section(self):
        section = self.section_mapper.items()
        return next((k[0].name for k, v in section if self.inject_solve in v), None)

    @property
    def summary_key(self):
        return (self.section, self.sobjs['options_prefix'])

    def __getattr__(self, attr):
        if attr in self.logobjs.keys():
            return getattr(self.value._obj, self.logobjs[attr].name)
        raise AttributeError(f"{attr} not found in PETSc return variables")


@dataclass
class PetscReturnVariable:
    name: str
    variable_type: None
    input_params: list
    output_param: str


# NOTE:
# In the future, this dictionary should be generated automatically from PETSc internals.
# For now, it serves as the reference for PETSc function metadata.
# If any of the PETSc function signatures change (e.g., names, input/output parameters),
# this dictionary must be updated accordingly.

petsc_return_variable_dict = {
    'kspgetiterationnumber': PetscReturnVariable(
        name='KSPGetIterationNumber',
        variable_type=PetscInt,
        input_params=['ksp'],
        output_param='kspiter'
    ),
    'snesgetiterationnumber': PetscReturnVariable(
        name='SNESGetIterationNumber',
        variable_type=PetscInt,
        input_params=['snes'],
        output_param='snesiter',
    )
}
