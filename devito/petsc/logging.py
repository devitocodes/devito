import os
from collections import namedtuple
from dataclasses import dataclass

from devito.types import CompositeObject

from devito.petsc.types import (
    PetscInt, PetscScalar, KSPType, KSPConvergedReason, KSPNormType
)
from devito.petsc.internals import petsc_type_to_ctype


class PetscEntry:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in self.kwargs.items():
            setattr(self, k, v)
        self._properties = {k.lower(): v for k, v in kwargs.items()}

    def __getitem__(self, key):
        return self._properties[key.lower()]

    def __len__(self):
        return len(self._properties)

    def __repr__(self):
        return f"PetscEntry({', '.join(f'{k}={v}' for k, v in self.kwargs.items())})"


class PetscSummary(dict):
    """
    # TODO: Actually print to screen when DEBUG of PERF is enabled
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
            for key in struct.query_functions
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
        # Collect the function names associated with this PetscInfo
        # instance (i.e., for a single PETScSolve).
        funcs = [
            petsc_return_variable_dict[f].name for f in petscinfo.query_functions
        ]
        values = [getattr(petscinfo, c) for c in funcs]
        return PetscEntry(**{k: v for k, v in zip(funcs, values)})

    def _add_properties(self):
        """
        For each function name in `self._functions` (e.g., 'KSPGetIterationNumber'),
        dynamically add a property to the class with the same name.

        Each property returns a dict mapping each PetscKey to the
        result of looking up that function on the corresponding PetscEntry,
        if the function exists on that entry.
        """
        def make_property(function):
            def getter(self):
                return {
                    k: getattr(v, function)
                    for k, v in self.items()
                    # Only include entries that have the function
                    if hasattr(v, function)
                }
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

    __rargs__ = ('name', 'pname', 'petsc_option_mapper', 'sobjs', 'section_mapper',
                 'inject_solve', 'query_functions')

    def __init__(self, name, pname, petsc_option_mapper, sobjs, section_mapper,
                 inject_solve, query_functions):

        self.petsc_option_mapper = petsc_option_mapper
        self.sobjs = sobjs
        self.section_mapper = section_mapper
        self.inject_solve = inject_solve
        self.query_functions = query_functions
        self.solve_expr = inject_solve.expr.rhs

        pfields = []

        # All petsc options needed to form the PetscInfo struct
        # e.g (kspits0, rtol0, atol0, ...)
        self._fields = [i for j in petsc_option_mapper.values() for i in j.values()]

        for petsc_option in self._fields:
            # petsc_type is e.g. 'PetscInt', 'PetscScalar', 'KSPType'
            petsc_type = str(petsc_option.dtype)
            ctype = petsc_type_to_ctype[petsc_type]
            pfields.append((petsc_option.name, ctype))

        super().__init__(name, pname, pfields)

    @property
    def fields(self):
        return self._fields

    @property
    def prefix(self):
        # If users provide an options prefix, use it in the summary;
        # otherwise, use the default one generated by Devito
        return self.solve_expr.user_prefix or self.solve_expr.formatted_prefix

    @property
    def section(self):
        section = self.section_mapper.items()
        return next((k[0].name for k, v in section if self.inject_solve in v), None)

    @property
    def summary_key(self):
        return (self.section, self.prefix)

    def __getattr__(self, attr):
        if attr not in self.petsc_option_mapper:
            raise AttributeError(f"{attr} not found in PETSc return variables")

        # Maps the petsc_option to its generated variable name e.g {'its': its0}
        obj_mapper = self.petsc_option_mapper[attr]

        def get_val(val):
            val = getattr(self.value._obj, val.name)
            # Decode the val if it is a bytes object
            return str(os.fsdecode(val)) if isinstance(val, bytes) else val

        # - If the function returns a single value (e.g., KSPGetIterationNumber),
        #   return that value directly.
        # - If the function returns multiple values (e.g., KSPGetTolerances),
        #   return a dictionary mapping each output name to its value,
        #   e.g., {'rtol': val0, 'atol': val1, ...}.
        info = {k: get_val(v) for k, v in obj_mapper.items()}
        if len(info) == 1:
            return info.popitem()[1]
        else:
            return info


@dataclass
class PetscReturnVariable:
    name: str
    variable_type: tuple
    input_params: str
    output_param: tuple[str]


# NOTE:
# In the future, this dictionary should be generated automatically from PETSc internals.
# For now, it serves as the reference for PETSc function metadata.
# If any of the PETSc function signatures change (e.g., names, input/output parameters),
# this dictionary must be updated accordingly.

# TODO: To be extended
petsc_return_variable_dict = {
    # KSP specific
    'kspgetiterationnumber': PetscReturnVariable(
        name='KSPGetIterationNumber',
        variable_type=(PetscInt,),
        input_params='ksp',
        output_param=('kspits',)
    ),
    'kspgettolerances': PetscReturnVariable(
        name='KSPGetTolerances',
        variable_type=(PetscScalar, PetscScalar, PetscScalar, PetscInt),
        input_params='ksp',
        output_param=('rtol', 'atol', 'divtol', 'max_it'),
    ),
    'kspgetconvergedreason': PetscReturnVariable(
        name='KSPGetConvergedReason',
        variable_type=(KSPConvergedReason,),
        input_params='ksp',
        output_param=('reason',),
    ),
    'kspgettype': PetscReturnVariable(
        name='KSPGetType',
        variable_type=(KSPType,),
        input_params='ksp',
        output_param=('ksptype',),
    ),
    'kspgetnormtype': PetscReturnVariable(
        name='KSPGetNormType',
        variable_type=(KSPNormType,),
        input_params='ksp',
        output_param=('kspnormtype',),
    ),
    # SNES specific -> will be extended when non-linear solvers are supported
    'snesgetiterationnumber': PetscReturnVariable(
        name='SNESGetIterationNumber',
        variable_type=(PetscInt,),
        input_params='snes',
        output_param=('snesits',),
    ),
}
