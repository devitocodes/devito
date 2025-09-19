
class InsertMode:
    """
    How the entries are combined with the current values in the vectors or matrices.
    Reference - https://petsc.org/main/manualpages/Sys/InsertMode/
    """
    insert_values = 'INSERT_VALUES'
    add_values = 'ADD_VALUES'


class ScatterMode:
    """
    Determines the direction of a scatter in `VecScatterBegin()` and `VecScatterEnd()`.
    Reference - https://petsc.org/release/manualpages/Vec/ScatterMode/
    """
    scatter_reverse = 'SCATTER_REVERSE'
    scatter_forward = 'SCATTER_FORWARD'
