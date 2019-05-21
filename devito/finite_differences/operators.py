from devito.logger import error



def div(func):
    """
    Divergence of the input func.
    Func can be a Function, VectorFunction or TensorFunction
    """
    from devito.types.tensor import VectorFunction, VectorTimeFunction
    to = getattr(func, 'time_order', 0)
    # Divergence of a VectorFunction is a Function (sum of derivatives of components)
    if func.is_VectorValued:
        return sum([getattr(func[i], 'd%s'%d.name) for i, d in enumerate(func.space_dimensions)])
    # Divergence of a TensorFunction is a VectorFunction (sum of derivatives of components of each column)
    elif func.is_TensorValued:
        comps = []
        vec_func = VectorTimeFunction if func.is_TimeDependent else VectorFunction
        for j, d in enumerate(func.space_dimensions):
            comps.append(sum([getattr(func[j, i], 'd%s'%d.name) for i, d in enumerate(func.space_dimensions)]))
        return vec_func(name='grad_%s'%func.name, grid=func.grid, space_order=func.space_order,
                        components=comps, time_order=to)
    # Divergence of a Function is a Function (sum of derivatives w.rt each dimensions)
    elif func.is_Function:
        return sum(getattr(func, 'd%s'%d.name) for d in func.grid.dimensions)     
    else:
        return 0


def grad(func):
    """
    Gradient of the input func.
    Func can be a Function or a VectorFunction
    """
    from devito.types.tensor import TensorFunction, TensorTimeFunction
    to = getattr(func, 'time_order', 0)
    # Gradient of a Vecotr is the tensor dfi/dxj
    if func.is_VectorValued:
        tens_func = TensorTimeFunction if func.is_TimeDependent else TensorFunction
        comps = []
        comps = [[getattr(f, 'd%s'%d.name) for d in func.space_dimensions] for f in func]
        return tens_func(name='grad_%s'%func.name, grid=func.grid, space_order=func.space_order,
                         components=comps, time_order=to, symmetric=False)     
    elif func.is_Function:
        comps = [getattr(func, 'd%s'%d.name) for d in func.dimensions if d.is_Space]
        vec_func = VectorTimeFunction if func.is_TimeDependent else VectorFunction
        return vec_func(name='grad_%s'%func.name, grid=func.grid, space_order=func.space_order,
                        components=comps, time_order=to)
    else:
        return 0

def curl(func):
    """
    Curl of the input func.
    Func can be a Function or a VectorFunction
    """
    from devito.types.tensor import VectorFunction, VectorTimeFunction
    if not func.is_VectorValued:
        error("Curl only defined for a Vector")
    if len(func.space_dimensions) != 3:
        error("Curl only defined in three dimensions")
    # The curl of a VectorFunction is a VectorFunction
    derivs = ['d%s'%d.name for d in func.space_dimensions]
    comp1 = getattr(func[2], derivs[1]) - getattr(func[1], derivs[2])
    comp2 = getattr(func[0], derivs[2]) - getattr(func[2], derivs[0])
    comp3 = getattr(func[1], derivs[0]) - getattr(func[0], derivs[1])
    vec_func = VectorTimeFunction if func.is_TimeDependent else VectorFunction
    to = getattr(func, 'time_order', 0)
    return vec_func(name='curl_%s'%func.name, grid=func.grid,
                    space_order=func.space_order, time_order=to,
                    components=[comp1, comp2, comp3])
    
def Laplacian(func):
    """
    Laplacian of the input func.
    Func can be a Function, VectorFunction or TensorFunction
    """
    from devito.types.tensor import VectorFunction, VectorTimeFunction
    to = getattr(func, 'time_order', 0)
    # Laplacian of a VectorFunction is a Function (sum of derivatives of components)
    if func.is_VectorValued:
        return sum([getattr(func[i], 'd%s2'%d.name) for i, d in enumerate(func.space_dimensions)])
    # Laplacian of a TensorFunction is a VectorFunction (sum of derivatives of components of each column)
    elif func.is_TensorValued:
        comps = []
        vec_func = VectorTimeFunction if func.is_TimeDependent else VectorFunction
        for j, d in enumerate(func.space_dimensions):
            comps.append(sum([getattr(func[i,j], 'd%s2'%d.name) for i, d in enumerate(func.space_dimensions)]))
        return vec_func(name='grad_%s'%func.name, grid=func.grid, space_order=func.space_order,
                        components=comps, time_order=to)
    # Laplacian of a Function is a Function (sum of derivatives w.rt each dimensions)
    elif func.is_Function:
        return sum(getattr(func, 'd%s2'%d.name) for d in func.grid.dimensions)     
    else:
        return 0

def diag(func, size=None):
    """
    Creates the diagonal Tensor with func on its diagonal
    """
    dim = size or len(func.indices)
    dim = dim-1 if func.is_TimeDependent else dim
    to = getattr(func, 'time_order', 0)

    from devito.types.tensor import TensorFunction, TensorTimeFunction
    tens_func = TensorTimeFunction if func.is_TimeDependent else TensorFunction

    comps = [[func if i==j else 0 for i in range(dim)] for j in range(dim)]
    return tens_func(name='diag', grid=func.grid, space_order=func.space_order,
                     components=comps, time_order=to, diagonal=True)     