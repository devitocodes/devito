from devito.types.tensor import tens_func

def D(self, shift=None):
    """
    Returns the result of matrix D applied over the TensorFunction.
    """
    if not self.is_TensorValued:
        raise TypeError("The object must be a Tensor object")
    
    M = self.tensor if self.shape[0] != self.shape[1] else self

    comps = []
    func = tens_func(self)
    for j, d in enumerate(self.space_dimensions):
        comps.append(sum([getattr(M[j, i], 'd%s' % d.name)
                        for i, d in enumerate(self.space_dimensions)]))
    return func._new(comps)

def S(self, shift=None):
    """
    Returns the result of transposed matrix D applied over the VectorFunction.
    """
    if not self.is_VectorValued:
        raise TypeError("The object must be a Vector object")
    
    derivs = ['d%s' % d.name for d in self.space_dimensions]
    
    comp = []
    comp.append(getattr(self[0], derivs[0]))
    comp.append(getattr(self[1], derivs[1]))
    if len(self.space_dimensions) == 3:
        comp.append(getattr(self[2], derivs[2]))
        comp.append(getattr(self[1], derivs[2]) + getattr(self[2], derivs[1])) 
        comp.append(getattr(self[0], derivs[2]) + getattr(self[2], derivs[0]))    
    comp.append(getattr(self[0], derivs[1]) + getattr(self[1], derivs[0]))    

    func = tens_func(self)

    return func._new(comp)
