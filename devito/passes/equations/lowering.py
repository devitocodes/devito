from devito.symbolics import retrieve_functions
from devito.types import Dimension

__all__ = ['add_eqns_from_subdomains']


def add_eqns_from_subdomains(expressions):
    """
    Create and add implicit expressions from subdomains.

    Implicit expressions are those not explicitly defined by the user
    but instead are requisites of some specified functionality.
    """
    processed = []
    for e in expressions:
        if e.subdomain:
            try:
                dims = [d.root for d in e.free_symbols if isinstance(d, Dimension)]
                sub_dims = [d.root for d in e.subdomain.dimensions]
                sub_dims.append(e.subdomain.implicit_dimension)
                dims = [d for d in dims if d not in frozenset(sub_dims)]
                dims.append(e.subdomain.implicit_dimension)
                grid = list(retrieve_functions(e, mode='unique'))[0].grid
                processed.extend([i.func(*i.args, implicit_dims=dims) for i in
                                  e.subdomain._create_implicit_exprs(grid)])
                dims.extend(e.subdomain.dimensions)
                new_e = Eq(e.lhs, e.rhs, subdomain=e.subdomain, implicit_dims=dims)
                processed.append(new_e)
            except AttributeError:
                processed.append(e)
        else:
            processed.append(e)
    return processed
