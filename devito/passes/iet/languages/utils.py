from devito.data import FULL

__all__ = ['make_clause_reduction']


def make_clause_reduction(reductions):
    """
    Build a string representing a reduction clause given a list of
    2-tuples `(symbol, ir.Operation)`.
    """
    args = []
    for i, r in reductions:
        if i.is_Indexed:
            f = i.function
            bounds = []
            for k, d in zip(i.indices, f.dimensions):
                if k.is_Number:
                    bounds.append('[%s]' % k)
                else:
                    # Languages such as OpenMP and OpenACC expect a range
                    # as input to a reduction clause, such as
                    # `reduction(+:f[0:f_vec->size[1]])`
                    bounds.append('[0:%s]' % f._C_get_field(FULL, d).size)
            args.append('%s%s' % (i.name, ''.join(bounds)))
        else:
            args.append(str(i))
    return 'reduction(%s:%s)' % (r.name, ','.join(args))
