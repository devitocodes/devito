import devito

from devito import SpaceDimension, TimeDimension

dims = {'i': SpaceDimension(name='i'),
        'j': SpaceDimension(name='j'),
        'k': SpaceDimension(name='k'),
        't0': TimeDimension(name='t0'),
        't1': TimeDimension(name='t1')}

print(dims)


from devito import Grid, Constant, Function, TimeFunction
from devito.types import Array, Scalar

grid = Grid(shape=(10, 10))
symbs = {'a': Scalar(name='a'),
         'b': Constant(name='b'),
         'c': Array(name='c', shape=(3,), dimensions=(dims['i'],)).indexify(),
         'd': Array(name='d',
                    shape=(3,3),
                    dimensions=(dims['j'],dims['k'])).indexify(),
         'e': Function(name='e',
                       shape=(3,3,3),
                       dimensions=(dims['t0'],dims['t1'],dims['i'])).indexify(),
         'f': TimeFunction(name='f', grid=grid).indexify()}
print(symbs)

from devito.ir.iet import Expression
from devito.ir.equations import DummyEq
from devito.tools import pprint

def get_exprs(a, b, c, d, e, f):
    return [Expression(DummyEq(a, b + c + 5.)),
            Expression(DummyEq(d, e - f)),
            Expression(DummyEq(a, 4 * (b * a))),
            Expression(DummyEq(a, (6. / b) + (8. * a)))]

exprs = get_exprs(symbs['a'],
                  symbs['b'],
                  symbs['c'],
                  symbs['d'],
                  symbs['e'],
                  symbs['f'])

pprint(exprs)

from devito.ir.iet import Iteration

def get_iters(dims):
    return [lambda ex: Iteration(ex, dims['i'], (0, 3, 1)),
            lambda ex: Iteration(ex, dims['j'], (0, 5, 1)),
            lambda ex: Iteration(ex, dims['k'], (0, 7, 1)),
            lambda ex: Iteration(ex, dims['t0'], (0, 4, 1)),
            lambda ex: Iteration(ex, dims['t1'], (0, 4, 1))]

iters = get_iters(dims)

def get_block1(exprs, iters):
    # Perfect loop nest:
    # for i
    #   for j
    #     for k
    #       expr0
    return iters[0](iters[1](iters[2](exprs[0])))

def get_block2(exprs, iters):
    # Non-perfect simple loop nest:
    # for i
    #   expr0
    #   for j
    #     for k
    #       expr1
    return iters[0]([exprs[0], iters[1](iters[2](exprs[1]))])

def get_block3(exprs, iters):
    # Non-perfect non-trivial loop nest:
    # for i
    #   for s
    #     expr0
    #   for j
    #     for k
    #       expr1
    #       expr2
    #   for p
    #     expr3
    return iters[0]([iters[3](exprs[0]),
                     iters[1](iters[2]([exprs[1], exprs[2]])),
                     iters[4](exprs[3])])

block1 = get_block1(exprs, iters)
block2 = get_block2(exprs, iters)
block3 = get_block3(exprs, iters)

pprint(block1), print('\n')
pprint(block2), print('\n')
pprint(block3)


from devito.ir.iet import Callable

kernels = [Callable('foo', block1, 'void', ()),
           Callable('foo', block2, 'void', ()),
           Callable('foo', block3, 'void', ())]

print('kernel no.1:\n' + str(kernels[0].ccode) + '\n')
print('kernel no.2:\n' + str(kernels[1].ccode) + '\n')
print('kernel no.3:\n' + str(kernels[2].ccode) + '\n')

