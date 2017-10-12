from devito.nodes import Iteration
from devito.cgen_utils import ccode
from devito import x
from sympy import symbols, Eq

def test_iteration():
    s, e = symbols("s e")
    it = Iteration([], x, (s, e, 1), offsets=(-1, 1))
    assert "for (int x = s + 1; x < e - 1; x += 1)" in str(ccode(it))

