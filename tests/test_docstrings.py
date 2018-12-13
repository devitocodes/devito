# Python provides the doctest module, which "searches for pieces of text that look
# like interactive Python sessions, and then executes those sessions to verify that
# they work exactly as shown".
#
# pytest integrates doctest: https://docs.pytest.org/en/latest/doctest.html
#
# Some good reasons to have a ``test_docstrings.py`` rather than calling
# ``pytest --doctest-modules`` directly:
# * inclusion in code coverage
# * skipping tests when using a devito backend (where they would fail, for
#   the most disparate reasons)

from os.path import abspath, dirname
from subprocess import check_call

import pytest
import devito

from conftest import skipif

pytestmark = skipif(['yask', 'ops'])

root = dirname(abspath(devito.__file__))


@pytest.mark.parametrize('module', [
    'dimension', 'equation', 'function', 'grid', 'operator',
    'data/decomposition', 'finite_differences/finite_difference',
    'ir/support/space'
])
def test_docstrings(module):
    assert check_call(["py.test", "--doctest-modules", "%s/%s.py" % (root, module)]) == 0
