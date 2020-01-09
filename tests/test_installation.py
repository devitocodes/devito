import platform
import os


def test_python_version():
    """
    Test that the CI runs with the expected python version
    """
    # expected version
    e_ver = os.environ.get("PYTHON_VERSION") or os.environ.get("TRAVIS_PYTHON_VERSION")
    # Installed version
    i_ver = platform.python_version()

    # Only for CI so ignore if non-CI
    assert e_ver is None or i_ver.startswith(e_ver)
