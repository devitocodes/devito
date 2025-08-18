import pytest
import warnings

from devito.warnings import warn, DevitoWarning


class NewWarning(UserWarning):
    """
    A custom warning class
    """
    pass


@pytest.fixture
def custom_warning():
    # Instance of custom warning class
    return NewWarning('** NEW ** A fun new kind of warning')


class TestDevitoWarnings:
    """
    In all cases check that the `DevitoWarning` type of Warning is raised
    __in all cases__, even if a custom type is provided
    """
    def test_raise(self):
        with pytest.warns(DevitoWarning) as w:
            warn('Let this be a warning to you')

        assert len(w) == 1
        assert 'DevitoWarning' in repr(w[0].message)
        assert w[0].filename == __file__

    def test_raise_from_user(self):
        with pytest.warns(DevitoWarning) as w:
            warn('Let this be another warning to you', UserWarning)

        assert len(w) == 1
        assert 'UserWarning:' in str(w[0].message)
        assert w[0].filename == __file__

    def test_raise_from_user_kw(self):
        with pytest.warns(DevitoWarning) as w:
            warn('Let this be another warning to you', category=UserWarning)

        assert len(w) == 1
        assert 'UserWarning:' in str(w[0].message)
        assert w[0].filename == __file__

    def test_raise_from_custom(self, custom_warning):
        with pytest.warns(DevitoWarning) as w:
            warn(custom_warning)

        assert len(w) == 1
        assert 'NewWarning:' in str(w[0].message)
        assert w[0].filename == __file__


class TestWarning:
    """
    Check that the custom DevitoWarning does not interfere with Python warnings
    """
    def test_raise(self):
        with pytest.warns(UserWarning):
            warnings.warn('Let this be a warning to you')

    def test_raise_devito(self):
        with pytest.warns(DevitoWarning):
            warnings.warn('Let this be another warning to you', DevitoWarning)

    def test_raise_devito_kw(self):
        with pytest.warns(DevitoWarning):
            warn('Let this be another warning to you', category=DevitoWarning)

    def test_raise_from_custom(self, custom_warning):
        with pytest.warns(NewWarning):
            warnings.warn(custom_warning)
