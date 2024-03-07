from devito.passes.iet.engine import iet_pass

__all__ = ['check_stability', 'error_mapper']


@iet_pass
def check_stability(iet, **kwargs):
    """
    Check if the simulation is stable. If not, return to Python as quickly as
    possible with an error code.
    """
    # TODO
    return iet, {}


error_mapper = {
    'Stability': 100,
    'KernelLaunch': 200,
}
