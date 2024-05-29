try:
    import pyrevolve as pyrevolve  # noqa
    from .checkpoint import *  # noqa
except ImportError:
    pass


class Noop:
    """ Dummy replacement in case pyrevolve isn't available. """

    def __init__(self, *args, **kwargs):
        raise ImportError("Missing required `pyrevolve`; cannot use checkpointing")


class NoopCheckpointOperator(Noop):
    pass


class NoopCheckpoint(Noop):
    pass


class NoopRevolver(Noop):
    pass
