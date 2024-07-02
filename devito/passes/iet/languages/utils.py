__all__ = ['joins']


def joins(*symbols):
    return ",".join(sorted([i.name for i in symbols]))
