__all__ = ['compose_nodes']


def compose_nodes(nodes):
    """Build an Iteration/Expression tree by nesting the nodes in ``nodes``."""
    l = list(nodes)

    body = l.pop(-1)
    while l:
        handle = l.pop(-1)
        body = handle._rebuild(body, **handle.args_frozen)

    return body
