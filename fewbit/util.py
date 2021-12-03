#   encoding: utf-8
#   filename: util.py

import torch as T

from functools import partial
from typing import Any, Callable

TraverseCallable = Callable[[Any, T.Tensor, bool], Any]


def estimate_memory_usage(variable: T.Tensor, saved_only=False) -> int:
    """Function estimate_memory_usage calculates memory used by all tensors
    reachebled from the specified one. It takes into account either explicitly
    saved tensors or implicitely saved (almost all).
    """
    if saved_only:
        args = (False, True)
    else:
        args = (True, False)
    return sum(ten.numel() * ten.element_size()
               for ten in teniter(variable, *args))


def traverse(variable: T.Tensor, callback: TraverseCallable):
    """Function traverse walks through backward gradients pass graph and calls
    callback on any encountered tensor. This means that tensor could may appear
    few times.

    :param variable: Root variable to traverse from.
    :param callback: Function called on saved tensor.
    """
    _traverse(variable.grad_fn, callback, set(), 0)


def _list_saved_tensors(edge, visited):
    for attr in dir(edge):
        if attr.startswith('_saved_'):
            if T.is_tensor(val := getattr(edge, attr)):
                yield val
            elif isinstance(val, tuple):
                yield from (ten for ten in val if T.is_tensor(ten))


def _traverse(node, fn, visited, depth):
    # Skip visited or undefined grad functions.
    if node is None or node in visited:
        return
    else:
        visited.add(node)

    # Custom Python function which instantiate `save_tensors` attribute.
    # NOTE There is no way to access saved tensors from here.

    # Custom Python function which instantiate `save_tensors` attribute.
    if hasattr(node, 'saved_tensors'):
        for ten in node.saved_tensors:
            fn(node, ten, True)

    # Obsolete function which uses `Variable`.
    if hasattr(node, 'variable'):
        # Since `Variable` is depreceted and `Variable.data` reffers to
        # underlying `Tensor` we should use it. In other words `Variable` is
        # proxy object to `Tensor`.
        fn(node, node.variable.data, False)

    # Built-in function like `relu` or `gelu`.
    for ten in _list_saved_tensors(node, visited):
        fn(node, ten, True)

    # Any PyTorch function.
    for child, _ in getattr(node, 'next_functions', ()):
        _traverse(child, fn, visited, depth + 1)


def teniter(variable: T.Tensor, include_ordinary=True, include_saved=False):
    """Function teniter returns iterator on tensors which could be traced from
    variable with `grad_fn` functions.
    """
    def dedup(state, parent, ten, saved):
        if tensor := state.get(id(ten)):
            ordinary = not saved or tensor[1]
            saved = saved or tensor[2]
            state[id(ten)] = (ten, ordinary, saved)
        else:
            state[id(ten)] = (ten, not saved, saved)

    state = {}
    traverse(variable, partial(dedup, state))

    def predicate(value):
        _, ordinary, saved = value
        if include_ordinary and include_saved:
            return ordinary or saved
        elif include_ordinary:
            return ordinary
        elif include_saved:
            return saved
        else:
            return False

    for ten, _, _ in filter(predicate, state.values()):
        yield ten
