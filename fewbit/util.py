#   encoding: utf-8
#   filename: util.py

import re

import torch as T

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from re import Pattern
from typing import Any, Callable, Optional

__all__ = ('HookedMemoryUsage', 'estimate_memory_usage', 'map_module',
           'memory_usage_hooks', 'teniter', 'traverse')

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


@dataclass
class HookedMemoryUsage:

    forward: Optional[int] = None

    backward: Optional[int] = None

    @property
    def value(self) -> Optional[int]:
        return self.backward or self.forward


@contextmanager
def memory_usage_hooks() -> HookedMemoryUsage:
    """Function memory_usage_hooks is a context manager which subscribes to
    pack/unpack events for saved tensors.
    """
    usage = HookedMemoryUsage()

    def pack(ten: T.Tensor) -> Any:
        acc = usage.forward if usage.forward else 0
        usage.forward = acc + ten.numel() * ten.element_size()
        return ten

    def unpack(ten: T.Tensor) -> T.Tensor:
        acc = usage.backward if usage.backward else 0
        usage.backward = acc + ten.numel() * ten.element_size()
        return ten

    with T.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield usage


def map_module(root: T.nn.Module,
               func: Callable[[T.nn.Module, str], T.nn.Module],
               patt: Optional[str] = None) -> T.nn.Module:
    """Function map_module applies function to each leaf of module tree which
    matches to a specified pattern.
    """
    @wraps(func)
    def func_safe(*args, **kwargs):
        node = func(*args, **kwargs)
        if not isinstance(node, T.nn.Module):
            raise ValueError('Mapped result should be toch.nn.Module type.')
        return node

    return _map_module(root, func_safe, re.compile(patt or r'.*'), '')


def _map_module(root: T.nn.Module, func: Callable[[T.nn.Module, str],
                                                  T.nn.Module], patt: Pattern,
                path: str) -> T.nn.Module:
    for name, child in root.named_children():
        node = _map_module(child, func, patt, f'{path}/{name}')
        if node != child:
            setattr(root, name, node)
    if patt.match(path or '/'):
        root = func(root, path or '/')
    return root


def convert_linear(module: T.nn.Linear, ctor, **kwargs) -> T.nn.Module:
    """Function convert_linear takes module and returns linear module with
    approximate matmul. Non-linear modules are returned intact.
    """
    if not isinstance(module, T.nn.Linear):
        return module

    # Construct new layer from existing Linear layer.
    layer = ctor(in_features=module.in_features,
                 out_features=module.out_features,
                 bias=module.bias is not None,
                 device=module.weight.device,
                 dtype=module.weight.dtype,
                 **kwargs)
    layer.weight = T.nn.Parameter(module.weight)
    if layer.bias is not None:
        layer.bias = T.nn.Parameter(module.bias)

    return layer
