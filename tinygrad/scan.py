from __future__ import annotations
from typing import Any, Callable

from tinygrad.tensor import Tensor


def _tree_map(fn:Callable[..., Any], *trees:Any) -> Any:
  tree = trees[0]
  if isinstance(tree, Tensor):
    if not all(isinstance(x, Tensor) for x in trees[1:]): raise TypeError("associative_scan tree structures must match")
    return fn(*trees)
  if isinstance(tree, dict):
    if not all(isinstance(x, dict) and x.keys() == tree.keys() for x in trees[1:]):
      raise TypeError("associative_scan tree structures must match")
    return {k:_tree_map(fn, *(x[k] for x in trees)) for k in tree}
  if isinstance(tree, (tuple, list)):
    if not all(type(x) is type(tree) and len(x) == len(tree) for x in trees[1:]):
      raise TypeError("associative_scan tree structures must match")
    values = [_tree_map(fn, *(x[i] for x in trees)) for i in range(len(tree))]
    return tuple(values) if isinstance(tree, tuple) else values
  raise TypeError(f"associative_scan only supports trees of Tensors, got {type(tree).__name__}")


def associative_scan(combine_fn:Callable[[Any, Any], Any], elems:Any, axis:int=0, reverse:bool=False) -> Any:
  """Inclusive parallel scan of a Tensor tree using an associative binary function.

  Uses a work-efficient Brent-Kung network: fewer than ``2*n`` elements pass through
  ``combine_fn`` for scan length ``n``, with logarithmic dependency depth.
  """
  if not callable(combine_fn): raise TypeError("combine_fn must be callable")
  leaves:list[Tensor] = []
  _tree_map(lambda x: leaves.append(x), elems)
  if not leaves: raise ValueError("associative_scan requires at least one Tensor")

  sizes = [x.shape[x._resolve_dim(axis)] for x in leaves]
  if not all(isinstance(size, int) for size in sizes): raise ValueError("associative_scan requires a concrete scan length")
  n = sizes[0]
  assert isinstance(n, int)
  if any(size != n for size in sizes[1:]): raise ValueError(f"associative_scan inputs must have the same scan length, got {sizes}")
  if n <= 1: return elems

  def scan_slice(x:Tensor, start:int, stop:int, step:int) -> Tensor:
    a = x._resolve_dim(axis)
    return x[(slice(None),)*a + (slice(start, stop, step),)]

  def update(x:Tensor, value:Tensor, start:int, stop:int, step:int) -> Tensor:
    a = x._resolve_dim(axis)
    target = scan_slice(x, start, stop, step)
    if value.shape != target.shape or value.dtype != x.dtype or value.device != x.device:
      raise ValueError("combine_fn must preserve the input tree structure and Tensor metadata")

    # Interleave stepped updates with padding, then place that span back in the full tensor.
    # This avoids advanced indexing and works for one-element tails at arbitrary scan lengths.
    if step != 1 and stop-start > 1:
      value = value.unsqueeze(a+1)
      value = value.pad_to(tuple(step if j == a+1 else None for j in range(value.ndim)))
      value = value.reshape(value.shape[:a] + (value.shape[a]*value.shape[a+1],) + value.shape[a+2:])
      value = value.shrink_to(tuple(stop-start if j == a else None for j in range(value.ndim)))
    pads = [(0, 0)] * x.ndim
    pads[a] = (start, n-stop)
    value = value.pad(tuple(pads))

    idx = type(x).arange(n).reshape((1,)*a + (n,) + (1,)*(x.ndim-a-1))
    mask = (idx >= start) & (idx < stop) & ((idx-start) % step == 0)
    return mask.where(value, x).contiguous()

  out = _tree_map(lambda x: x.flip(x._resolve_dim(axis)) if reverse else x, elems)
  stride = 1
  while stride < n:
    if 2*stride-1 < n:
      left = _tree_map(lambda x: scan_slice(x, stride-1, n-stride, 2*stride), out)
      right = _tree_map(lambda x: scan_slice(x, 2*stride-1, n, 2*stride), out)
      out = _tree_map(lambda x,y: update(x, y, 2*stride-1, n, 2*stride), out, combine_fn(left, right))
    stride *= 2

  stride //= 4
  while stride:
    if 3*stride-1 < n:
      left = _tree_map(lambda x: scan_slice(x, 2*stride-1, n-stride, 2*stride), out)
      right = _tree_map(lambda x: scan_slice(x, 3*stride-1, n, 2*stride), out)
      out = _tree_map(lambda x,y: update(x, y, 3*stride-1, n, 2*stride), out, combine_fn(left, right))
    stride //= 2

  return _tree_map(lambda x: x.flip(x._resolve_dim(axis)) if reverse else x, out)
