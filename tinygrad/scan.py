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

  The scan is recursively blocked. Each small block is fused into one lazy graph,
  while block prefixes are scanned hierarchically. This keeps logarithmic depth
  without launching one kernel for every prefix-network level.
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

  def move_front(x:Tensor) -> Tensor:
    a = x._resolve_dim(axis)
    return x if a == 0 else x.permute((a,)+tuple(i for i in range(x.ndim) if i != a))

  def move_back(x:Tensor, original:Tensor) -> Tensor:
    a = original._resolve_dim(axis)
    return x if a == 0 else x.permute(tuple(range(1, a+1))+(0,)+tuple(range(a+1, x.ndim)))

  def slice_tree(tree:Any, start:int, stop:int, dim:int) -> Any:
    return _tree_map(lambda x: x[(slice(None),)*dim + (slice(start, stop),)], tree)

  def checked_combine(left:Any, right:Any) -> Any:
    result = combine_fn(left, right)
    def check(value:Tensor, expected:Tensor) -> Tensor:
      if value.shape != expected.shape or value.dtype != expected.dtype or value.device != expected.device:
        raise ValueError("combine_fn must preserve the input tree structure and Tensor metadata")
      return value
    return _tree_map(check, result, right)

  def fused_hillis_steele(tree:Any, size:int, dim:int) -> Any:
    out, stride = tree, 1
    while stride < size:
      merged = checked_combine(slice_tree(out, 0, size-stride, dim), slice_tree(out, stride, size, dim))
      out = _tree_map(lambda x,y: x.cat(y, dim=dim), slice_tree(out, 0, stride, dim), merged)
      stride *= 2
    return out

  block = 8
  def recursive_scan(tree:Any, size:int) -> Any:
    if size <= block: return fused_hillis_steele(tree, size, 0)

    blocks = (size + block - 1) // block
    padded_size = blocks * block
    if padded_size != size:
      def pad_last(x:Tensor) -> Tensor:
        tail = x[-1:].expand((padded_size-size,)+x.shape[1:])
        return x.cat(tail, dim=0)
      tree = _tree_map(pad_last, tree)

    blocked = _tree_map(lambda x: x.reshape((blocks, block)+x.shape[1:]), tree)
    local = fused_hillis_steele(blocked, block, 1)
    totals = _tree_map(lambda x: x[:, -1], local)
    block_prefixes = recursive_scan(totals, blocks)

    offsets = _tree_map(lambda x: x[:-1].unsqueeze(1).expand((blocks-1, block)+x.shape[1:]), block_prefixes)
    fixed = checked_combine(offsets, _tree_map(lambda x: x[1:], local))
    local = _tree_map(lambda x,y: x[:1].cat(y, dim=0), local, fixed)
    return _tree_map(lambda x: x.reshape((padded_size,)+x.shape[2:])[:size], local)

  out = _tree_map(move_front, elems)
  if reverse: out = _tree_map(lambda x: x.flip(0), out)
  out = recursive_scan(out, n)
  if reverse: out = _tree_map(lambda x: x.flip(0), out)
  return _tree_map(move_back, out, elems)
