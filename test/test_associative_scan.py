import unittest
import numpy as np

from tinygrad import Tensor, associative_scan, dtypes


class TestAssociativeScan(unittest.TestCase):
  def test_add_lengths(self):
    for n in (0, 1, 2, 3, 5, 8, 9, 17, 31, 32, 33):
      values = np.arange(n, dtype=np.float32)
      np.testing.assert_allclose(associative_scan(lambda a,b: a+b, Tensor(values)).numpy(), np.cumsum(values))

  def test_axis_and_reverse(self):
    values = np.arange(30, dtype=np.float32).reshape(2, 5, 3)
    forward = associative_scan(lambda a,b: a+b, Tensor(values), axis=1).numpy()
    reverse = associative_scan(lambda a,b: a+b, Tensor(values), axis=-2, reverse=True).numpy()
    np.testing.assert_allclose(forward, np.cumsum(values, axis=1))
    np.testing.assert_allclose(reverse, np.flip(np.cumsum(np.flip(values, axis=1), axis=1), axis=1))

  def test_noncommutative_order(self):
    values = np.array([[[1, 1], [0, 1]], [[1, 0], [1, 1]], [[2, 0], [0, 1]], [[1, 2], [0, 1]]], dtype=np.float32)
    forward, reverse = np.empty_like(values), np.empty_like(values)
    forward[0] = values[0]
    for i in range(1, len(values)): forward[i] = forward[i-1] @ values[i]
    reverse[-1] = values[-1]
    for i in range(len(values)-2, -1, -1): reverse[i] = reverse[i+1] @ values[i]
    np.testing.assert_allclose(associative_scan(lambda a,b: a@b, Tensor(values)).numpy(), forward)
    np.testing.assert_allclose(associative_scan(lambda a,b: a@b, Tensor(values), reverse=True).numpy(), reverse)

  def test_affine_tree(self):
    a, b = Tensor([2, 3, 4, 5]), Tensor([1, 1, 1, 1])
    def compose(left, right):
      la, lb = left
      ra, rb = right
      return ra*la, ra*lb+rb
    out_a, out_b = associative_scan(compose, (a, b))
    self.assertEqual(out_a.tolist(), [2, 6, 24, 120])
    self.assertEqual(out_b.tolist(), [1, 4, 17, 86])

  def test_mamba_recurrence(self):
    rng = np.random.default_rng(0)
    a = rng.uniform(0.7, 0.99, (2, 3, 9, 4)).astype(np.float32)
    b = rng.uniform(-0.1, 0.1, (2, 3, 9, 4)).astype(np.float32)
    expected, state = np.empty_like(b), np.zeros_like(b[:, :, 0])
    for i in range(b.shape[2]):
      state = a[:, :, i]*state+b[:, :, i]
      expected[:, :, i] = state
    def compose(left, right):
      la, lb = left
      ra, rb = right
      return ra*la, ra*lb+rb
    _, actual = associative_scan(compose, (Tensor(a), Tensor(b)), axis=2)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-6)

  def test_nested_tree(self):
    elems = {"x":Tensor([1, 2, 3]), "state":[Tensor([2, 3, 4])]}
    out = associative_scan(lambda a,b: {"x":a["x"]+b["x"], "state":[a["state"][0]*b["state"][0]]}, elems)
    self.assertEqual(out["x"].tolist(), [1, 3, 6])
    self.assertEqual(out["state"][0].tolist(), [2, 6, 24])

  def test_gradient(self):
    x = Tensor(np.arange(7, dtype=np.float32), dtype=dtypes.float)
    grad = associative_scan(lambda a,b: a+b, x).sum().gradient(x)[0]
    np.testing.assert_allclose(grad.numpy(), np.arange(7, 0, -1, dtype=np.float32))

  def test_bounded_combine_work(self):
    work = []
    def combine(a, b):
      work.append(a.numel())
      return a+b
    associative_scan(combine, Tensor.arange(128))
    self.assertEqual((len(work), sum(work)), (9, 435))
    self.assertLess(sum(work), 4*128)

  def test_metadata_and_tree_errors(self):
    with self.assertRaises(ValueError):
      associative_scan(lambda a,b: (a[0]+b[0], a[1]+b[1]), (Tensor.arange(3), Tensor.arange(4)))
    with self.assertRaises(TypeError):
      associative_scan(lambda a,b: [a[0]+b[0]], (Tensor.arange(3),))
    with self.assertRaises(ValueError):
      associative_scan(lambda a,b: a.float()+b.float(), Tensor.arange(3))


if __name__ == "__main__": unittest.main()
