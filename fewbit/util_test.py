import torch as T

from unittest import TestCase

from fewbit.util import estimate_memory_usage, teniter, traverse


class TestTraverse(TestCase):

    def setUp(self):
        self.model1 = self.model = T.nn.Sequential(
            T.nn.Linear(8, 4),
            T.nn.Linear(4, 1),
        )

        self.model2 = T.nn.Sequential(
            T.nn.Linear(8, 4),
            T.nn.ReLU(),
            T.nn.Linear(4, 1),
        )

    def test_traverse(self):
        xs = T.randn((3, 8))
        ys = self.model(xs.requires_grad_())
        traverse(ys, lambda x, y, z: None)

    def test_teniter(self):
        xs = T.randn((3, 8))

        ys = self.model1(xs.requires_grad_())
        lhs = len(list(teniter(ys, False, True)))

        ys = self.model2(xs.requires_grad_())
        rhs = len(list(teniter(ys, False, True)))
        self.assertEqual(lhs + 1, rhs)

    def test_estimate_memory_usage(self):
        xs = T.randn((3, 8))
        ys = self.model(xs.requires_grad_())

        tot = 4 * (3 * 8 + 4 * 8 + 4 + 1 * 4 + 1)
        est = estimate_memory_usage(ys)
        self.assertEqual(tot, est)

    def test_estimate_memory_usage_saved(self):
        xs = T.randn((3, 8))

        ys = self.model1(xs.requires_grad_())
        lhs = estimate_memory_usage(ys, True)

        ys = self.model2(xs.requires_grad_())
        rhs = estimate_memory_usage(ys, True)

        # Number of bytes in intermediate layer (ReLU activation). It should be
        # equal size of input.
        size = 3 * 4 * 4
        diff = rhs - lhs
        self.assertEqual(size, diff)
