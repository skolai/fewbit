import torch as T

from unittest import TestCase

from fewbit.activation import gelu_op, gelu_py


class TestGELUs(TestCase):

    def test_forward_diff(self):
        inputs = T.tensor([[-1, 0, 1, float('nan')]])
        outgrads = T.ones_like(inputs)

        op_inputs = inputs.clone()
        op_inputs.requires_grad = True
        op_outputs = gelu_op(op_inputs)
        op_outputs.backward(outgrads)

        py_inputs = inputs.clone()
        py_inputs.requires_grad = True
        py_outputs = gelu_py(py_inputs)
        py_outputs.backward(outgrads)

        err = T.linalg.norm(op_inputs.grad - py_inputs.grad)
        self.assertAlmostEqual(err, 0, delta=1e-6)
