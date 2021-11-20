"""This script reproduces experimental results on for feed-forward neural
networks MNIST dataset from (Hendrycks, 2016).
"""

import logging

import jax
import jax.experimental.optimizers as optimizers
import jax.experimental.stax as stax
import jax.numpy as jnp
import tensorflow as tf

from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

from data import load_mnist_dataset
from tensorboard import TensorBoard, tensorboard

NOEPOCHES = 50
BATCH_SIZE = 128

N_HIDDEN = 128
N_LABELS = 10

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger = logging.getLogger(__file__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(sh)


def Dropout(rate, mode='train'):
    """Layer construction function for a dropout layer with given rate.
    """
    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.get('rng', None)
        if rng is None:
            msg = ('Dropout layer requires apply_fun to be called with a PRNG '
                   'key argument. That is, instead of `apply_fun(params, '
                   'inputs)`, call it like `apply_fun(params, inputs, rng)` '
                   'where `rng` is a jax.random.PRNGKey value.')
            raise ValueError(msg)
        if kwargs.get('mode', mode) == 'train':
            keep = jax.random.bernoulli(rng, rate, inputs.shape)
            return jnp.where(keep, inputs / rate, 0)
        else:
            return inputs
    return init_fun, apply_fun


def get_unit_by_name(unit):
    if unit == 'relu':
        return stax.Relu
    elif unit == 'elu':
        return stax.Elu
    elif unit == 'gelu':
        return stax.Gelu
    else:
        raise RuntimeError(f'Failed to find unit {unit}.')


def make_model(unit_name: str, proba: float, mode='train', n_hidden=N_HIDDEN):
    unit = get_unit_by_name(unit_name)
    block = [stax.Dense(n_hidden), unit, stax.Dropout(proba, mode)]
    return stax.serial(*block * 8, stax.Dense(10))


def jit(*args, **kwargs):
    return partial(jax.jit, *args, **kwargs)


@jit(static_argnames='agg')
def loss_entropy(y_true: jnp.ndarray, y_pred: jnp.ndarray, agg: bool = True):
    assert y_pred.ndim == 2
    ls = jax.nn.log_softmax(y_pred)
    ts = jnp.take_along_axis(ls, y_true[:, None], axis=-1)
    return -ts.mean() if agg else -ts


def evaluate(fn, testset: tf.data.Dataset, rng):
    for i, batch in enumerate(testset):
        xs, ys = (jnp.asarray(el) for el in batch)
        return fn(xs, ys, rng=rng)


def fit(model,
        trainset: tf.data.Dataset,
        testset: tf.data.Dataset,
        lr: float = 1e-3,
        tb: TensorBoard = None,
        seed: int = 42):
    rng = jax.random.PRNGKey(seed)
    model_init, model_apply = model
    _, model_state = model_init(rng, (-1, 784))

    opt_init, opt_update, opt_get = optimizers.adam(lr)
    opt_state = opt_init(model_state)

    @jax.jit
    @jax.value_and_grad
    def objective(state, xs, ys, **kwargs):
        ps = model_apply(state, xs, **kwargs)
        return loss_entropy(ys, ps)

    @jax.jit
    def step(step, opt_state, xs, ys, rng):
        model_state = opt_get(opt_state)
        value, grad = objective(model_state, xs, ys, rng=rng)
        return value, opt_update(step, grad, opt_state)

    @jax.jit
    def infer(state, xs, ys, rng):
        ps = model_apply(state, xs, rng=rng)
        return loss_entropy(ys, ps)

    for i, batch in enumerate(trainset):
        xs, ys = (jnp.asarray(el) for el in batch)
        value, opt_state = step(i, opt_state, xs, ys, rng)
        if i % 10 == 0:
            loss = evaluate(fn=partial(infer, opt_get(opt_state)),
                            testset=testset,
                            rng=rng)
            tb.scalar('cross-entropy/train', value, i)
            tb.scalar('cross-entropy/test', loss, i)
            logger.info('[%3d] cross-entropy: train=%e test=%e', i, value,
                        loss)

    return value, opt_get(opt_state)


def main(args: Namespace):
    trainset = load_mnist_dataset('data') \
        .cache() \
        .repeat(args.num_epoches) \
        .batch(args.batch_size)

    testset = load_mnist_dataset('data')
    testset = testset.take(len(testset)).cache().batch(len(testset))

    model = make_model(args.unit, args.proba)
    log_dir = Path(args.log_dir) / args.unit
    logger.info('write tensorboard logs to %s', log_dir)
    with tensorboard(log_dir) as tb:
        fit(model, trainset, testset=testset, lr=args.schedule, tb=tb)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', default='/tmp/log')
    parser.add_argument('--num-epoches', default=NOEPOCHES, type=int)
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int)
    parser.add_argument('unit',
                        choices=('relu', 'elu', 'gelu', 'silu'),
                        default='gelu',
                        nargs='?',
                        help='Type of activation unit.')
    parser.add_argument('schedule',
                        type=float,
                        default=1e-3,
                        nargs='?',
                        help='Optimiser learning rate.')
    parser.add_argument('proba',
                        type=float,
                        default=1.0,
                        nargs='?',
                        help='Dropout keep probability.')
    args = parser.parse_args()
    main(args)
