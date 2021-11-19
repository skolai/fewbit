import numpy as np

from gzip import open as gzopen
from pathlib import Path

__all__ = ('load_mnist', 'load_mnist_dataset')


def load_images(path: str):
    with gzopen(path, 'rb') as fin:
        header = fin.read(4 * 4)
        _, noimages, width, height = np.frombuffer(header, '>u4', 4)
        body = fin.read(noimages * width * height)
        img = np.frombuffer(body, 'u1')
        return img.reshape((noimages, width, height))


def load_labels(path: str):
    with gzopen(path, 'rb') as fin:
        header = fin.read(2 * 4)
        _, nolabels = np.frombuffer(header, '>u4', 2)
        body = fin.read(nolabels)
        labels = np.frombuffer(body, 'u1')
        return labels


def load_mnist(root: str, subset='train'):
    """Function load_mnist reads MNIST image dataset from a specified directory
    and returns pair of all images and all labels (60k train + 10k test).
    """
    if subset == 'train':
        prefix = 'train'
    elif subset == 'test':
        prefix = 't10k'
    else:
        raise RuntimeError(f'Unexpected value of `subset`: {subset}.')

    path = Path(root) / 'mnist'
    images = load_images(path / f'{prefix}-images-idx3-ubyte.gz')
    labels = load_labels(path / f'{prefix}-labels-idx1-ubyte.gz')
    return images, labels


def load_mnist_dataset(root: str, subset='train'):
    import tensorflow as tf
    images, labels = load_mnist(root, subset)
    images = images.reshape(-1, 28 * 28)  # Image size if 28 x 28.
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset
