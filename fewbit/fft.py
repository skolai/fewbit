import numpy as np
import torch as T

from typing import Literal, Optional

__all__ = ('dct', 'idct')

DCTNorm = Literal['backward', 'forward', 'ortho']
DCTType = Literal[1, 2, 3, 4]


def _dct2(x: T.Tensor, n: Optional[int], dim: int, norm: DCTNorm) -> T.Tensor:
    # 0a. Move DCT dimension to the last one.
    m = x.shape[dim]
    x = x.transpose(dim, -1)

    # 0b. Select multiplication factor for normalization regime.
    if norm == 'backward':
        mult = 2
    elif norm == 'forward':
        mult = 1
    elif norm == 'ortho':
        mult = np.sqrt(2)
    else:
        raise ValueError(f'Unexpected normalization regime: {norm}.')

    # 1. Shuffle elements [a b c d] -> [a c d b].
    x = T.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)

    # 2. Apply FFT and get [A B C B*].
    y = T.fft.fft(x, n, -1, norm)

    # 3. Multiply by 2 exp(-j pi k / 2 m) and take real part. We are relaying
    #    on broadcasting semantic i.e. the trailing dimensions are matched.
    angle = 0.5 * np.pi / m
    phases = angle * T.arange(m, dtype=x.dtype, device=x.device)
    y = mult * y.real * T.cos(phases) + mult * y.imag * T.sin(phases)

    # 4. Fix normalization of the first element.
    if norm == 'ortho':
        y[..., 0] /= mult

    # 5. Restore output tensor shape and exit.
    return y.transpose(dim, -1)


def _dct3(x: T.Tensor, n: Optional[int], dim: int, norm: DCTNorm) -> T.Tensor:
    # 0a. Move DCT dimension to the last one.
    if (m := x.shape[dim]) == 1:
        y = x.clone()
        return y
    x = x.transpose(dim, -1)

    # Select normalization coefficient.
    if norm == 'backward':
        mult = 1 / np.sqrt(2)
    elif norm == 'forward':
        mult = np.sqrt(2)
    elif norm == 'ortho':
        mult = 1
    else:
        raise ValueError(f'Unexpected normalization regime: {norm}.')

    # 1. Prepare phase transformation of input.
    angles = (0.5j * np.pi / m) * T.arange(m)
    phases = T.exp(angles) / np.sqrt(2)
    phases[0] *= np.sqrt(2)

    # 2. Permute elements to apply FFT procedure.
    fst, rest = x[..., :1], x[..., 1:]
    snd = (rest - 1j * rest.flip(-1)) * phases[1:]
    inp = T.cat([fst, snd], dim=-1)

    # 3. Apply FFT and take only real part.
    out = T.fft.ifft(inp, n, -1, norm).real

    # 4. Reshuffle result of FFT transformation.
    y_mid = (out.shape[-1] + 1) // 2
    y = T.zeros_like(out)
    y[..., :m:2] = mult * out[..., :y_mid]
    y[..., 1::2] = mult * out[..., y_mid:].flip(-1)

    # 5. Restore output tensor shape and exit.
    return y.transpose(dim, -1)


def dct(x: T.Tensor,
        type: DCTType = 2,
        n: Optional[int] = None,
        dim: int = -1,
        norm: DCTNorm = 'backward') -> T.Tensor:
    """Function dct calculates Discrete Cosine Transform (DCT) of arbitrary
    type sequence x in PyTorch.

    :param x: The input array.
    :param type: Type of DCT.
    :param n: Length of the transform.
    :param dim: Axis along whoch the DCT is computed.
    :param norm: Normalization mode.
    :return: The transformed input array.

    See reference papers.

    1. Ahmed N. et al - Discrete Cosine Transform // 1974.
    2. Makhoul J. - A Fast Cosine Transform in One and Two Dimensions // 1980.
    """
    if type == 2:
        return _dct2(x, n, dim, norm)
    elif type == 3:
        return _dct3(x, n, dim, norm)
    else:
        raise ValueError(f'Unexpected DCT type: {type}.')


def idct(x: T.Tensor,
         type: DCTType = 2,
         n: Optional[int] = None,
         dim: int = -1,
         norm: DCTNorm = 'backward') -> T.Tensor:
    if type == 2:
        return _dct3(x, n, dim, norm)
    elif type == 3:
        return _dct2(x, n, dim, norm)
    else:
        raise ValueError(f'Unexpected IDCT type: {type}.')
