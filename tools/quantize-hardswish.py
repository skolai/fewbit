#!/usr/bin/env python3
"""Simple script for manual quantization of gradients of hardswish activation
function.
"""

import logging

import numpy as np

from argparse import ArgumentParser
from pathlib import Path

from fewbit.approx import StepWiseFunction


parser = ArgumentParser(description=__doc__)
parser.add_argument('-o',
                    '--output',
                    type=Path,
                    help='Path to store quantization parameters.')
parser.add_argument('nobits', type=int, help='Number of quantization levels.')


def func(xs: np.ndarray) -> np.ndarray:
    return xs / 3 + 0.5


def quantize_hardswish(nobits: int,
                       x_min: float = -100.0,
                       x_max: float = 100.0):
    card = 2**nobits

    borders = np.empty(card + 1)
    borders[0] = x_min
    borders[-1] = x_max

    if nobits == 1:
        borders[1] = 0
    else:
        borders[1:-1] = np.linspace(-3, 3, card - 1)

    if nobits == 1:
        levels = np.array([
            (0.5 - 0.0) / (0 - x_min),
            (x_max - 0) / (x_max - 0),
        ])
    else:
        values = func(borders[1:-1])
        levels = np.empty(card)
        levels[0] = 0
        levels[-1] = 1
        levels[1:-1] = 0.5 * (values[1:] + values[:-1])

    return StepWiseFunction(borders, levels)


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    args = parser.parse_args()
    quant = quantize_hardswish(args.nobits)

    logging.info('resulting stepwise approximation is below\n%s', quant)

    if args.output:
        logging.info('save quantization to output file %s', args.output)
        case = f'hardswish{args.nobits:02d}'
        kwargs = {
            f'{case}-borders': quant.borders,
            f'{case}-levels': quant.levels,
        }

        if args.output.exists():
            logging.info('output file exists: try update it')
            try:
                with np.load(args.output) as npz:
                    kwargs_loaded = dict(npz)
                kwargs_loaded.update(**kwargs)
                kwargs = kwargs_loaded
            except Exception:
                logging.error('failed to load existing file: overwrite it')

        np.savez(args.output, **kwargs)

    logging.info('done.')


if __name__ == '__main__':
    main()
