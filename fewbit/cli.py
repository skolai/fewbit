#   encoding: utf_8
#   filename: cli.py
"""A command line utility (CLI) to generate few-bit quantizations for gradients
for activation functions.
"""

import inspect
import logging

from argparse import ArgumentParser, ArgumentTypeError, FileType
from importlib import import_module
from pathlib import Path
from sys import stderr
from typing import Optional

from .approx import approximate
from .version import version

__all__ = ('main', )

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARN,
    'error': logging.ERROR,
}


class PathType:
    def __init__(self, exists=False, not_dir=False, not_file=False):
        self.exists = exists
        self.check_dir = not_dir
        self.check_file = not_file

    def __call__(self, value: str) -> Path:
        path = Path(value)

        # If there is no check for path existance then exit.
        if not self.exists:
            return path

        # Check that path exists.
        if not path.exists():
            raise ArgumentTypeError(f'path does not exist: {path}')

        # Check type of a filesystem object referenced by path.
        if self.check_dir and path.is_dir():
            raise ArgumentTypeError(f'directory is not allowed: {path}')

        if self.check_file and path.is_file():
            raise ArgumentTypeError(f'file is not allowed: {path}')

        return path


def help_():
    parser.print_help()


def quantize(max_iters: int, output: Optional[Path], seed: Optional[None],
             nobits: int, spec: str):
    logging.info('loading (primal) function from spec %s', spec)
    module_name, func_name = spec.split(':', 1)
    module = import_module(module_name)
    func = getattr(module, func_name)

    # NOTE Since routine `approximate` defined in term of NumPy but we heavily
    # use PyTorch and its AD engine, we are forced to define bridging code
    # between PyTorch and NumPy. Another reason to use NumPy for approximation
    # routine is its ubiquitous.

    import numpy as np
    import torch as T

    def fn_prim(xs: np.ndarray) -> np.ndarray:
        return func(T.tensor(xs)).numpy()

    def fn(xs: np.ndarray) -> np.ndarray:
        ps = T.tensor(xs, requires_grad=True)
        qs = func(ps)
        qs.backward(T.ones_like(ps))
        return ps.grad.numpy()

    logging.info('run quantization procedure (%d bits)', nobits)

    quant, info = approximate(fn=fn,
                              fn_prim=fn_prim,
                              cardinality=2**nobits,
                              parity=False,
                              max_iters=max_iters,
                              beps=1e-6,
                              leps=1e-6,
                              domain=(-100, 100),
                              random_state=seed)

    if info['status'] != 'converged':
        logging.error('failed to converge in %d iterations', info['noiters'])
        exit(1)

    logging.info('converged in %d iterations', info['noiters'])
    logging.info('resulting stepwise approximation is below\n%s', quant)

    if output:
        logging.info('save quantization to output file %s', output)
        case = f'{func_name}{nobits:02d}'
        kwargs = {
            f'{case}-borders': quant.borders,
            f'{case}-levels': quant.levels,
        }

        if output.exists():
            logging.info('output file exists: try update it')
            try:
                with np.load(output) as npz:
                    kwargs_loaded = dict(npz)
                kwargs.update(**kwargs_loaded)
            except Exception:
                logging.error('failed to load existing file: overwrite it')

        np.savez(output, **kwargs)

    logging.info('done.')


def version_():
    print(f'fewbit version {version}')


def main():
    # Parse command line arguments. If no subcommand were run then show usage
    # and exit. We assume that only main parser (super command) has valid value
    # in func attribute.
    args = parser.parse_args()
    if args.func is None:
        parser.print_usage()
        return

    # Find CLI option or argument by parameter name of handling function.
    kwargs = {}
    spec = inspect.getfullargspec(args.func)
    for name in spec.args:
        kwargs[name] = getattr(args, name)

    # Set up basic logging configuration.
    if (stream := args.log_output) is None:
        stream = stderr

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=LOG_LEVELS[args.log_level],
                        stream=stream)

    # Invoke CLI command handler.
    args.func(**kwargs)


# yapf: disable
parser = ArgumentParser(description=__doc__)
parser.set_defaults(func=None)
parser.add_argument('--log-level', default='info', choices=sorted(LOG_LEVELS.keys()), help='set logger verbosity level')  # noqa: E501
parser.add_argument('--log-output', default=stderr, metavar='FILENAME', type=FileType('w'), help='set output file or stderr (-) for logging')  # noqa: E501

subparsers = parser.add_subparsers()

parser_help = subparsers.add_parser('help', add_help=False, help='Show this message and exit.')  # noqa: E501
parser_help.set_defaults(func=help_)

parser_quantize = subparsers.add_parser('quantize', help='Build and save few-bit approximation.')  # noqa: E501
parser_quantize.set_defaults(func=quantize)
parser_quantize.add_argument('-M', '--max-iters', type=int, default=10000, help='')  # noqa: E501
parser_quantize.add_argument('-o', '--output', type=PathType(False, not_dir=True), help='Path to store quantization parameters.')  # noqa: E501
parser_quantize.add_argument('-s', '--seed', type=int, default=None, help='')  # noqa: E501
parser_quantize.add_argument('nobits', type=int, help='Number of bits to use in quantization.')  # noqa: E501
parser_quantize.add_argument('spec', type=str, help='Qualified name of function to quantize (e.g. "torch.nn.functional:gelu").')  # noqa: E501

parser_version = subparsers.add_parser('version', add_help=False, help='Show version information.')  # noqa: E501
parser_version.set_defaults(func=version_)
# yapf: enable
