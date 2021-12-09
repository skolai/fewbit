#!/usr/bin/env python3

import inspect
import re

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from csv import DictWriter
from dataclasses import asdict, dataclass
from itertools import count, product
from json import dumps, loads
from os import mkfifo, unlink
from subprocess import Popen
from sys import executable, stdout, stderr
from time import monotonic_ns, sleep

parser = ArgumentParser()
parser.add_argument('-T',
                    '--output-type',
                    default='stdout',
                    choices=('stdout', 'csv'))
parser.add_argument('-d', '--delay', type=float, default=1.5)
parser.add_argument('-f', '--filter', type=str, default='.*')
parser.add_argument('-l', '--list', type=str, const='.*', nargs='?')
parser.add_argument('-r', '--run', type=str)
parser.add_argument('-s', '--socket', type=str, default='.benchmark.sock')
parser.add_argument('-t', '--timeout', type=int, default=600)
parser.add_argument('-o', '--output', type=str)

BENCHMARKS = OrderedDict()
ENTRYPOINT = __file__

CLS_IDX = count()


class Benchmark(ABC):

    @abstractmethod
    def run(self):
        """Method run must be defined by user and it should contain actually
        code to measure.
        """

    def setup(self):
        """Method setup prepares resources required for benchmarking code.
        """

    def teardown(self):
        """Method teardown frees all captured resources.
        """


@dataclass
class BenchmarkResult:

    name: str

    wall_time: int

    cuda_memory_usage: int

    @classmethod
    def from_json(cls, inp: bytes) -> 'BenchmarkResult':
        return cls(**loads(inp))

    def to_json(self) -> bytes:
        obj = asdict(self)
        res = dumps(obj, ensure_ascii=False, indent=None)
        return res


class BenchmarkFormatter(ABC):

    @abstractmethod
    def format(self, res: BenchmarkResult):
        pass


class CsvBenchmarkFormatter(BenchmarkFormatter):

    def __init__(self, path: str):
        self.fout = open(path, 'w')
        self.init = True

    def format(self, res: BenchmarkResult):
        if self.init:
            fields = sorted(asdict(res).keys())
            self.init = False
            self.writer = DictWriter(self.fout, fields)
            self.writer.writeheader()
        self.writer.writerow(asdict(res))


class FileBenchmarkFormatter(BenchmarkFormatter):

    def __init__(self, fout=stdout):
        self.fout = stdout
        self.prologue = True

    def format(self, res: BenchmarkResult):
        if self.prologue:
            self.prologue = False
            print('Name', 'Wall Time', 'CUDA Memory', file=self.fout)
        print(res.name, res.wall_time, res.cuda_memory_usage, file=self.fout)


def register(name: str, **kwargs):
    def decorator(cls_or_fn):
        # Wrap up plain function into stateless type.
        if isinstance(cls_or_fn, type) and issubclass(cls_or_fn, Benchmark):
            cls = cls_or_fn
        elif callable(cls_or_fn):
            cls_idx = next(CLS_IDX)
            cls = type(f'Benchmark{cls_idx}', (Benchmark, ),
                       {'run': lambda _: cls_or_fn()})
        else:
            raise ValueError('Parametrized decorator can be applied only to '
                             'callable or types derived from Benchmark type.')

        # Register route directly if there is no parameters.
        if not kwargs:
            BENCHMARKS[name] = (cls, {})
            return cls

        # Register all routes and it attributes.
        names, values = zip(*kwargs.items())
        for value in product(*values):
            opts = dict(zip(names, value))
            text = name.format(**opts)
            if text in BENCHMARKS:
                raise ValueError(f'Benchmark {text} is already registered.')
            BENCHMARKS[text] = (cls, opts)

        return cls

    return decorator


def search_benchmark(pattern: str):
    regexp = re.compile(pattern)
    for name, route in BENCHMARKS.items():
        if regexp.search(name):
            yield name, route


def spawn(name: str, timeout: int, socket: str):
    cmd = (executable, ENTRYPOINT, '--run', name)
    try:
        child = Popen(cmd, stdout=stderr)
        with open(socket) as fin:
            output = fin.readline()
        child.communicate(timeout=timeout)
    except Exception:
        child.kill()
        raise
    if child.returncode != 0:
        print('running benchmark', name, 'fails')
        exit(1)
    return BenchmarkResult.from_json(output)


def bench(name: str, route):
    # Measure GPU memory usage immideately after PyTorch initialisation.
    import torch as T
    cuda_memory_init = T.cuda.memory_allocated()

    # Do benchmark prologue.
    bench_ty, opts = route
    bench: Benchmark = bench_ty()
    for attr, value in opts.items():
        if hasattr(bench, attr):
            raise RuntimeError(f'Benchmark type already has attribute {attr}.')
        setattr(bench, attr, value)
    cuda_memory_setup = T.cuda.memory_allocated()
    bench.setup()

    # Envelop code needed to measure performance with performace counters.
    elapsed = -monotonic_ns()
    bench.run()  # Run benchmarking function.
    elapsed += monotonic_ns()
    cuda_memory_fini = T.cuda.max_memory_allocated()

    # Do benchmark epilogue.
    bench.teardown()
    return BenchmarkResult(name, elapsed, cuda_memory_fini - cuda_memory_init)


def run_benchmark(args: Namespace):
    if (route := BENCHMARKS.get(args.run)) is None:
        raise ValueError(f'There is no registered benchmark {args.run}.')
    with open(args.socket, 'w') as sock:
        res = bench(args.run, route)
        sock.write(res.to_json() + '\n')
        sock.flush()


def run_filter(args: Namespace):
    if args.output_type == 'csv':
        fmt = CsvBenchmarkFormatter(args.output)
    else:
        fmt = FileBenchmarkFormatter()

    # Do not fork process if only one benchmark is requested to run.
    benchmarks = list(search_benchmark(args.filter))
    print('total', len(benchmarks), 'benchmark(s) matched to pattern')
    if len(benchmarks) == 1:
        res = bench(*benchmarks[0])
        fmt.format(res)
        return

    # Run benchmarks in child subprocesses.
    global ENTRYPOINT
    ENTRYPOINT = inspect.stack()[-1].filename

    try:
        mkfifo(args.socket)
    except FileExistsError:
        print('named pipe', args.socket, 'is already exists')

    try:
        for name, _ in benchmarks:
            res = spawn(name, args.timeout, args.socket)
            fmt.format(res)
            sleep(args.delay)
    finally:
        unlink(args.socket)


def run(args: Namespace):
    if args.run is not None:
        run_benchmark(args)
    elif args.list is not None:
        for name, _ in search_benchmark(args.list):
            print(name)
    elif args.filter is not None:
        run_filter(args)


def main():
    run(parser.parse_args())


if __name__ == '__main__':
    main()
