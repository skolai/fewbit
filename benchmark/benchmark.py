#!/usr/bin/env python3

import inspect
import re

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from csv import DictWriter
from dataclasses import asdict, dataclass
from json import dumps, loads
from subprocess import PIPE, Popen
from sys import executable, stdout, stderr
from time import monotonic_ns

parser = ArgumentParser()
parser.add_argument('-F',
                    '--format',
                    default='stdout',
                    choices=('stdout', 'csv'))
parser.add_argument('-f', '--filter', type=str, default='.*')
parser.add_argument('-l', '--list', type=str, const='.*', nargs='?')
parser.add_argument('-r', '--run', type=str)
parser.add_argument('-t', '--timeout', type=int, default=600)
parser.add_argument('-o', '--output', type=str)

BENCHMARKS = OrderedDict()
ENTRYPOINT = __file__


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
        res = dumps(obj, ensure_ascii=False, indent=2)
        return res.encode('utf-8')

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


def register(name: str):
    def decorator(fn):
        if name in BENCHMARKS:
            raise ValueError(f'Benchmark {name} is already registered.')
        BENCHMARKS[name] = fn
        return fn

    return decorator


def search_benchmark(pattern: str):
    regexp = re.compile(pattern)
    for name, func in BENCHMARKS.items():
        if regexp.search(name):
            yield name, func


def spawn(name: str, timeout: int):
    cmd = (executable, ENTRYPOINT, '--run', name)
    try:
        child = Popen(cmd, stdout=PIPE, stderr=stderr)
        output, _ = child.communicate(timeout=timeout)
    except Exception:
        child.kill()
        raise
    if child.returncode != 0:
        print('running benchmark', name, 'fails')
        exit(1)
    return BenchmarkResult.from_json(output)


def bench(name, func):
    import torch as T
    cuda_memory = -T.cuda.max_memory_allocated()
    elapsed = -monotonic_ns()
    func()  # Run benchmarking function.
    elapsed += monotonic_ns()
    cuda_memory = T.cuda.max_memory_allocated()
    return BenchmarkResult(name, elapsed, cuda_memory)


def run(args: Namespace):
    if (name := args.run) is not None:
        if (func := BENCHMARKS.get(name)) is None:
            raise ValueError(f'There is no registered benchmark {name}.')
        res = bench(name, func)
        stdout.buffer.write(res.to_json())
        return

    if args.list is not None:
        for name, _ in search_benchmark(args.list):
            print(name)
        return

    if args.filter is not None:
        global ENTRYPOINT
        ENTRYPOINT = inspect.stack()[-1].filename

        if args.format == 'csv':
            fmt = CsvBenchmarkFormatter(args.output)
        else:
            fmt = FileBenchmarkFormatter()

        # Do not fork process if only one benchmark is requested to run.
        benchmarks = list(search_benchmark(args.filter))
        if len(benchmarks) == 1:
            res = bench(*benchmarks[0])
            fmt.format(res)
            return

        for name, _ in benchmarks:
            res = spawn(name, args.timeout)
            fmt.format(res)
        return


def main():
    run(parser.parse_args())


if __name__ == '__main__':
    main()
