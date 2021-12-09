# Few Bit: Benchmark

## Overview

Measuring memory usage is a bit tricky in case of GPU.
In fact what we want to call 'memory usage' is actually peak allocated memory.
Value of the metric is expected to be overestimated since memory fragmentation exists and objects do not fit (small/medium/large) extent well.
Another obstacle in memory measuring is memory leaking in sense that references to an unused objects persists in a code.

In order to prevent memory fragmentation and memory leaking as well as to maintain reproducibility, we measure memory usage during one epoch for a particular case in a sparate child process.
So, relying on subprocessing allows us to get some isolation guarantees.
We introduce simple single-module library to organize benchmarking code, maintain benchmark life-cycle (initialization, running, and finalization of benchmarking code) and manage subprocessing.

The library exposes a CLI which is similar to CLI of Google Benchmark/Test executables.
It runs the parent process which spawns exactly one child bprocess per case at a time.
A subprocess takes file lock to protect against occasional benchmark subprocesses (not implemented yet).
Also, a parent and child processes communicate over FIFO (named pipe).
In order to run multiple parent processes on different GPU, one can define visibility of GPUs with `CUDA_VISIBLE_GPU` envrironment variable.

The library measures memory usage with `allocated_memory` and `max_allocated_memory` functions from `torch.cuda` packages.
The former is invoked after PyTorch is imported.
The latter is invoked after benchmarking code finished.
The resulting memory usage is difference of measured values.

## CLI

As an example, in order to benchmark baseline GeLU against quantized GeLU with output to CSV-formatted file on RoBERTa model and MNLI pretrain task for batch size of 16 or 32, one can run the command below.

```bash
python bench-roberta.py -T csv -o mnli.csv -f 'RoBERTa/.*/MNLI/(16|32)'
```

## Usage

Library is simple enough in usage.
Everything one needs is to wrap up function or class with decorator and call library's `main` function.

```python
import benchmark


@benchmark.register(name='Sanity/Check')
def bench_sanity():
    """Function bench_sanity actually does nothing. It is just sanity check.
    """


@benchmark.register(name='RoBERTa/Baseline/{task}/{batch}',
                    task=['CoLA', 'MNLI'],
                    batch=[16, 32, 64, 128, 256, 512])
class BenchRoBERTaBaseline(BenchRoBERTa):
    """Class BenchRoBERTaBaseline is registered a benchmark with templated
    name. Its actual name will be determined in runtime. It renderrs template
    with substituting parameters (elements of both `task` and `batch`). The
    total number of benchmark cases to run is `len(taks) * len(batch) == 12`.
    """

    def setup(self):
        """Method setup does all required preparation for benchmarking code.
        For example, it can initialize model and optimizer or load and
        preprocess dataset. Variable `self` has attributes declared during
        benchmark registration (i.e. `task` and `batch`).
        """

    def run(self):
        """Method run fits a model defined in method `setup` to a dataset
        loaded in the same method. Actual benchmarking code have to be placed
        here.
        """

    def teardown(self):
        """Method teardown frees all required resources if necessary.
        """


if __name__ == '__main__':
    benchmark.main()
```
