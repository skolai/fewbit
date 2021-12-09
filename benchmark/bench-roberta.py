"""Pipeline in this benchmarks is based on that is described in notebook
`notebooks/memory-usage-roberta.ipynb`.
"""

import benchmark

from functools import partial

TASKS = [
    'CoLA', 'MNLI', 'MNLI-MM', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST2', 'STSB',
    'WNLI'
]


class BenchRoBERTa(benchmark.Benchmark):

    TASK_TO_KEYS = {
        'cola': ('sentence', None),
        'mnli': ('premise', 'hypothesis'),
        'mnli-mm': ('premise', 'hypothesis'),
        'mrpc': ('sentence1', 'sentence2'),
        'qnli': ('question', 'sentence'),
        'qqp': ('question1', 'question2'),
        'rte': ('sentence1', 'sentence2'),
        'sst2': ('sentence', None),
        'stsb': ('sentence1', 'sentence2'),
        'wnli': ('sentence1', 'sentence2'),
    }

    @staticmethod
    def compute_metric(task, metric, inputs):
        predictions, references = inputs
        if task != 'stsb':
            predictions = predictions.argmax(axis=1)
        else:
            predictions = predictions[..., 0]
        return metric.compute(predictions=predictions, references=references)

    @staticmethod
    def preprocess(tokenizer, lhs, rhs, sample):
        if rhs is None:
            return tokenizer(sample[lhs], truncation=True)
        return tokenizer(sample[lhs], sample[rhs], truncation=True)

    def setup(self):
        from datasets import load_dataset, load_metric
        from transformers import (RobertaTokenizerFast as Tokenizer,
                                  RobertaForSequenceClassification as Model,
                                  Trainer, TrainingArguments)

        # Normalize benchmark attributes.
        batch_size = self.batch
        task = self.task.lower()

        # Load and configure model output head.
        if task in ('mnli', 'mnli-mm'):
            num_labels = 3
        elif task == 'stsb':
            num_labels = 1
        else:
            num_labels = 2
        model_path = 'roberta-base'
        model = Model.from_pretrained(model_path, num_labels=num_labels)

        # Load tokenizer from checkpoint.
        tokenizer = Tokenizer.from_pretrained(model_path)

        # Make dataset preprocessor.
        keys = BenchRoBERTa.TASK_TO_KEYS[task]
        func = partial(BenchRoBERTa.preprocess, tokenizer, *keys)

        # Load and preprocess dataset.
        dataset_path = 'glue'
        dataset_name = 'mnli' if task == 'mnli-mm' else task
        dataset = load_dataset(dataset_path, dataset_name)
        dataset_encoded = dataset.map(func, batched=True)

        # Load dataset metric.
        metric = load_metric(dataset_path, dataset_name)
        metric_compute = partial(BenchRoBERTa.compute_metric, task,
                                 metric)

        # Initialize training driver.
        args = TrainingArguments(output_dir='roberta',
                                 save_strategy='no',
                                 learning_rate=2e-5,
                                 per_device_train_batch_size=batch_size,
                                 num_train_epochs=1,
                                 weight_decay=0.01,
                                 load_best_model_at_end=False,
                                 logging_strategy='no',
                                 log_level='warning',
                                 push_to_hub=False)
        self.trainer = Trainer(model=model,
                               args=args,
                               train_dataset=dataset_encoded['train'],
                               tokenizer=tokenizer,
                               compute_metrics=metric_compute)

    def run(self):
        self.trainer.train()


@benchmark.register(name='Sanity/Check')
def bench_sanity():
    """Function bench_sanity actually does nothing. It is just sanity check.
    """


@benchmark.register(name='RoBERTa/Baseline/{task}/{batch}',
                    task=TASKS,
                    batch=[16, 32, 64, 128, 256, 512])
class BenchRoBERTaBaseline(BenchRoBERTa):

    pass


@benchmark.register(name='RoBERTa/Quantized/{task}/{batch}',
                    task=TASKS,
                    batch=[16, 32, 64, 128, 256, 512])
class BenchRoBERTaQuantized(BenchRoBERTa):

    def setup(self):
        # Load and register PyTorch operators library.
        import fewbit  # noqa: F401
        import torch as T

        bounds = T.tensor([
            -2.39798704e+00, -7.11248159e-01, -3.26290283e-01, -1.55338428e-04,
            3.26182064e-01, 7.10855860e-01, 2.39811567e+00
        ], device=T.device('cuda'))

        levels = T.tensor([
            -0.00260009, -0.08883533, 0.1251944, 0.37204148, 0.6277958,
            0.87466175, 1.08880716, 1.00259936
        ], device=T.device('cuda'))

        def gelu3bit(xs):
            return T.ops.fewbit.gelu(xs, bounds, levels)

        super().setup()


if __name__ == '__main__':
    benchmark.main()
