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
        args = TrainingArguments(output_dir='../model/benchmark',
                                 evaluation_strategy='no',
                                 per_device_train_batch_size=batch_size,
                                 num_train_epochs=1,
                                 save_strategy='no',
                                 logging_strategy='no',
                                 log_level='warning',
                                 learning_rate=1e-5,
                                 weight_decay=0.1,
                                 adam_beta1=0.9,
                                 adam_beta2=0.98,
                                 adam_epsilon=1e-6,
                                 lr_scheduler_type='polynomial',
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
                    batch=[8, 16, 32, 64, 128])
class BenchRoBERTaBaseline(BenchRoBERTa):

    pass


@benchmark.register(name='RoBERTa/Randomized/{task}/{batch}/{rate}',
                    task=TASKS,
                    batch=[8, 16, 32, 64, 128],
                    rate=[0.1, 0.2, 0.5])
class BenchRoBERTaRandomized(BenchRoBERTa):

    def setup(self):
        super().setup()

        # Convert all linear modules to linear modules with randomized matrix
        # multiplication.
        from fewbit import LinearGRP
        from fewbit.util import convert_linear, map_module

        # NOTE Replace default Linear layer with ours with routine that traverses
        # module as a tree in post-order.
        def convert_layer(module, path):
            return convert_linear(module,
                                  LinearGRP,
                                  proj_dim_ratio=self.rate,
                                  proj_dim_min=3)

        model = map_module(self.trainer.model, convert_layer)

        # Now, override base trainer with converted linear layers.
        from transformers import Trainer
        self.trainer = Trainer(model=model,
                               args=self.trainer.args,
                               train_dataset=self.trainer.train_dataset,
                               tokenizer=self.trainer.tokenizer,
                               compute_metrics=self.trainer.compute_metrics)


if __name__ == '__main__':
    benchmark.main()
