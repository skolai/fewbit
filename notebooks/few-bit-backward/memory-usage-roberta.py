# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Memory Usage: RoBERTa Baseline Pipeline

# In this notebooks we builds benchmarking pipeline for RoBERTa model with
# HuggingFace amazing libraries.

# ## Prerequisites

# First of all, some Python packages are required.
# Also, it is needed to set (at least RO) access token to HuggingFace Hub.

# !pip install datasets transformers

from huggingface_hub import notebook_login  # noqa

notebook_login()

# ## Benchmark

# In fact we needs only a few imports which are more technical (measure memory
# usage) then ml-ish.

import torch as T

from functools import partial
from os import environ

# Suppress HuggingFace tokenizer warnings.
environ['TOKENIZERS_PARALLELISM'] = 'false'


# As we told above, all significant imports and ml-ish code are incapsulated
# inside object in order to prevent occasional resource acquisition.
# Lifecycle of a benchmark can be described as an ordered chain of method calls
# `setup()`, `run()`, and `teardown()`.

class BenchRoBERTa:

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

        # Load and configure model output head.
        if self.task in ('mnli', 'mnli-mm'):
            num_labels = 3
        elif self.task == 'stsb':
            num_labels = 1
        else:
            num_labels = 2
        model_path = 'roberta-base'
        model = Model.from_pretrained(model_path, num_labels=num_labels)

        # Load tokenizer from checkpoint.
        tokenizer = Tokenizer.from_pretrained(model_path)

        # Make dataset preprocessor.
        keys = BenchRoBERTa.TASK_TO_KEYS[self.task]
        func = partial(BenchRoBERTa.preprocess, tokenizer, *keys)

        # Load and preprocess dataset.
        dataset_path = 'glue'
        dataset_name = 'mnli' if self.task == 'mnli-mm' else self.task
        dataset = load_dataset(dataset_path, dataset_name)
        dataset_encoded = dataset.map(func, batched=True)

        # Load dataset metric.
        metric = load_metric(dataset_path, dataset_name)
        metric_compute = partial(BenchRoBERTa.compute_metric, self.task,
                                 metric)

        # Initialize training driver.
        args = TrainingArguments(output_dir='roberta',
                                 save_strategy='no',
                                 learning_rate=2e-5,
                                 per_device_train_batch_size=self.batch,
                                 num_train_epochs=1,
                                 weight_decay=0.01,
                                 load_best_model_at_end=False,
                                 push_to_hub=False)
        self.trainer = Trainer(model=model,
                               args=args,
                               train_dataset=dataset_encoded['train'],
                               tokenizer=tokenizer,
                               compute_metrics=metric_compute)

    def run(self):
        self.trainer.train()

    def teardown(self):
        pass


# Actual model fitting happens to be here.
# In order to get metrics we need to gather statistics of interests before
# invocation of `.run()` and after it.

bench = BenchRoBERTa()
bench.task = 'cola'
bench.batch = 16
bench.setup()

# %%time
memory = -T.cuda.memory_allocated()
bench.run()
memory += T.cuda.max_memory_allocated()

bench.teardown()

print(f'Memory Usage: {memory / 1024 ** 2:.1f} Mb')
