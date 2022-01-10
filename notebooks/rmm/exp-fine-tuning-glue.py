# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Few-Bit: Fine-tuning RoBERTa on GLUE

# Based on HuggingFace's tutorial on ["Fune-tuning on Classification Tasks"][1]
# and [pre-trained RoBERTa][2] model.
#
# [1]: https://huggingface.co/docs/transformers/notebooks
# [2]: https://huggingface.co/roberta-base

# + [markdown] id="kTCFado4IrIc"
# The GLUE Benchmark is a group of nine classification tasks on sentences or
# pairs of sentences which are
#
# - [CoLA][1] (abbrv. _Corpus of Linguistic Acceptability_) Determine if a
#   sentence is grammatically correct or not.is a  dataset containing sentences
#   labeled grammatically correct or not.
# - [MNLI][2] (abbrv. _Multi-Genre Natural Language Inference_) Determine if a
#   sentence entails, contradicts or is unrelated to a given hypothesis. (This
#   dataset has two versions, one with the validation and test set coming from
#   the same distribution, another called mismatched where the validation and
#   test use out-of-domain data.)
# - [MRPC][3] (abbrv. _Microsoft Research Paraphrase Corpus_) Determine if two
#   sentences are paraphrases from one another or not.
# - [QNLI][4] (abbrv. _Question-answering Natural Language Inference_)
#   Determine if the answer to a question is in the second sentence or not.
# - [QQP][5] (abbrv. _Quora Question Pairs2_) Determine if two questions are
#   semantically equivalent or not.
# - [RTE][6] (abbrv. _Recognizing Textual Entailment_) Determine if a sentence
#   entails a given hypothesis or not.
# - [SST-2][7] (abbrv. _Stanford Sentiment Treebank_) Determine if the sentence
#   has a positive or negative sentiment.
# - [STS-B][8] (abbrv. _Semantic Textual Similarity Benchmark_) Determine the
#   similarity of two sentences with a score from 1 to 5.
# - [WNLI][9] (abbrv. _Winograd Natural Language Inference_) Determine if a
#   sentence with an anonymous pronoun and a sentence with this pronoun
#   replaced are entailed or not.
#
# [1]: https://nyu-mll.github.io/CoLA/
# [2]: https://arxiv.org/abs/1704.05426
# [3]: https://www.microsoft.com/en-us/download/details.aspx?id=52398
# [4]: https://rajpurkar.github.io/SQuAD-explorer/
# [5]: https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs
# [6]: https://aclweb.org/aclwiki/Recognizing_Textual_Entailment
# [7]: https://nlp.stanford.edu/sentiment/index.html
# [8]: http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark
# [9]: https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html

import builtins

from argparse import ArgumentParser
from functools import partial
from os import environ, makedirs
from pathlib import Path
from typing import Optional

# Force using of fixed CUDA device.
if 'CUDA_VISIBLE_DEVICES' not in environ:
    environ['CUDA_VISIBLE_DEVICES'] = '0'

# +
from datasets import load_dataset, load_metric
from torch import manual_seed
from torch.utils.tensorboard import SummaryWriter
from transformers import (RobertaTokenizerFast as Tokenizer,
                          RobertaForSequenceClassification as Model,
                          Trainer, TrainerCallback, TrainingArguments)
from transformers.integrations import TensorBoardCallback
# -

from fewbit import LinearGRP
from fewbit.util import convert_linear, map_module

DEVICE = 'cuda'

SEED = 0x12c946425095e587
TASK = 'cola'

CACHE_DIR = Path('~/.cache/fewbit').expanduser()
DATA_DIR = Path('../data/huggingface')
LOG_DIR = Path('../log')
MODEL_DIR = Path('../model')

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

TASK_TO_HYPERPARAMS = {
    'cola': (16, 1e-5),
    'mnli': (16, 1e-5),
    'mnli-mm': (16, 1e-5),
    'mrpc': (16, 1e-5),
    'qnli': (16, 1e-5),  # NOTE We use batch size 16 instead of 32.
    'qqp': (32, 1e-5),
    'rte': (16, 2e-5),
    'sst2': (32, 2e-5),
    'stsb': (16, 1e-5),
    'wnli': (32, 1e-5),
}

# +
parser = ArgumentParser()

parser.add_argument('-c', '--cache-dir',
                    default=CACHE_DIR,
                    type=Path,
                    help='Directory to cache or original dataset files.')

parser.add_argument('-d', '--data-dir',
                    default=DATA_DIR,
                    type=Path,
                    help='Directory to cache preprocessed dataset files.')

parser.add_argument('-l', '--log-dir',
                    default=LOG_DIR,
                    type=Path,
                    help='Directory for TensorBoard logs.')

parser.add_argument('-m', '--model-dir',
                    default=MODEL_DIR,
                    type=Path,
                    help='Directory to save checkpoint files.')

parser.add_argument('-p', '--proj-dim-ratio',
                    default=None,
                    type=float,
                    help='Directory to save checkpoint files.')

parser.add_argument('-s', '--seed',
                    default=SEED,
                    type=int,
                    help='Random seed for reproducibility.')

parser.add_argument('task',
                    default=TASK,
                    choices=sorted(TASK_TO_HYPERPARAMS),
                    nargs='?',
                    help='GLUE task to learn.')


# +
def compute_metric(task, metric, inputs):
    predictions, references = inputs
    if task != 'stsb':
        predictions = predictions.argmax(axis=1)
    else:
        predictions = predictions[..., 0]
    return metric.compute(predictions=predictions, references=references)


def preprocess(tokenizer, lhs, rhs, sample):
    if rhs is None:
        args = (sample[lhs],)
    else:
        args = (sample[lhs], sample[rhs])
    return tokenizer(*args,
                     max_length=512,
                     padding=True,
                     truncation=True,
                     return_tensors='np')


def setup(task: str,
          cache_dir: Path = Path('cache'),
          data_dir: Path = Path('data'),
          model_dir: Path = Path('model'),
          callback: Optional[TrainerCallback] = None,
          proj_dim_ratio: Optional[float] = None):
    # Load and configure model output head.
    if task in ('mnli', 'mnli-mm'):
        num_labels = 3
    elif task == 'stsb':
        num_labels = 1
    else:
        num_labels = 2
    model_path = 'roberta-base'
    model = Model.from_pretrained(model_path, num_labels=num_labels)

    # NOTE Replace default Linear layer with ours with routine that traverses
    # module as a tree in post-order.
    def convert_layer(module, path):
        return convert_linear(module, LinearGRP, proj_dim_ratio=proj_dim_ratio)

    if proj_dim_ratio:
        model = map_module(model, convert_layer)

    print(model)

    # Load tokenizer from checkpoint.
    tokenizer = Tokenizer.from_pretrained(model_path)

    # Make dataset preprocessor.
    keys = TASK_TO_KEYS[task]
    func = partial(preprocess, tokenizer, *keys)

    # Load and preprocess dataset.
    dataset_path = 'glue'
    dataset_name = 'mnli' if task == 'mnli-mm' else task
    dataset = load_dataset(dataset_path, dataset_name, cache_dir=str(data_dir))
    dataset_cache = {key: str(cache_dir / f'glue-{task}-{key}.arrow')
                     for key in dataset.keys()}
    dataset_encoded = dataset.map(func,
                                  batched=True,
                                  cache_file_names=dataset_cache)

    # Load dataset metric.
    metric = load_metric(dataset_path, dataset_name)
    metric_compute = partial(compute_metric, task, metric)

    # Pick right evaluation metric.
    eval_metric_name = 'accuracy'
    if task == 'cola':
        eval_metric_name = 'matthews_correlation'
    elif task == 'stsb':
        eval_metric_name = 'pearson'

    # Pick right dataset for train/evaluation stage.
    dataset_train = dataset_encoded['train']
    dataset_eval = dataset_encoded.get('validation')
    if task == 'mnli-mm':
        dataset_eval = dataset_encoded['validation_mismatched']
    elif task == 'mnli':
        dataset_eval = dataset_encoded['validation_matched']

    # Get hyperparameters from task name.
    bs, lr = TASK_TO_HYPERPARAMS[task]

    # Make 6% of total steps as warm up steps.
    noepoches = 10
    warmup_steps = int(0.06 * len(dataset_train) * noepoches / bs)

    # Initialize training driver.
    args = TrainingArguments(
        output_dir=str(model_dir / f'glue-{task}'),
        evaluation_strategy='epoch',
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=noepoches,
        save_strategy='no',
        #save_strategy='epoch',
        #load_best_model_at_end=True,
        #metric_for_best_model=eval_metric_name,
        logging_strategy='epoch',
        log_level='warning',
        learning_rate=lr,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        lr_scheduler_type='polynomial',
        warmup_steps=warmup_steps,
        push_to_hub=False)

    trainer = Trainer(model=model.to(DEVICE),
                      args=args,
                      train_dataset=dataset_train,
                      eval_dataset=dataset_eval,
                      tokenizer=tokenizer,
                      compute_metrics=metric_compute,
                      callbacks=[callback])

    return trainer


# -

def train(task: str, cache_dir: Path, data_dir: Path, log_dir: Path,
          model_dir: Path, proj_dim_ratio: Optional[float], seed: int):
    makedirs(cache_dir, exist_ok=True)
    makedirs(log_dir, exist_ok=True)
    makedirs(model_dir, exist_ok=True)

    manual_seed(seed)

    tensorboard_sm = SummaryWriter(log_dir / task)
    tensorboard_cb = TensorBoardCallback(tensorboard_sm)

    trainer = setup(task, cache_dir, data_dir, model_dir, tensorboard_cb,
                    proj_dim_ratio)
    trainer.train()

    tensorboard_sm.flush()
    tensorboard_sm.close()

    return trainer


# +
# Check we are run by IPytton kernel.
if getattr(builtins, '__IPYTHON__', False):
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

# Run training finally!
train(**args.__dict__)
