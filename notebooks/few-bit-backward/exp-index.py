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

# # Few-Bit Backward: Experiments

# ## Table of Content

# ### I Correctness ([1][1a], 2)
#
# Impact of hyperparameters (number of bits) of GELU on a value of a loss
# function and metrics on both train and test sets in [fine-tuning][1a] regime
# or [from scratch][1b].
#
# ### II Universality and Limits of Applicability (3)
#
# Quality validation on different datasets and tasks (fine-tuning on GLUE?).
#
# ### III Computational Efficiency (4)
#
# Measure memory usage and execution time per one epoch for different batch
# sizes (fine-tuning RoBERTa on GLUE CoLA?).
#
# [1a]: ./exp-fine-tuning-glue.ipynb
# [1b]: ./exp-from-scratch.ipynb
