"""
File: roberta_token_dist
Created by Junwei (Ivy) Sun

This file contains python script that calculates average token length
of files using an RoBERTa model tokenizer. It also produces visualizations
that display token disributions of the documents.
"""

import json
import torch
import transformers
from transformers import (
    RobertaModel,
    RobertaTokenizerFast
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# global variables
MODEL_ID = "roberta-base"

# aataset loading and preprocessing
train_index = np.loadtxt("training_example_indices.txt")
test_index = np.loadtxt("testing_example_indices.txt")
label_matrix = pd.read_csv("label_matrix_merge_with_none_new.txt", sep=" ", header=None)
with open('removed_meta_all.json', 'r') as file:
    data = json.load(file)
text = []  # extract client texts from all samples into a list of strings
for files in data:
    paragraphs = ' '.join(data[files]['Client_Text_Replaced_Two'])
    text.append(paragraphs)

# initialize RoBERTa tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID, do_lower_case=True)

# tokenize each client text and calculate length
text_length = []
text_length_train = []
text_length_test = []
for count, i in enumerate(text):
    encoded = tokenizer.encode_plus(  # tokenize the input text
            i,
            add_special_tokens=True,
            return_attention_mask = True,
            return_token_type_ids=True,
            return_tensors = 'pt'
        )
    ids = encoded["input_ids"]
    text_length += [ids.shape[1]]
    if count in train_index:
        text_length_train += [ids.shape[1]]
    else:
        text_length_test += [ids.shape[1]]

print(f"The average token size under roberta tokenizer is {sum(text_length)/len(text_length)}")
bins = np.linspace(0, 20000, 500)

# histogram of all sample token sizes
plt.hist(text_length, bins)
plt.savefig('token_dist.pdf')
plt.clf()

# histogram of train and test sample token sizes stacked
plt.hist(text_length_train, bins, alpha=0.5, label="train")
plt.hist(text_length_test, bins, alpha=0.5, label="test")
plt.legend()
plt.savefig('token_dist_train_test.pdf')
