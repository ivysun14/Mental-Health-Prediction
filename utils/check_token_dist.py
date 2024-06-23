"""
File: roberta_token_dist
Created by Junwei (Ivy) Sun

This file contains python script that calculates average token length
of files using an RoBERTa model tokenizer. It also produces visualizations
that display token disributions of the documents.
"""

import json
from transformers import RobertaTokenizerFast, BertTokenizer, LongformerTokenizer
import numpy as np
import matplotlib.pyplot as plt


# daataset loading and preprocessing
train_index = np.loadtxt("training_example_indices.txt")
test_index = np.loadtxt("testing_example_indices.txt")
with open('./processed/removed_meta2_reduced.json', 'r') as file:
    data = json.load(file)
text = []  # extract client texts from all samples into a list of strings
for files in data:
    paragraphs = ' '.join(data[files]['Client_Text_Replaced_Two'])
    text.append(paragraphs)

# RoBERTa tokenizer
#tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", do_lower_case=True)
# BERT tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
# Longformer tokenizer
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

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

print(f"The average token size under longformer tokenizer is {sum(text_length)/len(text_length)}")


# histogram of train and test sample token sizes stacked
plt.figure(figsize=(10, 6))
bins = np.linspace(0, 20000, 300)
plt.hist(text_length_train, bins=bins, alpha=0.5, color='#80b1d3', label="Train Set")
plt.hist(text_length_test, bins=bins, alpha=0.7, color='#fdb462', label="Eval Set")
plt.legend()
plt.title('Distribution of Token Size For Samples in Longformer')
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig("token_longformer.png")