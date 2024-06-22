"""
File: dataset.py
Created by Junwei (Ivy) Sun

This file contains python class of data input to the RoBERTa model.
"""

import torch
from torch.utils.data import Dataset

## Class for transcript data
class TranscriptData(Dataset):
    '''
    Python class for psychotherapy transcripts data that
    will be used as input to the roberta model. Data of
    the class are tokenized by roberta tokenizer.
    '''
    def __init__(self, dataframe, tokenizer, max_len, slicing_cond):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = self.data.text
        self.anx_label = self.data.anxiety_label
        self.dep_label = self.data.depression_label
        self.max_len = max_len
        self.slice = slicing_cond
        if self.slice:
            self.doc_idx = self.data.doc_index  # use this to map back to original doc if slicing

    def __len__(self):
        '''
        Get the number of samples in the dataset
        '''
        return len(self.text)

    def __getitem__(self, index):
        '''
        Get one sample from the dataset at the given index
        '''
        text = str(self.text[index])

        inputs = self.tokenizer.encode_plus(  # tokenize the input text
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask = True,
            return_token_type_ids=True,
            return_tensors = 'pt'
        )

        ids = inputs['input_ids']  # the tokenized form of the input text
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        add_output = 0
        if self.slice:
            add_output = torch.tensor(self.doc_idx[index], dtype=torch.float)
        
        return {
            'ids': ids.clone().detach(),
            'mask': mask.clone().detach(),
            'token_type_ids': token_type_ids.clone().detach(),
            'anx_label': torch.tensor(self.anx_label[index], dtype=torch.float),
            'dep_label': torch.tensor(self.dep_label[index], dtype=torch.float),
            'doc_idx': add_output
        }