"""
File: model.py
Created by Junwei (Ivy) Sun

This file contains python classes of refined RoBERTa model, that
either has an additional head for binary classification, or multiple
heads for multi-label classifications.
"""

import torch
from transformers import RobertaModel

## Class for modified RoBERTa model: binary classification
class RobertaBinaryClass(torch.nn.Module):
    def __init__(self, model_id):
        super(RobertaBinaryClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained(model_id)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # contains the hidden states of all layers of the RoBERTa model
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # extract hidden state for the last layer
        hidden_state = output_1[0]
        # selects the first token's hidden state, often used for sentence-level tasks
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

## Class for modified RoBERTa model: multi-label classification
class RobertaMultiLabel(torch.nn.Module):
    def __init__(self, model_id):
        super(RobertaMultiLabel, self).__init__()
        self.l1 = RobertaModel.from_pretrained(model_id)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)  # 2 heads for (anxiety, depression)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output