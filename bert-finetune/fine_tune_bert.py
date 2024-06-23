#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:15:29 2024

@author: maguo
"""
####################LOAD PACKAGES##############################################
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import sigmoid
from transformers import BertTokenizer, BertForSequenceClassification
#from transformers import get_linear_schedule_with_warmup
import torch.nn as nn

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from tabulate import tabulate
from tqdm import tqdm
from tqdm import trange
import random
import sys

import spacy
nlp = spacy.load("en_core_web_sm")

import argparse
from torch.utils.tensorboard import SummaryWriter

###############################################################################
####################READ PARAMETERS############################################
argp = argparse.ArgumentParser()

argp.add_argument('mode', help="tune, train, tokenize, or test")
argp.add_argument('type', help="truncation or divide")
argp.add_argument('--max_token_len', default=510)
argp.add_argument('--overlap', default=0)
argp.add_argument("--label_path", default="filename.txt")
argp.add_argument("--test_label_path", default="filename_test.txt")
argp.add_argument("--train_data_path", default="removed_meta2_reduced.json")
argp.add_argument("--token_path", default="token")
argp.add_argument("--test_data_path", default="removed_meta2_reduced_test.json")
argp.add_argument("--lr_path", default="learning_rate.txt")
argp.add_argument("--epochs", default=15)
argp.add_argument("--batch_size", default=16)
argp.add_argument("--learning_rate", default=5e-5)

args = argp.parse_args()


MAX_LEN = args.max_token_len
OVERLAP = float(args.overlap)    
print(f"Training for overlap: {args.overlap}")
writer = SummaryWriter('runs/bert_experiment')

####################Prepare for Training#######################################

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    
    text = []
    for files in data:
        paragraphs = ' '.join(data[files]['Client_Text_Replaced_Two'])
        text.append(paragraphs)
        
    df_text = pd.DataFrame(text)
    return df_text
    
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )

def sentence_tokenizer(input_text):
    """Separate for sentences to keep entire sentence in the chunks"""
    doc = nlp(input_text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def divide_doc(input_text, tokenizer):
    """Divide document into smaller chunks each with at most MAX_LEN tokens"""
    sentences = sentence_tokenizer(input_text)
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    
    chunks = []
    overlap_length = int(MAX_LEN * OVERLAP)
    
    chunks = []
    current_chunk = []
    current_length = 0
    overlap_tokens = []

    for sentence in tokenized_sentences:
        # Start new chunk with overlap tokens if available
        if current_length == 0 and overlap_tokens:
            current_chunk.extend(overlap_tokens)
            current_length += len(overlap_tokens)
            overlap_tokens = []

        if current_length + len(sentence) > MAX_LEN:
            chunks.append(current_chunk)
            # Prepare for the next chunk with overlap, if there's room for it
            if overlap_length > 0 and len(current_chunk) >= overlap_length:
                overlap_tokens = current_chunk[-overlap_length:]
            current_chunk = []
            current_length = 0

        current_chunk.extend(sentence)
        current_length += len(sentence)

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def preprocessing_chuncks(input_text, tokenizer):
    """Return a list of processed chunks"""
    chunks = divide_doc(input_text, tokenizer)  # Using 510 to account for [CLS] and [SEP]
    
    # Process each chunk separately
    encoding_chunks = [tokenizer.encode_plus(
                            " ".join(chunk),  # Convert chunk back to string
                            add_special_tokens=True,
                            max_length=512,
                            padding='max_length',
                            return_attention_mask=True,
                            return_tensors='pt',
                            truncation=True
                       ) for chunk in chunks]
    
    # Combine the processed chunks
    """
    combined_encoding = {
        'input_ids': torch.cat([ec['input_ids'] for ec in encoding_chunks], dim=0),
        'attention_mask': torch.cat([ec['attention_mask'] for ec in encoding_chunks], dim=0)
    }
    # Note: token_type_ids are not needed for BERT models in single-sequence tasks
    """
    return encoding_chunks

def preprocessing(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 512,
                        padding='max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True
                   )

def preprocess_random(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  start_idx = random.randint(0, max(0, len(input_text) - 510))
  new_text = input_text[start_idx: ]
  return tokenizer.encode_plus(
                        new_text,
                        add_special_tokens = True,
                        max_length = 512,
                        padding='max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True
                   )

def prepare_data(df_text, label):

    input_ids = []
    attention_masks = []
    label_array = label.values[:, [1,2]]
    
    for sample in df_text[0]:
        # test random
        encoding_dict = preprocess_random(sample, tokenizer)
        input_ids.append(encoding_dict['input_ids']) 
        attention_masks.append(encoding_dict['attention_mask'])


    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    df_label = torch.tensor(label_array)
    
    bs = int(args.batch_size)
    train_data = TensorDataset(input_ids, attention_masks, df_label)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
    
    return train_dataloader

def prepare_chunked_data(df_text, label, group):
    """Save tokenized and chuncked data"""
    input_ids = []
    attention_masks = []
    document_ids = []
    chunk_labels = []
    label_array = label.values
    
    for doc_id, (sample, label) in enumerate(tqdm(zip(df_text[0], label_array), total=len(label_array))):
        
        encoding_dict = preprocessing_chuncks(sample, tokenizer)
        for chunk_encoding in encoding_dict:
            input_ids.append(chunk_encoding['input_ids'])
            attention_masks.append(chunk_encoding['attention_mask'])
            chunk_labels.append(label)  # Duplicate the label for each chunk
            document_ids.append(doc_id)  # Keep track of the original document ID for each chunk
    
    """
    for sample in df_text[0]:
        print("tokenizing ...")
        encoding_dict = preprocessing_chuncks(sample, tokenizer)
        input_ids.append(encoding_dict['input_ids']) 
        attention_masks.append(encoding_dict['attention_mask'])
    """

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    #df_label = torch.tensor(label_array, dtype=torch.float)
    chunk_labels = torch.tensor(chunk_labels, dtype=torch.float)
    
    save_dict = {
    'encoding_dict': encoding_dict,
    'attention_masks': attention_masks,
    'chunk_labels': chunk_labels,
    'document_ids': document_ids,
    'input_ids': input_ids
    }

    # Saving to disk
    torch.save(save_dict, f'{args.token_path}_{args.overlap}_{group}.pth')
    
def load_and_prepare(save_path):
    """Load the chuncked data, and pack then with dataloaders"""
    
    loaded_dict = torch.load(save_path, map_location=torch.device('cpu'))
    device = torch.device("cpu")
    
    input_ids = loaded_dict['input_ids']
    attention_masks = loaded_dict['attention_masks']
    chunk_labels = loaded_dict['chunk_labels']
    document_ids = loaded_dict['document_ids']
    document_ids = torch.tensor(document_ids)
    chunk_labels = torch.tensor(chunk_labels, dtype=torch.int64)
    chunk_labels = chunk_labels[:, [1,2]]
    
    
    bs = int(args.batch_size)
    train_data = TensorDataset(input_ids, attention_masks, chunk_labels, document_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
    
    return train_dataloader

def b_metrics_multi(preds, labels):
    '''
    Returns the following metrics:
      - accuracy    = (TP + TN) / N
      - precision   = TP / (TP + FP)
      - recall      = TP / (TP + FN)
      - specificity = TN / (TN + FP)
    '''
    probabilities = torch.sigmoid(preds)
    predictions = (probabilities > 0.5).int()
    
    predictions = predictions.int()
    labels = labels.int()
    
    TPi = (predictions & labels).sum(dim=0)    
    FPi = (predictions & ~labels).sum(dim=0)
    TNi = (~predictions & ~labels).sum(dim=0)
    FNi = (~predictions & labels).sum(dim=0)
    
    TP = TPi.sum()
    FP = FPi.sum()
    TN = TNi.sum()
    FN = FNi.sum()
    
    b_accuracy = (TP + TN) / (TP + FP + TN + FN)
    b_precision = TP / (TP + FP) if (TP + FP) > 0 else 'nan'
    b_recall = TP / (TP + FN) if (TP + FN) > 0 else 'nan'
    b_specificity = TN / (TN + FP) if (TN + FP) > 0 else 'nan'
    #f_1 = 2*(b_precision * b_recall)/(b_precision + b_recall)
    return b_accuracy, b_precision, b_recall, b_specificity

def b_metrics_sk(logits, labels):
            
    probabilities = sigmoid(logits.cpu()).numpy()
    
    # Convert probabilities to binary predictions based on a threshold of 0.5
    predictions = (probabilities > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    # Calculate F1-scores with different averaging methods
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_samples = f1_score(labels, predictions, average='samples', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Calculate AUROC with different averaging methods
    try:
        auroc_micro = roc_auc_score(labels, probabilities, average='micro')
        auroc_macro = roc_auc_score(labels, probabilities, average='macro')
        auroc_weighted = roc_auc_score(labels, probabilities, average='weighted')
    except:
        auroc_micro = 0
        auroc_macro = 0
        auroc_weighted = 0

    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_samples': f1_samples,
        'f1_weighted': f1_weighted,
        'auroc_micro': auroc_micro,
        'auroc_macro': auroc_macro,
        'auroc_weighted': auroc_weighted,
    }

def any_true(logits, all_document_ids, all_label_ids):
    # Assuming sigmoid function is defined elsewhere or using from scipy
    probabilities = sigmoid(logits.cpu()).numpy()    
    all_predictions = (probabilities > 0.5).astype(int)
    grouped_predictions = defaultdict(list)
    document_labels = {}
    
    for prediction, doc_id, label in zip(all_predictions, all_document_ids, all_label_ids):
        grouped_predictions[doc_id].append(prediction)
        document_labels[doc_id] = label
    
    # Aggregate predictions by checking if any prediction is True within each group
    final_predictions = []
    final_labels = []
    for doc_id, preds in grouped_predictions.items():
        # Use numpy.any to check if any prediction is True (1) within the group
        any_true_pred = np.any(preds).astype(int)
        final_predictions.append(any_true_pred)
        final_labels.append(document_labels[doc_id])
    
    return np.array(final_predictions), np.array(final_labels)

def majority_vote(logits, all_document_ids, all_label_ids):
    probabilities = sigmoid(logits.cpu()).numpy()    
    all_predictions = (probabilities > 0.5).astype(int)
    grouped_predictions = defaultdict(list)
    document_labels = {}
    
    for prediction, doc_id, label in zip(all_predictions, all_document_ids, all_label_ids):
        grouped_predictions[doc_id].append(prediction)
        document_labels[doc_id] = label
    
    # Perform a majority vote within each group
    final_predictions = []
    final_labels = []
    for doc_id, preds in grouped_predictions.items():
        most_common_pred = mode(preds)[0][0]
        final_predictions.append(most_common_pred)
        final_labels.append(document_labels[doc_id])
    
    return np.array(final_predictions), np.array(final_labels)
           
def train_multi_class(train_dataloader, validation_dataloader, group, num_class=2, learning=5e-5):
    #group denotes whether it is chuncked or not
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Load the BertForSequenceClassification model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = num_class,
        output_attentions = False,
        output_hidden_states = False,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning, eps = 1e-08)
    total_steps = len(train_dataloader) * int(args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps/10, eta_min=1e-10)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(int(args.epochs)):
        model.train()  # Set the model to training mode
        total_loss = 0
    
        # Progress bar (tqdm) can be wrapped around any iterable
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            batch = tuple(t.to(device) for t in batch)  # Move batch to device
            if group == "truncation":
                b_input_ids, b_input_mask, b_labels = batch
                #b_labels = b_labels.to(torch.int64)
            else:
                b_input_ids, b_input_mask, b_labels,_ = batch
                b_labels = b_labels.to(torch.float)
    
            model.zero_grad()  # Clear previously calculated gradients
    
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs.logits
            loss = loss_fn(logits, b_labels)
            total_loss += loss
    
            loss.backward()  # Perform backpropagation to calculate gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping to avoid exploding gradients
    
            optimizer.step()  # Update weights
            writer.add_scalar('Training loss', loss, epoch * len(train_dataloader) + step)

            scheduler.step()  # Update learning rate schedule
    
        # Calculate the average loss over the training data
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # ========== Validation ==========
    
        # Set model to evaluation mode
        model.eval()
    
        # Tracking variables 
        val_accuracy = []
        val_f1_micro = []
        val_f1_macro = []
        val_f1_samples = []
        val_f1_weighted = []
        val_auroc_micro = []
        val_auroc_macro = []
        val_auroc_weighted = []
        
        total_loss = 0
        
        if group == "truncation":
        
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                  # Forward pass
                    eval_output = model(b_input_ids, 
                                      token_type_ids = None, 
                                      attention_mask = b_input_mask)
                  
                #logits = eval_output.logits.detach().cpu().numpy()
                logits = eval_output.logits
                loss = loss_fn(logits, b_labels)
                total_loss += loss
                labels = b_labels.to('cpu').numpy()
                # Calculate validation metrics
                metrics = b_metrics_sk(logits, labels)
                val_accuracy.append(metrics["accuracy"])
                val_f1_micro.append(metrics["f1_micro"])
                val_f1_macro.append(metrics["f1_macro"])
                val_f1_samples.append(metrics["f1_samples"])
                val_f1_weighted.append(metrics["f1_weighted"])
                val_auroc_micro.append(metrics["auroc_micro"])
                val_auroc_macro.append(metrics["auroc_macro"])
                val_auroc_weighted.append(metrics["auroc_weighted"])
                
        else:
            all_predictions = torch.tensor([], dtype=torch.float32)
            all_label_ids = []
            all_document_ids = []
        
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_document_ids = batch
                with torch.no_grad():
                  # Forward pass
                    eval_output = model(b_input_ids, 
                                      token_type_ids = None, 
                                      attention_mask = b_input_mask)
                logits = eval_output.logits
                label_float = b_labels.float()
                loss = loss_fn(logits, label_float)
                total_loss += loss
                labels = b_labels.to('cpu').numpy()
                
                #TODO: FIX THIS CALCULATION
                all_predictions = torch.cat((all_predictions, logits.detach().cpu()), 0)
                all_label_ids.extend(labels)
                all_document_ids.extend(b_document_ids.to('cpu').numpy())
                
                # Majority vote or true
                final_pred, final_label = any_true(all_predictions, all_document_ids, all_label_ids)
                
                # Calculate validation metrics
                metrics = b_metrics_sk(logits, labels)
                val_accuracy.append(metrics["accuracy"])
                val_f1_micro.append(metrics["f1_micro"])
                val_f1_macro.append(metrics["f1_macro"])
                val_f1_samples.append(metrics["f1_samples"])
                val_f1_weighted.append(metrics["f1_weighted"])
                val_auroc_micro.append(metrics["auroc_micro"])
                val_auroc_macro.append(metrics["auroc_macro"])
                val_auroc_weighted.append(metrics["auroc_weighted"])
        
        print('\t - Validation loss: {:.4f}'.format(total_loss/len(validation_dataloader)))
        print('\t - Validation accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
        print('\t - Validation f1 micro: {:.4f}'.format(sum(val_f1_micro)/len(val_f1_micro)))
        print('\t - Validation f1 macro: {:.4f}'.format(sum(val_f1_macro)/len(val_f1_macro)))
        print('\t - Validation f1 samples: {:.4f}'.format(sum(val_f1_samples)/len(val_f1_samples)))
        print('\t - Validation f1 weighted: {:.4f}'.format(sum(val_f1_weighted)/len(val_f1_weighted)))
        print('\t - Validation auroc micro: {:.4f}'.format(sum(val_auroc_micro)/len(val_auroc_micro)))
        print('\t - Validation auroc macro: {:.4f}'.format(sum(val_auroc_macro)/len(val_auroc_macro)))
        print('\t - Validation auroc weighted: {:.4f}'.format(sum(val_auroc_weighted)/len(val_auroc_weighted)))
        model.save_pretrained(f"bert_model_{args.overlap}_{args.mode}_{args.type}_{learning}")

label_matrix = pd.read_csv(args.label_path, sep=" ", header=None)
df_train = load_data(args.train_data_path)

if args.mode == "tune":
    assert args.lr_path is not None
    df = pd.read_csv(args.lr_path, sep=" ", header=None)
    lrs_list = list(df[0])
    if args.type == "divide":
        for lr in lrs_list:
            torch.cuda.empty_cache()
            train_dataloader = load_and_prepare(f'{args.token_path}_{args.overlap}_train.pth')
            test_dataloader = load_and_prepare(f'{args.token_path}_{args.overlap}_test.pth')
            train_multi_class(train_dataloader, test_dataloader, args.type, learning=lr)
    else:
        for lr in lrs_list:
            train_dataloader = prepare_data(df_train, label_matrix)
            df_test = load_data(args.test_data_path)
            label_matrix_test = pd.read_csv(args.test_label_path, sep=" ", header=None)
            test_dataloader = prepare_data(df_test, label_matrix_test)

            torch.cuda.empty_cache()
            train_multi_class(train_dataloader, test_dataloader, args.type, learning=lr)
    
elif args.mode == "train":
    if args.type == "truncation":
        print("Training the BERT model with Truncation...")
        train_dataloader = prepare_data(df_train, label_matrix)
        df_test = load_data(args.test_data_path)
        label_matrix_test = pd.read_csv(args.test_label_path, sep=" ", header=None)
        test_dataloader = prepare_data(df_test, label_matrix_test)
        torch.cuda.empty_cache()
        lr = float(args.learning_rate)
        train_multi_class(train_dataloader, test_dataloader, args.type, learning=lr)
    else:
        lr = float(args.learning_rate)
        train_dataloader = load_and_prepare(f'{args.token_path}_{args.overlap}_train.pth')
        test_dataloader = load_and_prepare(f'{args.token_path}_{args.overlap}_test.pth')
        train_multi_class(train_dataloader, test_dataloader, args.type, learning=lr)
        
elif args.mode == "tokenize":
    assert args.type == "divide"
    prepare_chunked_data(df_train, label_matrix, "train")
    df_test = load_data(args.test_data_path)
    label_matrix_test = pd.read_csv(args.test_label_path, sep=" ", header=None)
    prepare_chunked_data(df_test, label_matrix_test, "test")
else:
    print("invalid argument")



