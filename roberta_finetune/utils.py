"""
File: utils.py
Created by Junwei (Ivy) Sun

This file contains utility functions for finetuning the
roberta model.
"""

import math
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def b_tp(preds, labels):
    '''Returns True Positives (TP): count of correct predictions of actual class 1'''
    return ((preds == labels) & (preds == 1)).sum()

def b_fp(preds, labels):
    '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
    return ((preds != labels) & (preds == 1)).sum()

def b_tn(preds, labels):
    '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
    return ((preds == labels) & (preds == 0)).sum()

def b_fn(preds, labels):
    '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
    return ((preds != labels) & (preds == 0)).sum()

def b_metrics(preds, labels):
    '''
    Returns the following metrics:
      - accuracy    = (TP + TN) / N
      - precision   = TP / (TP + FP)
      - recall      = TP / (TP + FN)
      - specificity = TN / (TN + FP)
      - f1_score    = (2 * Precision * Recall) / (Precision + Recall)
    '''
    tp = b_tp(preds, labels)
    tn = b_tn(preds, labels)
    fp = b_fp(preds, labels)
    fn = b_fn(preds, labels)
    b_accuracy = (tp + tn) / (tp + tn + fp + fn)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
    b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
    if b_precision != 'nan' and b_recall != 'nan':
        b_f1 = ((2 * b_precision * b_recall) / (b_precision + b_recall))
    else:
        b_f1 = 'nan'
    return b_accuracy, b_precision, b_recall, b_specificity, b_f1

def b_metrics_all(preds, preds_prob, labels, lr, epoch, placeholder):
    '''
    Wrapper function for all evaluation metrics. Can be called on
    all results combined or results from each epoch.
    
    Inputs:
        - preds: tensor with all predictions
        - preds_prob: tensor with all prediction logits
        - labels: tensor with true labels
        - lr: current lr, for output logging purpose
        - epoch: current epoch, for output logging purpose
        - placeholder: see main finetune script, but also just for output logging purpose
    '''
    # copy into numpy array
    preds_np = preds.detach().cpu().numpy()
    preds_prob_np = preds_prob.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    # calculate evaluation metrics
    accuracy = accuracy_score(labels_np, preds_np)
    f1_micro = f1_score(labels_np, preds_np, average='micro')
    f1_macro = f1_score(labels_np, preds_np, average='macro')
    f1_sample = f1_score(labels_np, preds_np, average='samples')
    f1_weighted = f1_score(labels_np, preds_np, average='weighted')
    auroc_micro = roc_auc_score(labels_np,  preds_prob_np, average='micro')
    auroc_macro = roc_auc_score(labels_np,  preds_prob_np, average='macro')
    auroc_weighted = roc_auc_score(labels_np,  preds_prob_np, average='weighted')
    # hand-calculate some for extra info
    b_accuracy, b_precision, b_recall, b_specificity, b_f1 = b_metrics(labels_np, preds_np)
    # print out into terminal
    print(f"Training metrics for {placeholder} using lr:{lr}, epoch={epoch}:")
    print(f"accuracy: {accuracy}.")
    print(f"f1_micro: {f1_micro}.")
    print(f"f1_macro: {f1_macro}.")
    print(f"f1_sample: {f1_sample}.")
    print(f"f1_weighted: {f1_weighted}.")
    print(f"auroc_micro: {auroc_micro}.")
    print(f"auroc_macro: {auroc_macro}.")
    print(f"auroc_weighted: {auroc_weighted}.")
    print(f"precision: {b_precision}")
    print(f"recall: {b_recall}")

    return

def lr_curve(tr_loss_list, test_loss_list, lr, mode):
    '''
    Draw learning curve for a specific learning rate.
    '''
    plt.plot(tr_loss_list, '--bo', label = "train loss")
    plt.plot(test_loss_list,  '--ro', label = "test loss")
    plt.legend()
    plt.savefig(f'lr_curve_{lr}_{mode}.pdf')
    plt.clf()

def slice_document(doc_string, max_len, tokenizer, overlap):
    '''
    Slice a document into a list of sub-strings with each substring
    containing maximum max_len tokens.
    
    Inputs:
        - doc_string: a string which is contents of a document
        - max_len: maximum token size each substring sallowed
        - tokenizer: tokenizer for a model
        - overlap: percentage of overlap between the substrings
        - doc_idx: unique index associated with this document, so can
                   perform result pooling after finetuning
    
    Returns:
        - sliced_doc: a document splitted into a list of sub-strings
    '''
    curr_pos = 0
    sliced_doc = []
    # encode doc_string into a list of token IDs
    tokenized_input = tokenizer.encode(
            doc_string,
            add_special_tokens=False,
            truncation=False,
            return_tensors = None)
    total_token_size = len(tokenized_input)
    overlap_pos = math.floor(overlap * (max_len - 2))
    # split the tokenized doc_string into substrings each of size (max_len-2), -2 for start and end tokens
    while curr_pos < total_token_size:
        # take the whole sequence when remaining sequence has no more than (max_len - 2) tokens
        if (curr_pos + (max_len - 2)) >= total_token_size:
            trunc_doc = tokenized_input[curr_pos:]
        else: # else take the next (max_len-2) tokens
            trunc_doc = tokenized_input[curr_pos:curr_pos+(max_len-2)]
        sliced_doc.append(trunc_doc)
        if (curr_pos + (max_len - 2)) >= total_token_size:
            break
        curr_pos += (max_len - 2 - overlap_pos)
    
    # sanity check
    recover_length = 0
    for i in sliced_doc:
        recover_length += (len(i) - overlap_pos)
    recover_length += overlap_pos
    assert(recover_length == total_token_size)

    # decode substrings back into words
    for idx, content in enumerate(sliced_doc):
        recovered_text = tokenizer.decode(content, skip_special_tokens=True)
        #print(f"{len(content)}, {len(recovered_text)}")
        sliced_doc[idx] = recovered_text
    
    return sliced_doc

def result_pooling(all_doc_idx, all_preds, all_preds_prob, all_labels):
    '''
    Generate a final prediction for a document based on majority vote
    from subdocuments.

    Returns:
        - true_preds: a 2D-tensor of pooled predictions for unique documents
        - ave_probs: a 2D-tensor of averaged probabilities for unique documents
        - true_labels: a 2D-tensor of true labels for unique documents
    '''
    true_preds = torch.empty(0, 2, device=all_doc_idx.device)
    ave_probs = torch.empty(0, 2, device=all_doc_idx.device)
    true_labels = torch.empty(0, 2, device=all_doc_idx.device)

    # unique docs and their counts, these two tensors have the same length
    unique_doc, inverse_indices, counts = torch.unique(all_doc_idx, return_inverse=True, return_counts=True)

    for i, doc in enumerate(unique_doc):
        # find true predictions through majority vote
        selected_rows = all_preds[inverse_indices == i]
        vote = (torch.sum(selected_rows, dim=0)).type(torch.float64)
        majority = ((torch.div(vote, counts[i]) > 0.5).float()).unsqueeze(dim=0)
        true_preds = (torch.cat((true_preds, majority), dim=0))
        # average prediction probabilities
        selected_probs = all_preds_prob[inverse_indices == i]
        average = torch.sum(selected_probs, dim=0)
        average = (torch.div(vote, counts[i])).unsqueeze(dim=0)
        ave_probs = (torch.cat((ave_probs, average), dim=0))
        # leave only one row per document as true labels
        rep_labels = all_labels[inverse_indices == i]
        rep_labels = torch.unique(rep_labels, dim=0)
        assert(rep_labels.shape[0] == 1)
        assert(rep_labels.shape[1] == 2)
        true_labels = (torch.cat((true_labels, rep_labels), dim=0))
    
    return true_preds, ave_probs, true_labels