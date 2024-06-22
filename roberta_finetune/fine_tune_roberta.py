"""
File: fine_tune_roberta
Created by Junwei (Ivy) Sun

This file contains python script that finetunes an RoBERTa model
on labeled psychotherapy transcripts. Based on the input command
line argument, either a binary classification on a specified psych
symptom will be performed or a multi-label classification will be
performed. In case of multilabel: there are in total four possible
outcomes: (anxiety, depression)
    - (1, 1)
    - (1, 0)
    - (0, 1)
    - (0, 0)
The evaluation metrics used are accuracy, F1 score, and AUROC.
"""

#!pip install torch transformers

## LIBRARIES NEEDED
## =================
import os
import sys
import json
import argparse
import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizerFast
import pandas as pd
import numpy as np
from model import RobertaBinaryClass, RobertaMultiLabel
from dataset import TranscriptData
from utils import b_metrics_all, lr_curve, slice_document, result_pooling
from train import train_multilabel, test_multilabel
## =================


## COMMAND LINE ARGUMENTS
## ======================
argp = argparse.ArgumentParser()
argp.add_argument('--mode', help="Choose hyperparam or evaluate", default="evaluate")
argp.add_argument('--subdocument', help='Proportion of overlap between subdocuments [0, 0.9]', default=None, type=float)
argp.add_argument('--finetune_lr', help="A list of lr to be used with mode hyperparam", nargs="*", type=float)
argp.add_argument('--optimal_lr', help="Optimal lr to be used with mode evaluate", default=1e-3, type=float)
argp.add_argument('--finetune_epoch', help="A max epoch number to be used with mode hyperparam", type=int)
argp.add_argument('--optimal_epoch', help="Optimal epoch to be used with mode evaluate", default=3, type=int)
argp.add_argument('--task', help="Choose binary or multi", default="multi")
argp.add_argument('--batch', help="Batch size", default=16, type=int)
argp.add_argument('--symptom', help="Choose anx or dep. In multilabel case this argument does not affect the script", default="anx")
argp.add_argument('--load_model_path', help="Input path for finetuned model", default=None)
argp.add_argument('--output_path', help="Output path for saving pipeline output", default=None)
args = argp.parse_args()
## ======================

## SANITY CHECK
## ======================
if args.mode != "hyperparam" and args.mode != "evaluate":
    sys.exit("Wrong mode input!")
if args.mode == "hyperparam" and not args.finetune_lr:
    sys.exit("Need to supply lr for hyperparameter tuning!")
if args.mode == "hyperparam" and not args.finetune_epoch:
    sys.exit("Need to supply epoch for hyperparameter tuning!")
if args.mode == "evaluate" and (args.finetune_lr or args.finetune_epoch):
    sys.exit("Only supply optimal lr and epoch when evaluating!")
if (args.subdocument is not None) and (args.subdocument < 0 or args.subdocument > 0.9):
    sys.exit("Invalid overlap proportion input!")
## ======================

## GLOBAL VARIABLES
## =================
MODEL_ID = "roberta-base"
MAX_LEN = 512  # in case of subdocument slicing, this will be the max size of each subdocument
SLICING = True if args.subdocument is not None else False
NUM_WORKERS = 1
SYMPTOM_NAME = args.symptom
if SYMPTOM_NAME == "anx":
    SYMPTOM = 1  # 0: none / 1: anxiety / 2: depression
elif SYMPTOM_NAME == "dep":
    SYMPTOM = 2  # 0: none / 1: anxiety / 2: depression
device = 'cuda' if cuda.is_available() else 'cpu'
if args.output_path and not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
## =================


# load dataset
print("Loading data...")
train_index = np.loadtxt("training_example_indices.txt")
test_index = np.loadtxt("testing_example_indices.txt")
label_matrix = pd.read_csv("label_matrix_merge_with_none_new.txt", sep=" ", header=None)
df_anx = label_matrix[1]
df_dep = label_matrix[2]
with open('removed_meta_all.json', 'r') as file:
    data = json.load(file)

# initialize RoBERTa tokenizer, perform subdocument slicing if necessary
print("Initializing tokenizer and preprocessing data...")
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID, do_lower_case=True)

# extract client texts from all samples into a list of strings
text = []
for files in data:
    paragraphs = ' '.join(data[files]['Client_Text_Replaced_Two'])
    text.append(paragraphs)

# perform subdocument slicing if needed
if SLICING:
    print("Performing subdocument slicing...")
    doc_counts = {}
    # split docs into list of list of strings
    for idx, each_doc in enumerate(text):
        text[idx] = slice_document(each_doc, MAX_LEN, tokenizer, overlap=args.subdocument)
        doc_counts[idx] = len(text[idx])
    # list of document indices corresponding to each substring
    doc_indices = [i for i, sublist in enumerate(text) for _ in sublist]
    # enumerate labels for each substring
    df1_anx = pd.concat([df_anx.loc[[idx]].repeat(rep) for idx, rep in doc_counts.items()])
    df1_anx = df1_anx.reset_index(drop=True)
    df2_dep = pd.concat([df_dep.loc[[idx]].repeat(rep) for idx, rep in doc_counts.items()])
    df2_dep = df2_dep.reset_index(drop=True)
    # flatten the list of lists
    flat_list = [substring for sublist in text for substring in sublist]
    # create pd dataframe
    df_text = pd.DataFrame({'text': flat_list, 'doc_index': doc_indices})
    df_all = pd.concat([df_text, df1_anx, df2_dep], axis=1)
    df_all.columns = ["text", "doc_index", "anxiety_label", "depression_label"]
    # split into train and test dataframe
    df_train = df_all[df_all['doc_index'].isin(train_index)].reset_index(drop=True)
    df_test = df_all[df_all['doc_index'].isin(test_index)].reset_index(drop=True)
else:
    # get train and test dataframe
    df_text = pd.DataFrame(text)
    df_all = pd.concat([df_text, df_anx, df_dep], axis=1)
    df_all.columns = ["text", "anxiety_label", "depression_label"]
    df_train = (df_all.iloc[train_index]).reset_index(drop=True)
    df_test = (df_all.iloc[test_index]).reset_index(drop=True)

# prepare data for training
print(f"TRAIN Dataset: {df_train.shape}")
print(f"TEST Dataset: {df_test.shape}")
training_set = TranscriptData(df_train, tokenizer, MAX_LEN, SLICING)
testing_set = TranscriptData(df_test, tokenizer, MAX_LEN, SLICING)
config_params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': NUM_WORKERS}
training_loader = DataLoader(training_set, **config_params)
testing_loader = DataLoader(testing_set, **config_params)

# initiating summary writer for tensorboard
print("Initializing writer for tensorboard...")
writer = SummaryWriter()

# run model base on mode: ["hyperparam", "evaluate"]
print(f"Running model under mode {args.mode}...")
if args.mode == "hyperparam":
    lr_list = args.finetune_lr  # a list of lr to try
    max_epoch = args.finetune_epoch  # an integer of epoch cap
elif args.mode == "evaluate":
    lr_list = [args.optimal_lr]  # with chosen opt this is a list of one lr
    max_epoch = args.optimal_epoch

for lr in lr_list:
    print(f"\n=========== Finetuning model with lr={lr} ===========")
    print("Initializing model and loss function...")
    loss_function = torch.nn.BCEWithLogitsLoss()
    if args.task == "binary":
        model = RobertaBinaryClass(model_id=MODEL_ID)
        #train_function = train_binary
        #test_function = test_binary
        placeholder = args.symptom
    elif args.task == "multi":
        model = RobertaMultiLabel(model_id=MODEL_ID)
        train_function = train_multilabel
        test_function = test_multilabel
        placeholder = args.task
    else:
        sys.exit("Invalid command line argument to --task")
        
    # make sure to move model to correct device
    model.to(device)
    print('Model on device: ', next(model.parameters()).device)
    
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(training_loader), eta_min=1e-8)
    tr_steps_accu = 0
    tr_loss_list = []
    test_loss_list = []
     
    for epoch in range(max_epoch):
        tr_labels_epoch, tr_preds_epoch, tr_preds_prob_epoch, tr_doc_idx, tr_steps_epoch, tr_loss_epoch = train_function(model,
                                                                                      lr,
                                                                                      loss_function,
                                                                                      optimizer,
                                                                                      scheduler,
                                                                                      training_loader,
                                                                                      SLICING,
                                                                                      tr_steps_accu,
                                                                                      writer,
                                                                                      device,
                                                                                      epoch)
        tr_steps_accu += tr_steps_epoch
        tr_loss_list += [tr_loss_epoch]
        # calculate evaluation metrics for this train epoch
        b_metrics_all(tr_preds_epoch, tr_preds_prob_epoch, tr_labels_epoch, lr, epoch, placeholder)
        if SLICING:
            pooled_preds, pooled_probs, pooled_labels = result_pooling(tr_doc_idx, tr_preds_epoch, tr_preds_prob_epoch, tr_labels_epoch)
            add_placeholder = f"{placeholder}, doc-slicing with overlap {args.subdocument}"
            b_metrics_all(pooled_preds, pooled_probs, pooled_labels, lr, epoch, add_placeholder)

        
        print(f"=========== Testing model with lr={lr}, epoch={epoch} ===========")
        # test the model on test data
        test_labels_epoch, test_preds_epoch, test_preds_prob_epoch, test_doc_idx, test_loss_epoch = test_function(model,
                                                                                loss_function,
                                                                                SLICING,
                                                                                device,
                                                                                testing_loader)
        test_loss_list += [test_loss_epoch]
        # calculate evaluation metrics for this test epoch
        b_metrics_all(test_preds_epoch, test_preds_prob_epoch, test_labels_epoch, lr, epoch, placeholder)
        if SLICING:
            pooled_preds, pooled_probs, pooled_labels = result_pooling(test_doc_idx, test_preds_epoch, test_preds_prob_epoch, test_labels_epoch)
            add_placeholder = f"{placeholder}, doc-slicing with overlap {args.subdocument}"
            b_metrics_all(pooled_preds, pooled_probs, pooled_labels, lr, epoch, add_placeholder)

    
    # plot learning curve for this lr
    lr_curve(tr_loss_list, test_loss_list, lr, args.mode)
    
    # save the model for future use
    output_model_file = f"{args.output_path}/roberta_{placeholder}_{lr}_{max_epoch}.bin"
    output_vocab_file = "./"
    model_to_save = model
    torch.save(model_to_save, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)
    print("********* All files saved *********")

writer.close()
print("=========== End of Script ===========")