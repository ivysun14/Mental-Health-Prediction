"""
File: train.py
Created by Junwei (Ivy) Sun

This file contains python functions of training an RoBERTa model on
binary or multi-label classification tasks, calculating the gradients,
and performing the backward propogation process to fine-tune the model.
"""

import torch
from tqdm import tqdm

'''
def train_binary(model, lr, loss_function, symptom_idx, optimizer, scheduler, training_loader, cummu_tr_steps, writer, device, epoch=1):
    ''''''
    Fine-tuning the model on the training data for 1 epoch and the specified
    symptom. If symptom_idx is 1, a binary classification on anxiety is used
    as the finetuning task. If symptom_idx is 2, a binary classification on
    depression is used as the finetuning task.

    @returns:
        all_labels: 1D-tensor of true labels of all samples processed in order
        all_predictions: 1D-tensor of predicted labels of all samples processed in order
    ''''''
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()  # Set the model to training mode

    all_labels = (torch.empty(0)).to(device)
    all_preds = (torch.empty(0)).to(device)
    all_preds_prob = (torch.empty(0)).to(device)
    
    # iterave over batches of data
    # step: index of the current batch
    for step, data in tqdm(enumerate(training_loader, 0), total=len(training_loader)):
        ids = data['ids'].squeeze(dim=1).to(device, dtype = torch.long) # torch.Size([64, 512])
        mask = data['mask'].squeeze(dim=1).to(device, dtype = torch.long)  # torch.Size([64, 512])
        token_type_ids = data['token_type_ids'].squeeze(dim=1).to(device, dtype = torch.long)
        if symptom_idx == 1:  # torch.Size([64])
            labels = data['anx_label'].to(device, dtype = torch.float)
        elif symptom_idx == 2:
            labels = data['dep_label'].to(device, dtype = torch.float)

        outputs = model.forward(ids, mask, token_type_ids)  # torch.Size([64, 1])
        loss = loss_function(outputs, labels.unsqueeze(dim=1))  # add unsqueeze to match shape for BCEWithLogitsLoss
        tr_loss += loss.item()
        preds_prob = torch.sigmoid(outputs)
        
        # convert logits to predictions using a threshold (e.g., 0.5)
        predictions = (preds_prob > 0.5).float()
        predictions = predictions.squeeze(dim=1)  # torch.Size([64])
        n_correct += ((predictions==labels).sum().item())
        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        # save labels and predictions
        all_labels = (torch.cat((all_labels, labels), dim=0)).to(device)
        all_preds = (torch.cat((all_preds, predictions), dim=0)).to(device)
        all_preds_prob = (torch.cat((all_preds_prob, preds_prob), dim=0)).to(device)

        # print out progress in log every 10 steps
        if (step % 10) == 0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = n_correct/nb_tr_examples 
            writer.add_scalar(f"loss/train_{lr}", loss_step, step+cummu_tr_steps)
            writer.add_scalar(f"accuracy/train_{lr}", accu_step, step+cummu_tr_steps)
            
        optimizer.zero_grad()  # zero out previous gradient calculation
        loss.backward()  # backpropogation to coompute gradient
        optimizer.step()  # update model parameters
        scheduler.step()  # update scheduler

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = n_correct/nb_tr_examples
    print(f"Training Loss Epoch {epoch}: {epoch_loss}")
    print(f"Training Accuracy Epoch {epoch}: {epoch_accu}")

    return all_labels, all_preds, all_preds_prob, nb_tr_steps, epoch_loss
'''

def train_multilabel(model, lr, loss_function, optimizer, scheduler, training_loader, slice_cond, cummu_tr_steps, writer, device, epoch=1):
    '''
    Fine-tuning the model on multilabel classification using training data for 1 epoch.
    This function supports subdocument slicing and result pooling.

    @returns:
        all_labels: 2D-tensor of true labels of all samples processed in order
        all_preds: 2D-tensor of predicted labels of all samples processed in order
        all_preds_prob: 2D-tensor of predicted logits of all samples processed in order
        all_doc_idx: 1D-tensor of each subdocument's corresponding original doc index
        nb_tr_steps: total number of batches processed
        epoch_loss: average batch loss of this epoch
    '''
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    all_labels = (torch.empty(0,2)).to(device)
    all_preds = (torch.empty(0,2)).to(device)
    all_preds_prob = (torch.empty(0,2)).to(device)
    # keep record of all corresponding orig doc index if slicing and pooling
    if slice_cond:
        all_doc_idx = (torch.empty(0)).to(device)
    else:
        all_doc_idx = None
    
    for step, data in tqdm(enumerate(training_loader, 0), total=len(training_loader)):
        ids = data['ids'].squeeze(dim=1).to(device, dtype = torch.long)
        mask = data['mask'].squeeze(dim=1).to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].squeeze(dim=1).to(device, dtype = torch.long)

        labels_anx = data['anx_label'].to(device, dtype = torch.float) # torch.Size([64])
        labels_dep = data['dep_label'].to(device, dtype = torch.float) # torch.Size([64])
        combined_labels = torch.stack((labels_anx, labels_dep), dim=1) # torch.Size([64, 2])
        outputs = model.forward(ids, mask, token_type_ids)  # torch.Size([64, 2])
        loss = loss_function(outputs, combined_labels)
        tr_loss += loss.item()
        preds_prob = torch.sigmoid(outputs)

        # convert logits to predictions using a threshold (e.g., 0.5)
        predictions = (preds_prob > 0.5).float()  # torch.Size([64, 2])
        n_correct += ((predictions==combined_labels).sum().item())
        nb_tr_steps += 1
        nb_tr_examples += (2*combined_labels.size(0))

        # save labels and predictions
        all_labels = (torch.cat((all_labels, combined_labels), dim=0)).to(device)
        all_preds = (torch.cat((all_preds, predictions), dim=0)).to(device)
        all_preds_prob = (torch.cat((all_preds_prob, preds_prob), dim=0)).to(device)
        if slice_cond:
            all_doc_idx = (torch.cat((all_doc_idx, data['doc_idx'].to(device)), dim=0)).to(device)

        # print out progress in log every 10 steps
        if (step % 10) == 0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = n_correct/nb_tr_examples
            writer.add_scalar(f"loss/train_{lr}", loss_step, step+cummu_tr_steps)
            writer.add_scalar(f"accuracy/train_{lr}", accu_step, step+cummu_tr_steps)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = n_correct/nb_tr_examples
    print(f"Training Loss Epoch {epoch}: {epoch_loss}")
    print(f"Training Accuracy Epoch {epoch}: {epoch_accu}")

    return all_labels, all_preds, all_preds_prob, all_doc_idx, nb_tr_steps, epoch_loss

'''
def test_binary(model, loss_function, symptom_idx, device, testing_loader):
    ''''''
    Test the model on the testing data on the specified symptom.
    If symptom_idx is 1, a binary classification on anxiety is performed.
    If symptom_idx is 2, a binary classification on depression is performed.

    @returns:
        all_labels: 1D-tensor of true labels of all samples processed in order
        all_predictions: 1D-tensor of predicted labels of all samples processed in order
    ''''''
    
    model.eval()  # set the model to evaluation mode
                  # disable operations like dropout and batch normalization
    n_correct = 0
    eval_loss = 0
    nb_eval_steps = 0
    nb_eval_examples = 0

    all_labels = (torch.empty(0)).to(device)
    all_preds = (torch.empty(0)).to(device)
    all_preds_prob = (torch.empty(0)).to(device)

    with torch.no_grad():  # disable gradient calculation
        for _, data in tqdm(enumerate(testing_loader, 0), total=len(testing_loader)):
            ids = data['ids'].squeeze(dim=1).to(device, dtype = torch.long)
            mask = data['mask'].squeeze(dim=1).to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].squeeze(dim=1).to(device, dtype=torch.long)
            if symptom_idx == 1:  # torch.Size([64])
                labels = data['anx_label'].to(device, dtype = torch.float)
            elif symptom_idx == 2:
                labels = data['dep_label'].to(device, dtype = torch.float)

            outputs = model.forward(ids, mask, token_type_ids)
            loss = loss_function(outputs, labels.unsqueeze(dim=1))
            eval_loss += loss.item()
            preds_prob = torch.sigmoid(outputs)

            # convert logits to predictions
            predictions = (preds_prob > 0.5).float()
            predictions = predictions.squeeze(dim=1)  # torch.Size([64])
            n_correct += ((predictions==labels).sum().item())
            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            # save labels and predictions
            all_labels = (torch.cat((all_labels, labels), dim=0)).to(device)
            all_preds = (torch.cat((all_preds, predictions), dim=0)).to(device)
            all_preds_prob = (torch.cat((all_preds_prob, preds_prob), dim=0)).to(device)
    
    total_loss = eval_loss/nb_eval_steps
    total_accu = n_correct/nb_eval_examples
    print(f"Validation Loss: {total_loss}")
    print(f"Validation Accuracy: {total_accu}")
    
    return all_labels, all_preds, all_preds_prob, total_loss
'''

def test_multilabel(model, loss_function, slice_cond, device, testing_loader):
    '''
    Test the model on the testing data on the specified symptom.

    @returns:
        all_labels: 2D-tensor of true labels of all samples processed in order
        all_predictions: 2D-tensor of predicted labels of all samples processed in order
    '''
    
    model.eval()
    n_correct = 0
    eval_loss = 0
    nb_eval_steps = 0
    nb_eval_examples = 0

    all_labels = (torch.empty(0, 2)).to(device)
    all_preds = (torch.empty(0, 2)).to(device)
    all_preds_prob = (torch.empty(0,2)).to(device)
    if slice_cond:
        all_doc_idx = (torch.empty(0)).to(device)
    else:
        all_doc_idx = None

    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0), total=len(testing_loader)):
            ids = data['ids'].squeeze(dim=1).to(device, dtype = torch.long)
            mask = data['mask'].squeeze(dim=1).to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].squeeze(dim=1).to(device, dtype=torch.long)
            
            labels_anx = data['anx_label'].to(device, dtype = torch.float) # torch.Size([64])
            labels_dep = data['dep_label'].to(device, dtype = torch.float) # torch.Size([64])
            combined_labels = torch.stack((labels_anx, labels_dep), dim=1) # torch.Size([64, 2])
            outputs = model.forward(ids, mask, token_type_ids)  # torch.Size([64, 2])
            loss = loss_function(outputs, combined_labels)
            eval_loss += loss.item()
            preds_prob = torch.sigmoid(outputs)

            # convert logits to predictions
            predictions = (preds_prob > 0.5).float()
            n_correct += ((predictions==combined_labels).sum().item())
            nb_eval_steps += 1
            nb_eval_examples += (2*combined_labels.size(0))

            # save labels and predictions
            all_labels = (torch.cat((all_labels, combined_labels), dim=0)).to(device)
            all_preds = (torch.cat((all_preds, predictions), dim=0)).to(device)
            all_preds_prob = (torch.cat((all_preds_prob, preds_prob), dim=0)).to(device)
            if slice_cond:
                all_doc_idx = (torch.cat((all_doc_idx, data['doc_idx'].to(device)), dim=0)).to(device)

    total_loss = eval_loss/nb_eval_steps
    total_accu = n_correct/nb_eval_examples
    print(f"Validation Loss: {total_loss}")
    print(f"Validation Accuracy: {total_accu}")
    
    return all_labels, all_preds, all_preds_prob, all_doc_idx, total_loss