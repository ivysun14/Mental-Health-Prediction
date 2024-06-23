import csv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def load_longformer_loss(filename):
    train_loss = []
    eval_loss = []
    epoch_train = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if 'loss' in row and row['loss']:
                train_loss.append(float(row['loss']))
                epoch_train.append(float(row['epoch']))
            if 'eval_loss' in row and row['eval_loss']:
                eval_loss.append(float(row['eval_loss']))
    return train_loss, eval_loss, epoch_train

def load_longformer_metrics(filename):
    accuracy = []
    f1 = []
    auroc = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if 'eval_accuracy' in row and row['eval_accuracy']:
                accuracy.append(float(row['eval_accuracy']))
            if 'eval_f1_weighted' in row and row['eval_f1_weighted']:
                f1.append(float(row['eval_f1_weighted']))
            if 'eval_roc_auc_weighted' in row and row['eval_roc_auc_weighted']:
                auroc.append(float(row['eval_roc_auc_weighted']))
    return accuracy, f1, auroc

def load_loss(filename):
    df = pd.read_csv(filename)
    selected_columns = ['training_loss', 'validation_loss', 'Epoch']
    df_selected = df[selected_columns]
    train_loss = df_selected['training_loss'].tolist()
    eval_loss = df_selected['validation_loss'].tolist()
    epoch = df_selected['Epoch'].tolist()
    return train_loss, eval_loss, epoch

def load_metrics(filename):
    df = pd.read_csv(filename)
    selected_columns = ['validation_accuracy', 'validation_f1_weighted', 'validation_auroc_weighted', 'Epoch']
    df_selected = df[selected_columns]
    accuracy = df_selected['validation_accuracy'].tolist()
    f1 = df_selected['validation_f1_weighted'].tolist()
    auroc = df_selected['validation_auroc_weighted'].tolist()
    epoch = df_selected['Epoch'].tolist()
    return accuracy, f1, auroc, epoch


# compare bert models
'''
bert_train_loss, bert_eval_loss, epoch = load_loss('./fine_tune_results/bert_truncation.csv')
pooling_train_loss, pooling_eval_loss, _ = load_loss('./fine_tune_results/bert_pooling.csv')
random_train_loss, random_eval_loss, _ = load_loss('./fine_tune_results/bert_random.csv')
pooling_true1_train_loss, pooling_true1_eval_loss, _ = load_loss('./fine_tune_results/bert_pooling_true1.csv')
plt.figure(figsize=(10, 6))
cmap = matplotlib.colormaps['Paired']
epoch_list = [epoch, epoch, epoch, epoch, epoch, epoch, epoch, epoch]
loss_list = [bert_train_loss, bert_eval_loss, pooling_train_loss, pooling_eval_loss, random_train_loss, random_eval_loss, pooling_true1_train_loss, pooling_true1_eval_loss]
label_list = ['Truncation Train Loss', 'Truncation Eval Loss', 'Pooling Train Loss', 'Pooling Eval Loss', 'Random Train Loss', 'Random Eval Loss', 'Pooling 1 True Train Loss', 'Pooling 1 True Eval Loss']
for i in range(len(epoch_list)):
    plt.plot(epoch_list[i][:8], loss_list[i][:8], label=label_list[i], color=cmap.colors[i])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss for Fine-Tuned BERT Models')
plt.legend()
plt.grid(True)
plt.savefig('./fine_tune_results/bert_train_loss.png')
'''

bert_accuracy, bert_f1, bert_auroc, epoch = load_metrics('./fine_tune_results/bert_truncation.csv')
pooling_accuracy, pooling_f1, pooling_auroc, _ = load_metrics('./fine_tune_results/bert_pooling.csv')
random_accuracy, random_f1, random_auroc, _ = load_metrics('./fine_tune_results/bert_random.csv')
pooling_true1_accuracy, pooling_true1_f1, pooling_true1_auroc, _ = load_metrics('./fine_tune_results/bert_pooling_true1.csv')
plt.figure(figsize=(15, 9))
cmap = matplotlib.colormaps['tab20c']
accuracy_list = [bert_accuracy, pooling_accuracy, random_accuracy, pooling_true1_accuracy]
f1_list = [bert_f1, pooling_f1, random_f1, pooling_true1_f1]
auroc_list = [bert_auroc, pooling_auroc, random_auroc, pooling_true1_auroc]
label_list = ['Truncation', 'Pooling', 'Random', 'Pooling 1 True']
for i in range(4):
    plt.plot(epoch[:8], accuracy_list[i][:8], label=label_list[i] + ' Accuracy', color=cmap.colors[i * 4 + 2], marker='o', linestyle='dotted')
    plt.plot(epoch[:8], f1_list[i][:8], label=label_list[i] + ' F1 score', color=cmap.colors[i * 4 + 1], marker='v', linestyle='dashed')
    plt.plot(epoch[:8], auroc_list[i][:8], label=label_list[i] + ' AUROC score', color=cmap.colors[i * 4], marker='h', linestyle='solid')
plt.ylim(0.0, 0.9)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Metric', fontsize=14)
plt.title('Performance for Fine-Tuned BERT Models', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.savefig('./fine_tune_results/bert_metrics.png')
plt.clf()

# compare three truncation models
'''
bert_train_loss, bert_eval_loss, epoch = load_loss('./fine_tune_results/bert_truncation.csv')
roberta_train_loss, roberta_eval_loss, _ = load_loss('./fine_tune_results/roberta_truncation.csv')
longformer_train_loss, longformer_eval_loss, longformer_train_epoch = load_longformer_loss('./fine_tune_results/longformer_truncation.csv')
plt.figure(figsize=(10, 6))
cmap = matplotlib.colormaps['Paired']
epoch_list = [epoch, epoch, epoch, epoch, longformer_train_epoch, epoch]
loss_list = [bert_train_loss, bert_eval_loss, roberta_train_loss, roberta_eval_loss, longformer_train_loss, longformer_eval_loss]
label_list = ['BERT Train Loss', 'BERT Eval Loss', 'RoBERTa Train Loss', 'RoBERTa Eval Loss', 'Longformer Train Loss', 'Longformer Eval Loss']
for i in range(len(epoch_list)):
    if label_list[i] == 'Longformer Train Loss':
        plt.plot(epoch_list[i][:18], loss_list[i][:18], label=label_list[i], color=cmap.colors[i])
    else:
        plt.plot(epoch_list[i][:10], loss_list[i][:10], label=label_list[i], color=cmap.colors[i])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss for Fine-Tuned Truncation Models')
plt.legend()
plt.grid(True)
plt.savefig('./fine_tune_results/truncation_train_test_loss.png')
'''

bert_accuracy, bert_f1, bert_auroc, epoch = load_metrics('./fine_tune_results/bert_truncation.csv')
roberta_accuracy, roberta_f1, roberta_auroc, _ = load_metrics('./fine_tune_results/roberta_truncation.csv')
longformer_accuracy, longformer_f1, longformer_auroc = load_longformer_metrics('./fine_tune_results/longformer_truncation.csv')
plt.figure(figsize=(15, 9))
cmap = matplotlib.colormaps['tab20c']
accuracy_list = [bert_accuracy, roberta_accuracy, longformer_accuracy]
f1_list = [bert_f1, roberta_f1, longformer_f1]
auroc_list = [bert_auroc, roberta_auroc, longformer_auroc]
label_list = ['BERT', 'RoBERTa', 'Longformer']
for i in range(3):
    plt.plot(epoch, accuracy_list[i], label=label_list[i] + ' Accuracy', color=cmap.colors[i * 4 + 2], marker='o', linestyle='dotted')
    plt.plot(epoch, f1_list[i], label=label_list[i] + ' F1 score', color=cmap.colors[i * 4 + 1], marker='v', linestyle='dashed')
    plt.plot(epoch, auroc_list[i], label=label_list[i] + ' AUROC score', color=cmap.colors[i * 4], marker='h', linestyle='solid')

plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Metric', fontsize=18)
plt.title('Performance for Fine-Tuned Truncation Models', fontsize=22)
plt.ylim(0.0, 0.9)
# plt.legend(fontsize=15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.savefig('./fine_tune_results/truncation.png')
plt.clf()

# compare pooling models
bert_accuracy, bert_f1, bert_auroc, epoch = load_metrics('./fine_tune_results/bert_pooling.csv')
roberta_accuracy, roberta_f1, roberta_auroc, _ = load_metrics('./fine_tune_results/roberta_pooling.csv')
longformer_accuracy, longformer_f1, longformer_auroc = load_longformer_metrics('./fine_tune_results/longformer_pooling.csv')
plt.figure(figsize=(15, 9))
cmap = matplotlib.colormaps['tab20c']
accuracy_list = [bert_accuracy, roberta_accuracy, longformer_accuracy]
f1_list = [bert_f1, roberta_f1, longformer_f1]
auroc_list = [bert_auroc, roberta_auroc, longformer_auroc]
label_list = ['BERT', 'RoBERTa', 'Longformer']
for i in range(3):
    if label_list[i] == 'Longformer':
        plt.plot(epoch[:10], accuracy_list[i], label=label_list[i] + ' Accuracy', color=cmap.colors[i * 4 + 2], marker='o',
                 linestyle='dotted')
        plt.plot(epoch[:10], f1_list[i], label=label_list[i] + ' F1 score', color=cmap.colors[i * 4 + 1], marker='v',
                 linestyle='dashed')
        plt.plot(epoch[:10], auroc_list[i], label=label_list[i] + ' AUROC score', color=cmap.colors[i * 4], marker='h',
                 linestyle='solid')
    else:
        plt.plot(epoch[:10], accuracy_list[i][:10], label=label_list[i] + ' Accuracy', color=cmap.colors[i * 4 + 2], marker='o', linestyle='dotted')
        plt.plot(epoch[:10], f1_list[i][:10], label=label_list[i] + ' F1 score', color=cmap.colors[i * 4 + 1], marker='v', linestyle='dashed')
        plt.plot(epoch[:10], auroc_list[i][:10], label=label_list[i] + ' AUROC score', color=cmap.colors[i * 4], marker='h', linestyle='solid')

plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Metric', fontsize=18)
plt.title('Performance for Fine-Tuned Boosted Models', fontsize=22)
plt.ylim(0.0, 0.9)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.savefig('./fine_tune_results/pooling.png')
plt.clf()