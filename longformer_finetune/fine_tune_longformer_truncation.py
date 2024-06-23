import numpy as np
import pandas as pd
import json
from pynvml import *
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EvalPrediction
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


def tokenize_texts_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    texts = []
    for entry_id, entry_data in data.items():
        if 'Client_Text_Replaced_Two' in entry_data and isinstance(entry_data['Client_Text_Replaced_Two'], list):
            # Concatenate strings within 'Client_Text' entry
            concatenated_text = " ".join(entry_data['Client_Text_Replaced_Two'])
            texts.append(concatenated_text)
        else:
            print(f"Entry {entry_id} does not contain valid 'Client_Text' data.")

    # Tokenize texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return inputs

class MyDataset(Dataset):
    def __init__(self, batch_encoding):
        self.batch_encoding = batch_encoding

    def __len__(self):
        return len(self.batch_encoding['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.batch_encoding['input_ids'][idx]),
            'attention_mask': torch.tensor(self.batch_encoding['attention_mask'][idx]),
            'labels': torch.tensor(self.batch_encoding['labels'][idx])
        }

# Implement the compute_metrics function for evaluation
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1_samples_average = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
    roc_auc_micro_average = roc_auc_score(y_true, y_pred, average='micro')
    roc_auc_macro_average = roc_auc_score(y_true, y_pred, average='macro')
    roc_auc_weighted_average = roc_auc_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'f1_weighted': f1_weighted_average,
               'f1_samples': f1_samples_average,
               'roc_auc_micro': roc_auc_micro_average,
               'roc_auc_macro': roc_auc_macro_average,
               'roc_auc_weighted': roc_auc_weighted_average,
               'accuracy': accuracy,
               }
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Print the CUDA device name
    print(torch.cuda.get_device_name(0))
    # Tell PyTorch to use the GPU
    device = torch.device("cuda")
else:
    print("CUDA is not available. Switching to CPU.")
    device = torch.device("cpu")


# Number of labels: anxiety, depression
num_labels = 2

# Tokenize texts
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
json_file_path = 'removed_meta2_reduced.json'
input_data = tokenize_texts_from_json(json_file_path)

# Load labels
label_file_path = 'label_matrix_merge_new.txt'
labels = np.loadtxt(label_file_path)
label_tensor = torch.tensor(labels)
input_data['labels'] = label_tensor

# Split the training set into train, eval, and test sets
train_indices = np.loadtxt('training_example_indices.txt')
test_indices = np.loadtxt('testing_example_indices.txt')
train_inputs = {key: value[train_indices] for key, value in input_data.items()}
eval_inputs = {key: value[test_indices] for key, value in input_data.items()}
train_inputs = tokenizer.pad(train_inputs, return_tensors="pt")
eval_inputs = tokenizer.pad(eval_inputs, return_tensors="pt")
train_data = MyDataset(train_inputs)
eval_data = MyDataset(eval_inputs)

# Load pretrained longformer
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                            num_labels=num_labels,
                                                            problem_type = 'multi_label_classification')
model.to(device)

# Set up the training arguments for the Trainer
arguments = TrainingArguments(
    output_dir="./longformer_results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    fp16=True,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    learning_rate=1e-5,
    lr_scheduler_type = "cosine",
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    eval_accumulation_steps=1,
    save_total_limit=3,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=80, num_training_steps=875)

trainer = Trainer(
    model=model,
    args=arguments,
    optimizers=(optimizer, lr_scheduler),
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data,
    compute_metrics=compute_metrics
)

trainer.train()

# Save the model and log
model.save_pretrained('./finetuned_longformer_ver10')
tokenizer.save_pretrained('./finetuned_longformer_tokenizer_ver10')
log_history = trainer.state.log_history
df = pd.DataFrame(log_history)
df.to_csv('log_history_ver10.csv', index=False)

# Plotting
train_loss = []
eval_loss = []
epoch_train = []
epoch_eval = []
for entry in log_history:
    if 'loss' in entry:
        train_loss.append(entry['loss'])
        epoch_train.append(entry['epoch'])
    if 'eval_loss' in entry:
        eval_loss.append(entry['eval_loss'])
        epoch_eval.append(entry['epoch'])
plt.figure(figsize=(10, 6))
plt.plot(epoch_train, train_loss, label='Training Loss', marker='o')
plt.plot(epoch_eval, eval_loss, label='Evaluation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss for Longformer')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('loss_curve_ver10.png')