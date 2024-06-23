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
from tqdm import tqdm
from scipy.stats import mode

# Only required for tokenizing chunks
# import spacy
# nlp = spacy.load("en_core_web_sm")


def load_data_from_json(json_file_path):
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

    df_text = pd.DataFrame(texts)
    return df_text


def sentence_tokenizer(input_text, nlp):
    """Separate for sentences to keep entire sentence in the chunks"""
    doc = nlp(input_text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

# sub-document slicing source: fine_tune_bert.py
def divide_doc(input_text, tokenizer, nlp):
    """Divide document into smaller chunks each with at most MAX_LEN tokens"""
    sentences = sentence_tokenizer(input_text, nlp)
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    print('tokenized_sentences length: ' + str(len(tokenized_sentences)))
    chunks = []
    max_chunk_length = 4094

    current_chunk = []
    current_length = 0

    for sentence in tokenized_sentences:
        if current_length + len(sentence) > max_chunk_length:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.extend(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def preprocessing_chuncks(input_text, tokenizer, nlp):
    """Return a list of processed chunks"""
    chunks = divide_doc(input_text, tokenizer, nlp)

    # Process each chunk separately
    encoding_chunks = [tokenizer.encode_plus(
        " ".join(chunk),  # Convert chunk back to string
        add_special_tokens=True,
        max_length=4096,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    ) for chunk in chunks]
    return encoding_chunks


def prepare_chunked_data(df_text, labels, tokenizer, nlp):
    """Save tokenized and chuncked data"""
    input_ids = []
    attention_masks = []
    document_ids = []
    chunk_labels = []

    for doc_id, (sample, label) in enumerate(tqdm(zip(df_text[0], labels), total=len(labels))):
        print(doc_id)
        encoding_chunks = preprocessing_chuncks(sample, tokenizer, nlp)
        print('number of chunks: ' + str(len(encoding_chunks)))
        for chunk_encoding in encoding_chunks:
            print('process chunk')
            input_ids.append(chunk_encoding['input_ids'])
            attention_masks.append(chunk_encoding['attention_mask'])
            chunk_labels.append(label)
            document_ids.append(doc_id)

    chunk_labels_array = np.array(chunk_labels)
    document_ids_array = np.array(document_ids)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    chunk_labels_array = torch.tensor(chunk_labels_array, dtype=torch.float)
    document_ids_array = torch.tensor(document_ids_array, dtype=torch.float)

    save_dict = {
        'input_ids': input_ids,
        'attention_masks': attention_masks,
        'labels': chunk_labels_array,
        'document_ids': document_ids_array
    }

    # Saving to disk
    torch.save(save_dict, 'tokenized_sub_document.pth')
    return save_dict


def tokenize_data(tokenizer, nlp):
    label_file_path = 'label_matrix_merge_new.txt'
    labels = np.loadtxt(label_file_path)
    json_file_path = 'removed_meta2_reduced.json'
    df_text = load_data_from_json(json_file_path)
    train_dict = prepare_chunked_data(df_text, labels, tokenizer, nlp)
    return train_dict


class MyDataset(Dataset):
    def __init__(self, batch_encoding):
        self.batch_encoding = batch_encoding

    def __len__(self):
        return len(self.batch_encoding['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.batch_encoding['input_ids'][idx]),
            'attention_masks': torch.tensor(self.batch_encoding['attention_masks'][idx]),
            'labels': torch.tensor(self.batch_encoding['labels'][idx])
        }


def majority_vote(predictions, document_ids, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    grouped_predictions = {}
    document_labels = {}
    document_ids = document_ids.numpy()

    for prediction, doc_id, label in zip(y_pred, document_ids, labels):
        if doc_id not in grouped_predictions:
            grouped_predictions[doc_id] = [prediction]
        else:
            grouped_predictions[doc_id].append(prediction)
        document_labels[doc_id] = label

    # Perform a majority vote within each group
    final_predictions = []
    final_labels = []
    for doc_id, preds in grouped_predictions.items():
        most_common_pred = mode(preds)[0]
        final_predictions.append(most_common_pred)
        final_labels.append(document_labels[doc_id])
    return np.array(final_predictions), np.array(final_labels)


# Implement the compute_metrics function for evaluation
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(y_pred, labels, threshold=0.5):
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


def compute_polled_metrics(doc_ids):
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        majority_vote_preds, majority_vote_labels = majority_vote(preds, doc_ids, p.label_ids)
        result = multi_label_metrics(
            y_pred=majority_vote_preds,
            labels=majority_vote_labels)
        print(result)
        return result

    return compute_metrics


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


tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

# Prepare for training by tokenizing input sequence into chunks. Only need to run one time.
# tokenize_data(tokenizer, nlp)

# Load tokenized data and split into train/eval set
input_data = torch.load('tokenized_sub_document.pth')
train_indices = np.loadtxt('training_example_indices.txt')
test_indices = np.loadtxt('testing_example_indices.txt')
document_ids = input_data['document_ids']
train_mask = np.isin(document_ids, train_indices)
train_indices_subdocuments = np.where(train_mask)
train_indices = np.array(train_indices_subdocuments).T
test_mask = np.isin(document_ids, test_indices)
test_indices_subdocuments = np.where(test_mask)
test_indices = np.array(test_indices_subdocuments).T
train_inputs = {key: value[train_indices] for key, value in input_data.items()}
eval_inputs = {key: value[test_indices] for key, value in input_data.items()}
for key in train_inputs:
    train_inputs[key] = train_inputs[key].squeeze()
for key in eval_inputs:
    eval_inputs[key] = eval_inputs[key].squeeze()
train_inputs.pop('document_ids')
train_data = MyDataset(train_inputs)
doc_ids = eval_inputs.pop('document_ids')
eval_data = MyDataset(eval_inputs)


# Number of labels: anxiety, depression
num_labels = 2

# Load pretrained longformer
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                            num_labels=num_labels,
                                                            problem_type = 'multi_label_classification')
model.to(device)

# Set up the training arguments for the Trainer
arguments = TrainingArguments(
    output_dir="./longformer_results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    fp16=True,
    per_device_eval_batch_size=1,
    num_train_epochs=8,
    learning_rate=1e-5,
    lr_scheduler_type = "cosine",
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    eval_accumulation_steps=1,
    save_total_limit=3
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1016)

trainer = Trainer(
    model=model,
    args=arguments,
    optimizers=(optimizer, lr_scheduler),
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data,
    compute_metrics=compute_polled_metrics(doc_ids)
)

trainer.train()

# Save the model and log
model.save_pretrained('./finetuned_longformer_pooling_ver2')
tokenizer.save_pretrained('./finetuned_longformer_tokenizer_pooling_ver2')
log_history = trainer.state.log_history
df = pd.DataFrame(log_history)
df.to_csv('log_history_pooling_ver2.csv', index=False)

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
plt.title('Training and Evaluation Loss for Longformer with Chunking and Pooling')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('loss_curve_pooling_ver2.png')