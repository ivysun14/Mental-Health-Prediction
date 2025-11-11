'''
File: GPT_Prompting

This file contains the script that connects to OpenAI's API and
prompt the model for anxiety + depression multi-label classification.
Currently two experiments are being conducted:
    1). Randomly sample 200 transcripts for one-time accuracy test
    2). Select a transcript and prompt for classification 200 times for stability
'''

import os
import sys
import json
import time
from openai import OpenAI
from openai import RateLimitError
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken
from sklearn.metrics import accuracy_score


## GLOBAL VARIABLES
## ================
MODEL = "gpt-4o-2024-05-13"           # gpt-3.5-turbo-0125, gpt-4-turbo(gpt-4-turbo-2024-04-09), gpt-4o(gpt-4o-2024-05-13)
INPUT_TOKEN_LIMIT = 128000              # 16385, 128000, 128000
OUTPUT_TOKEN_LIMIT = 4096
TEMPERATURE = 0.5                      # range 0-2, the higher the value the more random the response, default 1
SYMPTOM = ["anxiety", "depression"]    # symptoms wish to classify
FUNC_TOKEN = 0                         # total tokens taken up by functions
MAX_BACKOFF = 60                       # max wait time between API calls

# API SETUP
client = OpenAI(api_key="Input API key")
# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model(MODEL)
#enc = tiktoken.get_encoding("o200k_base")  # manually set tokenizer for gpt-4o
# define a function that instructs the LLM's response format
function = {
   "name": "multilabel",
   "description": "Perform multilabel classification for anxiety and depression for a given text",
   "parameters": {
       "type": "object",
       "properties": {
           "prediction": {
               "type": "array",
               "items": {
                   "type": "string",
                   "enum": [
                       "Positive",
                       "Negative"
                   ]
               },
               "description": "The predicted mental state label."
           }
       },
       "required": [
           "prediction"
       ]
   }
}

FUNC_TOKEN += len(enc.encode(json.dumps(function)))
print(f"The function is encoded for {FUNC_TOKEN} tokens.")


## INPUT FILES
## ===========
f = open('remove_cheating_words/removed_meta_all.json')                                 #  processed therapist and client texts
data = json.load(f)
f.close()
label_matrix = np.loadtxt("remove_cheating_words/label_matrix_merge_with_none_new.txt") # label matrix


## FUNCTIONS
## =========
def random_sample(label_mat, symptom = "anxiety", n_pos = 100, n_neg = 100):
    '''
    Randomly sample `n_pos` number of samples that has the symptom of interest
    and `n_neg` number of samples that do not have the symptom of interest. Sampling
    is done using the label matrix.

    Return:
        @rand_sample_plus: index of samples with symptom
        @rand_sample_minus: index of samples without symptom
        @concat_index: rand_sample_plus + rand_sample_minus
    '''
    
    # extract labels for the chosen symptom for use of random sampling below
    if symptom == "anxiety":
        index = 1  # symptom index (0: none / 1: anxiety / 2: depression)
    elif symptom == "depression":
        index = 2
    label_vector = label_mat[:,index].reshape(label_mat.shape[0])
    
    # randomly sample n_pos positive and n_neg negative samples
    sample_w_symptom = (np.where(label_vector == 1))[0]  # index of samples with + label
    print(f"Total number of {symptom} positive samples = {sample_w_symptom.shape}")
    sample_wo_symptom = (np.where(label_vector == 0))[0]  # index of samples with - label
    print(f"Total number of {symptom} positive samples = {sample_wo_symptom.shape}")
    rand_sample_plus = np.random.choice(sample_w_symptom, n_pos)  # choose samples with symptom
    print(f"The {n_pos} random samples drawed with {symptom} have index: {rand_sample_plus}")
    rand_sample_minus = np.random.choice(sample_wo_symptom, n_neg)  # choose samples without symptom
    print(f"The {n_neg} random samples drawed without {symptom} have index: {rand_sample_minus}")

    concat_index = np.concatenate((rand_sample_plus, rand_sample_minus))
    print(f"{concat_index.shape}\n")

    # save index of positive and negative samples chosen for future reference
    np.savetxt(f"pos_neg_examples/Pos_{symptom}_idx.txt", rand_sample_plus)
    np.savetxt(f"pos_neg_examples/Neg_{symptom}_idx.txt", rand_sample_minus)
    np.savetxt("pos_neg_examples/Concat_idx.txt", concat_index)
    
    return [rand_sample_plus, rand_sample_minus, concat_index]


### CLASSIFICATION ACCURACY
def prompt_GPT_accuracy(input_json, label_mat, all_idx, symptom = "anxiety", n_pos = 100, n_neg = 100):
    '''
    Prompt the GPT model to perform binary classificaton on the symptom of interest for a total
    of `n_pos + n_neg` number of examples usng OpenAI's API.

    Return:
        @true_labels: true labels of all examples queried
        @pred_labels: GPT-predicted labels of all examples queried
    '''
    
    print("=============================================")
    print(f"Querying GPT on {symptom} for n={n_pos + n_neg}......")
    print("=============================================")

    count = 0
    idx_count = 0
    correct_predict = 0
    true_labels = np.zeros((n_pos + n_neg, 1))  # save true labels
    pred_labels = np.zeros((n_pos + n_neg, 1))  # save GPT predicted labels

    if symptom == "anxiety":
        index = 1  # symptom index (0: none / 1: anxiety / 2: depression)
    elif symptom == "depression":
        index = 2
    label_vector = label_mat[:,index].reshape(label_mat.shape[0])

    SYSTEM = "You are an experienced psychiatrist who will diagnose mental health conditions from counseling transcripts."
    CLASS_DESCRIPTION = f"Classes: [Positive, Negative]\n`Positive` indicates the client shows the symptom of {symptom}. `Negative` indicates the client does not show the symptom of {symptom}."
    PROMPT = f"Classify the client text into one of the classes.\n{CLASS_DESCRIPTION}"

    for session in input_json:
        # obly query GPT when the sample is our chosen sample
        if count in all_idx:
            if label_vector[count] == 1:
                true_label = "Positive"
                true_labels[idx_count, 0] = 1   # record true label
            else:
                true_label = "Negative"

            print(f"Querying sample with index {count} and class {true_label}...")
        
            # process the client text into api-recognizable format
            client_text = input_json[session]['Client_Text_Replaced_Two']
            concat_text = f"{PROMPT}\nText:"
            token_size = len(enc.encode(SYSTEM)) + len(enc.encode(concat_text))
            for line in client_text:  # client_text: list[str]
                token_size += len(enc.encode(line))
                if (token_size + FUNC_TOKEN + OUTPUT_TOKEN_LIMIT) >= INPUT_TOKEN_LIMIT:
                    print(f"Reached maximum allowed token size considering tokens reserved for functions and outputs: {token_size}/{INPUT_TOKEN_LIMIT}")
                    break
                concat_text = concat_text + ' ' + line
            concat_text = concat_text + '\n' + 'Class: '
            print(f"The message has a total of {token_size}/{INPUT_TOKEN_LIMIT} tokens.")
        
            # feed the processed client text to content
            message = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": concat_text}]
        
            # write the concatenated text into a file
            if not os.path.isfile(f"pos_neg_examples/{true_label}_{symptom}_accuracy_example.txt"):
                with open(f"pos_neg_examples/{true_label}_{symptom}_accuracy_example.txt", "w") as file: 
                    file.write(concat_text)
            else:
                with open(f"pos_neg_examples/{true_label}_{symptom}_accuracy_example.txt", "a") as file: 
                    file.write("\n\n")
                    file.write(concat_text)
        
            # query gpt (exponential backoff time to ensure not hitting rate limit)
            backoff = 2
            while True:
                try:
                    completion = client.chat.completions.create(
                        model = MODEL,
                        temperature = TEMPERATURE,
                        max_tokens = OUTPUT_TOKEN_LIMIT,
                        messages = message,
                        functions=[function],
                        function_call={"name": "multilabel"}
                    )
                    break
                except RateLimitError as e:
                    print(f"Rate limit hit: {e}. Waiting {backoff}s before retry...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                except Exception as e:
                    print(f"Encountered error: {e}")
                    sys.exit(1)

            # construct a json object from the response
            # append current session's index into json object
            json_output = {}
            json_output = {
                "session_ID": count,
                "class": json.loads(completion.choices[0].message.function_call.arguments)["prediction"][0],
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }

            if json_output["class"] == true_label:
                correct_predict += 1
            if json_output["class"] == "Positive":
                pred_labels[idx_count, 0] = 1   # record predicted label
        
            # save chatgpt response
            if not os.path.isfile(f"GPT_output_{symptom}_accuracy.json"):
                with open(f"GPT_output_{symptom}_accuracy.json", "w") as outfile:
                    json.dump(json_output, outfile)
            else:
                with open(f"GPT_output_{symptom}_accuracy.json", "a") as outfile:
                    json.dump(json_output, outfile)

            idx_count += 1
            print(f"GPT answer to sample index {count} is {completion.choices[0].message.function_call}.")   
        
        count += 1

    np.savetxt(f"true_labels_{symptom}.txt", true_labels)
    np.savetxt(f"pred_labels_{symptom}.txt", pred_labels)
    print(f"The prediction accuracy on n={n_pos + n_neg} is {correct_predict/(n_pos + n_neg)}.\n")

    return [true_labels, pred_labels]

### CLASSIFICATION STABILITY
def prompt_GPT_stability(input_json, label_mat, symptom = "anxiety", idx = 777, iter = 200):
    '''
    Prompt the GPT model to perform binary classificaton on the symptom of interest for a single
    examples indicated by its index `idx` usng OpenAI's API.

    Return:
        @pos: number of times the example is classified as positive
        @neg: number of times the example is classified as negative
        @pred_labels: GPT-predicted labels in each iteration
    '''
    
    print("=============================================")
    print(f"Querying GPT on {symptom} for sample_idx={idx}......")
    print("=============================================")

    count = 0
    pos = 0  # number of positive predictions
    neg = 0  # number of negative predictions
    pred_labels = np.zeros((iter, 1))  # save GPT predicted labels for each iteration

    SYSTEM = "You are an experienced psychiatrist who will diagnose mental health conditions from counseling transcripts."
    CLASS_DESCRIPTION = f"Classes: [Positive, Negative]\n`Positive` indicates the client shows the symptom of {symptom}. `Negative` indicates the client does not show the symptom of {symptom}."
    PROMPT = f"Classify the client text into one of the classes.\n{CLASS_DESCRIPTION}"

    if symptom == "anxiety":
        index = 1  # symptom index (0: none / 1: anxiety / 2: depression)
    elif symptom == "depression":
        index = 2
    label_vector = label_mat[:,index].reshape(label_mat.shape[0])
    if label_vector[idx] == 1:
        true_label = "Positive"
    else:
        true_label = "Negative"

    for session in input_json:
        # obly query GPT when the sample is our chosen sample
        if count == idx:
            print(f"Querying sample with index {count} and class {true_label}...")
        
            # process the client text into api-recognizable format
            client_text = input_json[session]['Client_Text_Replaced_Two']
            concat_text = f"{PROMPT}\nText:"
            token_size = len(enc.encode(SYSTEM)) + len(enc.encode(concat_text))
            for line in client_text:  # client_text: list[str]
                token_size += len(enc.encode(line))
                if (token_size + FUNC_TOKEN + OUTPUT_TOKEN_LIMIT) >= INPUT_TOKEN_LIMIT:
                    print(f"Reached maximum allowed token size considering tokens reserved for functions and outputs: {token_size}/{INPUT_TOKEN_LIMIT}")
                    break
                concat_text = concat_text + ' ' + line
            concat_text = concat_text + '\n' + 'Class: '
            print(f"The message has a total of {token_size}/{INPUT_TOKEN_LIMIT} tokens.")
        
            # feed the processed client text to content
            message = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": concat_text}]
        
            # write the concatenated text into a file
            if not os.path.isfile(f"pos_neg_examples/stability_example.txt"):
                with open(f"pos_neg_examples/stability_example.txt", "w") as file: 
                    file.write(concat_text)
        
            # query gpt (exponential backoff time to ensure not hitting rate limit)
            for i in range(iter):
                backoff = 2
                while True:
                    try:
                        completion = client.chat.completions.create(
                            model = MODEL,
                            temperature = TEMPERATURE,
                            max_tokens = OUTPUT_TOKEN_LIMIT,
                            messages = message,
                            functions=[function],
                            function_call={"name": "multilabel"}
                        )
                        break
                    except RateLimitError as e:
                        print(f"Rate limit hit: {e}. Waiting {backoff}s before retry...")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, MAX_BACKOFF)
                    except Exception as e:
                        print(f"Encountered error: {e}")
                        sys.exit(1)

                # construct a json object from the response
                # append current session's index into json object
                json_output = {}
                json_output = {
                    "session_ID": count,
                    "class": json.loads(completion.choices[0].message.function_call.arguments)["prediction"][0],
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }
            
                if json_output["class"] == "Positive":
                    pos += 1
                    pred_labels[i, 0] = 1   # record predicted label
                elif json_output["class"] == "Negative":
                    neg += 1
        
                # save chatgpt response
                if not os.path.isfile(f"GPT_output_{symptom}_stability.json"):
                    with open(f"GPT_output_{symptom}_stability.json", "w") as outfile:
                        json.dump(json_output, outfile)
                else:
                    with open(f"GPT_output_{symptom}_stability.json", "a") as outfile:
                        json.dump(json_output, outfile)
    
        count += 1

    np.savetxt(f"pred_labels_{symptom}_stability.txt", pred_labels)
    print(f"Positive prediction: {pos}/{iter}.")
    print(f"Negative prediction: {neg}/{iter}.\n")

    return [pos, neg, pred_labels]


# random sample total n = 200 transcripts
pos_idx, neg_idx, concat_idx = random_sample(label_matrix, symptom = SYMPTOM[0], n_pos = 100, n_neg = 100)
# perform anxiety and depression classification
anxiety_true, anxiety_pred = prompt_GPT_accuracy(data, label_matrix, concat_idx, symptom = SYMPTOM[0], n_pos = 100, n_neg = 100)
depression_true, depression_pred = prompt_GPT_accuracy(data, label_matrix, concat_idx, symptom = SYMPTOM[1], n_pos = 100, n_neg = 100)
# combine the results to assess multi-label accuracy
y_preds = np.concatenate((anxiety_pred, depression_pred), axis=1)
y_true = np.concatenate((anxiety_true, depression_true), axis=1)
print(y_preds.shape)
print(y_true.shape)
accuracy = accuracy_score(y_true, y_preds)
print(f"The multilabel accuracy for {MODEL} is: {accuracy}")


# stability test
pos_anxiety_count, neg_anxiety_count, anxiety_stab_preds = prompt_GPT_stability(data, label_matrix, symptom = SYMPTOM[0], idx = 777, iter = 200)
pos_depression_count, neg_depression_count, depression_stab_preds = prompt_GPT_stability(data, label_matrix, symptom = SYMPTOM[1], idx = 777, iter = 200)
print(f"Positive, Negative: {pos_anxiety_count}, {neg_anxiety_count}")
print(f"Positive, Negative: {pos_depression_count}, {neg_depression_count}")
print(f"True label: {label_matrix[777, :]}")
# count number of examples in category 0/0, 0/1, 1/0, 1/1
stability_preds = np.concatenate((anxiety_stab_preds, depression_stab_preds), axis=1)
print(stability_preds.shape)
elements, repeats = np.unique(stability_preds, return_counts=True, axis=0)
elements = elements.astype(int)
print(f"The unique label combinations are: {elements}")
print(f"Each with repeats: {repeats}")

# draw confusion matrix
confusion_matrix = np.zeros((2, 2), dtype=int)
# rows = depression, cols = anxiety
for i, (anx, dep) in enumerate(elements):
    confusion_matrix[dep, anx] = repeats[i]
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=['-', '+'], yticklabels=['-', '+'])
plt.xlabel('Anxiety Diagnosis')
plt.ylabel('Depression Diagnosis')
plt.title('Label Distribution')
plt.tight_layout()
plt.savefig(f'stability_{MODEL}.png')

print(f"\n****Script complete***")

