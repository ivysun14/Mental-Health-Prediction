"""
Created by Junwei (Ivy) Sun
Date: 12/02/2023

This file creates a naive word frequency feature matrix with shape (n, d) from all client
texts in Counsling and Psychotherapy Transcripts, Volumn I and II. The input data are from
processed/meta_cleaned.json. Two output matrices are produced which are the raw frequency count
matrix (feature_matrix_naive.txt) and the frequency normalized by row sum (feature_matrix_naive
_normalized.txt)

Note:
- n: number of counsling/psychotherapy sessions. Here we treat each individual session as
a single data point, because even for the same client between sessions, the individual's
mental state might change.
- d: number of unique words in the entirity of the volumns.

"""

import json
import collections
from string import punctuation
import numpy as np

np.random.seed(1234)

# Take a single therapy session and eliminate punctions in client texts, output a list of words separated by white spaces.
# Attention: this list of words is NOT unique!
#
def process_session(session_data):
    """
    Inputs:
        -- session_data: a dictionary representing all data from a single counsling/psychotherapy session (i.e one training data point)
    
    Return:
        -- word_list: a list of words excluding punctuations present in the client text of this particular therapy session
    """
    
    word_list = []

    num_conversation = len(session_data['Client_Text']) # how many lines has the client spoken
    for i in range(num_conversation): # loop over each line
        client_text = session_data['Client_Text'][i]
        # eliminate punctuations
        for c in punctuation:
            client_text = client_text.replace(c, ' ')
        client_text = client_text.lower().split()
        word_list += client_text
    
    return word_list


# Assemble a dictionary of unique words from client texts across ALL therapy sessions
#
def assemble_dictionary(json_obj):
    """
    Inputs:
        -- json_obj: a processed json file in the form of a dictionary
    
    Return:
        -- word_dict: a dictionary of words assembled from all client texts across ALL sessions
                this dictionary has (key, value) pairs representing (word, index):
                    -- word: a unique word
                    -- index: keeps track of how many unique words there are in the dictionary, NOT word frequency
    """

    word_dict = {}
    idx = 0

    for session in json_obj:
        # punctuation-eliminated words from client
        words = process_session(json_obj[session])
        for word in words:
            if word not in word_dict:
                word_dict[word] = idx
                idx += 1
    
    return word_dict


# Create a feature matrix that will be used as input to the training algorithm.
#
def generate_feature_matrix(json_obj, word_dict):
    """
    Inputs:
        -- json_obj: a processed json file in the form of a dictionary
        -- word_dict: the dictionary of unique words outputted by assemble_dictionary()
    
    Return:
        -- feature_matrix: a 2D numpy array of shape (n, d)
                Each row is a feature vector with word frequency, indicating how many times a dictionary word has appeared in a session:
                    -- n: each line represent a single therapy session
                    -- d: each column represent a unique word from the 'Client_Text' section of the json file
    """

    n = len(json_obj)
    d = len(word_dict)
    matrix = np.zeros((n, d))

    data_point = 0
    for session in json_obj:
        words = process_session(json_obj[session]) # a list of non-unique words from a single session
        counter = collections.Counter(words) # a dictionary of unique words with frequency for the session
        for word, frequency in counter.items():
            if word in word_dict.keys():
                value = word_dict[word]
                matrix[data_point][value] += frequency
        data_point += 1
    
    return matrix


# return JSON object as a dictionary
f = open('processed/meta_cleaned.json')
data = json.load(f)
f.close()

# there are in total 3503 sessions
print(len(data))

# create dictionary and matrix
unique_words_dictionary = assemble_dictionary(data)
feature_matrix = generate_feature_matrix(data, unique_words_dictionary)
print(f'The dimension of the feature matrix is {feature_matrix.shape}')
print(f'A single feature vector inside the feature matrix looks like:\n {feature_matrix[0]}')

np.savetxt("feature_matrix_naive.txt", feature_matrix)
print(feature_matrix)

# normalize frequency counts by row
row_sum = np.sum(feature_matrix, axis = 1)
feature_matrix_normalized = feature_matrix / row_sum.reshape(-1,1)

# save results
np.savetxt("feature_matrix_naive_normalized.txt", feature_matrix_normalized)
print(feature_matrix_normalized)

# create json object from dictionary
json = json.dumps(unique_words_dictionary)
f = open("unique_words_dictionary.json", "w")
f.write(json)
f.close()