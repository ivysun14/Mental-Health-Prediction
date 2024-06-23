# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:52:56 2023

@author: Siqi Ma
"""
import json
from string import punctuation
import numpy as np
import collections

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet

f = open('processed/meta_cleaned.json')
data = json.load(f)
f.close()

def process_session_stem(session_data):
    porter=PorterStemmer()
    word_list = []
    
    try:
        num_conversation = len(session_data['Client_Text']) # how many lines has the client spoken
        for i in range(num_conversation): # loop over each line
            client_text = session_data['Client_Text'][i]
            # eliminate punctuations
            for c in punctuation:
                client_text = client_text.replace(c, ' ')
            
            token_words=word_tokenize(client_text)
            for word in token_words:
                word_list.append(porter.stem(word))
                
        flag = True
        
        if num_conversation == 0:
            flag = False
    except:
        flag = False
    
    return word_list, flag

def process_session_sym(session_data):
    word_list = []
    try:
        num_conversation = len(session_data['Client_Text']) # how many lines has the client spoken
        for i in range(num_conversation): # loop over each line
            client_text = session_data['Client_Text'][i]
            # eliminate punctuations
            for c in punctuation:
                client_text = client_text.replace(c, ' ')
            
            client_text = client_text.lower().split()
            word_list += client_text
        
        flag = True
        
        if num_conversation == 0:
            flag = False
    except:
        flag = False
    
    
    return word_list, flag

def assemble_dictionary_stem(json_obj):
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
        words, flag = process_session_stem(json_obj[session])
        if flag:
            for word in words:
                if word not in word_dict:
                    word_dict[word] = idx
                    idx += 1
    return word_dict

def getsym(list_obj):
    output = []
    for syns in list_obj:
        for w in syns.lemmas():
            output.append(w.name())
    uniq = set(output)
    return uniq

def assemble_dictionary_sym(json_obj):
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

    for session in json_obj:
        # punctuation-eliminated words from client
        words, flag = process_session_sym(json_obj[session])
        if flag:
            for word in words:
                syns = wordnet.synsets(word)
                syn_w = getsym(syns)
                if word not in word_dict:
                    word_dict[word] = syn_w
    return word_dict
                    
def generate_feature_matrix_stem(json_obj, word_dict):
    porter=PorterStemmer()
    n = len(json_obj)
    d = len(word_dict)
    feature_matrix = np.zeros((n, d))
    error_word = []
    error_session = []

    data_point = 0
    for session in json_obj:
        words, flag = process_session_stem(json_obj[session]) # a list of non-unique words from a single session
        if flag:
            words = tuple(words)
            counter = collections.Counter(words) # a dictionary of unique words with frequency for the session
            for word, frequency in counter.items():
                word_stem = porter.stem(word)
                if word_stem in word_dict.keys():
                    value = word_dict[word]
                    feature_matrix[data_point][value] += frequency
                else:
                    error_word.append(word_stem)
        else:
            error_session.append(session)
        data_point += 1
    return feature_matrix, error_session

def combine_sym(word_dict):
    new_dict = {}
    count = 0
    # for every uniq word
    for key in word_dict.keys():
        # search if the uniq word exist in any lists
        newlist = []
        for item in word_dict:
            if key in word_dict[item]:
                newlist += word_dict[item]
            
        # search if the word is in new dict
        for keys in new_dict:
            if key in new_dict[keys]:
                new_dict[keys] = newlist
            else:
                index = str(count)
                new_dict[index] = newlist
                count += 1
    return new_dict

def generate_feature_matrix_sym(json_obj, word_dict):
    
    porter=PorterStemmer()
    n = len(json_obj)
    d = len(word_dict)
    feature_matrix = np.zeros((n, d))
    error_word = []
    error_session = []

    data_point = 0
    for session in json_obj:
        words, flag = process_session_stem(json_obj[session]) # a list of non-unique words from a single session
        if flag:
            words = tuple(words)
            counter = collections.Counter(words) # a dictionary of unique words with frequency for the session
            for word, frequency in counter.items():
                word_stem = porter.stem(word)
                if word_stem in word_dict.keys():
                    value = word_dict[word]
                    feature_matrix[data_point][value] += frequency
                else:
                    error_word.append(word_stem)
        else:
            error_session.append(session)
        data_point += 1
    return feature_matrix, error_session



def return_stem():
    unique_words_dictionary = assemble_dictionary_stem(data)
    feature_matrix, error_session = generate_feature_matrix_stem(data, unique_words_dictionary)
    
    np.savetxt("./feature_matrix_stem.txt", feature_matrix)
    
    # create json object from dictionary
    output = json.dumps(unique_words_dictionary)
    f = open("stem_words_dictionary.json", "w")
    f.write(output)
    f.close()
    
    return error_session

#error_session = return_stem()
#print(error_session)

unique_words_dictionary = assemble_dictionary_stem(data)
feature_matrix, error_session = generate_feature_matrix_stem(data, unique_words_dictionary)

np.savetxt("./feature_matrix_stem.txt", feature_matrix)

# create json object from dictionary
output = json.dumps(unique_words_dictionary)
f = open("stem_words_dictionary_12.json", "w")
f.write(output)
f.close()


#normalize counts
total_length = np.sum(feature_matrix, axis = 1)
normalized = feature_matrix
for i in range(feature_matrix.shape[0]):
    normalized[i,] = feature_matrix[i, :]/(total_length[i])
    
np.savetxt("./feature_matrix_stem_normalized.txt", normalized)
import pandas as pd
data = pd.read_csv("dic_features/NRC_feature_matrix.csv")
data2 = pd.read_csv("dic_features/MOESM.csv")

data['concreteness_mean'] = data2['mean']

data.drop("Unnamed: 0", axis=1, inplace=True)
data.drop("disgust", axis=1, inplace=True)
data.drop("negative", axis=1, inplace=True)
data.drop("sadness", axis=1, inplace=True)
data.drop("positive", axis=1, inplace=True)
data.drop("surprise", axis=1, inplace=True)
data.drop("joy", axis=1, inplace=True)
data.drop("anticipation", axis=1, inplace=True)
data.drop("trust", axis=1, inplace=True)
data.drop("anger", axis=1, inplace=True)
data.drop("fear", axis=1, inplace=True)
data.drop("length", axis=1, inplace=True)

data_matrix = data.values

all_features = np.column_stack((feature_matrix, data_matrix))
np.savetxt("./feature_matrix_total.txt", all_features)







