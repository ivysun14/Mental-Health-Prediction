#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:48:59 2024

@author: maguo
"""
import json
import pandas as pd
import re


# Read in the file
file_path = 'symptom_dictionary_new'
with open(file_path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame([data]).T

# Replace symptom words by #
data = json.load(open("meta_cleaned.json"))

def replace_symptoms(sentences, symptoms):
    for i, sentence in enumerate(sentences):
        for symptom in symptoms:
            # Create a replacement string of '#' characters of the same length as the symptom
            replacement = '#' * len(symptom)
            # Replace the symptom in the sentence with the replacement string, ignoring case
            sentences[i] = re.sub(r'\b{}\b'.format(symptom), replacement, sentence, flags=re.IGNORECASE)
    return sentences

temp = ["anxiety", "depression"]
symptom_list = list(df.index)

for file in data:
    print("procesing", file)
    #replaced = replace_symptoms(data[file]["Client_Text"], symptom_list)
    rep = replace_symptoms(data[file]["Client_Text"], temp)
    #data[file]["Client_Text_Replaced_All"] = replaced
    data[file]["Client_Text_Replaced_Two"] = rep
    

# Remove symptom words entirely
with open("removed_meta_all.json", "w") as fout:
    json.dump(data, fout, indent=4)