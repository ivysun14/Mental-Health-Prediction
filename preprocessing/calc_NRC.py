#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:57:37 2023

@author: Siqi Ma
"""
import json
import pandas as pd
import re


out_dir = "dic_features/NRC_feature_matrix.csv"
NRC_dir = "dic_features/NRC_dic.json"
data_dir = 'processed/meta_cleaned.json'

def process_sentence(NRC_dic, sentence, emo):
    result = 0
    word_list = re.split("; |, | |\?|\"|\'|\*|. ", sentence)
    for word in word_list:
        try:
            result += NRC_dic[word.lower()][emo]
        except:
            result += 0
    if result != 0:
        result = result/len(word_list)
    return result

def calc_length(sentence):
    return len(sentence.split(" "))

def calc_mean(l):
    if len(l) > 0:
        return sum(l)/len(l)
    else:
        return float('nan')

def process_NRC(NRC_dir, out_dir, data_dir):
    NRC_dic = json.load(open(NRC_dir))
    data = json.load(open(data_dir))
    docs = list(data.keys())
    emos = list(NRC_dic['aback'].keys())
    
    feature = {}
    
    for doc in docs:
        print("processing" + str(doc))
        try:
            sentences = data[doc]["Client_Text"]
            dic = {}
            for emo in emos:
                temp = []
                length = []
                for sentence in sentences:
                    temp.append(process_sentence(NRC_dic, sentence, emo))
                    length.append(calc_length(sentence))
                dic[emo] = temp
                dic['length'] = length
            feature[doc] = dic
        except:
            feature[doc] = {}
            
    with open("dic_features/NRC_feature.json", "w") as fout:
        json.dump(feature, fout, indent = 4)
    
    df = pd.DataFrame.from_dict(feature, orient="index")
    
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11 = ([] for i in range(11))

    
    for i in range(len(df)):
        l1.append(calc_mean(df.iloc[i]["disgust"]))
        l2.append(calc_mean(df.iloc[i]["negative"]))
        l3.append(calc_mean(df.iloc[i]["sadness"]))
        l4.append(calc_mean(df.iloc[i]["positive"]))
        l5.append(calc_mean(df.iloc[i]["surprise"]))
        l6.append(calc_mean(df.iloc[i]["joy"]))
        l7.append(calc_mean(df.iloc[i]["anticipation"]))
        l8.append(calc_mean(df.iloc[i]["trust"]))
        l9.append(calc_mean(df.iloc[i]["trust"]))
        l10.append(calc_mean(df.iloc[i]["trust"]))
        l11.append(calc_mean(df.iloc[i]["length"]))
            
    df["disgust_mean"] = l1
    df["negative_mean"] = l2
    df["sadness_mean"] = l3
    df["positive_mean"] = l4
    df["surprise_mean"] = l5
    df["job_mean"] = l6
    df["anticipation_mean"] = l7
    df["trust_mean"] = l8
    df["anger_mean"] = l9
    df["fear_mean"] = l10
    df["length"] = l11
    
    print(df[df['fear_mean'].isna()])
    
    df.to_csv(out_dir, index=True)
    

def process_sentence_MOESM(MOESM_dic, sentence):
    result = 0
    word_list = re.split("; |, | |\?|\"|\'|\*|. ", sentence)
    for word in word_list:
        try:
            result += MOESM_dic[word.lower()]
        except:
            result += 0
    if result != 0:
        result = result / len(word_list)
    return result

def MOESM(out_dir, data_dir):
    MOESM = pd.read_csv("dic_features/13428_2013_403_MOESM1_ESM.csv")
    data = json.load(open(data_dir))
    docs = list(data.keys())
    
    new = MOESM[["Word","Percent_known"]]
    
    MOESM_dic = {}
    
    feature = {}
    
    for i in range(len(new)):
        MOESM_dic[new["Word"][i]] = new['Percent_known'][i]
    
    for doc in docs:
        print("processing" + str(doc))
        try:
            sentences = data[doc]["Client_Text"]
            dic = {}
            temp = []
            for sentence in sentences:
                temp.append(process_sentence_MOESM(MOESM_dic, sentence))
            dic['percent_known'] = temp
            dic['mean'] = calc_mean(temp)
            feature[doc] = dic
        except:
            feature[doc] = {}
        
    df = pd.DataFrame.from_dict(feature, orient="index")
    
    df.to_csv(out_dir, index=True)


process_NRC(NRC_dir, out_dir, data_dir)

MOESM("dic_features/MOESM.csv", data_dir)




