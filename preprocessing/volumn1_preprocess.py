'''
Created by Junwei (Ivy) Sun

This file contains code that performs preliminaty preprocessing for Volumn 1 data.
After this preprocess step only therapy sessions are kept in the publication metadata
csv (book transcripts are eliminated), and Volumn I and II publication csv are combined
together to make further downstream processing easier.
'''

import pandas as pd

# read in publication metadata file for volumn 1
df = pd.read_csv('publication_metadata_volumn1_full.csv')
print(df)

# keep only therapy session metadata, eliminate all other rows
with open("volumn1_transcripts_filename.txt", encoding="utf-8") as fp:
    session_names = fp.readlines()
    session_names = [line.strip() for line in session_names]

print(session_names)

for row in range(df.shape[0]):
    if df["file_name"][row] not in session_names:
        df = df.drop(row)

print(df)
df.to_csv('publication_metadata_volumn1_filtered.csv', index = False)

# combine volumn1 and volumn2 metadata csv files
metadata_volumn1 = pd.read_csv('publication_metadata_volumn1_filtered.csv')
metadata_volumn2 = pd.read_csv('publication_metadata_volumn2.csv')
columns_volumn1 = list(metadata_volumn1.columns)
columns_volumn2 = list(metadata_volumn2.columns)

# rename according to volumn2's ways of expression
metadata_volumn1 = metadata_volumn1.rename(columns={"Client_Age_Range": "Client_Age"})

# organize columns in metadata files to keep
columns_to_keep = ["file_name", "Entity_ID","Abstract", "Client_Age", "Client_Gender",
                    "Client_Marital_Status", "Client_Sexual_Orientation", "Psyc_Subjects",
                    "Symptoms", "Therapies", "Therapist", 'Real_Title']
# These five metadata columns do not exist in volumn 1 collection:
# "Previous_Session_ID", "Next_Session_ID", "CTIV_category", "Race_of_Therapist", "Race_of_Client"

# combine
metadata_volumn1 = metadata_volumn1[columns_to_keep]
metadata_volumn2 = metadata_volumn2[columns_to_keep]
print(metadata_volumn1)
print(metadata_volumn2)
df = pd.concat([metadata_volumn1, metadata_volumn2], ignore_index=True)
print(df)

df.to_csv('publication_metadata_combined.csv', index = False)
