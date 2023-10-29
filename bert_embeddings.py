#data loading 
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

# read the datasets
non_sarcastic_df = pd.read_csv('C:\\Users\\elenh\\phase_1\\raw_news.csv')
sarcastic_df = pd.read_csv('C:\\Users\\elenh\\phase_1\\raw_sarc.csv')

# remove duplicates from "title" column 
non_sarcastic_df = non_sarcastic_df.drop_duplicates(subset="title", keep="first")
sarcastic_df = sarcastic_df.drop_duplicates(subset="title", keep="first")

print("non_sarcastic_df size after removing duplicates:", len(non_sarcastic_df))
print("sarcastic_df size after removing duplicates:", len(sarcastic_df))

# find the minimum between the datasets
min_size = min(len(non_sarcastic_df), len(sarcastic_df))

#sample them to have equal number of samples
non_sarcastic_sampled = non_sarcastic_df.sample(n=min_size, random_state=42)
sarcastic_sampled = sarcastic_df.sample(n=min_size, random_state=42)

#concatenate the datasets (sarcastic , non sarcastic)
total_df = pd.concat([non_sarcastic_sampled, sarcastic_sampled], axis=0)


total_df = total_df.dropna() #remove Nan values

print("total_df size:", len(total_df))

#shuffle the rows of the total dataset
shuffled_df = total_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(shuffled_df)


#%%
!pip install transformers
!pip install tensorflow

#%%
#get embeddings 
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import os

#initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
model = TFAutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

#function to extract sentence embeddings with argument a list of sentences
def get_sentence_embedding(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="tf", max_length=512)
    outputs = model(encoded_input.input_ids, attention_mask=encoded_input.attention_mask) #attention mask tensors are extracted from encoded_input

    return outputs.pooler_output  #fixed-size representation of each sentence

titles = [str(title) for title in shuffled_df["title"]]
targets = [int(target) for target in shuffled_df["target"]]

embeddings = get_sentence_embedding(titles)

# Create a DataFrame that includes the embeddings, titles, and targets
embeddings_df = pd.DataFrame(embeddings.numpy())
embeddings_df['title'] = titles
embeddings_df['target'] = targets

print(embeddings_df)

# create the directory if it does not exist
directory = 'C:/Users/elenh/Documents'
if not os.path.exists(directory):
    os.makedirs(directory)

filepath = 'raw_data_embeddings_final.csv'

#save the embeddings DataFrame as a CSV file
embeddings_df.to_csv(filepath, index=False, header=True)