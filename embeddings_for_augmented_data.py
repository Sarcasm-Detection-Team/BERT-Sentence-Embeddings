#embeddings for augmented data 
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import os

#set a batch size for processing the data. Otherwise , a memory error occured
batch_size = 16

#using the same code as above for getting embeddings 
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
model = TFAutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

def get_sentence_embedding(sentences):
    embeddings = []

    # Process the data in batches
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]

        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf", max_length=512)
        outputs = model(encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
        batch_embeddings = outputs.pooler_output.numpy().tolist()
        embeddings.extend(batch_embeddings)

    return embeddings


#load the files again for augmanting them 
non_sarcastic_df = pd.read_csv('raw_news.csv')
sarcastic_df = pd.read_csv('raw_sarc.csv')

non_sarcastic_df = non_sarcastic_df.drop_duplicates(subset="title", keep="first")
sarcastic_df = sarcastic_df.drop_duplicates(subset="title", keep="first")

print("non_sarcastic_df size after removing duplicates:", len(non_sarcastic_df))
print("sarcastic_df size after removing duplicates:", len(sarcastic_df))

desired_size = 3000 #for each label 

#sample the two dataframes with the desired size
non_sarcastic_sampled = non_sarcastic_df.sample(n=desired_size, replace=True, random_state=42)
sarcastic_sampled = sarcastic_df.sample(n=desired_size, replace=True, random_state=42)

aug_df = pd.concat([non_sarcastic_sampled, sarcastic_sampled], axis=0) #concatenate them

aug_df = aug_df.dropna()
print("aug_df size:", len(aug_df))

'''This code was made , cause we met difficulties with the memory when running the 
code for getting the embeddings of the augmented data .As a solution , we split the dataframe into
4 smaller pieces , and then generated the embeddings '''

num_splits = 4  #define the number of smaller sets
subset_size = len(aug_df) // num_splits #define the size from the number of splits 

#create the different dataframes 
aug_dfs = [aug_df[i * subset_size : (i + 1) * subset_size].copy() for i in range(num_splits)]

for i, df in enumerate(aug_dfs):
    print(f"aug_df{i+1} size:", len(df))

'''this for loop is similar to the code that extracts and saves the embeddings (as csv files).
In this case it's implemented on the different dataframes '''
for i, df in enumerate(aug_dfs):
    titles = [str(title) for title in df["title"]]
    targets = [int(target) for target in df["target"]]

    embeddings = get_sentence_embedding(titles)

    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['title'] = titles
    embeddings_df['target'] = targets


    filepath = f'aug_data_embeddings_df{i+1}.csv'
    embeddings_df.to_csv(filepath, index=False, header=True)

#%%
#concatenate augmented_data csvs'
import pandas as pd
import os

file_paths = [
    'aug_data_embeddings_df1.csv',
    'aug_data_embeddings_df2.csv',
    'aug_data_embeddings_df3.csv',
    'aug_data_embeddings_df4.csv'
]

dfs = []

# read each CSV file and append its dataframe to the list
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)

total_df = pd.concat(dfs)

total_csv_path = 'total_aug_data_embeddings.csv'

# save the total dataframe as a CSV file
total_df.to_csv(total_csv_path, index=False)