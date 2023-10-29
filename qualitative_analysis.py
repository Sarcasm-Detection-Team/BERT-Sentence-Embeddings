#checking out similarity matrix on raw data-downsampled
from sklearn.metrics.pairwise import cosine_similarity

data_df = pd.read_csv("raw_data_embeddings_final.csv")
embeddings = data_df.iloc[:, :768].values

similarity_matrix = cosine_similarity(embeddings)

print(similarity_matrix)

#%%
# Calculate similarity between sarcastic sentences-10 most similar
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

data_df = pd.read_csv("raw_data_embeddings_final.csv")

#filter the data , so we get only sarcastic sentences
filtered_data = data_df[data_df['target'] == 1]

# get the embeddings of those sentences
embeddings = filtered_data.iloc[:, :768].values

#calculate cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

#get the pairs over a defined threshold(here 0.85) , so we dont charge the machine with added computational load -we dont need the pairs below 0.85 for now
similar_pairs = [(i, j) for i in range(len(similarity_matrix))
                 for j in range(i + 1, len(similarity_matrix[i]))
                 if similarity_matrix[i, j] >= 0.85]

# sort the pairs
similar_pairs.sort(key=lambda x: similarity_matrix[x[0], x[1]], reverse=True)

#empty list for storing
similar_pairs_list = []

#iterate over the similar pairs and extract the corresponding titles and similarity scores
for pair in similar_pairs[:10]:
    title1 = filtered_data['title'].iloc[pair[0]]
    title2 = filtered_data['title'].iloc[pair[1]]
    similarity_score = similarity_matrix[pair[0], pair[1]]

    # store the pair and similarity score in the list (as a dictionary)
    similar_pairs_list.append({'title1': title1, 'title2': title2, 'similarity_score': similarity_score})

#create a DataFrame from the list of similar pairs
similar_pairs_df = pd.DataFrame(similar_pairs_list)

print(similar_pairs_df)

#%%
# Calculate similarity between non sarcastic sentences-10 most similar

'''This is the same code as the previous one , but now we compute the 
10 most similar headlines for the non sarcastic sentences '''

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

data_df = pd.read_csv("raw_data_embeddings_final.csv")

filtered_data = data_df[data_df['target'] == 0]
embeddings = filtered_data.iloc[:, :768].values

similarity_matrix = cosine_similarity(embeddings)

similar_pairs = [(i, j) for i in range(len(similarity_matrix))
                 for j in range(i + 1, len(similarity_matrix[i]))
                 if similarity_matrix[i, j] >= 0.85]

similar_pairs.sort(key=lambda x: similarity_matrix[x[0], x[1]], reverse=True)

similar_pairs_list = []

for pair in similar_pairs[:10]:
    title1 = filtered_data['title'].iloc[pair[0]]
    title2 = filtered_data['title'].iloc[pair[1]]
    similarity_score = similarity_matrix[pair[0], pair[1]]

    similar_pairs_list.append({'title1': title1, 'title2': title2, 'similarity_score': similarity_score})

similar_pairs_df = pd.DataFrame(similar_pairs_list)

print(similar_pairs_df)

#%%
#most frequent words from clusters -sarcastic
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_df = pd.read_csv("raw_data_embeddings_final.csv")

#extract the target values from df and corresponding embeddings and sentences
target_values = data_df['target'].values
target_1_indices = np.where(target_values == 1)[0]
target_1_embeddings = data_df.iloc[target_1_indices, :768].values
target_1_titles = data_df.iloc[target_1_indices]['title'].values

k = 5  #number of clusters

#initialize and apply KMeans clustering on the embeddings
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(target_1_embeddings)

#create a dict
cluster_groups = defaultdict(list)

#add the titles to their corresponding clusters in the dictionary
for title, label in zip(target_1_titles, cluster_labels):
    cluster_groups[label].append(title)

# Get the 10 most frequent words from each cluster
vectorizer = CountVectorizer(stop_words='english') #initialize the CountVectorizer as our vectorizer
for cluster, titles in cluster_groups.items():
    print(f"Cluster {cluster}:")
    print("Titles:")
    for title in titles:
        print(title)
    print("Most frequent words:")
    cluster_text = ' '.join(titles)  #concatenate all the headlines together
    word_counts = vectorizer.fit_transform([cluster_text]) #vectorize 
    words = list(vectorizer.vocabulary_.keys())  #get the vocabulary of the words
    counts = np.asarray(word_counts.sum(axis=0)).ravel() #calculate the toal count of each word
    top_indices = counts.argsort()[-10:][::-1] #sort the indices and get the top 10 words
    top_words = [words[i] for i in top_indices]  #retrieve the actual words 
    top_word_counts = [counts[i] for i in top_indices] #retrieve the counts of the words
    
    for word, count in zip(top_words, top_word_counts):
        print(f"Word: {word}, Frequency: {count}")
    print()

#apply PCA for visualization of the clusters (in a 2d space) - Reduce initial dimensions
pca = PCA(n_components=2) #initialize PCA
embeddings_2d = pca.fit_transform(target_1_embeddings) #apply PCA to the sarcastic embeddings 


for cluster, titles in cluster_groups.items():
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster] #get the indices of the embeddings in the current cluster:
    cluster_embeddings = embeddings_2d[cluster_indices]   #extract the corresponding transformed embeddings 
    plt.scatter(cluster_embeddings[:, 0], cluster_embeddings[:, 1], label=f"Cluster {cluster}")  #plot the 2d embeddings

plt.title("Visualization of Sarcastic headlines")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.savefig("5_cluster_groups_sarc.png") #save the figure 
plt.show()


#%%
#most frequent words in clusters , non sarcastic
'''This is the same code as above , for getting the 10 most frequent words for 
each cluster , and then applying PCA for visualizing the data points in a 2d space , 
but for non sarcastic headlines .'''

from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

data_df = pd.read_csv("raw_data_embeddings_final.csv")
target_values = data_df['target'].values
target_0_indices = np.where(target_values == 0)[0]
target_0_embeddings = data_df.iloc[target_0_indices, :768].values
target_0_titles = data_df.iloc[target_0_indices]['title'].values

k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(target_0_embeddings)

cluster_groups = defaultdict(list)

for title, label in zip(target_0_titles, cluster_labels):
    cluster_groups[label].append(title)

# Get the 10 most frequent words from each cluster
vectorizer = CountVectorizer(stop_words='english')
for cluster, titles in cluster_groups.items():
    print(f"Cluster {cluster}:")
    print("Titles:")
    for title in titles:
        print(title)
    print("Most frequent words:")
    cluster_text = ' '.join(titles)
    word_counts = vectorizer.fit_transform([cluster_text])
    words = list(vectorizer.vocabulary_.keys())
    counts = np.asarray(word_counts.sum(axis=0)).ravel()
    top_indices = counts.argsort()[-10:][::-1]
    top_words = [words[i] for i in top_indices]
    top_word_counts = [counts[i] for i in top_indices]
    for word, count in zip(top_words, top_word_counts):
        print(f"Word: {word}, Frequency: {count}")
    print()


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(target_0_embeddings)


for cluster, titles in cluster_groups.items():
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
    cluster_embeddings = embeddings_2d[cluster_indices]
    plt.scatter(cluster_embeddings[:, 0], cluster_embeddings[:, 1], label=f"Cluster {cluster}")

plt.title("Visualization of Non Sarcastic Headlines")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.savefig("5_cluster_groups_non_sarc.png")
plt.show()