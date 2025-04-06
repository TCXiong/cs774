import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import faiss


# Load data
tb1_path = os.path.join("small_files", "amazon.csv")
tb2_path = os.path.join("small_files", "best_buy.csv")

table1 = pd.read_csv(tb1_path)
table2 = pd.read_csv(tb2_path)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for table1
table1['combined'] = table1['Brand'] + " " + table1['Name'] + " " + table1['Features'].fillna('')

# Create embeddings for table2
table2['combined'] = table2['Brand'] + " " + table2['Name'] + " " + table2['Features'].fillna('')


embeddings1 = model.encode(table1['combined'].tolist(), show_progress_bar=True).astype('float32')
embeddings2 = model.encode(table2['combined'].tolist(), show_progress_bar=True).astype('float32')
print(type(embeddings1))
print(embeddings1.shape)
print(embeddings2.shape)

# Create FAISS index
dimension = embeddings1.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings1)

k = 3  # number of nearest neighbors to retrieve
D, I = index.search(embeddings2, k)

# Create candidate pairs
candidate_pairs = []
for i in range(len(table2)):
    for j in range(k):
        candidate_pairs.append((table2.iloc[i]['ID'], table1.iloc[I[i][j]]['ID']))