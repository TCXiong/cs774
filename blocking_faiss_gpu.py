import os
import re
import time
import numpy as np
import pandas as pd
import faiss
import faiss.contrib.torch_utils  # Enables GPU support
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ==== Data Loading ====
def load_data(tb1_path, tb2_path):
    table1 = pd.read_csv(tb1_path)
    table2 = pd.read_csv(tb2_path)
    # table1 = pd.read_csv(tb1_path, encoding='cp1252')
    # table2 = pd.read_csv(tb2_path, encoding='cp1252')
    # table1 = pd.read_csv(tb1_path, encoding='latin1')
    # table2 = pd.read_csv(tb2_path, encoding='utf-8-sig')  # fixes BOM issue
    return table1, table2

# ==== Preprocessing ====
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s-]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_table(df, columns_to_concat):
    for col in columns_to_concat:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe.")
        df[col] = df[col].fillna('')

    processed_text = ''
    for col in columns_to_concat:
        processed_text += df[col].apply(preprocess) + ' '
    
    df['processed'] = processed_text.str.strip()
    df['processed'] = df['processed'].replace('', 'unknown')
    return df

def preprocess_tables(table1, table2, table1_columns, table2_columns):
    table1 = preprocess_table(table1, table1_columns)
    table2 = preprocess_table(table2, table2_columns)
    return table1, table2

# ==== Embedding ====
def encode_texts(model, texts, batch_size=64):
    print("\nEncoding texts into embeddings...")
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    elapsed = time.time() - start_time
    print(f"Embedding generation completed in {elapsed:.2f} seconds.")

    if np.isnan(embeddings).any():
        print("Warning: NaN values detected in embeddings - replacing with zeros")
        embeddings = np.nan_to_num(embeddings)
    return embeddings.astype(np.float32)

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    norms[norms == 0] = 1e-10
    return embeddings / norms[:, np.newaxis]

# ==== FAISS (GPU) Indexing ====
def build_faiss_index(embeddings, batch_size=1000):
    print("\nBuilding FAISS index on GPU...")
    start_time = time.time()
    dimension = embeddings.shape[1]
    if dimension == 0:
        raise ValueError("Embedding dimension is 0 - check your input data")

    try:
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatIP(dimension)
        index = faiss.index_cpu_to_gpu(res, 0, index_flat)

        for i in tqdm(range(0, len(embeddings), batch_size), desc="Adding batches to FAISS"):
            batch = embeddings[i:i+batch_size]
            if not np.isnan(batch).any():
                index.add(batch.astype(np.float32))
            else:
                print(f"Skipping batch {i} due to NaN values")

        elapsed = time.time() - start_time
        print(f"FAISS GPU index built in {elapsed:.2f} seconds.")
        return index
    except Exception as e:
        print(f"Error creating FAISS GPU index: {e}")
        return None

# ==== Matching ====
def find_matches(query_embeddings, table1, target_df, target_embeddings, index=None, top_k=30):
    if index is None:
        raise ValueError("FAISS index is not available. Matching cannot proceed.")
    print("\nFinding matches...")
    matches = []
    start_time = time.time()
    for i, query_embedding in enumerate(tqdm(query_embeddings, desc="Matching queries")):
        if np.isnan(query_embedding).any():
            print(f"Skipping query {i} due to NaN values")
            continue

        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        distances = distances[0]
        indices = indices[0]

        for score, idx in zip(distances, indices):
            matches.append({
                'left_id': table1.iloc[i]['ID'],
                'right_id': target_df.iloc[idx]['ID'],
                'similarity_score': score,
                'table1_text': table1.iloc[i]['processed'],
                'table2_text': target_df.iloc[idx]['processed']
            })
    elapsed = time.time() - start_time
    print(f"Matching completed in {elapsed:.2f} seconds.")
    return matches

# ==== Save Results ====
def save_results(matches_df, index=None, matches_file="entity_matches.csv", index_file="entity_matching_index.faiss"):
    matches_df.to_csv(matches_file, index=False)
    print(f"\nSaved matches to {matches_file}.")
    if index:
        faiss.write_index(faiss.index_gpu_to_cpu(index), index_file)
        print(f"Saved FAISS index to {index_file}.")

# ==== Main Pipeline for Abt-Buy ====
# def main():
#     # File paths
#     tb1_path = os.path.join("Abt-Buy", "Abt.csv")
#     tb2_path = os.path.join("Abt-Buy", "Buy.csv")
#     ground_truth_path = os.path.join("Abt-Buy", "abt_buy_perfectMapping.csv")

#     # Load and preprocess data
#     table1, table2 = load_data(tb1_path, tb2_path)
#     table1_columns = ["name"]
#     table2_columns = ["name"]
#     table1, table2 = preprocess_tables(table1, table2, table1_columns, table2_columns)

#     # Load ground truth
#     ground_truth = pd.read_csv(ground_truth_path)
#     ground_truth_set = set(zip(ground_truth['idAbt'], ground_truth['idBuy']))

#     model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

#     start_time = time.time()
#     embeddings1 = encode_texts(model, table1['processed'].tolist())
#     embeddings2 = encode_texts(model, table2['processed'].tolist())
#     embeddings1 = normalize_embeddings(embeddings1)
#     embeddings2 = normalize_embeddings(embeddings2)
#     embedding_time = time.time() - start_time

#     k_values = [1, 5, 10, 15, 20, 25, 30]
#     results = []

#     for k in k_values:
#         start_time = time.time()
#         index = build_faiss_index(embeddings2)
#         build_time = time.time() - start_time

#         start_time = time.time()
#         matches = find_matches(embeddings1, table1, table2, embeddings2, index=index, top_k=k)
#         search_time = time.time() - start_time

#         matches_df = pd.DataFrame(matches)
#         predicted_set = set(zip(matches_df['left_id'], matches_df['right_id']))
#         true_positives = predicted_set & ground_truth_set

#         recall = len(true_positives) / len(ground_truth_set)
#         precision = len(true_positives) / len(predicted_set) if len(predicted_set) > 0 else 0
#         f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#         results.append({
#             'k': k,
#             'recall': recall,
#             'precision': precision,
#             'f1_score': f1_score,
#             'embedding_time': embedding_time,
#             'build_time': build_time,
#             'search_time': search_time,
#             'total_pairs': len(predicted_set),
#             'true_positives': len(true_positives)
#         })

#     results_df = pd.DataFrame(results)
#     print("\nEvaluation Results:")
#     print(results_df.to_string(index=False))
#     results_df.to_csv("apt-buy_blocking_evaluation_gpu_results.csv", index=False)

# if __name__ == "__main__":
#     main()



def main():
    # File paths
    tb1_path = os.path.join("files", "amazon.csv")
    tb2_path = os.path.join("files", "best_buy.csv")
    ground_truth_path = os.path.join("files", "labeled_data.csv")

    # Load and preprocess data
    table1, table2 = load_data(tb1_path, tb2_path)
    table1_columns = ["Brand", "Name"]
    table2_columns = ["Brand", "Name"]
    table1, table2 = preprocess_tables(table1, table2, table1_columns, table2_columns)

    # Load ground truth (with special handling for header row)
    ground_truth_df = pd.read_csv(ground_truth_path, skiprows=5)
    ground_truth_matches = set(zip(
        ground_truth_df.loc[ground_truth_df['gold'] == 1, 'ltable.ID'],
        ground_truth_df.loc[ground_truth_df['gold'] == 1, 'rtable.ID']
    ))
    n_ltable = ground_truth_df['ltable.ID'].nunique()
    n_rtable = ground_truth_df['rtable.ID'].nunique()
    total_possible_pairs = n_ltable * n_rtable

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    # Time embedding generation
    start_time = time.time()
    embeddings1 = encode_texts(model, table1['processed'].tolist())
    embeddings2 = encode_texts(model, table2['processed'].tolist())
    embeddings1 = normalize_embeddings(embeddings1)
    embeddings2 = normalize_embeddings(embeddings2)
    embedding_time = time.time() - start_time

    # Test different k values
    k_values = [1, 5, 10, 15, 20, 25, 30]
    results = []

    for k in k_values:
        # Time index building
        start_time = time.time()
        index = build_faiss_index(embeddings2)
        build_time = time.time() - start_time

        # Time matching
        start_time = time.time()
        matches = find_matches(embeddings1, table1, table2, embeddings2, index=index, top_k=k)
        search_time = time.time() - start_time

        matches_df = pd.DataFrame(matches)
        candidate_set = set(zip(matches_df['left_id'], matches_df['right_id']))

        # Compute metrics
        true_positives = ground_truth_matches.intersection(candidate_set)
        recall = len(true_positives) / len(ground_truth_matches) if len(ground_truth_matches) > 0 else 0.0
        precision = len(true_positives) / len(candidate_set) if len(candidate_set) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        reduction_ratio = 1 - (len(candidate_set) / total_possible_pairs)

        results.append({
            'k': k,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            'reduction_ratio': reduction_ratio,
            'embedding_time': embedding_time,
            'build_time': build_time,
            'search_time': search_time,
            'total_pairs': len(candidate_set),
            'true_positives': len(true_positives)
        })

        # Show sample matches for k=30 (original value)
        if k == 30:
            print("\nSample matches (k=30):")
            print(matches_df.sort_values(by='similarity_score', ascending=False).head())

    # Print results
    results_df = pd.DataFrame(results)
    print(f"\nTotal ground-truth matches: {len(ground_truth_matches)}")
    print("\nEvaluation Results:")
    print(results_df.to_string(index=False, float_format="{:0.4f}".format))

    # Save results
    results_df.to_csv("amazon_bestbuy_evaluation_results.csv", index=False)
    print("\nResults saved to amazon_bestbuy_blocking_evaluation_gpu_results.csv")

if __name__ == "__main__":
    main()