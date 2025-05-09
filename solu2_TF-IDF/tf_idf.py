import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import heapq
from pathlib import Path


def preprocess_data(df):
    """Combine all columns into a single string and preprocess"""
    df['combined'] = df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df['processed'] = df['combined'].str.lower().replace(r'[^a-z0-9\s]', '', regex=True)
    return df

def generate_ngrams(text, n=3):
    """Generate n-grams from text"""
    return [text[i:i+n] for i in range(len(text)-n+1)]

def build_inverted_index(df, ngram_range=(3, 3)):
    """Build inverted index from dataset"""
    inverted_index = defaultdict(set)
    for idx, row in df.iterrows():
        ngrams = generate_ngrams(row['processed'], n=ngram_range[0])
        for ngram in ngrams:
            inverted_index[ngram].add(idx)
    return inverted_index

def find_candidates(source_df, target_df, inverted_index, top_k=5, ngram_range=(3, 3)):
    """Find candidate matches using inverted index and TF-IDF similarity"""
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range)
    all_text = pd.concat([target_df['processed'], source_df['processed']])
    vectorizer.fit(all_text)
    
    tfidf_target = vectorizer.transform(target_df['processed'])
    tfidf_source = vectorizer.transform(source_df['processed'])
    
    candidates = {}
    
    for source_idx, source_row in source_df.iterrows():
        source_ngrams = generate_ngrams(source_row['processed'], n=ngram_range[0])
        candidate_ids = set()
        for ngram in source_ngrams:
            candidate_ids.update(inverted_index.get(ngram, set()))
        
        if not candidate_ids:
            continue
            
        similarities = cosine_similarity(
            tfidf_source[source_idx],
            tfidf_target[list(candidate_ids)]
        ).flatten()
        
        top_matches = heapq.nlargest(
            top_k,
            zip(similarities, candidate_ids),
            key=lambda x: x[0]
        )
        
        candidates[source_idx] = [(score, target_id) 
                                for score, target_id in top_matches 
                                if score > 0]
    
    return candidates


def create_candidate_df(larger_df, smaller_df, candidate_matches):
    """Create a DataFrame with all columns from both datasets for candidates"""
    candidate_list = []
    
    for large_idx, matches in candidate_matches.items():
        if not matches:
            continue
        
        large_record = larger_df.loc[large_idx].to_dict()
        for score, small_idx in matches:
            small_record = smaller_df.loc[small_idx].to_dict()
            candidate_entry = {
                'source_index': large_idx,
                'target_index': small_idx,
                'similarity_score': score
            }
            for col, val in large_record.items():
                candidate_entry[f'source_{col}'] = val
                
            for col, val in small_record.items():
                candidate_entry[f'target_{col}'] = val
            
            candidate_list.append(candidate_entry)
    
    return pd.DataFrame(candidate_list)

# Main execution flowå
if __name__ == "__main__":
    larger_df = pd.read_csv(Path('/Users/vijaypatel/Desktop/cs774/cs774/data/abt/Abt.csv'))
    smaller_df = pd.read_csv(Path('/Users/vijaypatel/Desktop/cs774/cs774/data/abt/Buy.csv'))
    
    # if len(df1) > len(df2):
    #     larger_df, smaller_df = df1, df2
    #     print(f"Dataset1 is larger ({len(df1)} records), Dataset2 is smaller ({len(df2)} records)")
    # else:
    #     larger_df, smaller_df = df2, df1
    #     print(f"Dataset2 is larger ({len(df2)} records), Dataset1 is smaller ({len(df1)} records)")
    
    larger_df = preprocess_data(larger_df)
    smaller_df = preprocess_data(smaller_df)
    
    inverted_index = build_inverted_index(smaller_df, ngram_range=(3, 5))
    
    candidate_matches = find_candidates(larger_df, smaller_df, inverted_index, top_k=36)
    

    candidate_df = create_candidate_df(larger_df, smaller_df, candidate_matches)
    
    candidate_df.to_csv(Path('/Users/vijaypatel/Desktop/cs774/cs774/results/blocking_results.csv'), index=False)
    print("Candidate matches saved to candidate_matches.csv")

