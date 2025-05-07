import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz
from collections import defaultdict
import csv
import time

def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove non-alphanumeric characters, and trim."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def filter_short_records(df: pd.DataFrame, min_words: int = 2) -> pd.DataFrame:
    """
    Filter out records where title or author is too short.
    """
    df = df[df["title"].apply(lambda x: len(normalize_text(x).split()) >= min_words)]
    df = df[df["authors"].apply(lambda x: len(normalize_text(x).split()) >= min_words)]
    return df.reset_index(drop=True)  # Reset the index after filtering

def group_by_title_tokens(df: pd.DataFrame, num_tokens: int = 3) -> defaultdict:
    """
    Group records by the first few tokens of their title for blocking purposes.
    """
    groups = defaultdict(list)
    for i, row in df.iterrows():
        title_tokens = normalize_text(row["title"]).split()[:num_tokens]
        if title_tokens:  # Make sure we have tokens to use
            group_key = " ".join(title_tokens)
            groups[group_key].append(i)  # Store indices of records that share the same group key
    return groups

def fuzzy_text_match(s1: str, s2: str, threshold: int = 80) -> bool:
    """Levenshtein-based fuzzy match above a given threshold."""
    return fuzz.ratio(s1, s2) >= threshold

def load_mapping(filepath: str = "DBLP-Scholar_perfectMapping.csv") -> set:
    """
    Load the perfect mapping of (idDBLP, idScholar) pairs.
    """
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip()
    return set((row["idDBLP"], row["idScholar"]) for _, row in df.iterrows())

def rule_based_blocking_optimized(dblp_data: pd.DataFrame, scholar_data: pd.DataFrame) -> list:
    """
    Perform blocking and matching, after applying preprocessing filters to reduce the number of comparisons.
    """
    # Filter short titles/authors
    dblp_data = filter_short_records(dblp_data)
    scholar_data = filter_short_records(scholar_data)

    # Group by title tokens to reduce the comparisons
    dblp_groups = group_by_title_tokens(dblp_data)
    scholar_groups = group_by_title_tokens(scholar_data)
    
    candidates = []
    
    # Debug information
    print(f"DBLP data shape after filtering: {dblp_data.shape}")
    print(f"Scholar data shape after filtering: {scholar_data.shape}")
    print(f"Number of DBLP groups: {len(dblp_groups)}")
    print(f"Number of Scholar groups: {len(scholar_groups)}")
    
    # Count matching groups
    matching_groups = set(dblp_groups.keys()) & set(scholar_groups.keys())
    print(f"Number of matching groups: {len(matching_groups)}")

    # Perform matching within groups
    for group_key in matching_groups:
        dblp_group = dblp_groups[group_key]
        scholar_group = scholar_groups[group_key]
        
        for dblp_idx in dblp_group:
            if dblp_idx >= len(dblp_data):
                continue  # Skip if index is out of bounds
                
            dblp_row = dblp_data.iloc[dblp_idx]
            dblp_title = normalize_text(dblp_row["title"])
            dblp_authors = normalize_text(dblp_row["authors"])
            dblp_id = dblp_row["id"]

            for scholar_idx in scholar_group:
                if scholar_idx >= len(scholar_data):
                    continue  # Skip if index is out of bounds
                
                scholar_row = scholar_data.iloc[scholar_idx]
                scholar_title = normalize_text(scholar_row["title"])
                scholar_authors = normalize_text(scholar_row["authors"])
                scholar_id = scholar_row["id"]

                # Fuzzy matching on title and authors
                title_fuzzy_match = fuzzy_text_match(dblp_title, scholar_title, threshold=80)
                author_fuzzy_match = fuzzy_text_match(dblp_authors, scholar_authors, threshold=70)

                if title_fuzzy_match and author_fuzzy_match:
                    candidates.append((dblp_id, scholar_id))

    return candidates

def evaluate_blocking(predicted: list, true_pairs: set) -> None:
    """
    Compute and print precision, recall, and F1 score.
    """
    pred_set = set(predicted)
    tp = len(pred_set & true_pairs)
    fn = len(true_pairs - pred_set)
    fp = len(pred_set - true_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print("\n=== Blocking Evaluation ===")
    print(f"Total candidates generated: {len(predicted)}")
    print(f"True matches in gold set : {len(true_pairs)}")
    print(f"True positives found      : {tp}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

def main():
    start_time = time.time()  # Start timing the process

    try:
        # Load the datasets
        print("Loading datasets...")
        dblp_df = pd.read_csv("DBLP1.csv", quoting=csv.QUOTE_ALL, encoding="ISO-8859-1")
        scholar_df = pd.read_csv("Scholar.csv", quoting=csv.QUOTE_ALL, encoding="ISO-8859-1")
        
        # Clean up column names to remove any BOM or quotes
        dblp_df.columns = [col.strip().strip('"').strip("'").replace('ï»¿', '') for col in dblp_df.columns]
        scholar_df.columns = [col.strip().strip('"').strip("'").replace('ï»¿', '') for col in scholar_df.columns]
        
        # Specifically handle the Scholar dataset's 'id' column
        if '"id' in scholar_df.columns:
            scholar_df = scholar_df.rename(columns={'"id': 'id'})
        elif 'ï»¿"id"' in scholar_df.columns:
            scholar_df = scholar_df.rename(columns={'ï»¿"id"': 'id'})
        elif 'ï»¿id' in scholar_df.columns:
            scholar_df = scholar_df.rename(columns={'ï»¿id': 'id'})
        
        print(f"DBLP dataset shape: {dblp_df.shape}")
        print(f"Scholar dataset shape: {scholar_df.shape}")
        print(f"DBLP columns: {dblp_df.columns.tolist()}")
        print(f"Scholar columns: {scholar_df.columns.tolist()}")
        
        # Check if required columns exist
        required_columns = ["id", "title", "authors"]
        for col in required_columns:
            if col not in dblp_df.columns:
                # Try alternative column names with quotes or different formats
                possible_alternatives = [f'"{col}"', f'"{col}', f'{col}"', f'ï»¿{col}', f'ï»¿"{col}"']
                found = False
                for alt in possible_alternatives:
                    if alt in dblp_df.columns:
                        print(f"Renaming column '{alt}' to '{col}' in DBLP dataset")
                        dblp_df = dblp_df.rename(columns={alt: col})
                        found = True
                        break
                if not found:
                    print(f"Error: Column '{col}' missing from DBLP dataset")
                    print(f"Available columns: {dblp_df.columns.tolist()}")
                    return
            
            if col not in scholar_df.columns:
                # Try alternative column names with quotes or different formats
                possible_alternatives = [f'"{col}"', f'"{col}', f'{col}"', f'ï»¿{col}', f'ï»¿"{col}"']
                found = False
                for alt in possible_alternatives:
                    if alt in scholar_df.columns:
                        print(f"Renaming column '{alt}' to '{col}' in Scholar dataset")
                        scholar_df = scholar_df.rename(columns={alt: col})
                        found = True
                        break
                if not found:
                    print(f"Error: Column '{col}' missing from Scholar dataset")
                    print(f"Available columns: {scholar_df.columns.tolist()}")
                    return
        
        # Load the gold standard mappings
        print("Loading gold standard mappings...")
        gold_pairs = load_mapping("DBLP-Scholar_perfectMapping.csv")
        print(f"Number of gold standard pairs: {len(gold_pairs)}")
        
        # Perform blocking and matching
        print("Performing blocking and matching...")
        candidates = rule_based_blocking_optimized(dblp_df, scholar_df)

        if candidates:
            print("First 5 candidate pairs:", candidates[:5])
        else:
            print("No candidate pairs found!")

        # Evaluate the results
        evaluate_blocking(candidates, gold_pairs)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Calculate and print the total time taken
        end_time = time.time()  # End timing
        print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()