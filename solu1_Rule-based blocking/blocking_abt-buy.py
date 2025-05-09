import pandas as pd
import numpy as np
import re
import csv
import time
from typing import List, Tuple, Set, Dict
from collections import defaultdict

# Try importing fuzzywuzzy, with a helpful error message if not installed
try:
    from fuzzywuzzy import fuzz
except ImportError:
    print("Error: The 'fuzzywuzzy' package is required but not installed.")
    print("Please install it using: pip install fuzzywuzzy python-Levenshtein")
    exit(1)

def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove non-alphanumeric characters, and trim."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def filter_short_records(df: pd.DataFrame, min_words: int = 2) -> pd.DataFrame:
    """Filter out records where name is too short."""
    df = df[df["name"].apply(lambda x: len(normalize_text(x).split()) >= min_words)]
    return df.reset_index(drop=True)

def extract_key_tokens(text: str, num_tokens: int = 3) -> List[str]:
    """Extract important tokens from text by removing common stop words."""
    # Simple list of stop words that often appear in product descriptions
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of'}
    
    tokens = normalize_text(text).split()
    filtered_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # If we don't have enough tokens after filtering, use original tokens
    if len(filtered_tokens) < num_tokens:
        return tokens[:num_tokens]
    return filtered_tokens[:num_tokens]

def group_by_name_tokens(df: pd.DataFrame, num_tokens: int = 3) -> Dict[str, List[int]]:
    """Group records by their key name tokens for blocking."""
    groups = defaultdict(list)
    
    for i, row in df.iterrows():
        # Extract key tokens from the product name
        key_tokens = extract_key_tokens(row["name"], num_tokens)
        
        if key_tokens:
            # Create multiple group keys for better recall
            for token in key_tokens:
                groups[token].append(i)
    
    return groups

def fuzzy_text_match(s1: str, s2: str, threshold: int = 85) -> bool:
    """Levenshtein-based fuzzy match above a given threshold."""
    return fuzz.ratio(s1, s2) >= threshold

def load_mapping(filepath: str = "abt_buy_perfectMapping.csv") -> Set[Tuple[int, int]]:
    """Load the perfect mapping of (idAbt, idBuy) pairs."""
    try:
        df = pd.read_csv(filepath, encoding="ISO-8859-1")
        df.columns = df.columns.str.strip()
        return set((row["idAbt"], row["idBuy"]) for _, row in df.iterrows())
    except Exception as e:
        print(f"Error loading mapping file {filepath}: {str(e)}")
        return set()

def optimized_blocking(abt_data: pd.DataFrame, buy_data: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    Perform blocking using token-based groups and fuzzy matching.
    This reduces the comparison space significantly.
    """
    start_time = time.time()
    print("Starting improved blocking procedure...")
    
    # Filter short product names
    abt_filtered = filter_short_records(abt_data)
    buy_filtered = filter_short_records(buy_data)
    
    print(f"After filtering: Abt {abt_filtered.shape}, Buy {buy_filtered.shape}")
    
    # Group by name tokens to reduce comparisons
    abt_groups = group_by_name_tokens(abt_filtered)
    buy_groups = group_by_name_tokens(buy_filtered)
    
    print(f"Created {len(abt_groups)} token groups for Abt")
    print(f"Created {len(buy_groups)} token groups for Buy")
    
    # Find overlapping tokens between datasets
    common_tokens = set(abt_groups.keys()) & set(buy_groups.keys())
    print(f"Found {len(common_tokens)} common tokens between datasets")
    
    # Progress tracking
    total_comparisons = sum(len(abt_groups[token]) * len(buy_groups[token]) 
                           for token in common_tokens)
    print(f"Will perform approximately {total_comparisons} comparisons")
    progress_interval = max(1, total_comparisons // 20)  # Report progress at 5% intervals
    
    # Store candidates in a set to avoid duplicates
    candidates = set()
    comparisons_done = 0
    last_report = 0
    
    # Match within token groups
    for token in common_tokens:
        abt_indices = abt_groups[token]
        buy_indices = buy_groups[token]
        
        for abt_idx in abt_indices:
            if abt_idx >= len(abt_filtered):
                continue
                
            abt_row = abt_filtered.iloc[abt_idx]
            abt_name = normalize_text(abt_row["name"])
            abt_descr = normalize_text(abt_row["description"]) if "description" in abt_filtered.columns else ""
            abt_id = abt_row["id"]
            
            for buy_idx in buy_indices:
                if buy_idx >= len(buy_filtered):
                    continue
                
                comparisons_done += 1
                if comparisons_done - last_report >= progress_interval:
                    print(f"Progress: {comparisons_done/total_comparisons:.1%} complete")
                    last_report = comparisons_done
                
                buy_row = buy_filtered.iloc[buy_idx]
                buy_name = normalize_text(buy_row["name"])
                buy_descr = normalize_text(buy_row["description"]) if "description" in buy_filtered.columns else ""
                buy_id = buy_row["id"]
                
                # Apply multiple matching criteria
                
                # 1. Fuzzy name match - primary criterion
                name_match = fuzzy_text_match(abt_name, buy_name, threshold=80)
                
                # If names match somewhat, check for additional evidence
                if name_match:
                    # Add this pair as a candidate
                    candidates.add((abt_id, buy_id))
                    continue
                
                # 2. If names don't match well enough but descriptions have strong overlap
                if abt_descr and buy_descr:
                    # Check if model numbers or specific identifiers are shared
                    # Extract alphanumeric tokens that might be model numbers
                    abt_tokens = set(re.findall(r'[a-z0-9]{5,}', abt_descr))
                    buy_tokens = set(re.findall(r'[a-z0-9]{5,}', buy_descr))
                    
                    # If they share specific identifiers (likely model numbers)
                    shared_identifiers = abt_tokens & buy_tokens
                    if shared_identifiers and len(abt_name.split()) > 2 and len(buy_name.split()) > 2:
                        candidates.add((abt_id, buy_id))
    
    end_time = time.time()
    print(f"Blocking completed in {end_time - start_time:.2f} seconds")
    print(f"Generated {len(candidates)} candidate pairs")
    
    return list(candidates)

def evaluate_blocking(predicted: List[Tuple[int, int]], true_pairs: Set[Tuple[int, int]]) -> None:
    """Compute and print precision, recall, and F1 score."""
    pred_set = set(predicted)
    tp = len(pred_set & true_pairs)
    fn = len(true_pairs - pred_set)
    fp = len(pred_set - true_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    
    print("\n=== Blocking Evaluation ===")
    print(f"Total candidates generated: {len(predicted)}")
    print(f"True matches in gold set  : {len(true_pairs)}")
    print(f"True positives found      : {tp}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    # Additional insights
    if len(true_pairs) > 0:
        print(f"\nMissed {len(true_pairs) - tp} out of {len(true_pairs)} true matches ({(1-recall)*100:.1f}%)")
    
    if fp > 0:
        print(f"Generated {fp} false positives ({fp/len(predicted)*100:.1f}% of candidates)")

def main():
    start_time = time.time()
    
    try:
        # Load the datasets
        print("Loading datasets...")
        abt_df = pd.read_csv("Abt.csv", quoting=csv.QUOTE_ALL, encoding="ISO-8859-1")
        buy_df = pd.read_csv("Buy.csv", quoting=csv.QUOTE_ALL, encoding="ISO-8859-1")
        
        print(f"Abt dataset: {abt_df.shape[0]} records, columns: {list(abt_df.columns)}")
        print(f"Buy dataset: {buy_df.shape[0]} records, columns: {list(buy_df.columns)}")
        
        # Check for required columns
        required_cols = ["id", "name", "description", "price"]
        for col in required_cols:
            if col not in abt_df.columns:
                print(f"Warning: Column '{col}' missing from Abt dataset")
            if col not in buy_df.columns:
                print(f"Warning: Column '{col}' missing from Buy dataset")
        
        # Load the gold standard mappings
        true_pairs = load_mapping("abt_buy_perfectMapping.csv")
        print(f"Loaded {len(true_pairs)} gold standard matching pairs")
        
        # Perform the improved blocking
        candidates = optimized_blocking(abt_df, buy_df)
        
        # If no candidates were found, something is wrong
        if not candidates:
            print("Warning: No candidate pairs were generated!")
        else:
            print(f"First 5 candidate pairs: {candidates[:5]}")
        
        # Evaluate blocking results
        evaluate_blocking(candidates, true_pairs)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        end_time = time.time()
        print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()