# Entity Matching with Scalable Blocking Techniques

This project explores efficient blocking strategies for entity matching in large-scale datasets (e.g., e-commerce product catalogs). We implement and evaluate three approaches:  
1. **Rule-based blocking** (heuristic filters),  
2. **N-gram inverted index with TF-IDF**,  
3. **FAISS-based semantic search with Sentence Transformers**.  

## ⚠️ Important Notes  
- **Datasets**: Due to size constraints, the `.csv` files are not included in this repository. To run the code:  
  - Download the datasets from [source link, if any]  
  - Update file paths in the code (e.g., `data_path = "your/local/path.csv"`).  
- **Dependencies**: If `requirements.txt` misses any packages, install them manually.  

---

## Main Files Overview

### 1. `blocking_abt-buy.py`  `blocking_dblp-scholar.py`
**Purpose**: Implements heuristic blocking using exact/fuzzy string matching on product attributes (name, brand).  
**Key Features**:  
- Normalizes text (lowercase, remove punctuation).  
- Applies Levenshtein distance for fuzzy matching (threshold=80%).  
- Filters junk entries (e.g., short product names).  

---

### 2. `tf_idf.py`  
**Purpose**: Two-stage blocking using character-level n-grams and TF-IDF cosine similarity.  
**Key Features**:  
- Builds an inverted index for fast candidate retrieval.  
- Uses TF-IDF vectors with character n-grams (e.g., n=5).  
- Retains pairs with similarity > 0.8.  

---

### 3. `blocking_with_faiss.ipynb`  `blocking_faiss_gpu.py`
**Purpose**: Leverages Sentence-BERT embeddings and FAISS for GPU-accelerated semantic blocking.  
**Key Features**:  
- Encodes text using `all-MiniLM-L6-v2` embeddings.  
- Builds a FAISS index for approximate nearest neighbor search.  
- Supports CPU/GPU execution (via `faiss-gpu`).  

---

## Setup  
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
