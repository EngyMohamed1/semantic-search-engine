# Smart Semantic Search Engine

A desktop search engine that understands **meaning**, not just keywords. Built with Word2Vec embeddings, fuzzy matching, and real-time query processing over a custom document corpus.

---

## Features

- **Semantic Search** — finds relevant documents based on meaning using Word2Vec cosine similarity
- **Smart Spell Correction** — automatically corrects typos using fuzzy matching against the trained vocabulary
- **Auto-Complete Suggestions** — word completion and phrase detection as you type
- **Exact Match Boosting** — hybrid ranking that combines semantic scores with exact phrase detection
- **Sentence Highlighting** — shows which sentences in each document contain your query
- **Smart Caching** — retrains model only when documents change (MD5 hash detection)
- **Desktop GUI** — built with Tkinter, updates results instantly on every keystroke

---

## How It Works

```
documents/ (txt files)
      ↓
preprocessing.py   → tokenization, stopword removal, sentence splitting
      ↓
train_model.py     → trains Word2Vec, computes document vectors, caches results
      ↓
search_engine.py   → cosine similarity search over document vectors
      ↓
main.py            → GUI, spell correction, autocomplete, ranked results
```

1. Documents are preprocessed and tokenized
2. A Word2Vec model is trained on the corpus (vector size=150, window=10, epochs=40)
3. Each document is represented as the mean of its word vectors
4. At query time: the query vector is compared to all document vectors via cosine similarity
5. Results are re-ranked by combining semantic score + exact phrase match bonus

---

## Project Structure

```
├── main.py             # GUI and search orchestration
├── search_engine.py    # Cosine similarity search
├── train_model.py      # Word2Vec training and document vectorization
├── preprocessing.py    # Text cleaning and tokenization
├── documents/          # Your .txt document corpus (add your own)
├── word2vec_english.model
├── doc_vectors.pkl
├── processed_data.pkl
└── documents_hash.txt  # Used to detect corpus changes
```

---

## Setup

```bash
pip install gensim nltk rapidfuzz
python -m nltk.downloader stopwords punkt
```

Add your `.txt` files to a `documents/` folder, then run:

```bash
python main.py
```

The model trains automatically on first run and re-trains only if documents change.

---

## Tech Stack

- **Python** — core language
- **Gensim (Word2Vec)** — semantic embeddings
- **NLTK** — tokenization and stopword removal
- **RapidFuzz** — fast fuzzy string matching for spell correction
- **Tkinter** — desktop GUI
- **NumPy** — vector operations and cosine similarity
- **Pickle** — model and data caching
