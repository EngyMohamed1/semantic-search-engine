# search_engine.py
from gensim.models import Word2Vec
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

_model = None
_doc_vectors = None

def _load():
    global _model, _doc_vectors
    if _model is None:
        print("Loading model and vectors...", end=" ")
        _model = Word2Vec.load("word2vec_english.model")
        with open("doc_vectors.pkl", "rb") as f:
            _doc_vectors = pickle.load(f)
        print("Done!")

def clean_query(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return words

def search(query, top_k=10):
    _load()
    q_tokens = clean_query(query)
    
    if not q_tokens:
        return [("No meaningful words in query!", 0.0)]
        
    q_vecs = []
    for w in q_tokens:
        if w in _model.wv:
            q_vecs.append(_model.wv[w])
    
    if not q_vecs:
        return [("No matching words found in model!", 0.0)]
        
    query_vec = np.mean(q_vecs, axis=0)
    
    results = []
    for doc, vec in _doc_vectors.items():
        if np.all(vec == 0):
            continue
        sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
        results.append((doc, float(sim)))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]