# preprocessing.py
import os
import re
import pickle
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def load_documents(folder="documents"):
    docs = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs[filename] = f.read()
    return docs


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return ' '.join(words), words

def preprocess_and_save():
    docs = load_documents()
    
    processed_sentences = []
    cleaned_docs = {}
    doc_tokens = {}

    for name, text in docs.items():
        cleaned_text, tokens = clean_text(text)
        cleaned_docs[name] = cleaned_text
        doc_tokens[name] = tokens
        
        sentences = sent_tokenize(text.lower())
        for sent in sentences:
            _, sent_tokens = clean_text(sent)
            if len(sent_tokens) > 1:
                processed_sentences.append(sent_tokens)

    with open("processed_data.pkl", "wb") as f:
        pickle.dump({
            "cleaned_docs": cleaned_docs,
            "doc_tokens": doc_tokens,
            "sentences": processed_sentences
        }, f)

    print("Preprocessing done and saved!")
    return processed_sentences, cleaned_docs, doc_tokens