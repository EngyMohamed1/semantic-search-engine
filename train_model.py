# train_model.py
from gensim.models import Word2Vec
import pickle
import numpy as np
import os
import hashlib
from preprocessing import preprocess_and_save  # <-- مهم جدًا

def get_documents_hash(folder="documents"):
    """يحسب هاش لكل ملفات النصوص عشان نعرف لو اتغيرت"""
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' not found!")
        return "no_folder"
        
    hash_md5 = hashlib.md5()
    filenames = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    
    if not filenames:
        return "empty_folder"
        
    for filename in filenames:
        path = os.path.join(folder, filename)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        hash_md5.update(filename.encode())
    return hash_md5.hexdigest()

def train_and_prepare():
    current_hash = get_documents_hash()
    hash_file = "documents_hash.txt"
    
    should_reprocess = True
    if os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            old_hash = f.read().strip()
        if old_hash == current_hash:
            should_reprocess = False

    # لو مفيش تغيير → نحمل الجاهز بس
    if not should_reprocess and \
       os.path.exists("processed_data.pkl") and \
       os.path.exists("word2vec_english.model") and \
       os.path.exists("doc_vectors.pkl"):
        
        print("No changes detected. Loading existing data and model...")
        with open("processed_data.pkl", "rb") as f:
            data = pickle.load(f)
        sentences = data["sentences"]
        doc_tokens = data["doc_tokens"]
        
    else:
        # لو في تغيير أو أول مرة → كل حاجة من الأول
        print("Documents changed or first run → Reprocessing everything from scratch...")
        sentences, _, doc_tokens = preprocess_and_save()
        
        print("Training Word2Vec model...")
        model = Word2Vec(
            sentences=sentences,
            vector_size=150,
            window=10,
            min_count=1,
            workers=6,
            epochs=40,
            sg=1
        )
        model.save("word2vec_english.model")
        print("Model trained and saved!")

        print("Computing document vectors...")
        doc_vectors = {}
        for doc_name, tokens in doc_tokens.items():
            if not tokens:
                doc_vectors[doc_name] = np.zeros(150)
                continue
            vectors = [model.wv[w] for w in tokens if w in model.wv]
            if vectors:
                doc_vectors[doc_name] = np.mean(vectors, axis=0)
            else:
                doc_vectors[doc_name] = np.zeros(150)

        with open("doc_vectors.pkl", "wb") as f:
            pickle.dump(doc_vectors, f)
        print("Document vectors saved!")

        # حفظ الهاش الجديد
        with open(hash_file, "w", encoding="utf-8") as f:
            f.write(current_hash)
        
        print("All done! New hash saved.")

    # مهم نرجّع القيم دي عشان main.py يستخدمها
    return sentences, doc_tokens

# عشان تقدر تشغل الملف لوحده للتجربة
if __name__ == "__main__":
    train_and_prepare()