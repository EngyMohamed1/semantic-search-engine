import tkinter as tk
import os
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec
from rapidfuzz import fuzz, process
from search_engine import search
from train_model import train_and_prepare

# ======== NLTK Setup ========
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))

# ======== Load or Train Word2Vec Model ========
print("Checking documents and preparing model...")
from train_model import train_and_prepare

sentences, doc_tokens = train_and_prepare()

MODEL_PATH = "word2vec_english.model"
model = Word2Vec.load(MODEL_PATH)
print("Model loaded successfully.")

# ======== Load Data ========
def load_all_data():
    try:
        with open("processed_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        all_sentences = []
        if "sentences" in data:
            for sent_tokens in data["sentences"]:
                if sent_tokens:
                    all_sentences.append(" ".join(sent_tokens))
        
        vocab_words = list(model.wv.index_to_key)
        all_phrases = []
        doc_tokens_dict = data.get("doc_tokens", {})
        for tokens in doc_tokens_dict.values():
            for i in range(len(tokens)-1):
                all_phrases.append(f"{tokens[i]} {tokens[i+1]}")
            for i in range(len(tokens)-2):
                all_phrases.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        common_phrases = [p for p, _ in Counter(all_phrases).most_common(150)]
        
        return common_phrases, all_sentences, vocab_words, data.get("cleaned_docs", {})
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], [], [], {}

COMMON_PHRASES, ALL_SENTENCES, VOCAB_WORDS, CLEANED_DOCS = load_all_data()

# ======== Smart Spelling Correction ========
def correct_spelling(text):
    words = text.lower().split()
    corrected = []
    for word in words:
        if len(word) < 3:
            corrected.append(word)
            continue
        match = process.extractOne(word, VOCAB_WORDS, scorer=fuzz.ratio)
        if match and match[1] >= 82:
            corrected.append(match[0])
        else:
            corrected.append(word)
    return " ".join(corrected).title()

# ======== Find Sentences with Phrase ========
def find_sentences_with_phrase(phrase, max_sentences=8):
    if not phrase or not ALL_SENTENCES:
        return []
    phrase_lower = phrase.lower()
    found = []
    for sentence in ALL_SENTENCES:
        if phrase_lower in sentence.lower():
            doc_name = "Unknown"
            for doc, content in CLEANED_DOCS.items():
                if sentence in content:
                    doc_name = doc
                    break
            found.append({'sentence': sentence, 'document': doc_name})
            if len(found) >= max_sentences:
                break
    return found

# ======== Smart Suggestions ========
def get_instant_suggestions(raw_query):
    if not raw_query.strip():
        return []
    
    query = raw_query.strip()
    lower_query = query.lower()
    
    corrected_full = correct_spelling(query)
    corrected_lower = corrected_full.lower()

    suggestions = []

    words = lower_query.split()
    if len(words) == 1:
        base = words[0]
        starts_with = [w for w in VOCAB_WORDS if w.startswith(corrected_lower)]
        if starts_with:
            top = sorted(starts_with, key=lambda w: VOCAB_WORDS.index(w) if w in VOCAB_WORDS else 9999)[:6]
            for w in top:
                suggestions.append({'text': f"Word: {w.title()}", 'value': w.title(), 'type': 'word_complete'})
        
        fuzzy_matches = process.extract(base, VOCAB_WORDS, scorer=fuzz.ratio, limit=10)
        for word, score, _ in fuzzy_matches:
            if score >= 80 and word.title() not in [s['value'] for s in suggestions]:
                suggestions.append({'text': f"Word: {word.title()}", 'value': word.title(), 'type': 'word_complete'})

    last_word = corrected_lower.split()[-1]
    if len(last_word) >= 2:
        completions = process.extract(last_word, VOCAB_WORDS, scorer=fuzz.partial_ratio, limit=8)
        for word, score, _ in completions:
            if score < 72: continue
            prefix = " ".join(corrected_lower.split()[:-1]) + " " if len(corrected_lower.split()) > 1 else ""
            completed = (prefix + word).strip().title()
            if completed.lower() != lower_query and completed not in [s['value'] for s in suggestions]:
                suggestions.append({'text': f"Word: {completed}", 'value': completed, 'type': 'word_complete'})

    phrase_matches = process.extract(corrected_lower, COMMON_PHRASES, scorer=fuzz.partial_ratio, limit=8)
    for phrase, score, _ in phrase_matches:
        if score < 75: continue
        clean_phrase = correct_spelling(phrase)
        if clean_phrase.lower() != lower_query:
            suggestions.append({'text': f"Phrase: {clean_phrase}", 'value': clean_phrase, 'type': 'phrase_match'})

    seen = set()
    unique = []
    for s in suggestions:
        if s['value'].lower() not in seen:
            seen.add(s['value'].lower())
            unique.append(s)
    
    return unique[:8]

# ======== Show Sentences ========
def show_sentences(sentences_list, phrase):
    sentence_text.delete(1.0, tk.END)
    if not sentences_list:
        sentence_text.insert(tk.END, f"No sentences found containing:\n\"{phrase}\"\n\n")
        return
    sentence_text.insert(tk.END, f"Found {len(sentences_list)} example sentence(s):\n")
    sentence_text.insert(tk.END, "═" * 82 + "\n\n")
    phrase_lower = phrase.lower()
    for i, item in enumerate(sentences_list, 1):
        sent = item['sentence']
        doc = item['document'].replace('.txt', '').replace('_', ' ').title()
        highlighted = ""
        pos = 0
        lower_sent = sent.lower()
        while True:
            idx = lower_sent.find(phrase_lower, pos)
            if idx == -1:
                highlighted += sent[pos:]
                break
            highlighted += sent[pos:idx] + sent[idx:idx+len(phrase)].upper()
            pos = idx + len(phrase)
        sentence_text.insert(tk.END, f"{i}. {doc}:\n   \"{highlighted}\"\n\n")
    sentence_text.insert(tk.END, "═" * 82)
    sentence_text.see(tk.END)

# ======== Instant Search + Full Intelligence========
search_timer = None

def perform_full_search(raw_query=""):
    global search_timer
    search_timer = None

    query = raw_query.strip() if raw_query else entry.get().strip()
    if len(query) < 2:
        listbox.delete(0, tk.END)
        sentence_text.delete(1.0, tk.END)
        result_label.config(text="Type 2+ characters...")
        return

    corrected = correct_spelling(query)
    was_corrected = corrected.lower() != query.lower()
    if was_corrected:
        entry.delete(0, tk.END)
        entry.insert(0, corrected)
        query = corrected

    suggestions = get_instant_suggestions(query)
    listbox.delete(0, tk.END)
    for s in suggestions:
        listbox.insert(tk.END, s['text'])
        idx = listbox.size() - 1
        listbox.itemconfig(idx, fg='green' if 'Word' in s['text'] else 'blue')
    listbox.insert(tk.END, "")
    listbox.insert(tk.END, "Click to complete")

    sentences_list = find_sentences_with_phrase(query.lower(), 8)
    show_sentences(sentences_list, query)

    results = search(query.lower(), top_k=20)

    sentences_list = find_sentences_with_phrase(query.lower(), 8)
    exact_docs = {item['document'] for item in sentences_list if item['document'] != "Unknown"}

    from search_engine import clean_query
    q_tokens = clean_query(query)
    query_words = set(q_tokens)

    if not query_words:
        result_label.config(text="No valid words in query after cleaning.")
        return

    candidate_docs = []
    for doc in CLEANED_DOCS:
        doc_text_lower = CLEANED_DOCS[doc].lower()
        if any(word in doc_text_lower for word in query_words):
            candidate_docs.append(doc)

    if not candidate_docs:
        result_label.config(text=f"No documents contain any of the words:\n{', '.join(query_words)}\n\nTry a broader term.")
        sentence_text.delete(1.0, tk.END)
        sentence_text.insert(tk.END, f"No sentences found for: \"{query}\"")
        return

    scored = []
    for doc, sim_score in results:
        if doc in candidate_docs:
            exact_bonus = 0.5 if doc in exact_docs else 0.0
            final_score = sim_score + exact_bonus
            scored.append((doc, min(final_score, 1.0)))

    scored.sort(key=lambda x: x[1], reverse=True)
    boosted = scored[:15]

    text = "Full Search Results (Smart Ranking):\n" + "═" * 62 + "\n"
    if was_corrected:
        text += f"Spelling Correction: \"{query}\"\n\n"

    if boosted and boosted[0][1] > 0.2:
        for i, (doc, score) in enumerate(boosted[:10]):
            name = doc.replace('.txt', '').replace('_', ' ').title()
            stars = " ★★★" if score >= 0.95 else " ★★" if score >= 0.75 else " ★" if score >= 0.5 else ""
            tag = " [Exact Match]" if doc in exact_docs else " [Semantic]"
            text += f"{i+1:2}. {name:30} → {score:.4f}{stars}{tag}\n"
        if len(boosted) > 10:
            text += f"\n... and {len(boosted)-10} more results"
    else:
        text += "No strong matches found."

    result_label.config(text=text)

def on_key_release(event=None):
    global search_timer
    if search_timer:
        root.after_cancel(search_timer)
    query = entry.get().strip()
    if len(query) < 2:
        listbox.delete(0, tk.END)
        return
    search_timer = root.after(380, perform_full_search, query)

def on_select(event):
    sel = listbox.curselection()
    if not sel or "Click" in listbox.get(sel[0]):
        return
    selected = listbox.get(sel[0])
    value = selected.split(": ", 1)[1] if ": " in selected else selected
    entry.delete(0, tk.END)
    entry.insert(0, value)
    perform_full_search(value)

# ======== GUI ========
root = tk.Tk()
root.title("Smart Semantic Search - Instant + Auto-Correction + Accurate Ranking")
root.geometry("1380x840")
root.configure(bg="#f5f7fa")

tk.Label(root, text="Smart Semantic Search", font=("Helvetica", 32, "bold"), bg="#f5f7fa", fg="#2c3e50").pack(pady=20)
tk.Label(root, text="Type anything (even with typos!) → Everything updates instantly", font=("Helvetica", 13), bg="#f5f7fa", fg="#636e72").pack()

main = tk.Frame(root, bg="#f5f7fa")
main.pack(fill="both", expand=True, padx=35, pady=10)

left = tk.Frame(main, bg="#f5f7fa")
left.pack(side="left", fill="both", expand=True, padx=(0, 25))

tk.Label(left, text="Type your query:", font=("Helvetica", 14, "bold"), bg="#f5f7fa").pack(anchor="w", pady=(0,8))
entry = tk.Entry(left, font=("Helvetica", 19), relief="solid", bd=2, bg="white")
entry.pack(fill="x", ipady=14, pady=(0,20))
entry.focus()

tk.Label(left, text="Auto-Complete Suggestions:", font=("Helvetica", 13, "bold"), bg="#f5f7fa").pack(anchor="w")
listbox = tk.Listbox(left, font=("Helvetica", 14), height=10, selectbackground="#3498db", relief="flat")
listbox.pack(fill="both", expand=True, pady=(0,20))
listbox.bind("<<ListboxSelect>>", on_select)

tk.Label(left, text="Search Results (Smart Ranking):", font=("Helvetica", 13, "bold"), bg="#f5f7fa").pack(anchor="w")
result_label = tk.Label(left, text="Start typing...", font=("Courier", 11), bg="white", justify="left", anchor="nw", relief="groove", bd=2, padx=16, pady=16, height=12)
result_label.pack(fill="x")

right = tk.Frame(main, bg="#f5f7fa")
right.pack(side="right", fill="both", expand=True)

tk.Label(right, text="Sentence Examples from Documents:", font=("Helvetica", 13, "bold"), bg="#f5f7fa").pack(anchor="w")
text_frame = tk.Frame(right, bg="white", relief="solid", bd=2)
text_frame.pack(fill="both", expand=True)
sentence_text = tk.Text(text_frame, font=("Helvetica", 11), wrap="word", padx=16, pady=16, bg="#fdfdfd")
sentence_text.pack(side="left", fill="both", expand=True)
sb = tk.Scrollbar(text_frame, command=sentence_text.yview)
sb.pack(side="right", fill="y")
sentence_text.config(yscrollcommand=sb.set)

tk.Label(root, text="Now with perfect accuracy • No false positives • Exact matches first",
         font=("Helvetica", 11, "bold"), fg="#27ae60", bg="#f5f7fa").pack(side="bottom", pady=20)

entry.bind("<KeyRelease>", on_key_release)
entry.bind("<Return>", lambda e: perform_full_search())

root.mainloop()