"""
TextLens v0.1.1 - egyszerű szöveges keresőmotor.

Futtatás:
    python app.py            # TF-IDF alapú keresés (alapértelmezett)
    python app.py --log      # v0.1-es logaritmikus súlyozású keresés

Csak ezt a fájlt kell futtatnod, minden más (előfeldolgozás, vektorizáció,
keresés) innen, ebből a fájlból van meghívva.
"""
import math
import os
import re
import string
import sys
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")

STOPWORDS = {"a", "an", "the", "is", "of", "to", "in", "and", "for"}


# ============================================================
# 1. SZÖVEGTISZTÍTÁS ÉS TOKENIZÁCIÓ
# ============================================================

def clean_text(text):
    """Kisbetűsítés, írásjelek és felesleges whitespace eltávolítása."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    """Szavakra bontás szóköz mentén."""
    return text.split()


def remove_stopwords(tokens):
    """Kiszűri a gyakori, keveset mondó szavakat."""
    return [t for t in tokens if t not in STOPWORDS]


def preprocess(text):
    """Teljes lánc: tisztítás -> tokenizáció -> stopwordök kiszűrése."""
    return remove_stopwords(tokenize(clean_text(text)))


# ============================================================
# 2. LOGARITMIKUS SÚLYOZÁS (v0.1)
# ============================================================

def load_documents_as_counters(folder):
    """{fájlnév: Counter(szó -> gyakoriság)}"""
    index = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".txt"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                index[name] = Counter(preprocess(f.read()))
    return index


def word_weight(frequency):
    """weight = 1 + log10(frequency)"""
    return 1 + math.log10(frequency)


def score_document_log(query_tokens, word_counts):
    """A keresett szavak súlyainak összege egy dokumentumban."""
    score = 0.0
    for word in query_tokens:
        if word_counts[word] > 0:
            score += word_weight(word_counts[word])
    return score


def search_log(query, index):
    """Rangsorolt találatok logaritmikus súlyozással."""
    query_tokens = list(dict.fromkeys(preprocess(query)))
    results = []
    for name, word_counts in index.items():
        score = score_document_log(query_tokens, word_counts)
        if score > 0:
            results.append((name, score))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


# ============================================================
# 3. VEKTORIZÁCIÓ (indexelés, one-hot, Bag of Words)
# ============================================================

def build_vocab(token_lists):
    """{szó: index}, ábécésorrendben."""
    words = sorted({w for tokens in token_lists for w in tokens})
    return {word: i for i, word in enumerate(words)}


def index_text(text, vocab):
    """Szöveg -> indexlista. Ismeretlen szó: -1."""
    return [vocab.get(word, -1) for word in preprocess(text)]


def one_hot(word, vocab):
    """Egy szó = csupa 0, egyetlen 1-es a szó helyén."""
    vector = [0] * len(vocab)
    if word in vocab:
        vector[vocab[word]] = 1
    return vector


def bag_of_words(texts):
    """(count_vect, X_count): analyzer=preprocess, így ugyanaz a tisztítás
    és stopword-szűrés, mint fentebb."""
    count_vect = CountVectorizer(analyzer=preprocess)
    X_count = count_vect.fit_transform(texts)
    return count_vect, X_count


# ============================================================
# 4. TF-IDF ÉS TF-IDF ALAPÚ KERESÉS (v0.1.1)
# ============================================================

def build_tfidf(texts):
    """(count_vect, transformer, X_tfidf) az egész láncon végigmenve."""
    count_vect, X_count = bag_of_words(texts)
    transformer = TfidfTransformer()
    X_tfidf = transformer.fit_transform(X_count)
    return count_vect, transformer, X_tfidf


def load_texts(folder):
    """(fájlnevek, szövegek) ugyanabban a sorrendben."""
    names = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
    texts = [open(os.path.join(folder, n), encoding="utf-8").read() for n in names]
    return names, texts


def build_tfidf_index(folder):
    """Egyszer felépítjük: (fájlnevek, count_vect, X_tfidf)."""
    names, texts = load_texts(folder)
    count_vect, _, X_tfidf = build_tfidf(texts)
    return names, count_vect, X_tfidf


def search_tfidf(query, names, count_vect, X_tfidf):
    """A keresett szavak TF-IDF értékeinek összege dokumentumonként."""
    columns = count_vect.transform([query]).indices
    if len(columns) == 0:
        return []
    scores = X_tfidf[:, columns].sum(axis=1).A1
    results = [(name, float(s)) for name, s in zip(names, scores) if s > 0]
    results.sort(key=lambda r: r[1], reverse=True)
    return results


# ============================================================
# MAIN - csak ezt kell futtatni
# ============================================================

def print_results(results):
    print()
    if not results:
        print("Nem található releváns dokumentum.\n")
        return
    for rank, (name, score) in enumerate(results, start=1):
        print(f"{rank}. {name}")
        print(f"   Relevancia: {score:.2f}\n")


def main():
    use_log = "--log" in sys.argv

    if use_log:
        index = load_documents_as_counters(DOCS_DIR)
        run_search = lambda q: search_log(q, index)
        count = len(index)
    else:
        names, count_vect, X_tfidf = build_tfidf_index(DOCS_DIR)
        run_search = lambda q: search_tfidf(q, names, count_vect, X_tfidf)
        count = len(names)

    print("TEXTLENS")
    print(f"{count} dokumentum betöltve, mód: {'log' if use_log else 'TF-IDF'}. (Kilépés: üres sor)\n")

    while True:
        query = input("Keresés: ").strip()
        if not query:
            break
        print_results(run_search(query))


if __name__ == "__main__":
    main()
