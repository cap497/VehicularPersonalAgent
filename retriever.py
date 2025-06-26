import os
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
from embedder import get_embedding
import faiss

def load_documents(folder="knowledge_base"):
    docs = []
    for path in Path(folder).glob("*.txt"):
        with open(path, 'r', encoding='utf-8') as f:
            docs.append((path.stem, f.read()))
    return docs

def retrieve_documents(query, method="bm25", k=3):
    docs = load_documents()
    corpus = [text for _, text in docs]

    if method == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
        q_vec = vectorizer.transform([query])
        scores = (matrix @ q_vec.T).toarray().flatten() # type: ignore
    else:
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        q_tokens = query.lower().split()
        scores = bm25.get_scores(q_tokens)

    indices = np.argsort(scores)[::-1][:k]
    return [docs[i][0] for i in indices]

def retrieve_faiss_chunks(query: str, doc_ids: list[str], index_dir="faiss_indexes", top_k=5):
    """
    Searches relevant FAISS indexes and returns top-k similar chunks with their doc_id and scores.
    """
    query_embedding = get_embedding(query).astype("float32") # type: ignore
    all_results = []

    for doc_id in doc_ids:
        index_path = os.path.join(index_dir, f"{doc_id}.faiss")
        chunks_path = os.path.join(index_dir, f"{doc_id}.pkl")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            continue

        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        D, I = index.search(np.array([query_embedding]), top_k)
        for score, idx in zip(D[0], I[0]):
            if idx < len(chunks):
                chunk_text = chunks[idx]
                all_results.append((doc_id, chunk_text, score))

    # Sort globally by score (lower distance = better)
    all_results.sort(key=lambda x: x[2])
    return all_results[:top_k]

def format_as_mcp(doc_chunks):
    blocks = []
    for i, (doc_id, chunk) in enumerate(doc_chunks):
        blocks.append(f"### Document {i+1}: {doc_id}\n{chunk.strip()}")
    return "\n\n".join(blocks)
