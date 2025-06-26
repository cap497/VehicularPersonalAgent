import os
import faiss
import pickle
import numpy as np
from embedder import get_embedding
from chunker import chunk_text
from pathlib import Path

def index_document(doc_path: str, output_dir="faiss_indexes"):
    os.makedirs(output_dir, exist_ok=True)
    name = Path(doc_path).stem

    with open(doc_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    chunks = chunk_text(full_text)

    if not chunks:
        print(f"⚠️ Skipping empty document: {name}")
        return

    embeddings = [get_embedding(chunk) for chunk in chunks]
    embedding_matrix = np.vstack(embeddings).astype("float32")

    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix) # type: ignore

    faiss.write_index(index, f"{output_dir}/{name}.faiss")
    with open(f"{output_dir}/{name}.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Indexed {name} with {len(chunks)} chunks.")

def index_all_documents(base_dir="knowledge_base", output_dir="faiss_indexes"):
    """
    Index all `.txt` documents in the specified folder into FAISS (1 index per file).
    """
    base = Path(base_dir)
    if not base.exists():
        print(f"❌ knowledge_base folder not found: {base_dir}")
        return

    for file in base.glob("*.txt"):
        index_document(str(file), output_dir)
