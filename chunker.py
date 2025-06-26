import re
from typing import List

def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter using punctuation rules.
    Replace with nltk.sent_tokenize() if you want better accuracy.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def chunk_text(text: str, max_sentences: int = 3) -> List[str]:
    """
    Combines sentences into paragraph-like chunks of up to `max_sentences`.
    """
    sentences = split_sentences(text)
    chunks = []
    temp_chunk = []

    for sentence in sentences:
        temp_chunk.append(sentence)
        if len(temp_chunk) >= max_sentences:
            chunks.append(" ".join(temp_chunk))
            temp_chunk = []

    if temp_chunk:  # any remaining sentence(s)
        chunks.append(" ".join(temp_chunk))

    return chunks
