from __future__ import annotations

import numpy as np
import nltk
import re

from models import cache_models
from helpers import logger

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

def clean_text(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")  # fix ligatures
    text = text.replace("\n", " ")                     # remove line breaks
    text = re.sub(r"\s+", " ", text)                  # normalize spaces
    text = re.sub(r"-\s+", "", text)                  # fix broken hyphen words
    return text.strip()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def sentences_from_text(text: str) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    
    # remove very short/broken fragments
    return [s.strip() for s in sentences if len(s.split()) > 4]


def semantic_chunk_auto(
    text: str,
    percentile: float = 20.0,    
    min_chunk_sentences: int = 3,
    max_chunk_sentences: int = 7,
) -> list[str]:

    text = clean_text(text) 

    sentences = sentences_from_text(text)
    if len(sentences) <= 2:
        return [text]

    passages = ["passage: " + s for s in sentences]

    model = cache_models.get_sentence_trasformer_model()

    embeddings = model.encode(passages, batch_size=64, show_progress_bar=False)

    # similarity between consecutive sentences
    similarities = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    threshold = float(np.percentile(similarities, percentile))

    split_indices: set[int] = set()
    for i, sim in enumerate(similarities):
        if sim < threshold:
            split_indices.add(i + 1)

    chunks: list[list[str]] = []
    current: list[str] = []


    for i, sentence in enumerate(sentences):
        if i in split_indices and len(current) >= min_chunk_sentences:
            chunks.append(current)
            current = []

        current.append(sentence)

        if len(current) >= max_chunk_sentences:
            chunks.append(current)
            current = []

    if current:
        chunks.append(current)

    
    return [" ".join(chunk) for chunk in chunks]

# Re-ranker functions with cross encoder to optimize the chunks that are going to the LLM to generate answer.
def re_rank_chunks(query: str, chunks: list, top_k=5) -> list[dict]:
    logger.logging.info('PROCESS:STARTED: Re-ranking process started...')
    model = cache_models.get_cross_encoder_model()
    pairs = [(query, doc['text']) for doc in chunks]
    scores = model.predict(pairs)
    
    ranked_indices = np.argsort(scores)[::-1] 
    ranked_indices = ranked_indices[:top_k]
    ranked_docs = []

    for i in ranked_indices:
        chunks[i]['score'] = float(scores[i])
        ranked_docs.append(chunks[i])

    logger.logging.info(f'PROCESS:COMPLETED: Re-ranking process completed, re-ranked {len(ranked_docs)} chunks')
    return ranked_docs

def perform_chunking(text: str) -> list[str]:
    return semantic_chunk_auto(text)
