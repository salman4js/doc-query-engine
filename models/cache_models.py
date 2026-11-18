# This helper file returns the model required.
# Caches the model the first it loads, and returns the same instance of the model every other time.
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

@lru_cache(maxsize=1)
def get_sentence_trasformer_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

@lru_cache(maxsize=1)
def get_semantic_score_checker_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@lru_cache(maxsize=1)
def get_cross_encoder_model():
    return CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L-2",
        max_length=512
    )
