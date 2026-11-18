from models import cache_models

def embed_chunks(
    chunks: list[dict],
    batch_size: int = 64,
) -> list[dict]:
    model = cache_models.get_sentence_trasformer_model()

    # Extract all texts and encode in one batch
    texts = [chunk.get("text") for chunk in chunks]
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False)

    # Attach the vector back to each chunk's data
    results = []
    for chunk, vector in zip(chunks, vectors):
        results.append({
            "text":   chunk.get("text"),
            "page":   chunk.get("page"),
            "vector": vector.tolist(),
        })

    return results
