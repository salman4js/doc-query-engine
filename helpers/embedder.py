from sentence_transformers import SentenceTransformer

def embed_chunks(
    chunks: list[dict],
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    batch_size: int = 64,
) -> list[dict]:
    model = SentenceTransformer(model_name)

    # Extract all texts and encode in one batch
    texts = [chunk.get("text") for chunk in chunks]
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Attach the vector back to each chunk's data
    results = []
    for chunk, vector in zip(chunks, vectors):
        results.append({
            "text":   chunk.get("text"),
            "page":   chunk.get("page"),
            "vector": vector.tolist(),
        })

    return results
