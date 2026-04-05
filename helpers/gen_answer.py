import chromadb
from sentence_transformers import SentenceTransformer
from helpers import llm_response


def query_chromadb(
    question: str,
    collection_name: str = "pdf_chunks",
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    n_results: int = 3,
    min_similiarity_score: int = 0.3
) -> list[dict]:

    # --- Step 1: embed the question ---
    model = SentenceTransformer(model_name)
    question_vector = model.encode("query: " + question).tolist()

    # --- Step 2: connect to ChromaDB ---
    client = chromadb.PersistentClient(path="./chromadb_store")
    collection = client.get_or_create_collection(collection_name)

    # --- Step 3: search for similar chunks ---
    results = collection.query(
        query_embeddings = [question_vector],
        n_results        = n_results,
        include          = ["documents", "metadatas", "distances"],
    )

    # --- Step 4: format and return results ---
    matches = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0], # [0.87], [0.31] -> distance .50
    ):
        similiarity = round(1 - distance, 4)

        if similiarity < min_similiarity_score:
            continue
            
        matches.append({
            "text":       doc,
            "metadata":   meta,
            "similarity": similiarity,  # distance → similarity score
        })

    return matches

def generate_answer(question: str) -> str:
    matches_from_db = query_chromadb(question)
    llm_response.generate_response(question, matches_from_db)