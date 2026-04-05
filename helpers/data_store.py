import chromadb
from sentence_transformers import SentenceTransformer

def get_data() -> chromadb.Collection:

    # Connect to the same store you created
    client = chromadb.PersistentClient(path="./chromadb_store")
    collection = client.get_or_create_collection("pdf_chunks")

    # --- 1. Total count ---
    print(f"Total chunks: {collection.count()}")

    # --- 4. Fetch ALL records ---
    all_data = collection.get(include=["documents", "metadatas", "embeddings"])
    for i, (doc, meta, embedding) in enumerate(zip(all_data["documents"], all_data["metadatas"], all_data['embeddings'])):
        print(f"\nChunk {i+1}")
        print(f"  Text    : {doc}")
        print(f"  Page    : {meta['page']}")
        print(f" Embedding   : {embedding}")

    # --- 6. List all collections in the store ---
    print("\nAll collections:", client.list_collections())


def store_in_chromadb(
    chunks: list[dict],
    collection_name: str = "pdf_chunks",
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    batch_size: int = 64,
) -> chromadb.Collection:

    # Local persistent ChromaDB stored on disk
    client = chromadb.PersistentClient(path="./chromadb_store")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}, # Using cosine similiarities
    )

    # Embed all chunks in one batch
    model = SentenceTransformer(model_name)
    texts   = [chunk.get("text") for chunk in chunks]
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Build parallel lists ChromaDB expects
    ids, embeddings, documents, metadatas = [], [], [], []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        ids.append(chunk.get("doc_id", "doc") + f"_chunk_{i}")
        embeddings.append(vector.tolist())
        documents.append(chunk.get("text"))
        metadatas.append({
            "page": chunk.get("page", 0),
            "chunk_index": i,
        })

    # Upsert into ChromaDB
    collection.upsert(
        ids        = ids,
        embeddings = embeddings,
        documents  = documents,
        metadatas  = metadatas,
    )

    print(f"Stored {len(chunks)} chunks into '{collection_name}'")
    print(f"Total records in collection: {collection.count()}")

    return collection

def wipe_out():
    client = chromadb.PersistentClient(path="./chromadb_store")
    try:
        collection = client.get_collection(name="pdf_chunks")
        if collection:
            client.delete_collection("pdf_chunks")
            print("Collection deleted.")
    except Exception as e:
        print(f"Collection does not exist or error occurred: {e}")
