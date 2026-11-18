import chromadb
from rank_bm25 import BM25Okapi
from helpers import llm_response
from helpers import chunker
from helpers import logger
from utils import utils
from hallucination_checker import semantic_checker
from models import cache_models

RETRY_ATTEMPTS = 0

MAX_RETRY_ATTEMPTS = 1

def get_chroma_collection(collection_name: str = "pdf_chunks"):
    client = chromadb.PersistentClient(path="./chromadb_store")
    return client.get_or_create_collection(collection_name)

def query_chromadb(
    question: str,
    n_results: int = 5,
    min_similiarity_score: int = 0.3
) -> tuple[dict[list], list[str]]:

    logger.logging.info('PROCESS:STARTED: Vector Similiarity Search...')
    # --- Step 1: embed the question ---
    model = cache_models.get_sentence_trasformer_model()
    question_vector = model.encode("query: " + question).tolist()

    # --- Step 2: Get the ChromaDB collection---
    collection = get_chroma_collection()

    # --- Step 3: search for similar chunks ---
    results = collection.query(
        query_embeddings = [question_vector],
        n_results        = n_results,
        include          = ["documents", "metadatas", "distances"],
    )

    # --- Step 4: format and return results ---
    matches = []
    chunk_ids = []
    for id, doc, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0], # [0.87], [0.31] -> distance .50
    ):
        similiarity = round(1 - distance, 4)

        if similiarity < min_similiarity_score:
            continue
            
        matches.append({
            "id": id,
            "text": doc,
            "metadata": meta,
            "similarity": similiarity,  # distance → similarity score
        })

        # Keeping track of chunk ids separately to use for reciprocal rank fusion
        chunk_ids.append(id)
    logger.logging.info(f"PROCESS:COMPLETED: Vector Similiarity Search found {len(chunk_ids)} matches")
    return matches, chunk_ids

def retrieve_bm25_chunks(question:str) -> tuple[list[str], list[str]]:
    # Get the unfiltered chunks from the chromaDB collection.
    logger.logging.info('PROCESS:STARTED: BM25 Keyword Search Started...')
    collection = get_chroma_collection()
    chunks = collection.get(include=["documents"])
    # For every chunks in chromaDB
    bm25_ids  = chunks["ids"]
    bm25_docs = chunks["documents"]
    bm25 = BM25Okapi([d.lower().split() for d in bm25_docs])
    # Get score for each chunks.
    bm25_scores   = bm25.get_scores(question.lower().split())
    # Get the top 20 results from the chunks
    bm25_top      = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    bm25_ranked_doc = []
    bm25_ranked_ids = []
    for i in bm25_top:
        if bm25_scores[i] > 0:
            bm25_ranked_ids.append(bm25_ids[i])
            bm25_ranked_doc.append(bm25_docs[i])

    # Return the top chunks
    logger.logging.info(f'PROCESS:COMPLETED: BM25 Keyword Search found {len(bm25_ranked_ids)} matches')
    return bm25_ranked_doc, bm25_ranked_ids

def generate_answer(question: str, model_name: str) -> str:
    global RETRY_ATTEMPTS
    logger.logging.info('PROCESS:STARTED: ANSWER GENERATOR --- Initiating RAG pipeline process...')
    final_fused_chunks = []
    # Query re-writing and multi query generation
    queries = llm_response.rewrite_and_generate_multi_query(question=question, model_name=model_name)
    # Get chunks for all the queries generated.
    for query in queries:
        # Generate relevant chunks using dense (Vector similiarities)
        vs_chunks, vs_chunks_ids = query_chromadb(query)
        # Generate relevant chunks using BM25 (Keyword similiarites)
        bm25_hits_chunks, bm25_chunks_hit_ids = retrieve_bm25_chunks(query)
        # Perform reciprocal rank fushion to perform ranking from vs_chunks and bm25_chunks
        if not len(vs_chunks) == 0 or len(bm25_hits_chunks) == 0:
            fused_ids_score = utils.reciprocal_rank_fusion([vs_chunks_ids, bm25_chunks_hit_ids])
            fused_ids = [id for id, score in fused_ids_score]
            if len(fused_ids) == 0:
                fused_chunks = get_chroma_collection().get(ids=vs_chunks_ids, include=["documents", "metadatas"])
            else:
                fused_chunks = get_chroma_collection().get(ids=fused_ids, include=["documents", "metadatas"])
        else:
            if len(vs_chunks) == 0:
                fused_chunks = get_chroma_collection().get(ids=bm25_chunks_hit_ids, include=["documents", "metadatas"])
            else:
                fused_chunks = get_chroma_collection().get(ids=vs_chunks_ids, include=["documents", "metadatas"])
            
        results = []
        
        # Deserialize fused chunks into re_ranking suitable input chunks
        for id, doc, meta in zip(
            fused_chunks["ids"],
            fused_chunks["documents"],
            fused_chunks["metadatas"]
        ):
            results.append({
                "id": id,
                "text": doc,
                "metadata": meta
            })

        ranked_docs = chunker.re_rank_chunks(query=query, chunks=results)
        final_fused_chunks.append(ranked_docs)

    # Flatten the lists of lists into single lists of dict
    final_fused_chunks = [item for sublist in final_fused_chunks for item in sublist]
    
    # Perform retrieval grading with threshold of 0.5
    final_fused_chunks = utils.retrieval_grader(final_fused_chunks)
    if len(final_fused_chunks) == 0 and MAX_RETRY_ATTEMPTS > RETRY_ATTEMPTS:
        # Retry by generating different variants of questions
        logger.logging.info('PROCESS:STARTED: RETRY ANSWER GENERATION --- No valid chunks retrieved, retrying the whole process again...')
        RETRY_ATTEMPTS += 1
        generate_answer(question=question, model_name=model_name)
        return
    
    # If there are no chunks present, prompt that there are no chunks and ask for the document.
    if len(final_fused_chunks) == 0:
        logger.logging.info('PROCESS:ABORTED: ANSWER GENERATOR --- No chunks found, answer is not in the document...')
        print('No chunks found, run --store command with the file to create chunks')
        return False
    else:
        # chunk_sentences is needed for hallucination checker!
        chunk_sentences = ""
        for chunk in final_fused_chunks:
            chunk_sentences += chunk['text']

        logger.logging.info('PROCESS:RUNNING: ANSWER GENERATOR --- Sending the finalized chunks to LLM...')
        claims = llm_response.generate_response(question, model_name, final_fused_chunks)
        logger.logging.info('PROCESS:COMPLETED: ANSWER GENERATOR --- Answer generated...')
        semantic_score = semantic_checker.SemanticScoreChecker().compute_score(chunk_sentences, claims)
        logger.logging.info('PROCESS:STARTED: HALLUCINATION SCORE CHECKER --- Started...')
        print(f"Hallucination Score for the generated answer is {semantic_score['semantic_score']}")
        logger.logging.info(f'PROCESS:COMPLETED: HALLUCINATION SCORE CHECKER --- Finished with score of {semantic_score['semantic_score']}')
        return True