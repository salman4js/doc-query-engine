from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from helpers import logger

CHAT_HISTORY = []

# Keyword search tokenizer
def tokenize(text: str) -> list[str]:
    _tokenizer = RegexpTokenizer(r"\w+")
    _stemmer   = PorterStemmer()
    _stopwords = set(stopwords.words("english"))
    tokens = _tokenizer.tokenize(text.lower())
    return [_stemmer.stem(t) for t in tokens if t not in _stopwords]

# Reciprocal rank fusion method impl
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60, top_n: int = 5,) -> list[tuple[str, float]]:
    logger.logging.info('PROCESS:STARTED: Reciprocal Rank Fusion Started...')
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, id_ in enumerate(ranked_ids, start=1):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = fused[:top_n]
    logger.logging.info(f'PROCESS:COMPLETED: Reciprocal Rank Fusion filtered out final {len(top)} chunks')
    return top

# Retrieval grading utility function
def retrieval_grader(chunks: list[dict], min_score=0.2) -> list[dict]:
    logger.logging.info(f'PROCESS:STARTED: RETRIEVAL GRADE Started with {len(chunks)}')
    new_chunks = []
    for chunk in chunks:
        if chunk['score'] > min_score:
            new_chunks.append(chunk)
        else:
            logger.logging.info(f'PROCESS:STATUS: RETRIEVAL GRADER -- Discarded chunk {chunk['id']}. Score is below minimium threshold')
    logger.logging.info(f'PROCESS:COMPLETED: RETRIEVAL GRADE Started with {len(new_chunks)}')
    return new_chunks

# System prompt for the LLM
def get_sys_prompt() -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are the AI that uses the context to answer the question. "
            "If the answer is not in the context, say "
            "'The question is not relevant to the document uploaded'."
            "Never ever answer anything out of the context"
        )
    }

# System prompt for re-writing query and generating multi-query use-case.
def get_multi_query_sys_prompt() -> dict[str, str]:
    return """
        You are an AI assistant specialized in multi-query generation
        for Retrieval-Augmented Generation (RAG).

        Your task is to generate exactly 2 alternative search queries
        based on the original user question.

        Follow these rules strictly:

        1. Preserve the exact intent and meaning of the original question.
        2. Generate queries that explore different wording or search
        perspectives without changing the topic.
        3. Do not introduce new subjects, assumptions, or unrelated aspects.
        4. Do not turn a general question into a question about a
        specific aspect unless the original question implies it.
        5. Keep the original entities, names, and important keywords.
        6. Make every generated query independently useful for
        retrieving relevant documents.
        7. Avoid repetitive or nearly identical queries.

        Return exactly 2 queries in valid JSON format:

        {
            "queries": [
                "query 1",
                "query 2"
            ]
        }

        Return only JSON without explanations.
    """
# User prompt for the LLM
def get_user_prompt(context:str, question:str) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"""
            Use the following context to answer the question.
            Context: {context}
            Question: {question}
            Answer:
        """
    }

# Method to store chat history
def store_history(chat_dict: dict):
    CHAT_HISTORY.append(chat_dict)

# Method to retrieve chat history
def get_history() -> list[dict]:
    if len(CHAT_HISTORY) <= 4:
        return CHAT_HISTORY
    else:
        chat_history = CHAT_HISTORY[-2:]
        chat_history.insert(0, get_sys_prompt())
        return chat_history

def show_help_commands():
    print("Please pass --read, --store, --delete or --answer")
    print("Example: python main.py --store --file report.pdf")
    print("--read: Gives all the colletion's chunks from the database")
    print("--store: Expects a file with '--file' command. Performs chunking, embedding of the file passed and store the results to the database")
    print("--delete: Deletes all the collection's data from the database (Wipes out everything)")
    print("--answer: Expects a question from the user with '--question'")
    print("--model_name: Expects a model name from the user with '--model_name'")


def show_no_arguments_command():
    print("No arguments has been passed, run 'python3 main.py --help' to see available commands")

def show_no_model_msg():
    print("Model name must be mandatory to generate answers")
    print("Example: ")
    print("To get terminal-level quick answer")
    print("python3 main.py --answer --model_name 'model_name' --question 'your_question_here'")
    print("or")
    print("To get fully functioned chat history based chat-bot")
    print("python3 main.py --answer --model_name 'model_name' --start_qa_bot")
    print('Run python3 main.py --help for more information')

# Get Ollama Model URL's
def get_chat_url() -> str:
    return "http://localhost:11434/api/chat"

def get_generate_utl() -> str:
    return "http://localhost:11434/api/generate"