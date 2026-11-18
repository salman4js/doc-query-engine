import argparse

# Disabling logging from hugging face and sentence transformer
import os
os.environ["HF_HUB_OFFLINE"] = "1"                 # no network checks (the HF_TOKEN warning goes away)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"   # hides "Loading weights" bar
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()                   # hides the LOAD REPORT / warnings
hf_logging.disable_progress_bar()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from helpers import data_store
from helpers import gen_answer
from helpers import document_extractor
from qa_system import qa_system
from utils import utils

def parse_args():
    parser = argparse.ArgumentParser(description="PDF RAG pipeline")
    # Prevents adding default help command by the parser
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--read",
        action="store_true",        
        help="Read the data from chromaDB",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="Embed chunks and store in ChromaDB",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Path to the PDF file",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete all the collection from DB"
    )

    parser.add_argument(
        "--answer",
        action="store_true",
        help="Get the question from the user"
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Question to perform the similiarity search from the database"
    )

    parser.add_argument(
        "--help",
        action="store_true",
        help="Shows the available commands to the user"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Ollama model name to be passed to generate answers by uploaded document chunks"
    )

    parser.add_argument(
        "--start_qa_bot",
        action="store_true",
        help="Starts the question-answer to answer question from the document uplaoded"
    )

    # parser.add_argument(
    #     "--collection",
    #     type=str,
    #     default="pdf_chunks",
    #     help="ChromaDB collection name (default: pdf_chunks)",
    # )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.help:
        utils.show_help_commands()
        return

    # Must pass at least one flag
    if not args.read and not args.store and not args.delete and not args.answer and not args.start_qa_bot:
        utils.show_no_arguments_command()
        return
    
    if args.store and not args.read:
        if not args.file:
            print('Specify the file name to read...')
            return
        document_extractor.extract(args.file)

    if args.read and not args.store:
        data_store.get_data()
    
    if args.delete and not args.read and not args.store:
        data_store.wipe_out()

    if args.start_qa_bot:
        if not args.model_name:
            utils.show_no_model_msg()
            return
        qa_system.start_qa_bot(args.model_name)
        return
    
    if args.answer and not args.delete and not args.read and not args.store:
        if not args.question:
            print('Please ask any question from the document uploaded...')
            return
        
        if not args.model_name:
            utils.show_no_model_msg()
            return
        gen_answer.generate_answer(args.question, args.model_name)


if __name__ == "__main__":
    from helpers import logger
    logger.logging.info('Application started')
    main()