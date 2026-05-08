import argparse
from helpers import text_extracter
from helpers import chunker
from helpers import embedder
from helpers import data_store
from helpers import gen_answer
from utils import utils


def document_extraction(file: str):
    
    # First, perform text extraction.
    extracted_texts = text_extracter.extract_text(file)
    chunks = []
    
    if len(extracted_texts) >= 1:
        
        for index, texts in enumerate(extracted_texts, 1):
            # convert the extracted texts into chunks (using semantic chunking)
            chunks_per_page = chunker.perform_chunking(texts)
            if len(chunks_per_page) >= 1:
                for chunk in chunks_per_page:
                    chunks_dict = {}
                    chunks_dict['text'] = chunk
                    chunks_dict['page'] = index
                    chunks.append(chunks_dict)
                    

        if len(chunks) > 0:
            vector_embeddings = embedder.embed_chunks(chunks)
            data_store.store_in_chromadb(vector_embeddings)
        else:
            print('Something went wrong while performing chunking...')


    else:
        print('Extracted text is not found')

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
    if not args.read and not args.store and not args.delete and not args.answer:
        utils.show_no_arguments_command()
        return
    
    if args.store and not args.read:
        if not args.file:
            print('Specify the file name to read...')
            return
        document_extraction(args.file)

    if args.read and not args.store:
        data_store.get_data()
    
    if args.delete and not args.read and not args.store:
        data_store.wipe_out()
    
    if args.answer and not args.delete and not args.read and not args.store:
        if not args.question:
            print('Please ask any question from the document uploaded...')
            return
        
        if not args.model_name:
            utils.show_no_model_msg()
            return
        gen_answer.generate_answer(args.question, args.model_name)


if __name__ == "__main__":
    main()