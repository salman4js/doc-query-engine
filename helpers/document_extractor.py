from helpers import text_extracter
from helpers import chunker
from helpers import embedder
from helpers import data_store

def extract(file: str):
    
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