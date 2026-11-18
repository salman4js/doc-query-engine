from helpers import gen_answer
from helpers import logger


EXIT_PROGRAM_KEYWORDS = ['quit', 'exit']

def start_qa_bot(model_name: str):
    logger.logging.info('Starting QA-System...')
    print('QA-System: How can I help you today?')
    while True:
        message = input("You: ")
        logger.logging.info(f"User Input: {message}")
        if message.lower() in EXIT_PROGRAM_KEYWORDS:
            print('QA-System: Bye!')
            logger.logging.info('QA System stopped...')
            break
        print('QA-System:')
        is_resp_done = gen_answer.generate_answer(message, model_name) 
        if not is_resp_done:
            break
