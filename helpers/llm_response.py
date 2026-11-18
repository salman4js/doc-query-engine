import requests
import json
from utils import utils
from helpers import logger

def generate_response(question: str, model_name: str, matches: list[dict]) -> str:

    context = "\n\n".join([m.get("text", str(m)) for m in matches])
    payload = {
        "model": model_name,
        "stream": True
    }

    history = utils.get_history()

    if len(history) == 0:
        utils.store_history(utils.get_sys_prompt())
        utils.store_history(utils.get_user_prompt(context=context, question=question))
        payload['messages'] = utils.get_history()
    else:
        # Add user question to history
        history.append(utils.get_user_prompt(context=context, question=question))
        payload['messages'] = history

    # Generate response from the LLM
    try:
        response = requests.post(utils.get_chat_url(), json=payload, stream=True)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return ""

        full_response = ""

        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)

                    if "message" in json_data:
                        content = json_data["message"].get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            full_response += content

                    if json_data.get("done"):
                        break

                except json.JSONDecodeError:
                    print(f"\nFailed to parse line: {line}")

        print()  # newline after streaming

        # Store assistant response into history
        utils.store_history({
            "role": "assistant",
            "content": full_response
        })
        # Return the full_response to perform hallucination check!
        return full_response

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return ""


def rewrite_and_generate_multi_query(question: str, model_name:str) -> list[str]:
    logger.logging.info('PROCESS:STARTED: QUERY RE-WRITER --- Generating multiple variants of user questions...')
    payload = {
        "model": model_name,
        "system": utils.get_multi_query_sys_prompt(),
        "prompt": f"Generate 2 semantically diverse queries for: {question}",
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(utils.get_generate_utl(), json=payload, stream=False)

        # If any error occurs, return the original question without stopping the application
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            logger.logging.info('PROCESS:ABORTED: QUERY RE-WRITER --- Failed to generate variants of user question, proceeding with original question...')
            return [question]

        result = response.json()
        generated_result = json.loads(result["response"])
        queries = generated_result.get("queries", [])
        queries.insert(0, question)
        logger.logging.info(f'PROCESS:COMPLETED: QUERY RE-WRITER --- Variants generation completed, generated {len(queries)} variants...')
        return queries

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        logger.logging.info('PROCESS:ABORTED: QUERY RE-WRITER --- Failed to generate variants of user question, proceeding with original question...')
        return [question]