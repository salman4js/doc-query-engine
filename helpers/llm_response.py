import requests
import json

url = "http://localhost:11434/api/chat"

def generate_response(question: str, matches: list[dict]) -> str:

    context = "\n\n".join([m.get("text", str(m)) for m in matches])
    payload = {
        "model": "celestia",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the AI that uses the context to answer the question. "
                    "If the answer is not in the context, say "
                    "'The question is not relevant to the document uploaded'."
                )
            },
            {
                "role": "user",
                "content": f"""
                    Use the following context to answer the question.

                    Context:
                    {context}

                    Question: {question}
                    Answer:
                """
            }
        ],
        "stream": True
    }

    try:
        response = requests.post(url, json=payload, stream=True)

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

        # Return the full_response to perform hallucination check!
        return full_response

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return ""