from dotenv import load_dotenv
import os
import logging
import requests
import json

# Load environment variables from the .env file
load_dotenv()

logger = logging.getLogger('RAG_Project')


def query_llm(prompt, model_id="DeepSeek-V3-0324", api_token=None):
    """
    Query the Hugging Face OpenAI-style router endpoint for DeepSeek-V3-0324.

    Parameters:
        prompt (str): The structured prompt string to send to the LLM.
        model_id (str): The model identifier on Hugging Face (e.g. "DeepSeek-V3-0324").
        api_token (str): The Hugging Face API token for authentication.

    Returns:
        str: The LLM's text response.
    """
    if not api_token:
        logger.warning("No API token provided. Ensure your token is set.")
        return None

    # Updated API endpoint for OpenAI-compatible chat completion
    api_url = "https://router.huggingface.co/sambanova/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }

    try:
        logger.info("Sending request to the LLM API...")
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info("LLM API response received.")
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error querying the LLM: {e}")
        return None


if __name__ == "__main__":
    # For demonstration: using a sample prompt created in Phase 7.
    sample_prompt = (
        "User Query:\nWhat are some interesting cat facts?\n\n"
        "Relevant Context:\n"
        "Chunk 1 (('cat_facts.txt.processed.txt.chunks.txt', 0)):\nCats have flexible bodies and quick reflexes.\n\n"
        "Chunk 2 (('cat_facts.txt.processed.txt.chunks.txt', 1)):\nThey can jump up to six times their body length.\n\n"
        "Please use the above context to provide a detailed answer to the query."
    )

    # Retrieve the Hugging Face API token from environment variables.
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN", None)

    # Debug: print the token (be cautious with tokens in production!)
    print(hf_token)

    # Query the LLM using the structured prompt.
    llm_response = query_llm(sample_prompt, api_token=hf_token)

    print("LLM Response:")
    print(json.dumps(llm_response, indent=2))
