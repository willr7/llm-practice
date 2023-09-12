import requests
import json
import numpy as np
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

LLM_URL = os.getenv("LLM_URL")
TRANSFORMER_URL = os.getenv("TRANSFORMER_URL")


def make_llm_request(prompt):
    url = LLM_URL
    payload = json.dumps(
        {
            "inputs": prompt,
            "parameters": {
                "do_sample": True,
                "top_p": 0.7,
                "temperature": 0.3,
                "top_k": 50,
                "max_new_tokens": 100,
                "repetition_penalty": 1.03,
            },
        }
    )

    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)
    response = response.json()

    if not "Prediction" in response.keys():
        return response
    return response["Prediction"][0]["generated_text"][len(prompt) :]


def make_transformer_request(text):
    """Takes in a list of sentences and returns a list of vectors

    Args:
        text (list): a list of sentences

    Returns:
        list: a list of vectors which represent the input sentence
    """

    url = TRANSFORMER_URL

    payload = json.dumps({"inputs": text})
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    loaded_response = json.loads(json.loads(response.text)["body"])["Prediction"]
    return loaded_response


def embed_docs(docs: List[str]) -> List[List[float]]:
    out = make_transformer_request(docs)
    embeddings = np.mean(np.array(out), axis=1)
    return embeddings.tolist()
