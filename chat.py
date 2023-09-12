from aws_api_wrapper import embed_docs, make_llm_request
import pinecone
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_KEY=os.getenv("PINECONE_KEY")
PINECONE_ENV=os.getenv("PINECONE_ENV")

pinecone.init(
    api_key=PINECONE_KEY,
    environment=PINECONE_ENV
)

index_name = 'retrieval-augmentation-aws'

index = pinecone.Index(index_name)

question = "What is Joe Biden doing about the cost of medicine?"

# extract embeddings for the questions
query_vec = embed_docs(question)[0]

# query pinecone
res = index.query(query_vec, top_k=5, include_metadata=True)

contexts = [match.metadata['text'] for match in res.matches]

max_section_len = 1000
separator = "\n"

def construct_context(contexts: List[str]) -> str:
    chosen_sections = []
    chosen_sections_len = 0

    for text in contexts:
        text = text.strip()
        # Add contexts until we run out of space.
        chosen_sections_len += len(text) + 2
        if chosen_sections_len > max_section_len:
            break
        chosen_sections.append(text)
    concatenated_doc = separator.join(chosen_sections)
    # print(
    #     f"With maximum sequence length {max_section_len}, selected top {len(chosen_sections)} document sections: \n{concatenated_doc}"
    # )
    return concatenated_doc

context_str = construct_context(contexts=contexts)

prompt_template = """Answer the following QUESTION based on the CONTEXT
given. If you do not know the answer and the CONTEXT doesn't
contain the answer truthfully say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


running = True
while running:
    question = input("Question: ")
    if question == "quit": running = False
    query = prompt_template.replace("{context}", context_str).replace("{question}", question)
    
    answer = make_llm_request(query)
    print(f"Answer: {answer}")