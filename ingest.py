import pinecone
from aws_api_wrapper import embed_docs
import time
import pandas as pd
from typing import List
from tqdm.auto import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_KEY=os.getenv("PINECONE_KEY")
PINECONE_ENV=os.getenv("PINECONE_ENV")

# TODO
# Find a better way to split up sentences
# Right now, they are separated line breaks
df_knowledge = pd.read_csv("state_of_the_union.txt", header=None, sep=" \n\n", engine='python')

pinecone.init(
    api_key=PINECONE_KEY,
    environment=PINECONE_ENV
)

index_name = 'retrieval-augmentation-aws'

if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

pinecone.create_index(
    name=index_name,
    dimension=384,
    metric='cosine'
)

# wait for index to finish initialization
while not pinecone.describe_index(index_name).status['ready']:
    time.sleep(1)

batch_size = 2  # can increase but needs larger instance size otherwise instance runs out of memory
vector_limit = 1000

answers = df_knowledge[:vector_limit]
index = pinecone.Index(index_name)

for i in tqdm(range(0, len(answers), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(answers))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]
    # create metadata batch
    metadatas = [{'text': text} for text in answers[0][i:i_end]]
    # create embeddings
    texts = answers[0][i:i_end].tolist()
    embeddings = embed_docs(texts)
    # create records list for upsert
    records = zip(ids, embeddings, metadatas)
    # upsert to Pinecone
    index.upsert(vectors=records)

print(index.describe_index_stats())