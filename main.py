from time import time

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.legacy.postprocessor import FlagEmbeddingReranker
from llama_index.llms.openai import OpenAI

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

Settings.llm = OpenAI(api_key="sk-123456", model="qwen-turbo",
                      api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# build index
index = VectorStoreIndex.from_documents(documents=documents)

rerank = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=5)

# Firstly, we try with rerank
query_engine = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[rerank]
)

now = time()

response = query_engine.query(
    "Which grad schools did the author apply for and why?",
)
print(f"Elapsed: {round(time() - now, 2)}s")

print(response)

print(response.get_formatted_sources(length=200))

# # Next, we try without rerank
#
# query_engine = index.as_query_engine(similarity_top_k=10)
#
# now = time()
# response = query_engine.query(
#     "Which grad schools did the author apply for and why?",
# )
#
# print(f"Elapsed: {round(time() - now, 2)}s")
#
# print(response)
#
# print(response.get_formatted_sources(length=200))
