# 引入本地嵌入工具
from .embedding_utils import compute_embeddings_local
import importlib.util
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from .embedding_utils import compute_embeddings

# 定义一个本地的、免费的 Sentence-Transformer 模型
# 这是一个非常流行的、用于科研的轻量级嵌入模型
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 修改 get_embedding 函数
def get_embedding(client, text):
    # Get the embedding for the text
    """
    embedding = compute_embeddings(
        text, True, "text-embedding-3-large", client)
    """
    # 删除了对 OpenAI client 的依赖，使用本地模型计算嵌入
    # compute_embeddings_local 函数位于 embedding_utils.py 中
    embedding = compute_embeddings_local([text], True, LOCAL_EMBEDDING_MODEL)
    emb = np.array(embedding[0])
    return emb

"""
def get_embeddings_in_batch(client, texts):
    # Using the embeddings.create method to fetch embeddings for multiple texts in one request
    response = client.embeddings.create(
        input=texts,  # Input is a list of texts
        model="text-embedding-3-large"   # Specify the model you are using
    )
    # Extracting the embeddings from the response object
    embeddings = np.array([res.embedding for res in response.data])
    return embeddings
"""
# 修改 get_embeddings_in_batch 函数
def get_embeddings_in_batch(client, texts):
    # 删除了对 OpenAI client 的依赖
    embeddings = compute_embeddings_local(texts, True, LOCAL_EMBEDDING_MODEL)
    return np.array(embeddings)

if __name__ == "__main__":
    client = OpenAI()  # automatically uses OPENAI_API_KEY env var
    text = "What are you up to today?"
    embedding = get_embedding(client, text)
    print(embedding)
    print(type(embedding))
