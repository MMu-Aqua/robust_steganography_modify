import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# --- [核心修复] 全局单例模式，防止重复加载 ---
_EMBEDDING_MODEL = None
_EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_embedding_model(engine):
    """
    单例模式加载 Embedding 模型：
    如果模型已经加载过，直接返回；否则加载并缓存。
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        # 如果没有指定具体的模型名，或者传入的是旧的 engine 参数，强制使用本地轻量模型
        if not engine or "text-embedding" in engine:
            engine = "all-MiniLM-L6-v2"
            
        print(f"Loading embedding model: {engine} on {_EMBEDDING_DEVICE}...")
        try:
            _EMBEDDING_MODEL = SentenceTransformer(engine, device=_EMBEDDING_DEVICE)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise e
    return _EMBEDDING_MODEL

def compute_embeddings_local(texts, normalize, engine):
    # 获取全局缓存的模型
    model = get_embedding_model(engine)

    # 计算 embeddings
    embeddings = model.encode(texts)
    
    # 归一化
    if normalize:
        embeddings = [normalize_embedding(e) for e in embeddings]
    
    return embeddings

# --- [兼容层] 修复 ImportError 的关键 ---
#即使 get_embedding.py 还在调用这个旧函数，它现在也会安全地转发给本地模型

def compute_embeddings(texts, normalize, engine, client=None):
    """
    兼容旧代码的接口。
    忽略 client 参数，直接调用本地模型。
    """
    # 强制使用本地模型逻辑
    return compute_embeddings_local(texts, normalize, "all-MiniLM-L6-v2")

def compute_embeddings_concurrently(texts, engine, client=None):
    """兼容旧代码的并发接口（不再真正并发，直接批量计算）"""
    return np.array(compute_embeddings(texts, True, engine, client))

# --- 辅助函数 ---

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

def compute_embeddings_local_pair(texts1, texts2, normalize, engine):
    model = get_embedding_model(engine)
    embeddings_1 = model.encode(texts1)
    embeddings_2 = model.encode(texts2)
    
    if normalize:
        embeddings_1 = [normalize_embedding(e) for e in embeddings_1]
        embeddings_2 = [normalize_embedding(e) for e in embeddings_2]
    
    return [embeddings_1, embeddings_2]

def get_embedding(text, model, client, max_retries=5, backoff_factor=1.5):
    # 兼容接口：直接调用本地计算
    return compute_embeddings_local([text], True, "all-MiniLM-L6-v2")[0]

def get_embeddings_in_batch(client, texts):
    # 兼容接口：批量计算
    return np.array(compute_embeddings_local(texts, True, "all-MiniLM-L6-v2"))