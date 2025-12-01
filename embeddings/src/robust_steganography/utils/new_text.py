# This file contains the code to sample a new message
import re
import torch
import numpy as np
import openai

# 导入本地模型
from watermark.models.gpt2 import GPT2Model
# 引用本地嵌入计算工具
from .embedding_utils import compute_embeddings_local
_LOCAL_MODEL = None

# 自动检测设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

#新增
def get_local_model():
    """惰性加载本地模型，只加载一次"""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        print(f"Loading local GPT-2 model (gpt2) on {DEVICE}...")
        # 默认加载 GPT-2 小模型，速度较快
        _LOCAL_MODEL = GPT2Model(model_name='gpt2') 
        # 将模型移动到 GPU
        _LOCAL_MODEL._model.to(DEVICE)
    return _LOCAL_MODEL


def clean_response(text):
    # Regex to find the last full sentence ending with ., !, or ?
    match = re.search(r'([.!?])[^.!?]*$', text)
    if match:
        return text[:match.end()].strip()
    else:
        return text.strip()

# [核心创新] Logit 引导函数
def guide_logits(model, context_tokens, logits, target_bits, hash_fn, top_k=10):
    """
    通过预测候选词的 Embedding 来调整 Logits，引导生成方向。
    """
    # 1. 选出概率最高的 Top-K 个候选词
    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, top_k)
    
    # 2. 构建 K 个候选句子 (Current Context + Candidate Token)
    candidates = []
    current_text = model.tokenizer.decode(context_tokens[0])
    
    for idx in top_k_indices:
        token_id = idx.item()
        token_str = model.tokenizer.decode([token_id])
        candidates.append(current_text + token_str)
    
    # 3. 批量计算候选句子的 Embeddings (利用 GPU 加速)
    # 使用 all-MiniLM-L6-v2 (384维)
    candidate_embeddings = compute_embeddings_local(candidates, normalize=True, engine="all-MiniLM-L6-v2")
    
    # 4. 计算每个候选句子与目标哈希的匹配程度
    # RandomProjectionHash 的原理是: sign(Emb @ Matrix) == Bits
    # 我们希望: Emb @ Matrix 的符号 与 目标Bits (0/1 -> -1/+1) 一致
    # Score = sum( (Emb @ Matrix) * (2*Bits - 1) )
    # 分数越高，说明 Embedding 越符合目标哈希的方向
    
    rand_matrix = hash_fn.rand_matrix # (384, num_bits)
    target_signs = 2 * np.array(target_bits) - 1 # [0, 1] -> [-1, 1]
    
    # [K, 384] @ [384, num_bits] -> [K, num_bits]
    projections = candidate_embeddings @ rand_matrix 
    
    # 计算匹配分数: 投影值与目标符号同号则为正，异号则为负
    # [K, num_bits] * [num_bits] -> [K, num_bits] -> Sum -> [K]
    scores = np.sum(projections * target_signs, axis=1)
    
    # 5. 将分数转化为 Logit 偏置 (Bias) 加回到原始 Logits 上
    # alpha 是引导强度，可以调整 (例如 5.0 - 20.0)
    alpha = 3.0 
    scores_tensor = torch.tensor(scores, device=DEVICE)
    
    # 只更新 Top-K 的 logits，其他 token 保持原样 (或设为负无穷)
    new_logits = logits.clone()
    # 先把 Top-K 以外的词概率压低，专注于我们在 K 个里面选最好的
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[top_k_indices] = False
    new_logits[mask] = -float('inf')
    
    # 给 Top-K 加上引导分
    new_logits[top_k_indices] += alpha * scores_tensor
    print(f"Top 1 score: {scores.max().item()}")
    return new_logits
# 修改 generate_response 签名，接收 hash_fn 和 target_bits

# --- 核心生成函数：修改为使用本地模型 ---
# 移除了 client 参数，因为它不再是 OpenAI client
def generate_response(model, conversation_history, hash_fn=None, target_bits=None, system_prompt=None, max_length=100):
    
    # 注意：本地模型通常不能很好地理解 system_prompt，但我们留着它
    # 1. 构造历史上下文
    prompt = "\n".join(conversation_history)
    
    # 2. Tokenize
    #inputs = model.tokenizer(prompt, return_tensors='pt')['input_ids']
    #确保输入也在 GPU 上
    inputs = model.tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

    # [关键修复]：严格计算剩余空间
    # GPT-2 上下文最大长度# 通常是 1024
    ctx_limit = model.context_length  
    
    # 我们计划生成 max_length 这么长，所以输入必须留出空间
    # 安全起见，我们再多留 4 个 token 的余量
    safe_input_limit = ctx_limit - max_length - 4
    
    # 如果当前输入超过了安全限制，进行截断，只保留最后面的 safe_input_limit 个 token
    if inputs.shape[1] > safe_input_limit:
        inputs = inputs[:, -safe_input_limit:]

    # 初始化输出 token 列表
    output_tokens = inputs
    
    # --- 您的 Logit 引导循环就将发生在这里！ ---
    # 3. 逐个 Token 生成
    for _ in range(max_length):
        # 这里的 output_tokens 始终已经被限制在 1024 长度以内
        context_tokens = output_tokens
        
        with torch.no_grad(): # 暂时禁用梯度，未来需要移除这行
            # 模型计算在 GPU 进行
            outputs = model._model(context_tokens)
            # 获取下一个 token 的概率分布（Logits）
            logits = outputs.logits[0, -1, :]
            # --- [插入引导逻辑] ---
            if hash_fn is not None and target_bits is not None:
                # 使用引导后的 Logits 覆盖原始 Logits
                logits = guide_logits(model, context_tokens, logits, target_bits, hash_fn)
            # --------------------
            probabilities = torch.softmax(logits, dim=-1)

        # 这里是您未来插入引导逻辑的地方！
        # 例如：perturbed_probabilities = guide_logits(probabilities, current_embedding, target_hash)

        # 采样下一个 token
        next_token = torch.multinomial(probabilities, 1).item()
        
        #如果生成了结束符，强制停止！防止无限循环
        if next_token == model.tokenizer._tokenizer.eos_token_id:
            break

        # 将新 token 加入序列
        #新生成的 token 也要在 GPU 上
        token_tensor = torch.tensor([[next_token]]).to(DEVICE)
        output_tokens = torch.cat((output_tokens, token_tensor), dim=1)
        # 双重保险：如果在循环中意外达到了 1024，立即停止
        if output_tokens.shape[1] >= ctx_limit:
            break

       
    
    # 4. 解码新生成的文本
    generated_text = model.tokenizer.decode(output_tokens[0])
    
    # 这里的切片处理可能需要根据 prompt 长度调整，简单起见先以此为例
    # 注意：decode 可能会包含 prompt，这里假设 prompt 长度对应的字符数
    # 更稳健的做法是只 decode 新生成的 tokens:
    new_tokens = output_tokens[0, inputs.shape[1]:]
    new_response = model.tokenizer.decode(new_tokens)
    
    # 清理和返回
    text = clean_response(new_response)
    if not text: return "..."
    return text

if __name__ == "__main__":
    pass
'''
    # Example usage:
    conversation_history = [
        "What are you up to today?",
        "Nothing much, I'm just working on a project.",
        "Do you want me to take a look? We can grab some coffee."
    ]

    client = openai.OpenAI()
    response = generate_response(client, conversation_history)
    print("Generated response:", response)
'''
    
