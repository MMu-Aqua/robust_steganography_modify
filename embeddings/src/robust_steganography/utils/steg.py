import logging
import concurrent.futures
import numpy as np
from .new_text import generate_response
from .get_embedding import get_embeddings_in_batch

def sample_concurrent(
    model, 
    desired_bits,  # List of bits (the chunk)
    history, 
    hash_fn, 
    k=1, # 在Logit引导模式下，k不再代表并发线程数，保留此参数是为了兼容接口
    system_prompt="You are having a casual conversation.",
    max_length=200,
    alpha=3.0,  # 新增默认参数
    max_attempts=1000  # [修改点] 默认值改大，或者允许外部传入
):
    """
    使用带有 Logit 引导的生成函数来尝试生成符合 desired_bits 的文本。
    不再使用多线程并发，而是使用“生成-检查”的循环。由于有引导，通常 1-2 次即可成功。
    """
    sampled_bits = None
    attempts = 0
    
    # [关键] 复制原始历史记录，用于每次重置
    base_history = list(history) 
    
    while attempts < max_attempts:
        attempts += 1
        
        # --- [策略 1: 动态扰动 (Input-level)] ---
        # 如果不是第一次尝试，就在 Prompt 后面加空格
        # 第1次: 无空格
        # 第2次: " "
        # 第3次: "  "
        # 这样每次输入模型的 Prompt 都不一样，生成的文本自然也会不同！
        current_history = base_history.copy()
        if attempts > 1 and len(current_history) > 0:
            # 在最后一条历史消息后面追加 (attempts-1) 个空格
            # 这是一个对人类不可见，但能改变模型生成的有效技巧
            current_history[-1] = current_history[-1] + " " * (attempts - 1)
        # --- [策略 2: 混合回退 (Output-level)] ---
        # 第 1 次尝试：使用强引导 (alpha=3.0)，争取“精确制导”一次过
        # 第 2+ 次尝试：如果失败，说明引导方向可能陷入局部最优，切换回随机模式 (alpha=0.0)
        current_alpha = alpha if attempts == 1 else 0.0
        
        # 打印日志方便观察策略切换
        if attempts == 2:
            print(f"  [Fallback] Guidance failed, switching to Random Sampling (alpha=0)...")

        # 生成文本
        message = generate_response(
            model, 
            current_history, # 使用带扰动的历史
            hash_fn=hash_fn,
            target_bits=desired_bits,
            system_prompt=system_prompt,
            max_length=max_length,
            alpha=current_alpha # 使用动态调整的 alpha
        )
        
        # --- 验证环节 ---
        # 即使有引导，我们也需要验证生成的文本是否真的符合哈希要求
        # client 传 None，因为 get_embeddings_in_batch 现在使用本地模型
        embeddings = get_embeddings_in_batch(None, [message])
        
        # 计算生成文本的实际哈希
        emb = np.array(embeddings[0]).reshape(1, -1)
        sampled_bits = hash_fn(emb)
        
        print(f"Attempt {attempts}: Target={desired_bits}, Got={sampled_bits}")
        print('Generated message:', message) # 调试时可取消注释
        
        # 如果哈希匹配，直接返回
        if np.array_equal(sampled_bits, desired_bits):
            # 成功！返回生成的文本（去除可能添加的尾部空格）
            return message.rstrip(), attempts  # <--- 修改这里，返回 tuple
            
    # 如果实在不行，返回最后一次的结果（交由纠错码处理）
    print(f"Warning: Failed after {max_attempts} attempts.")
    # 失败时，返回 (文本, 次数)
    return message, attempts

def encode(
    model, 
    chunks, 
    history, 
    hash_fn, 
    k=1, 
    system_prompt="You are having a casual conversation.", 
    max_length=200, 
    alpha=3.0, 
    max_attempts=1000
):
    """
    编码主函数：遍历所有数据块，依次生成隐写文本。
    """
    cover_text = []
    for chunk in chunks:
        # 对每个数据块进行采样（引导生成）
        response, _ = sample_concurrent(
            model, 
            chunk,
            history, 
            hash_fn,
            k=k,
            system_prompt=system_prompt,
            max_length=max_length,
            alpha=alpha,
            max_attempts=max_attempts # [修改点] 传递参数
        )
        # 将生成的文本加入历史记录，用于下一次生成
        history.append(response)
        cover_text.append(response)
        # 这里 encode 函数接口通常只返回文本，为了兼容性我们只返回文本列表
        # 如果需要统计，可以在这里打印或记录
        
    return cover_text
    