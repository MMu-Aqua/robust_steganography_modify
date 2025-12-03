import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../src")
sys.path.append(src_dir)

from robust_steganography.core.hash_functions import RandomProjectionHash
from robust_steganography.utils.new_text import get_local_model
from robust_steganography.utils.steg import sample_concurrent 

def calculate_ppl_with_context(history_text, new_text, model, device):
    """
    计算 PPL 时必须包含上文 (Context)，否则分数会虚高。
    我们只计算 'new_text' 部分的 Loss，但需要 'history_text' 作为条件。
    """
    tokenizer = model.tokenizer
    
    # 1. 拼接全文
    full_text = history_text + new_text
    
    # 2. 编码
    encodings = tokenizer(full_text, return_tensors='pt').to(device)
    input_ids = encodings.input_ids
    
    # 3. 找到 new_text 开始的位置
    # 简单做法：编码 history 看看有多长
    history_ids = tokenizer(history_text, return_tensors='pt').input_ids
    start_loc = history_ids.size(1)
    
    # 4. 构造 labels (只计算后半部分的 loss)
    target_ids = input_ids.clone()
    target_ids[:, :start_loc] = -100 # -100 表示忽略这部分的 Loss

    # 5. 计算 Loss
    with torch.no_grad():
        outputs = model._model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    # 6. 转换为 PPL
    ppl = torch.exp(neg_log_likelihood)
    return ppl.item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running PPL Experiment on {device}...")
    
    model = get_local_model()
    num_trials = 20   
    test_bits = 4     # 4-bit 难度适中，适合测质量
    max_len = 100     # 生成 100 个词足够了
    
    hash_fn = RandomProjectionHash(embedding_dim=384, num_bits=test_bits) 
    
    # 使用一个固定的 Prompt
    base_prompt = "The development of artificial intelligence has led to"
    history = [base_prompt] 
    
    results = []
    alphas = [0.0, 3.0] 
    
    for alpha in alphas:
        method_name = "Ours (Guided)" if alpha > 0 else "Baseline (Random)"
        print(f"\n--- Testing {method_name} (Alpha={alpha}) ---")
        
        ppl_scores = []
        
        for _ in tqdm(range(num_trials)):
            target_bits = np.random.randint(0, 2, size=test_bits).tolist()
            
            try:
                # 生成文本
                # 注意：我们不需要在这里拼接 prompt，sample_concurrent 会处理
                text, _ = sample_concurrent(
                    model, target_bits, history, hash_fn, 
                    alpha=alpha, max_length=max_len, max_attempts=100
                )
                
                # 计算 PPL (传入 Prompt 和 新生成的 Text)
                # sample_concurrent 返回的是完整文本还是只有新生成的？
                # 根据 new_text.py，它返回的是 clean_response(new_response) 即只有新部分
                # 但 steg.py 可能会做处理。为了保险，我们假设返回的是 content。
                
                # 注意：steg.py 返回的 text 可能包含 prompt 也可能不包含
                # 现在的 generate_response 逻辑是只返回新内容。
                # 但为了防止拼接错误，我们打印一下看看
                # print(f"DEBUG: {text[:20]}...")
                
                if not text: continue

                ppl = calculate_ppl_with_context(base_prompt, " " + text, model, device)
                ppl_scores.append(ppl)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        if ppl_scores:
            avg_ppl = np.mean(ppl_scores)
            print(f"  -> Avg PPL: {avg_ppl:.2f}")
            
            results.append({
                "Method": method_name,
                "Alpha": alpha,
                "Avg_PPL": avg_ppl
            })
        
    df = pd.DataFrame(results)
    print("\n=== PPL Results ===")
    print(df)
    df.to_csv(os.path.join(current_dir, "ppl_results_2.csv"), index=False)

if __name__ == "__main__":
    main()