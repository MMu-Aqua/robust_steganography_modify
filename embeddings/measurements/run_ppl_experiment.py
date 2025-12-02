# 文件名: embeddings/measurements/run_ppl_experiment.py
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

def calculate_ppl(text, model, device):
    """使用 GPT-2 计算文本的困惑度 (Perplexity)"""
    tokenizer = model.tokenizer
    # 编码文本
    encodings = tokenizer(text, return_tensors='pt').to(device)
    
    # 这里的 max_length 是模型的窗口大小，GPT-2 是 1024
    max_length = model.context_length
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model._model(input_ids, labels=target_ids)
            
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running PPL Experiment on {device}...")
    
    # 配置
    model = get_local_model()
    num_trials = 20   # 生成 20 条样本取平均
    test_bits = 4     # 使用中等难度
    max_len = 200     # [关键] 生成长文本以评估质量
    
    # 1. 准备哈希
    hash_fn = RandomProjectionHash(embedding_dim=384, num_bits=test_bits) 
    target_bits = np.random.randint(0, 2, size=test_bits).tolist()
    history = ["The advancement of artificial intelligence has"]
    
    results = []
    alphas = [0.0, 3.0] # 对比组
    
    for alpha in alphas:
        method_name = "Ours (Guided)" if alpha > 0 else "Baseline (Random)"
        print(f"\n--- Testing {method_name} (Alpha={alpha}) ---")
        
        ppl_scores = []
        
        for _ in tqdm(range(num_trials)):
            # 生成文本
            try:
                # 注意：这里我们不需要强制 limit 很高，因为只测质量
                text, _ = sample_concurrent(
                    model, target_bits, history, hash_fn, 
                    alpha=alpha, max_length=max_len, max_attempts=100
                )
                
                # 计算 PPL
                # 注意：GPT-2 的 PPL 计算需要完整的句子，我们只计算新生成部分的 PPL
                # 或者计算 历史+生成 的整体 PPL，这里简单起见算整体
                ppl = calculate_ppl(text, model, device)
                ppl_scores.append(ppl)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        avg_ppl = np.mean(ppl_scores)
        print(f"  -> Avg PPL: {avg_ppl:.2f}")
        
        results.append({
            "Method": method_name,
            "Alpha": alpha,
            "Avg_PPL": avg_ppl
        })
        
    # 保存
    df = pd.DataFrame(results)
    print("\n=== PPL Results ===")
    print(df)
    df.to_csv(os.path.join(current_dir, "ppl_results_1.csv"), index=False)

if __name__ == "__main__":
    main()