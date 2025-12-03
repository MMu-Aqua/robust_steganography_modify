import sys
import os
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../src")
sys.path.append(src_dir)

from robust_steganography.core.hash_functions import RandomProjectionHash
from robust_steganography.core.encoder import MinimalEncoder
from robust_steganography.utils.new_text import get_local_model
from robust_steganography.utils.steg import sample_concurrent 
from robust_steganography.utils.get_embedding import get_embeddings_in_batch # [新增] 用于验证

def run_single_trial(alpha, num_bits):
    """运行一次实验，必须验证哈希是否正确"""
    hash_fn = RandomProjectionHash(embedding_dim=384, num_bits=num_bits) 
    target_bits = np.random.randint(0, 2, size=num_bits).tolist()
    
    model = get_local_model()
    history = ["The technology industry is evolving rapidly."]
    
    start_time = time.time()
    
    # [关键设置] 给足够多的尝试次数，让 Baseline 必须跑通
    # 8 bits 理论平均需要 256 次，我们给 1000 次上限
    # 注意：这需要 steg.py 的 sample_concurrent 支持 max_attempts 参数
    # 如果没改 steg.py，默认是 15，那这里怎么传都没用，Baseline 永远会失败
    limit = 10000
    
    attempts = 0
    try:
        response, attempts = sample_concurrent(
            model, 
            target_bits, 
            history, 
            hash_fn, 
            alpha=alpha,
            max_length=30,
            max_attempts=limit # [需要修改 steg.py 支持此参数，或者手动去 steg.py 改默认值]
        )
        
        # [核心修复] 验证结果是否正确！
        embeddings = get_embeddings_in_batch(None, [response])
        emb = np.array(embeddings[0]).reshape(1, -1)
        sampled_bits = hash_fn(emb)
        
        if np.array_equal(sampled_bits, target_bits):
            success = True
        else:
            success = False # 即使跑完了，没对也不算！
            
    except Exception as e:
        print(f"Error: {e}")
        success = False
        # 核心解释：
        # 如果程序崩得太快，导致 sample_concurrent 甚至没来得及返回 attempts
        # 此时本地变量 attempts 依然是初始值 0。
        # 如果我们就这样把 0 记入统计数据，会导致 "Avg Attempts"（平均尝试次数）被错误地拉低。
        # 比如：[100次, 200次, 0次(其实是报错)] -> 平均 100次。这会让你看起来“莫名其妙地变快了”。
        
        # 所以，我们强制规定：只要报错失败，就当它是“最坏情况”
        # 把它记为“跑满了上限（limit）次”，这样能保证实验数据的诚实性（惩罚失败）。
        if attempts == 0: 
            attempts = limit
        
    end_time = time.time()
    duration = end_time - start_time
    
    return duration, success, attempts

def main():
    results = []
    
    # 测试范围
    bits_list = [1, 2, 3, 4, 5, 6, 7, 8] 
    alphas = [0.0, 3.0]
    num_trials = 20 # 8 bits 很慢，可以适当减少次数，比如 5 次
    
    print(f"Starting Experiments on {torch.cuda.get_device_name(0)}")
    
    for n_bits in bits_list:
        print(f"\n====== Testing Bits = {n_bits} ======")
        for alpha in alphas:
            method_name = "Ours (Guided)" if alpha > 0 else "Baseline (Random)"
            durations = []
            all_attempts = [] # [新增] 用于记录尝试次数
            success_count = 0
            
            # 这里的 tqdm 会显示每一步的耗时
            for _ in tqdm(range(num_trials), desc=f"{method_name}"):
                dur, success, attempts = run_single_trial(alpha, num_bits=n_bits)
                all_attempts.append(attempts)
                if success:
                    durations.append(dur)
                    success_count += 1
                else:
                    # 失败时不计入 duration (不使用惩罚时间)，以免拉偏平均值
                    # 但因为我们记录了 Success Rate，所以数据依然诚实
                    pass

            # 计算统计数据
            if durations:
                avg_time = np.mean(durations)
            else:
                avg_time = 0.0 # 避免除以零
            
            # [新增] 计算平均尝试次数
            avg_attempts = np.mean(all_attempts) if all_attempts else 0
            
            print(f"  -> Avg Time: {avg_time:.4f}s | Avg Attempts: {avg_attempts:.1f} | Success: {success_count}/{num_trials}")
            
            results.append({
                "Num_Bits": n_bits,
                "Method": method_name,
                "Alpha": alpha,
                "Avg_Time": avg_time,
                "Avg_Attempts": avg_attempts, # [新增] 最强指标
                "Success_Rate": success_count/num_trials
            })

    df = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(df)
    
    output_file = os.path.join(current_dir, "Lookahead_run_experiments_8bit_5_20trial_max10000.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()