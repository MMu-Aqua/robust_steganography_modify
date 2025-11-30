# This file contains the code to sample a new message
import torch
import openai
import re
# 导入本地模型
from watermark.models.gpt2 import GPT2Model
import torch
# --- 引入本地模型（新增全局变量） ---
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


'''
def generate_response(client, conversation_history, system_prompt="You are a highly dynamic conversational model tasked with generating responses that are extremely varied in tone, content, and structure.", max_length=100):
    # Prepare the prompt from the conversation history
    prompt = "\n".join(conversation_history) + "\n"

    try:
        # Generate a response using GPT-4o mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Original model name preserved
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_length,  # Use passed in max_length
            temperature=1.0,  # Preserved original temperature
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
        )
        
        # Extract and return the generated response text
        text = response.choices[0].message.content.strip()
        text = clean_response(text)
        return text

    except Exception as e:
        return f"An error occurred: {e}"
'''
# --- 核心生成函数：修改为使用本地模型 ---
# 移除了 client 参数，因为它不再是 OpenAI client
def generate_response(model, conversation_history, system_prompt=None, max_length=100):
    
    # 注意：本地模型通常不能很好地理解 system_prompt，但我们留着它
    # 1. 构造历史上下文
    prompt = "\n".join(conversation_history)
    
    # 2. Tokenize
    #inputs = model.tokenizer(prompt, return_tensors='pt')['input_ids']
    #确保输入也在 GPU 上
    inputs = model.tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEVICE)

    # [关键修复]：严格计算剩余空间
    # GPT-2 上下文最大长度
    ctx_limit = model.context_length  # 通常是 1024
    
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
    
