# This file contains the code to sample a new message

import openai
import re
# 导入本地模型
from watermark.models.gpt2 import GPT2Model
import torch
# --- 引入本地模型（新增全局变量） ---
_LOCAL_MODEL = None

#新增
def get_local_model():
    """惰性加载本地模型，只加载一次"""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        print("Loading local GPT-2 model (gpt2)...")
        # 默认加载 GPT-2 小模型，速度较快
        _LOCAL_MODEL = GPT2Model(model_name='gpt2') 
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
    inputs = model.tokenizer(prompt, return_tensors='pt')['input_ids']
    
    # [新增的 1024 上下文窗口截断逻辑]
    context_length = model.context_length # 1024
    if inputs.shape[1] > context_length:
        # 只保留模型能够处理的最后 1024 个 token
        inputs = inputs[:, -context_length:]

    # 初始化输出 token 列表
    output_tokens = inputs
    
    # --- 您的 Logit 引导循环就将发生在这里！ ---
    # 3. 逐个 Token 生成
    for _ in range(max_length):
        # 这里的 output_tokens 始终已经被限制在 1024 长度以内
        context_tokens = output_tokens
        
        with torch.no_grad(): # 暂时禁用梯度，未来需要移除这行
            # 获取下一个 token 的概率分布（Logits）
            logits = model._model(context_tokens)['logits'][0, -1, :] 
            probabilities = torch.softmax(logits, dim=-1)

        # 这里是您未来插入引导逻辑的地方！
        # 例如：perturbed_probabilities = guide_logits(probabilities, current_embedding, target_hash)

        # 采样下一个 token
        next_token = torch.multinomial(probabilities, 1).item()
        
        # 将新 token 加入序列
        token_tensor = torch.tensor([[next_token]])
        output_tokens = torch.cat((output_tokens, token_tensor), dim=1)
        # 实时截断 output_tokens，防止它在循环内继续增长超过限制
        if output_tokens.shape[1] > context_length:
             output_tokens = output_tokens[:, -context_length:]

        # 检查是否生成了结束符（可选，GPT-2可能不会生成）
        #if next_token == model.tokenizer._tokenizer.eos_token_id:
        #    break
    
    # 4. 解码新生成的文本
    generated_text = model.tokenizer.decode(output_tokens[0])
    
    # 5. 只返回新生成的文本部分（去除历史记录）
    new_response = generated_text[len(prompt):]
    
    # 清理和返回
    text = clean_response(new_response)
    return text

if __name__ == "__main__":

    # Example usage:
    conversation_history = [
        "What are you up to today?",
        "Nothing much, I'm just working on a project.",
        "Do you want me to take a look? We can grab some coffee."
    ]

    client = openai.OpenAI()
    response = generate_response(client, conversation_history)
    print("Generated response:", response)
