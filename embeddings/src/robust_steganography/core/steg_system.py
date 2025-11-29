import openai
from .error_correction import RepetitionCode, ConvolutionalCode
from .hash_functions import RandomProjectionHash, PCAHash, OracleHash
from .encoder import Encoder, StandardEncoder
from .simulation import Simulator
from ..utils.steg import encode
from ..utils.get_embedding import get_embeddings_in_batch
from ..utils.paraphrase import paraphrase_message
from typing import Any
# 引入本地模型加载函数
from ..utils.new_text import get_local_model

class StegSystem:
    def __init__(self, 
                 client,
                 hash_function,
                 error_correction,
                 encoder: Encoder = None,
                 system_prompt=None,
                 chunk_length=200,
                 simulator=None):
        self.client = client
        self.hash_fn = hash_function
        self.ecc = error_correction
        self.encoder = encoder or StandardEncoder()
        self.system_prompt = system_prompt
        self.chunk_length = chunk_length
        self.simulator = simulator
        
        # Get hash output length
        self.hash_output_length = getattr(hash_function, 'output_length')
        
        if self.simulator and not isinstance(hash_function, OracleHash):
            raise ValueError("Simulation mode can only be used with OracleHash, "
                           f"not {type(hash_function).__name__}")

    '''
    def hide_message(self, data: Any, history):
        # Get raw bits from encoder
        m_bits = self.encoder.encode(data)
        
        # Let the ECC handle any necessary padding
        m_encoded = self.ecc.encode(m_bits)
        
        # Convert to chunks of size hash_output_length
        m_chunks = [m_encoded[i:i + self.hash_output_length] 
                    for i in range(0, len(m_encoded), self.hash_output_length)]
        
        if self.simulator:
            cover_texts = []
            for desired_bits in m_chunks:
                while True:
                    text = self.simulator.generate_dummy_text()
                    embedding = self.simulator.get_embedding(text)
                    hash_bits = self.hash_fn(embedding)
                    if all(h == d for h, d in zip(hash_bits, desired_bits)):
                        cover_texts.append(text)
                        break
            return cover_texts
        
        # Normal mode - use real API calls
        cover_text = encode(
            self.client, 
            m_chunks,
            history, 
            self.hash_fn,
            system_prompt=self.system_prompt,
            max_length=self.chunk_length
        )
        paraphrases = [paraphrase_message(self.client, text) for text in cover_text]
        return paraphrases
     '''  
     
    def hide_message(self, data: Any, history):
    
        #Hides a secret message in generated text using local LLM and embedding-based rejection sampling.

        # 1. 消息编码和纠错 (保留核心逻辑)
        # Get raw bits from encoder
        m_bits = self.encoder.encode(data)
        
        # Let the ECC handle any necessary padding
        m_encoded = self.ecc.encode(m_bits)
        
        # Convert to chunks of size hash_output_length
        m_chunks = [m_encoded[i:i + self.hash_output_length] 
                    for i in range(0, len(m_encoded), self.hash_output_length)]
        
        # 2. 模拟器模式 (保留原版代码)
        if self.simulator:
            cover_texts = []
            for desired_bits in m_chunks:
                while True:
                    # generate_dummy_text and get_embedding will use the Simulator's internal oracle/memory
                    text = self.simulator.generate_dummy_text()
                    embedding = self.simulator.get_embedding(text)
                    hash_bits = self.hash_fn(embedding)
                    
                    if all(h == d for h, d in zip(hash_bits, desired_bits)):
                        cover_texts.append(text)
                        break
            return cover_texts
        
        # 3. 正常模式 (替换为本地模型逻辑)
        
        # 加载本地的 GPT-2/NanoGPT 模型实例
        # 注意：get_local_model() 已经被修改为返回 GPT2Model
        llm_model = get_local_model() 
        
        # 调用 encode 函数进行生成和拒绝采样
        # encode 函数（位于 steg.py）将使用 llm_model 来调用 new_text.py 中的生成逻辑
        cover_text = encode(
            llm_model,                  # <--- 传递本地模型实例
            m_chunks,
            history, 
            self.hash_fn,
            system_prompt=self.system_prompt,
            max_length=self.chunk_length
        )
        
        # 移除了原版中最后的付费 paraphrase 步骤 (paraphrases = [...])
        return cover_text

    def recover_message(self, stego_texts):
        if self.simulator:
            embeddings = [self.simulator.get_embedding(text) for text in stego_texts]
            bits_encoded = [self.hash_fn(emb, corrupt=True) for emb in embeddings]
        else:
            embeddings = get_embeddings_in_batch(self.client, stego_texts)
            bits_encoded = [self.hash_fn(emb) for emb in embeddings]
        
        m_bits = self.ecc.decode(bits_encoded)
        
        return self.encoder.decode(m_bits)