#!/usr/bin/env python3
"""
第一步：基础模型加载器
测试7B模型是否能正常加载和推理
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BasicModelLoader:
    """基础模型加载器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """加载模型"""
        logger.info(f"正在加载模型: {self.model_name}")
        
        try:
            # 直接加载到CPU，不使用量化
            logger.info("尝试直接加载到CPU...")
            
            # 加载tokenizer
            logger.info("加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 加载模型
            logger.info("加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            
            logger.info("模型加载成功！")
            return True
            
        except Exception as e:
            logger.error(f"GPU加载失败: {e}")
            logger.info("尝试使用CPU加载...")
            
            try:
                # 如果GPU加载失败，尝试CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                logger.info("模型在CPU上加载成功！")
                return True
                
            except Exception as e2:
                logger.error(f"CPU加载也失败: {e2}")
                return False
    
    def test_inference(self, prompt: str = "你好，请介绍一下自己") -> str:
        """测试模型推理"""
        if self.model is None or self.tokenizer is None:
            logger.error("模型未加载")
            return "模型未加载"
        
        logger.info(f"测试推理，输入: {prompt}")
        
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取新生成的部分
            new_response = response[len(prompt):].strip()
            
            logger.info(f"推理结果: {new_response}")
            return new_response
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return f"推理失败: {e}"
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        if self.model is None:
            return {"status": "未加载"}
        
        info = {
            "model_name": self.model_name,
            "status": "已加载",
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "parameters": sum(p.numel() for p in self.model.parameters())
        }
        
        return info


def main():
    """主函数"""
    logger.info("=== 第一步：基础模型加载测试 ===")
    
    # 创建模型加载器
    loader = BasicModelLoader()
    
    # 加载模型
    success = loader.load_model()
    
    if success:
        # 获取模型信息
        info = loader.get_model_info()
        logger.info(f"模型信息: {info}")
        
        # 测试推理
        test_prompts = [
            "你好，请介绍一下自己",
            "什么是强化学习？",
            "请解释一下工具调用的概念"
        ]
        
        for prompt in test_prompts:
            logger.info(f"\n--- 测试提示: {prompt} ---")
            response = loader.test_inference(prompt)
            logger.info(f"响应: {response}")
    
    else:
        logger.error("模型加载失败，无法进行测试")


if __name__ == "__main__":
    main()
