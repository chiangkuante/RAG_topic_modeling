import os
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
import time

# LLM API 套件
import openai
import google.generativeai as genai
import anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """LLM配置類"""
    provider: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30

class BaseLLM(ABC):
    """LLM基類"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._setup_client()
    
    @abstractmethod
    def _setup_client(self):
        """設置LLM客戶端"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成文本"""
        pass

class OpenAILLM(BaseLLM):
    """OpenAI LLM實現"""
    
    def _setup_client(self):
        try:
            openai.api_key = self.config.api_key
            self.client = openai.OpenAI(api_key=self.config.api_key)
            logger.info("OpenAI客戶端初始化成功")
        except Exception as e:
            logger.error(f"OpenAI客戶端初始化失敗：{e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                timeout=kwargs.get('timeout', self.config.timeout)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI生成失敗：{e}")
            raise
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
                time.sleep(0.1)  # 避免速率限制
            except Exception as e:
                logger.error(f"批量生成中的錯誤：{e}")
                results.append("")
        return results

class GeminiLLM(BaseLLM):
    """Google Gemini LLM實現"""
    
    def _setup_client(self):
        try:
            genai.configure(api_key=self.config.api_key)
            self.client = genai.GenerativeModel(self.config.model)
            logger.info("Gemini客戶端初始化成功")
        except Exception as e:
            logger.error(f"Gemini客戶端初始化失敗：{e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', self.config.temperature),
                max_output_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            )
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini生成失敗：{e}")
            raise
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
                time.sleep(0.1)  # 避免速率限制
            except Exception as e:
                logger.error(f"批量生成中的錯誤：{e}")
                results.append("")
        return results

class ClaudeLLM(BaseLLM):
    """Anthropic Claude LLM實現"""
    
    def _setup_client(self):
        try:
            self.client = anthropic.Anthropic(api_key=self.config.api_key)
            logger.info("Claude客戶端初始化成功")
        except Exception as e:
            logger.error(f"Claude客戶端初始化失敗：{e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude生成失敗：{e}")
            raise
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
                time.sleep(0.1)  # 避免速率限制
            except Exception as e:
                logger.error(f"批量生成中的錯誤：{e}")
                results.append("")
        return results

class LLMManager:
    """LLM管理器"""
    
    SUPPORTED_PROVIDERS = {
        'openai': {
            'class': OpenAILLM,
            'models': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview']
        },
        'gemini': {
            'class': GeminiLLM,
            'models': ['gemini-pro', 'gemini-pro-vision']
        },
        'claude': {
            'class': ClaudeLLM,
            'models': ['claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
        }
    }
    
    def __init__(self):
        self.active_llm = None
    
    def get_supported_providers(self) -> Dict[str, List[str]]:
        """獲取支持的提供者和模型"""
        return {provider: info['models'] for provider, info in self.SUPPORTED_PROVIDERS.items()}
    
    def create_llm(self, provider: str, model: str, api_key: str, **kwargs) -> BaseLLM:
        """創建LLM實例"""
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"不支持的LLM提供者：{provider}")
        
        if model not in self.SUPPORTED_PROVIDERS[provider]['models']:
            raise ValueError(f"{provider} 不支持模型：{model}")
        
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs
        )
        
        llm_class = self.SUPPORTED_PROVIDERS[provider]['class']
        llm = llm_class(config)
        
        return llm
    
    def set_active_llm(self, llm: BaseLLM):
        """設置活躍的LLM"""
        self.active_llm = llm
        logger.info(f"設置活躍LLM：{llm.config.provider} - {llm.config.model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """使用活躍LLM生成文本"""
        if self.active_llm is None:
            raise ValueError("請先設置活躍的LLM")
        
        return self.active_llm.generate(prompt, **kwargs)
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """使用活躍LLM批量生成文本"""
        if self.active_llm is None:
            raise ValueError("請先設置活躍的LLM")
        
        return self.active_llm.generate_batch(prompts, **kwargs)
    
    def test_connection(self, provider: str, model: str, api_key: str) -> bool:
        """測試LLM連接"""
        try:
            llm = self.create_llm(provider, model, api_key)
            test_prompt = "Hello, this is a test. Please respond with 'Connection successful.'"
            response = llm.generate(test_prompt, max_tokens=50)
            return "successful" in response.lower() or len(response) > 0
        except Exception as e:
            logger.error(f"連接測試失敗：{e}")
            return False