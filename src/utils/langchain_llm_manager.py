"""
LangChain-based LLM Manager
统一管理不同LLM提供商的LangChain集成
"""

from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum

from langchain_core.language_models import BaseLanguageModel  
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"

@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1000

class LangChainLLMManager:
    """基於LangChain的LLM管理器"""
    
    SUPPORTED_MODELS = {
        LLMProvider.OPENAI: {
            'models': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview', 'gpt-4o'],
            'class': ChatOpenAI
        },
        LLMProvider.GEMINI: {
            'models': ['gemini-pro', 'gemini-1.5-pro'],
            'class': ChatGoogleGenerativeAI
        },
        LLMProvider.CLAUDE: {
            'models': ['claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'claude-3-opus-20240229'],
            'class': ChatAnthropic
        }
    }
    
    def __init__(self):
        self.active_llm: Optional[BaseLanguageModel] = None
        self.config: Optional[LLMConfig] = None
        
    def create_llm(self, provider: str, model: str, api_key: str, **kwargs) -> BaseLanguageModel:
        """創建LangChain LLM實例"""
        provider_enum = LLMProvider(provider)
        
        if provider_enum not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的LLM提供者：{provider}")
        
        models = self.SUPPORTED_MODELS[provider_enum]['models']
        if model not in models:
            raise ValueError(f"{provider} 不支持模型：{model}")
        
        llm_class = self.SUPPORTED_MODELS[provider_enum]['class']
        
        # 準備參數
        llm_kwargs = {
            'model': model,
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1000),
        }
        
        # 根據提供者設置API密鑰
        if provider_enum == LLMProvider.OPENAI:
            llm_kwargs['openai_api_key'] = api_key
        elif provider_enum == LLMProvider.GEMINI:
            llm_kwargs['google_api_key'] = api_key
        elif provider_enum == LLMProvider.CLAUDE:
            llm_kwargs['anthropic_api_key'] = api_key
        
        try:
            llm = llm_class(**llm_kwargs)
            logger.info(f"創建LLM成功：{provider} - {model}")
            return llm
        except Exception as e:
            logger.error(f"創建LLM失敗：{e}")
            raise
    
    def set_active_llm(self, llm: BaseLanguageModel, config: LLMConfig = None):
        """設置活躍的LLM"""
        self.active_llm = llm
        self.config = config
        logger.info(f"設置活躍LLM：{config.provider.value if config else 'unknown'} - {config.model if config else 'unknown'}")
    
    def get_active_llm(self) -> BaseLanguageModel:
        """獲取活躍的LLM"""
        if self.active_llm is None:
            raise ValueError("請先設置活躍的LLM")
        return self.active_llm
    
    def create_chain(self, prompt_template: Union[str, PromptTemplate], 
                    output_parser: Optional[BaseOutputParser] = None) -> RunnableSequence:
        """創建LangChain鏈"""
        if self.active_llm is None:
            raise ValueError("請先設置活躍的LLM")
        
        # 準備prompt
        if isinstance(prompt_template, str):
            prompt = PromptTemplate.from_template(prompt_template)
        else:
            prompt = prompt_template
        
        # 準備輸出解析器
        if output_parser is None:
            output_parser = StrOutputParser()
        
        # 創建鏈
        chain = prompt | self.active_llm | output_parser
        
        return chain
    
    def invoke_chain(self, chain: RunnableSequence, inputs: Dict[str, Any]) -> Any:
        """執行鏈"""
        try:
            result = chain.invoke(inputs)
            return result
        except Exception as e:
            logger.error(f"鏈執行失敗：{e}")
            raise
    
    def test_connection(self, provider: str, model: str, api_key: str) -> bool:
        """測試LLM連接"""
        try:
            llm = self.create_llm(provider, model, api_key)
            
            # 創建測試鏈
            test_prompt = PromptTemplate.from_template("Say 'Connection successful' if you can read this.")
            test_chain = test_prompt | llm | StrOutputParser()
            
            # 執行測試
            result = test_chain.invoke({})
            return "successful" in result.lower() or len(result) > 0
            
        except Exception as e:
            logger.error(f"連接測試失敗：{e}")
            return False
    
    def get_supported_providers(self) -> Dict[str, List[str]]:
        """獲取支持的提供者和模型"""
        return {
            provider.value: info['models'] 
            for provider, info in self.SUPPORTED_MODELS.items()
        }