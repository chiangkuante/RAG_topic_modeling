import os
from pathlib import Path
from typing import Dict, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """配置管理類"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_default_config()
        
        if self.config_path.exists():
            self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """載入默認配置"""
        return {
            "data": {
                "sample_size": None,
                "min_word_length": 2,
                "max_documents": 10000
            },
            "vectorizer": {
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384,
                "batch_size": 32
            },
            "llm": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "timeout": 30
            },
            "topic_modeling": {
                "num_topics": 10,
                "max_iterations": 5,
                "quality_threshold": 0.7,
                "num_rounds": 5
            },
            "evaluation": {
                "top_k_words": 20,
                "similarity_threshold": 0.5
            },
            "ui": {
                "max_display_docs": 100,
                "chart_height": 400
            }
        }
    
    def _load_config(self):
        """從文件載入配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # 遞歸更新配置
            self._update_config(self.config, file_config)
            logger.info(f"配置已從 {self.config_path} 載入")
            
        except Exception as e:
            logger.error(f"載入配置文件失敗：{e}")
    
    def _update_config(self, default_config: Dict, file_config: Dict):
        """遞歸更新配置"""
        for key, value in file_config.items():
            if key in default_config:
                if isinstance(value, dict) and isinstance(default_config[key], dict):
                    self._update_config(default_config[key], value)
                else:
                    default_config[key] = value
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到 {self.config_path}")
        except Exception as e:
            logger.error(f"保存配置文件失敗：{e}")
    
    def get(self, key_path: str, default=None):
        """獲取配置值，支持點號分隔的路徑"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value):
        """設置配置值，支持點號分隔的路徑"""
        keys = key_path.split('.')
        config = self.config
        
        # 導航到父級配置
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 設置值
        config[keys[-1]] = value
    
    def get_data_config(self) -> Dict[str, Any]:
        """獲取數據配置"""
        return self.config.get("data", {})
    
    def get_vectorizer_config(self) -> Dict[str, Any]:
        """獲取向量化配置"""
        return self.config.get("vectorizer", {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """獲取LLM配置"""
        return self.config.get("llm", {})
    
    def get_topic_modeling_config(self) -> Dict[str, Any]:
        """獲取主題建模配置"""
        return self.config.get("topic_modeling", {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """獲取評估配置"""
        return self.config.get("evaluation", {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """獲取UI配置"""
        return self.config.get("ui", {})

# 全局配置實例
config = Config()