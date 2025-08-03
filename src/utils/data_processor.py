import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.stop_words = set(ENGLISH_STOP_WORDS)
        self.data = None
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """讀取CSV文件"""
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"成功載入數據，共 {len(self.data)} 行")
            return self.data
        except FileNotFoundError:
            logger.error(f"找不到文件：{file_path}")
            raise
        except Exception as e:
            logger.error(f"載入數據時發生錯誤：{e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """清理單個文本"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 轉換為小寫
        text = text.lower()
        
        # 移除特殊字符和數字，只保留字母和空格
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # 移除多餘空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """移除停用詞"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        return ' '.join(filtered_words)
    
    def preprocess_data(self, text_column: str = None) -> pd.DataFrame:
        """預處理數據"""
        if self.data is None:
            raise ValueError("請先載入數據")
        
        # 自動檢測文本列
        if text_column is None:
            text_columns = self.data.select_dtypes(include=['object']).columns
            if len(text_columns) == 0:
                raise ValueError("找不到文本列")
            text_column = text_columns[0]  # 使用第一個文本列
            logger.info(f"自動選擇文本列：{text_column}")
        
        if text_column not in self.data.columns:
            raise ValueError(f"找不到列：{text_column}")
        
        # 創建處理後的數據副本
        processed_data = self.data.copy()
        
        # 清理文本
        processed_data['cleaned_text'] = processed_data[text_column].apply(self.clean_text)
        
        # 移除停用詞
        processed_data['processed_text'] = processed_data['cleaned_text'].apply(self.remove_stopwords)
        
        # 移除空文本
        processed_data = processed_data[processed_data['processed_text'].str.len() > 0]
        
        logger.info(f"預處理完成，剩餘 {len(processed_data)} 行有效數據")
        
        return processed_data
    
    def get_text_stats(self, data: pd.DataFrame, text_column: str = 'processed_text') -> Dict:
        """獲取文本統計信息"""
        if text_column not in data.columns:
            raise ValueError(f"找不到列：{text_column}")
        
        texts = data[text_column].tolist()
        word_counts = [len(text.split()) for text in texts]
        
        stats = {
            'total_documents': len(texts),
            'avg_words_per_doc': np.mean(word_counts),
            'median_words_per_doc': np.median(word_counts),
            'min_words': np.min(word_counts),
            'max_words': np.max(word_counts),
            'total_words': sum(word_counts)
        }
        
        return stats
    
    def sample_data(self, data: pd.DataFrame, sample_size: int = None, random_state: int = 42) -> pd.DataFrame:
        """對數據進行採樣"""
        if sample_size is None or sample_size >= len(data):
            return data
        
        sampled_data = data.sample(n=sample_size, random_state=random_state)
        logger.info(f"採樣完成，從 {len(data)} 行中選擇了 {len(sampled_data)} 行")
        
        return sampled_data