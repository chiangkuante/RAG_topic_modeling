import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self.model = None
        self.index = None
        self.texts = []
        self.embeddings = None
        
    def load_model(self):
        """載入嵌入模型"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"成功載入模型：{self.model_name}")
        except Exception as e:
            logger.error(f"載入模型失敗：{e}")
            raise
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """將文本編碼為向量"""
        if self.model is None:
            self.load_model()
        
        logger.info(f"開始向量化 {len(texts)} 個文本")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="向量化進度"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        logger.info(f"向量化完成，向量維度：{embeddings.shape}")
        
        return embeddings
    
    def build_index(self, texts: List[str], save_path: str = None) -> None:
        """建立FAISS索引"""
        self.texts = texts
        self.embeddings = self.encode_texts(texts)
        
        # 建立FAISS索引
        self.index = faiss.IndexFlatIP(self.dimension)  # 使用內積相似度
        
        # 正規化向量以使用餘弦相似度
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"成功建立FAISS索引，包含 {self.index.ntotal} 個向量")
        
        if save_path:
            self.save_index(save_path)
    
    def search(self, query: str, k: int = 10) -> Tuple[List[str], List[float], List[int]]:
        """搜索相似文本"""
        if self.index is None or self.model is None:
            raise ValueError("請先建立索引")
        
        # 編碼查詢
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding, k)
        
        # 返回結果
        similar_texts = [self.texts[idx] for idx in indices[0]]
        similarities = scores[0].tolist()
        indices_list = indices[0].tolist()
        
        return similar_texts, similarities, indices_list
    
    def get_embeddings(self) -> np.ndarray:
        """獲取所有嵌入向量"""
        return self.embeddings
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """計算相似度矩陣"""
        if self.embeddings is None:
            raise ValueError("請先建立索引")
        
        # 計算餘弦相似度矩陣
        similarity_matrix = np.dot(self.embeddings, self.embeddings.T)
        return similarity_matrix
    
    def save_index(self, save_path: str) -> None:
        """保存索引到文件"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, f"{save_path}.faiss")
        
        # 保存其他信息
        metadata = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'texts': self.texts,
            'embeddings': self.embeddings
        }
        
        with open(f"{save_path}.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"索引已保存到：{save_path}")
    
    def load_index(self, load_path: str) -> None:
        """從文件載入索引"""
        # 載入FAISS索引
        self.index = faiss.read_index(f"{load_path}.faiss")
        
        # 載入其他信息
        with open(f"{load_path}.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_name = metadata['model_name']
        self.dimension = metadata['dimension']
        self.texts = metadata['texts']
        self.embeddings = metadata['embeddings']
        
        # 載入模型
        self.load_model()
        
        logger.info(f"索引已從 {load_path} 載入，包含 {self.index.ntotal} 個向量")

class TopicVectorizer:
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
    
    def vectorize_topics(self, topics: List[str]) -> np.ndarray:
        """將主題列表向量化"""
        if self.vector_db.model is None:
            self.vector_db.load_model()
        
        topic_embeddings = self.vector_db.model.encode(topics, convert_to_numpy=True)
        faiss.normalize_L2(topic_embeddings)
        
        return topic_embeddings
    
    def compute_topic_similarity(self, topics: List[str], document_embeddings: np.ndarray = None) -> Dict:
        """計算主題與文檔的相似度"""
        if document_embeddings is None:
            document_embeddings = self.vector_db.embeddings
        
        topic_embeddings = self.vectorize_topics(topics)
        
        # 計算主題與所有文檔的相似度
        similarities = np.dot(topic_embeddings, document_embeddings.T)
        
        # 計算統計信息
        results = {
            'topic_embeddings': topic_embeddings,
            'similarities': similarities,
            'mean_similarities': np.mean(similarities, axis=1),
            'max_similarities': np.max(similarities, axis=1),
            'min_similarities': np.min(similarities, axis=1),
            'std_similarities': np.std(similarities, axis=1)
        }
        
        return results