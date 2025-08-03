from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime

from ..utils.vectorizer import VectorDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass  
class RetrievalResult:
    """檢索結果數據類"""
    query: str
    documents: List[str]
    scores: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime

class DocumentRetriever:
    """文檔檢索器 - 負責從向量數據庫檢索相關文檔"""
    
    def __init__(self, vector_database: VectorDatabase, top_k: int = 10):
        self.vector_db = vector_database
        self.top_k = top_k
        
    def retrieve(self, query: str, k: Optional[int] = None) -> RetrievalResult:
        """檢索相關文檔"""
        k = k or self.top_k
        
        try:
            similar_texts, similarities, indices = self.vector_db.search(query, k=k)
            
            result = RetrievalResult(
                query=query,
                documents=similar_texts,
                scores=similarities,
                metadata={
                    'retrieval_method': 'vector_similarity',
                    'top_k': k,
                    'avg_score': np.mean(similarities),
                    'min_score': np.min(similarities),
                    'max_score': np.max(similarities),
                    'indices': indices
                },
                timestamp=datetime.now()
            )
            
            logger.info(f"檢索完成: 查詢='{query}', 找到{len(similar_texts)}個相關文檔")
            return result
            
        except Exception as e:
            logger.error(f"檢索失敗: {e}")
            raise

class TopicRetrievalAgent:
    """主題檢索代理 - 簡化的檢索代理"""
    
    def __init__(self, vector_database: VectorDatabase, top_k: int = 10):
        self.retriever = DocumentRetriever(vector_database, top_k)
        
    def forward(self, query: str, context: str = "") -> RetrievalResult:
        """前向傳播 - 執行檢索任務"""
        
        # 使用向量檢索獲取相關文檔
        retrieval_result = self.retriever.retrieve(query)
        
        logger.info(f"檢索代理處理查詢: {query}")
        
        return retrieval_result

class MultiQueryRetriever:
    """多查詢檢索器 - 支持多個查詢的並行檢索"""
    
    def __init__(self, retrieval_agent: TopicRetrievalAgent):
        self.agent = retrieval_agent
    
    def retrieve_multiple(self, queries: List[str], k: int = 10) -> List[RetrievalResult]:
        """批量檢索多個查詢"""
        results = []
        
        for query in queries:
            try:
                result = self.agent.forward(query)
                results.append(result)
            except Exception as e:
                logger.error(f"查詢 '{query}' 檢索失敗: {e}")
                # 創建空結果
                empty_result = RetrievalResult(
                    query=query,
                    documents=[],
                    scores=[],
                    metadata={'error': str(e)},
                    timestamp=datetime.now()
                )
                results.append(empty_result)
        
        return results
    
    def aggregate_results(self, results: List[RetrievalResult], max_docs: int = 20) -> Dict:
        """聚合多個檢索結果"""
        all_documents = []
        all_scores = []
        all_indices = []
        query_mapping = {}
        
        for i, result in enumerate(results):
            indices = result.metadata.get('indices', list(range(len(result.documents))))
            for j, (doc, score, idx) in enumerate(zip(result.documents, result.scores, indices)):
                if idx not in query_mapping:  # 避免重複文檔
                    all_documents.append(doc)
                    all_scores.append(score)
                    all_indices.append(idx)
                    query_mapping[idx] = {
                        'query_index': i,
                        'query': result.query,
                        'rank_in_query': j
                    }
        
        # 按分數排序並限制數量
        sorted_indices = np.argsort(all_scores)[::-1][:max_docs]
        
        aggregated = {
            'documents': [all_documents[i] for i in sorted_indices],
            'scores': [all_scores[i] for i in sorted_indices],
            'indices': [all_indices[i] for i in sorted_indices],
            'query_mapping': {all_indices[i]: query_mapping[all_indices[i]] for i in sorted_indices},
            'total_unique_docs': len(all_documents),
            'queries': [r.query for r in results]
        }
        
        return aggregated

class ContextualRetriever:
    """上下文感知檢索器 - 基於上下文優化檢索結果"""
    
    def __init__(self, retrieval_agent: TopicRetrievalAgent):
        self.agent = retrieval_agent
        self.context_history = []
    
    def retrieve_with_context(self, query: str, context: str = "", use_history: bool = True) -> RetrievalResult:
        """帶上下文的檢索"""
        
        # 構建增強查詢
        enhanced_query = query
        if context:
            enhanced_query = f"Context: {context}\nQuery: {query}"
        
        if use_history and self.context_history:
            recent_context = " ".join(self.context_history[-3:])  # 使用最近3次的上下文
            enhanced_query = f"Previous context: {recent_context}\n{enhanced_query}"
        
        # 執行檢索
        result = self.agent.forward(enhanced_query)
        
        # 更新上下文歷史
        self.context_history.append(query)
        if len(self.context_history) > 10:  # 保持歷史長度
            self.context_history.pop(0)
        
        return result
    
    def clear_context(self):
        """清除上下文歷史"""
        self.context_history.clear()
        logger.info("上下文歷史已清除")

class RetrievalEvaluator:
    """檢索評估器 - 評估檢索質量"""
    
    def __init__(self):
        pass
    
    def evaluate_retrieval_quality(self, result: RetrievalResult) -> Dict:
        """評估檢索質量"""
        if not result.documents:
            return {
                'quality_score': 0.0,
                'coverage_score': 0.0,
                'diversity_score': 0.0,
                'error': 'No documents retrieved'
            }
        
        # 計算質量指標
        avg_score = np.mean(result.scores)
        score_variance = np.var(result.scores)
        
        # 計算文檔多樣性（基於長度差異）
        doc_lengths = [len(doc.split()) for doc in result.documents]
        length_diversity = np.std(doc_lengths) / (np.mean(doc_lengths) + 1e-8)
        
        # 計算覆蓋度（前k個結果的分數分佈）
        if len(result.scores) > 1:
            coverage_score = (result.scores[0] - result.scores[-1]) / (result.scores[0] + 1e-8)
        else:
            coverage_score = 1.0
        
        quality_metrics = {
            'quality_score': avg_score,
            'score_variance': score_variance,
            'coverage_score': coverage_score,
            'diversity_score': length_diversity,
            'num_documents': len(result.documents),
            'score_distribution': {
                'mean': avg_score,
                'std': np.std(result.scores),
                'min': np.min(result.scores),
                'max': np.max(result.scores)
            }
        }
        
        return quality_metrics