import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

from ..utils.vectorizer import VectorDatabase, TopicVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EffectivenessMetrics:
    """有效性評估指標"""
    mean_similarity: float
    max_similarity: float
    min_similarity: float
    std_similarity: float
    median_similarity: float
    coverage_score: float
    topic_document_matrix: np.ndarray
    similarity_distribution: Dict[str, float]
    top_matching_docs: List[Dict]
    evaluation_timestamp: datetime

class EffectivenessEvaluator:
    """有效性評估器 - 基於餘弦相似度評估主題與文檔的相關性"""
    
    def __init__(self, vector_database: VectorDatabase):
        self.vector_db = vector_database
        self.topic_vectorizer = TopicVectorizer(vector_database)
        
    def evaluate_topic_effectiveness(self, topics: List[str]) -> EffectivenessMetrics:
        """評估主題有效性"""
        logger.info(f"開始評估 {len(topics)} 個主題的有效性")
        
        try:
            # 獲取文檔嵌入向量
            document_embeddings = self.vector_db.get_embeddings()
            if document_embeddings is None:
                raise ValueError("向量數據庫中沒有文檔嵌入")
            
            # 計算主題與文檔的相似度
            similarity_results = self.topic_vectorizer.compute_topic_similarity(
                topics, document_embeddings
            )
            
            # 計算詳細指標
            metrics = self._compute_detailed_metrics(
                topics, similarity_results, document_embeddings
            )
            
            logger.info(f"有效性評估完成，平均相似度: {metrics.mean_similarity:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"有效性評估失敗: {e}")
            return self._create_empty_metrics(len(topics))
    
    def _compute_detailed_metrics(self, 
                                topics: List[str],
                                similarity_results: Dict,
                                document_embeddings: np.ndarray) -> EffectivenessMetrics:
        """計算詳細的有效性指標"""
        
        # 基本統計指標
        similarities = similarity_results['similarities']  # shape: (n_topics, n_documents)
        mean_similarities = similarity_results['mean_similarities']
        
        # 計算統計值
        all_similarities = similarities.flatten()
        
        # 計算覆蓋分數 - 每個文檔與最相關主題的相似度
        max_similarities_per_doc = np.max(similarities, axis=0)
        coverage_score = np.mean(max_similarities_per_doc)
        
        # 相似度分佈統計
        similarity_distribution = self._compute_similarity_distribution(all_similarities)
        
        # 找出每個主題的最佳匹配文檔
        top_matching_docs = self._find_top_matching_documents(similarities, topics)
        
        return EffectivenessMetrics(
            mean_similarity=float(np.mean(mean_similarities)),
            max_similarity=float(np.max(mean_similarities)),
            min_similarity=float(np.min(mean_similarities)),
            std_similarity=float(np.std(mean_similarities)),
            median_similarity=float(np.median(mean_similarities)),
            coverage_score=float(coverage_score),
            topic_document_matrix=similarities,
            similarity_distribution=similarity_distribution,
            top_matching_docs=top_matching_docs,
            evaluation_timestamp=datetime.now()
        )
    
    def _compute_similarity_distribution(self, similarities: np.ndarray) -> Dict[str, float]:
        """計算相似度分佈統計"""
        return {
            'percentile_25': float(np.percentile(similarities, 25)),
            'percentile_50': float(np.percentile(similarities, 50)),
            'percentile_75': float(np.percentile(similarities, 75)),
            'percentile_90': float(np.percentile(similarities, 90)),
            'percentile_95': float(np.percentile(similarities, 95)),
            'skewness': float(self._compute_skewness(similarities)),
            'kurtosis': float(self._compute_kurtosis(similarities))
        }
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """計算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """計算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _find_top_matching_documents(self, 
                                   similarities: np.ndarray, 
                                   topics: List[str],
                                   top_k: int = 3) -> List[Dict]:
        """找出每個主題的最佳匹配文檔"""
        top_matches = []
        
        for topic_idx, topic in enumerate(topics):
            # 獲取該主題與所有文檔的相似度
            topic_similarities = similarities[topic_idx]
            
            # 找出最相似的文檔索引
            top_doc_indices = np.argsort(topic_similarities)[::-1][:top_k]
            
            # 構建匹配信息
            matches = []
            for rank, doc_idx in enumerate(top_doc_indices):
                similarity_score = topic_similarities[doc_idx]
                
                # 獲取文檔內容（如果可用）
                doc_content = ""
                if hasattr(self.vector_db, 'texts') and doc_idx < len(self.vector_db.texts):
                    doc_content = self.vector_db.texts[doc_idx][:200] + "..."
                
                matches.append({
                    'rank': rank + 1,
                    'document_index': int(doc_idx),
                    'similarity_score': float(similarity_score),
                    'document_preview': doc_content
                })
            
            top_matches.append({
                'topic_index': topic_idx,
                'topic': topic,
                'matches': matches,
                'avg_top_similarity': float(np.mean([m['similarity_score'] for m in matches]))
            })
        
        return top_matches
    
    def _create_empty_metrics(self, num_topics: int) -> EffectivenessMetrics:
        """創建空的指標對象"""
        return EffectivenessMetrics(
            mean_similarity=0.0,
            max_similarity=0.0,
            min_similarity=0.0,
            std_similarity=0.0,
            median_similarity=0.0,
            coverage_score=0.0,
            topic_document_matrix=np.zeros((num_topics, 1)),
            similarity_distribution={},
            top_matching_docs=[],
            evaluation_timestamp=datetime.now()
        )
    
    def generate_effectiveness_report(self, metrics: EffectivenessMetrics, topics: List[str]) -> Dict[str, Any]:
        """生成有效性評估報告"""
        
        # 評估等級
        effectiveness_grade = self._grade_effectiveness(metrics.mean_similarity)
        
        # 主題排名
        topic_rankings = self._rank_topics_by_effectiveness(metrics, topics)
        
        # 建議
        recommendations = self._generate_effectiveness_recommendations(metrics)
        
        report = {
            'summary': {
                'overall_effectiveness': effectiveness_grade,
                'mean_similarity': metrics.mean_similarity,
                'coverage_score': metrics.coverage_score,
                'num_topics_evaluated': len(topics)
            },
            'detailed_metrics': {
                'similarity_statistics': {
                    'mean': metrics.mean_similarity,
                    'max': metrics.max_similarity,
                    'min': metrics.min_similarity,
                    'std': metrics.std_similarity,
                    'median': metrics.median_similarity
                },
                'distribution': metrics.similarity_distribution,
                'coverage_score': metrics.coverage_score
            },
            'topic_rankings': topic_rankings,
            'top_matches': metrics.top_matching_docs,
            'recommendations': recommendations,
            'metadata': {
                'evaluation_method': 'cosine_similarity',
                'timestamp': metrics.evaluation_timestamp.isoformat(),
                'evaluator_version': '1.0'
            }
        }
        
        return report
    
    def _grade_effectiveness(self, mean_similarity: float) -> str:
        """評估有效性等級"""
        if mean_similarity >= 0.8:
            return "優秀"
        elif mean_similarity >= 0.7:
            return "良好"
        elif mean_similarity >= 0.6:
            return "中等"
        elif mean_similarity >= 0.4:
            return "較差"
        else:
            return "很差"
    
    def _rank_topics_by_effectiveness(self, 
                                    metrics: EffectivenessMetrics, 
                                    topics: List[str]) -> List[Dict]:
        """按有效性對主題排名"""
        # 計算每個主題的平均相似度
        topic_scores = np.mean(metrics.topic_document_matrix, axis=1)
        
        # 排序
        sorted_indices = np.argsort(topic_scores)[::-1]
        
        rankings = []
        for rank, topic_idx in enumerate(sorted_indices):
            rankings.append({
                'rank': rank + 1,
                'topic_index': int(topic_idx),
                'topic': topics[topic_idx],
                'effectiveness_score': float(topic_scores[topic_idx]),
                'max_similarity': float(np.max(metrics.topic_document_matrix[topic_idx])),
                'min_similarity': float(np.min(metrics.topic_document_matrix[topic_idx]))
            })
        
        return rankings
    
    def _generate_effectiveness_recommendations(self, metrics: EffectivenessMetrics) -> List[str]:
        """生成有效性改進建議"""
        recommendations = []
        
        if metrics.mean_similarity < 0.5:
            recommendations.append("整體相似度較低，建議重新生成更相關的主題")
        
        if metrics.std_similarity > 0.2:
            recommendations.append("主題間有效性差異較大，建議平衡主題質量")
        
        if metrics.coverage_score < 0.6:
            recommendations.append("文檔覆蓋度不足，建議增加更多樣化的主題")
        
        # 檢查相似度分佈
        if metrics.similarity_distribution.get('percentile_90', 0) < 0.7:
            recommendations.append("高相似度匹配較少，建議優化主題特異性")
        
        if not recommendations:
            recommendations.append("有效性表現良好，可考慮進一步優化細節")
        
        return recommendations

class TopicDocumentMatcher:
    """主題-文檔匹配器"""
    
    def __init__(self, effectiveness_evaluator: EffectivenessEvaluator):
        self.evaluator = effectiveness_evaluator
    
    def find_representative_documents(self, 
                                    topics: List[str], 
                                    top_k: int = 5) -> Dict[str, List[Dict]]:
        """為每個主題找出最具代表性的文檔"""
        
        # 評估有效性
        metrics = self.evaluator.evaluate_topic_effectiveness(topics)
        
        representative_docs = {}
        
        for topic_idx, topic in enumerate(topics):
            # 獲取該主題的相似度分數
            topic_similarities = metrics.topic_document_matrix[topic_idx]
            
            # 找出最相似的文檔
            top_indices = np.argsort(topic_similarities)[::-1][:top_k]
            
            docs = []
            for rank, doc_idx in enumerate(top_indices):
                similarity = topic_similarities[doc_idx]
                
                # 獲取文檔內容
                doc_content = ""
                if (hasattr(self.evaluator.vector_db, 'texts') and 
                    doc_idx < len(self.evaluator.vector_db.texts)):
                    doc_content = self.evaluator.vector_db.texts[doc_idx]
                
                docs.append({
                    'rank': rank + 1,
                    'document_index': int(doc_idx),
                    'similarity_score': float(similarity),
                    'content': doc_content,
                    'content_preview': doc_content[:300] + "..." if len(doc_content) > 300 else doc_content
                })
            
            representative_docs[topic] = docs
        
        return representative_docs
    
    def compute_topic_overlap_matrix(self, topics: List[str]) -> np.ndarray:
        """計算主題間的重疊矩陣"""
        
        metrics = self.evaluator.evaluate_topic_effectiveness(topics)
        similarity_matrix = metrics.topic_document_matrix
        
        # 計算主題間的文檔重疊度
        overlap_matrix = np.zeros((len(topics), len(topics)))
        
        for i in range(len(topics)):
            for j in range(len(topics)):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    # 計算兩個主題在文檔相似度上的相關性
                    correlation = np.corrcoef(similarity_matrix[i], similarity_matrix[j])[0, 1]
                    overlap_matrix[i, j] = correlation if not np.isnan(correlation) else 0.0
        
        return overlap_matrix