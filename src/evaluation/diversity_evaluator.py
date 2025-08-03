import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from collections import Counter, defaultdict
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiversityMetrics:
    """多樣性評估指標"""
    diversity_score: float
    unique_words_ratio: float
    topic_overlap_score: float
    semantic_diversity: float
    lexical_diversity: float
    top_k_uniqueness: float
    word_distribution_entropy: float
    topic_similarity_matrix: np.ndarray
    unique_word_analysis: Dict[str, Any]
    evaluation_timestamp: datetime

class DiversityEvaluator:
    """多樣性評估器 - 評估主題的多樣性和獨特性"""
    
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must'])
    
    def evaluate_topic_diversity(self, topics: List[str]) -> DiversityMetrics:
        """評估主題多樣性"""
        logger.info(f"開始評估 {len(topics)} 個主題的多樣性")
        
        try:
            # 詞彙級別多樣性分析
            lexical_metrics = self._evaluate_lexical_diversity(topics)
            
            # 語義級別多樣性分析
            semantic_metrics = self._evaluate_semantic_diversity(topics)
            
            # Top-K獨特詞彙分析
            topk_metrics = self._evaluate_topk_uniqueness(topics)
            
            # 綜合多樣性分數
            overall_diversity = self._compute_overall_diversity(
                lexical_metrics, semantic_metrics, topk_metrics
            )
            
            # 構建完整指標
            metrics = DiversityMetrics(
                diversity_score=overall_diversity,
                unique_words_ratio=lexical_metrics['unique_words_ratio'],
                topic_overlap_score=lexical_metrics['overlap_score'],
                semantic_diversity=semantic_metrics['semantic_diversity'],
                lexical_diversity=lexical_metrics['lexical_diversity'],
                top_k_uniqueness=topk_metrics['uniqueness_score'],
                word_distribution_entropy=lexical_metrics['entropy'],
                topic_similarity_matrix=semantic_metrics['similarity_matrix'],
                unique_word_analysis=topk_metrics['word_analysis'],
                evaluation_timestamp=datetime.now()
            )
            
            logger.info(f"多樣性評估完成，整體分數: {overall_diversity:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"多樣性評估失敗: {e}")
            return self._create_empty_metrics(len(topics))
    
    def _evaluate_lexical_diversity(self, topics: List[str]) -> Dict[str, float]:
        """評估詞彙級別多樣性"""
        
        # 提取所有詞彙
        all_words = []
        topic_words = []
        
        for topic in topics:
            # 清理和分詞
            words = self._tokenize_topic(topic)
            topic_words.append(set(words))
            all_words.extend(words)
        
        # 基本統計
        total_words = len(all_words)
        unique_words = set(all_words)
        unique_count = len(unique_words)
        
        # 計算獨特詞彙比例
        unique_words_ratio = unique_count / total_words if total_words > 0 else 0
        
        # 計算主題間詞彙重疊
        overlap_scores = []
        for i in range(len(topic_words)):
            for j in range(i + 1, len(topic_words)):
                intersection = len(topic_words[i].intersection(topic_words[j]))
                union = len(topic_words[i].union(topic_words[j]))
                overlap = intersection / union if union > 0 else 0
                overlap_scores.append(overlap)
        
        avg_overlap = np.mean(overlap_scores) if overlap_scores else 0
        overlap_score = 1 - avg_overlap  # 重疊越少，多樣性越高
        
        # 計算詞彙分佈熵
        word_counts = Counter(all_words)
        entropy = self._compute_entropy(word_counts, total_words)
        
        # 詞彙多樣性綜合分數
        lexical_diversity = (unique_words_ratio * 0.4 + overlap_score * 0.4 + 
                           min(entropy / 10, 1.0) * 0.2)
        
        return {
            'unique_words_ratio': unique_words_ratio,
            'overlap_score': overlap_score,
            'entropy': entropy,
            'lexical_diversity': lexical_diversity,
            'total_words': total_words,
            'unique_words': unique_count
        }
    
    def _evaluate_semantic_diversity(self, topics: List[str]) -> Dict[str, Any]:
        """評估語義級別多樣性"""
        
        if len(topics) < 2:
            return {
                'semantic_diversity': 1.0,
                'similarity_matrix': np.eye(len(topics))
            }
        
        try:
            # 使用TF-IDF向量化主題
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000
            )
            
            tfidf_matrix = vectorizer.fit_transform(topics)
            
            # 計算主題間相似度矩陣
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 計算語義多樣性（相似度越低，多樣性越高）
            # 只考慮上三角矩陣（避免重複和自相似）
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            avg_similarity = np.mean(upper_triangle)
            semantic_diversity = 1 - avg_similarity
            
            return {
                'semantic_diversity': max(0, semantic_diversity),
                'similarity_matrix': similarity_matrix,
                'avg_similarity': avg_similarity
            }
            
        except Exception as e:
            logger.error(f"語義多樣性計算失敗: {e}")
            return {
                'semantic_diversity': 0.5,
                'similarity_matrix': np.eye(len(topics))
            }
    
    def _evaluate_topk_uniqueness(self, topics: List[str]) -> Dict[str, Any]:
        """評估Top-K詞彙獨特性"""
        
        # 提取每個主題的關鍵詞
        topic_keywords = []
        all_keywords = []
        
        for topic in topics:
            keywords = self._extract_topic_keywords(topic, self.top_k)
            topic_keywords.append(keywords)
            all_keywords.extend(keywords)
        
        # 計算詞彙頻率
        keyword_counts = Counter(all_keywords)
        total_keywords = len(all_keywords)
        
        # 計算獨特性分數
        unique_keywords = sum(1 for count in keyword_counts.values() if count == 1)
        uniqueness_score = unique_keywords / total_keywords if total_keywords > 0 else 0
        
        # 分析詞彙分佈
        word_analysis = {
            'total_keywords': total_keywords,
            'unique_keywords': unique_keywords,
            'repeated_keywords': total_keywords - unique_keywords,
            'keyword_frequency_distribution': dict(Counter(keyword_counts.values())),
            'most_common_keywords': keyword_counts.most_common(10),
            'topics_keyword_overlap': self._analyze_keyword_overlap(topic_keywords)
        }
        
        return {
            'uniqueness_score': uniqueness_score,
            'word_analysis': word_analysis
        }
    
    def _tokenize_topic(self, topic: str) -> List[str]:
        """分詞處理"""
        # 轉為小寫並移除標點
        topic_clean = re.sub(r'[^\w\s]', ' ', topic.lower())
        
        # 分詞
        words = topic_clean.split()
        
        # 過濾停用詞和短詞
        filtered_words = [
            word for word in words 
            if len(word) > 2 and word not in self.stop_words
        ]
        
        return filtered_words
    
    def _extract_topic_keywords(self, topic: str, k: int) -> List[str]:
        """提取主題關鍵詞"""
        words = self._tokenize_topic(topic)
        
        # 簡單策略：按長度和位置權重排序
        word_scores = {}
        for i, word in enumerate(words):
            # 位置權重（前面的詞更重要）
            position_weight = 1.0 / (i + 1)
            # 長度權重（較長的詞可能更具體）
            length_weight = min(len(word) / 10, 1.0)
            word_scores[word] = position_weight * 0.7 + length_weight * 0.3
        
        # 排序並返回前k個
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_words[:k]]
    
    def _compute_entropy(self, word_counts: Counter, total_words: int) -> float:
        """計算詞彙分佈熵"""
        if total_words == 0:
            return 0.0
        
        entropy = 0
        for count in word_counts.values():
            prob = count / total_words
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _analyze_keyword_overlap(self, topic_keywords: List[List[str]]) -> Dict[str, int]:
        """分析關鍵詞重疊情況"""
        overlap_analysis = defaultdict(int)
        
        # 統計每個關鍵詞出現在多少個主題中
        keyword_topic_count = defaultdict(int)
        for keywords in topic_keywords:
            for keyword in set(keywords):  # 使用set避免同一主題內重複計算
                keyword_topic_count[keyword] += 1
        
        # 按重疊程度分類
        for keyword, count in keyword_topic_count.items():
            if count == 1:
                overlap_analysis['unique'] += 1
            elif count == 2:
                overlap_analysis['shared_by_2'] += 1
            elif count <= len(topic_keywords) // 2:
                overlap_analysis['shared_by_few'] += 1
            else:
                overlap_analysis['shared_by_many'] += 1
        
        return dict(overlap_analysis)
    
    def _compute_overall_diversity(self, 
                                 lexical_metrics: Dict,
                                 semantic_metrics: Dict,
                                 topk_metrics: Dict) -> float:
        """計算整體多樣性分數"""
        
        # 權重分配
        weights = {
            'lexical': 0.3,
            'semantic': 0.4,
            'topk': 0.3
        }
        
        # 組合分數
        overall_score = (
            lexical_metrics['lexical_diversity'] * weights['lexical'] +
            semantic_metrics['semantic_diversity'] * weights['semantic'] +
            topk_metrics['uniqueness_score'] * weights['topk']
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _create_empty_metrics(self, num_topics: int) -> DiversityMetrics:
        """創建空的多樣性指標"""
        return DiversityMetrics(
            diversity_score=0.0,
            unique_words_ratio=0.0,
            topic_overlap_score=0.0,
            semantic_diversity=0.0,
            lexical_diversity=0.0,
            top_k_uniqueness=0.0,
            word_distribution_entropy=0.0,
            topic_similarity_matrix=np.eye(num_topics),
            unique_word_analysis={},
            evaluation_timestamp=datetime.now()
        )
    
    def generate_diversity_report(self, metrics: DiversityMetrics, topics: List[str]) -> Dict[str, Any]:
        """生成多樣性評估報告"""
        
        # 評估等級
        diversity_grade = self._grade_diversity(metrics.diversity_score)
        
        # 主題相似度分析
        similarity_analysis = self._analyze_topic_similarities(
            metrics.topic_similarity_matrix, topics
        )
        
        # 改進建議
        recommendations = self._generate_diversity_recommendations(metrics)
        
        report = {
            'summary': {
                'overall_diversity': diversity_grade,
                'diversity_score': metrics.diversity_score,
                'num_topics_evaluated': len(topics)
            },
            'detailed_metrics': {
                'lexical_diversity': {
                    'unique_words_ratio': metrics.unique_words_ratio,
                    'topic_overlap_score': metrics.topic_overlap_score,
                    'lexical_score': metrics.lexical_diversity,
                    'word_entropy': metrics.word_distribution_entropy
                },
                'semantic_diversity': {
                    'semantic_score': metrics.semantic_diversity,
                    'avg_topic_similarity': float(np.mean(metrics.topic_similarity_matrix[np.triu_indices_from(metrics.topic_similarity_matrix, k=1)]))
                },
                'topk_uniqueness': {
                    'uniqueness_score': metrics.top_k_uniqueness,
                    'keyword_analysis': metrics.unique_word_analysis
                }
            },
            'topic_similarity_analysis': similarity_analysis,
            'recommendations': recommendations,
            'metadata': {
                'evaluation_method': 'multi_level_diversity',
                'top_k': self.top_k,
                'timestamp': metrics.evaluation_timestamp.isoformat(),
                'evaluator_version': '1.0'
            }
        }
        
        return report
    
    def _grade_diversity(self, diversity_score: float) -> str:
        """評估多樣性等級"""
        if diversity_score >= 0.8:
            return "優秀"
        elif diversity_score >= 0.7:
            return "良好"
        elif diversity_score >= 0.6:
            return "中等"
        elif diversity_score >= 0.4:
            return "較差"
        else:
            return "很差"
    
    def _analyze_topic_similarities(self, similarity_matrix: np.ndarray, topics: List[str]) -> Dict[str, Any]:
        """分析主題相似度"""
        
        # 找出最相似的主題對
        most_similar_pairs = []
        similarity_values = []
        
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                similarity = similarity_matrix[i, j]
                similarity_values.append(similarity)
                most_similar_pairs.append({
                    'topic1_index': i,
                    'topic1': topics[i],
                    'topic2_index': j,
                    'topic2': topics[j],
                    'similarity': float(similarity)
                })
        
        # 排序找出最相似的對
        most_similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'most_similar_pairs': most_similar_pairs[:5],
            'similarity_statistics': {
                'mean': float(np.mean(similarity_values)),
                'std': float(np.std(similarity_values)),
                'max': float(np.max(similarity_values)),
                'min': float(np.min(similarity_values))
            },
            'high_similarity_count': sum(1 for s in similarity_values if s > 0.7),
            'low_similarity_count': sum(1 for s in similarity_values if s < 0.3)
        }
    
    def _generate_diversity_recommendations(self, metrics: DiversityMetrics) -> List[str]:
        """生成多樣性改進建議"""
        recommendations = []
        
        if metrics.diversity_score < 0.5:
            recommendations.append("整體多樣性較低，建議重新生成更具差異性的主題")
        
        if metrics.unique_words_ratio < 0.6:
            recommendations.append("詞彙重複度較高，建議使用更多樣化的表達方式")
        
        if metrics.semantic_diversity < 0.5:
            recommendations.append("主題語義相似度過高，建議增加不同角度的主題")
        
        if metrics.top_k_uniqueness < 0.4:
            recommendations.append("關鍵詞獨特性不足，建議提高主題特異性")
        
        # 基於詞彙分析的建議
        if metrics.unique_word_analysis.get('shared_by_many', 0) > 5:
            recommendations.append("過多關鍵詞在多個主題中重複，建議細化主題區分")
        
        if not recommendations:
            recommendations.append("多樣性表現良好，可考慮進一步優化細節")
        
        return recommendations

class TopicClusterAnalyzer:
    """主題聚類分析器"""
    
    def __init__(self, diversity_evaluator: DiversityEvaluator):
        self.evaluator = diversity_evaluator
    
    def analyze_topic_clusters(self, topics: List[str], threshold: float = 0.7) -> Dict[str, Any]:
        """分析主題聚類情況"""
        
        # 評估多樣性獲取相似度矩陣
        metrics = self.evaluator.evaluate_topic_diversity(topics)
        similarity_matrix = metrics.topic_similarity_matrix
        
        # 基於閾值聚類
        clusters = self._cluster_by_threshold(similarity_matrix, topics, threshold)
        
        # 分析聚類結果
        cluster_analysis = self._analyze_clusters(clusters, topics)
        
        return {
            'clusters': clusters,
            'cluster_analysis': cluster_analysis,
            'similarity_threshold': threshold,
            'num_clusters': len(clusters)
        }
    
    def _cluster_by_threshold(self, similarity_matrix: np.ndarray, topics: List[str], threshold: float) -> List[List[int]]:
        """基於閾值的簡單聚類"""
        
        visited = set()
        clusters = []
        
        for i in range(len(topics)):
            if i in visited:
                continue
            
            # 開始新聚類
            cluster = [i]
            visited.add(i)
            
            # 找出所有與當前主題相似度超過閾值的主題
            for j in range(i + 1, len(topics)):
                if j not in visited and similarity_matrix[i, j] >= threshold:
                    cluster.append(j)
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _analyze_clusters(self, clusters: List[List[int]], topics: List[str]) -> Dict[str, Any]:
        """分析聚類結果"""
        
        cluster_info = []
        for i, cluster_indices in enumerate(clusters):
            cluster_topics = [topics[idx] for idx in cluster_indices]
            
            cluster_info.append({
                'cluster_id': i,
                'size': len(cluster_indices),
                'topic_indices': cluster_indices,
                'topics': cluster_topics,
                'representative_topic': cluster_topics[0] if cluster_topics else ""
            })
        
        # 統計信息
        cluster_sizes = [len(cluster) for cluster in clusters]
        
        analysis = {
            'cluster_details': cluster_info,
            'statistics': {
                'num_clusters': len(clusters),
                'avg_cluster_size': np.mean(cluster_sizes),
                'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
                'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
                'singleton_clusters': sum(1 for size in cluster_sizes if size == 1),
                'large_clusters': sum(1 for size in cluster_sizes if size > 3)
            }
        }
        
        return analysis