from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime
import json
import re

from .reasoning_agent import TopicResult
from ..utils.vectorizer import VectorDatabase, TopicVectorizer
from ..utils.llm_api import LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """評估結果數據類"""
    overall_score: float
    individual_scores: Dict[str, float]
    criteria_scores: Dict[str, float]
    feedback: str
    recommendations: List[str]
    should_iterate: bool
    metadata: Dict[str, Any]
    timestamp: datetime

class TopicEvaluationAgent:
    """主題評估代理 - 評估主題質量並提供改進建議"""
    
    def __init__(self, vector_database: VectorDatabase, quality_threshold: float = 7.0, llm_manager: LLMManager = None):
        self.vector_db = vector_database
        self.topic_vectorizer = TopicVectorizer(vector_database)
        self.quality_threshold = quality_threshold
        self.llm_manager = llm_manager
        
    def forward(self, topic_result: TopicResult, context: str = "") -> EvaluationResult:
        """主要評估流程"""
        
        try:
            if self.llm_manager and self.llm_manager.active_llm:
                # 使用LLM進行評估
                overall_score, feedback, recommendations = self._llm_evaluate_topics(topic_result, context)
            else:
                # 使用基礎評估方法
                overall_score, feedback, recommendations = self._basic_evaluate_topics(topic_result, context)
            
            # 執行個別主題評估
            individual_scores = self._evaluate_individual_topics_basic(topic_result.topics, topic_result.topic_descriptions)
            
            # 計算標準化評估
            criteria_scores = self._compute_criteria_scores_basic(individual_scores)
            
            # 決定是否需要迭代
            should_iterate = overall_score < self.quality_threshold
            
            evaluation_result = EvaluationResult(
                overall_score=overall_score,
                individual_scores=individual_scores,
                criteria_scores=criteria_scores,
                feedback=feedback,
                recommendations=recommendations,
                should_iterate=should_iterate,
                metadata={
                    'evaluation_method': 'direct_llm_evaluation' if self.llm_manager else 'basic_evaluation',
                    'quality_threshold': self.quality_threshold,
                    'num_topics_evaluated': len(topic_result.topics),
                    'original_topic_metadata': topic_result.metadata
                },
                timestamp=datetime.now()
            )
            
            logger.info(f"評估完成：整體分數 {overall_score:.2f}, 需要迭代: {should_iterate}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"評估過程出錯: {e}")
            # 返回默認評估結果
            return self._create_default_evaluation(topic_result)
    
    def _evaluate_individual_topics(self, topics: List[str], context: str) -> Dict[str, float]:
        """評估個別主題"""
        individual_scores = {}
        
        for i, topic in enumerate(topics):
            try:
                quality_result = self.evaluate_quality(
                    topic=topic,
                    context=context[:1000]  # 限制上下文長度
                )
                
                # 計算綜合分數
                relevance = self._parse_score(quality_result.relevance_score)
                clarity = self._parse_score(quality_result.clarity_score)
                specificity = self._parse_score(quality_result.specificity_score)
                
                combined_score = (relevance + clarity + specificity) / 3
                individual_scores[f"topic_{i+1}"] = combined_score
                
            except Exception as e:
                logger.error(f"評估主題 {i+1} 失敗: {e}")
                individual_scores[f"topic_{i+1}"] = 5.0  # 默認中等分數
        
        return individual_scores
    
    def _compute_criteria_scores(self, topic_result: TopicResult, 
                               individual_scores: Dict[str, float]) -> Dict[str, float]:
        """計算標準化評估指標"""
        
        # 計算向量相似度指標
        try:
            similarity_scores = self._compute_vector_similarity(topic_result.topics)
        except Exception as e:
            logger.error(f"計算向量相似度失敗: {e}")
            similarity_scores = {'avg_similarity': 0.5, 'diversity_score': 0.5}
        
        # 計算文本質量指標
        text_quality = self._compute_text_quality(topic_result.topics)
        
        # 整合所有指標
        criteria_scores = {
            'relevance': np.mean(list(individual_scores.values())),
            'diversity': similarity_scores['diversity_score'],
            'coherence': similarity_scores['avg_similarity'],
            'text_quality': text_quality,
            'confidence': np.mean(topic_result.confidence_scores) if topic_result.confidence_scores else 0.8
        }
        
        return criteria_scores
    
    def _compute_vector_similarity(self, topics: List[str]) -> Dict[str, float]:
        """計算主題間的向量相似度"""
        topic_embeddings = self.topic_vectorizer.vectorize_topics(topics)
        
        # 計算相似度矩陣
        similarity_matrix = np.dot(topic_embeddings, topic_embeddings.T)
        
        # 計算多樣性（相似度越低，多樣性越高）
        # 去除對角線（自相似度）
        off_diagonal = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        avg_similarity = np.mean(off_diagonal)
        diversity_score = max(0, 1 - avg_similarity)  # 轉換為多樣性分數
        
        return {
            'avg_similarity': avg_similarity,
            'diversity_score': diversity_score,
            'similarity_matrix': similarity_matrix.tolist()
        }
    
    def _compute_text_quality(self, topics: List[str]) -> float:
        """計算文本質量分數"""
        quality_scores = []
        
        for topic in topics:
            # 長度適中性
            length_score = min(1.0, len(topic.split()) / 5)  # 理想長度5個詞
            
            # 包含有意義的詞彙
            meaningful_words = len([w for w in topic.split() if len(w) > 2])
            meaningful_score = min(1.0, meaningful_words / len(topic.split()))
            
            # 避免過度一般化
            specificity_score = 1.0 - (topic.lower().count('general') + topic.lower().count('various')) * 0.2
            
            topic_quality = (length_score + meaningful_score + specificity_score) / 3
            quality_scores.append(topic_quality)
        
        return np.mean(quality_scores)
    
    def _format_topics(self, topics: List[str], descriptions: List[str] = None) -> str:
        """格式化主題列表"""
        formatted = []
        
        for i, topic in enumerate(topics):
            if descriptions and i < len(descriptions):
                formatted.append(f"{i+1}. {topic}: {descriptions[i]}")
            else:
                formatted.append(f"{i+1}. {topic}")
        
        return "\n".join(formatted)
    
    def _get_evaluation_criteria(self) -> str:
        """獲取評估標準"""
        return """
        評估標準：
        1. 相關性 (Relevance): 主題是否與文檔內容密切相關
        2. 清晰度 (Clarity): 主題描述是否清晰易懂
        3. 具體性 (Specificity): 主題是否具體，避免過於寬泛
        4. 多樣性 (Diversity): 主題之間是否有足夠的差異
        5. 完整性 (Completeness): 主題是否覆蓋了重要內容
        6. 一致性 (Consistency): 主題風格和格式是否一致
        
        評分標準：
        - 9-10分: 優秀，完全滿足要求
        - 7-8分: 良好，基本滿足要求
        - 5-6分: 普通，需要改進
        - 3-4分: 較差，有明顯問題
        - 1-2分: 很差，需要重新生成
        """
    
    def _parse_score(self, score_text: str) -> float:
        """解析分數文本"""
        try:
            # 提取數字
            import re
            numbers = re.findall(r'\d+\.?\d*', str(score_text))
            if numbers:
                score = float(numbers[0])
                return min(10.0, max(0.0, score))  # 限制在0-10範圍
            return 5.0  # 默認分數
        except:
            return 5.0
    
    def _parse_recommendations(self, recommendations_text: str) -> List[str]:
        """解析建議文本"""
        try:
            lines = recommendations_text.strip().split('\n')
            recommendations = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('Recommendations:'):
                    # 移除編號
                    line = re.sub(r'^\d+\.?\s*', '', line)
                    # 移除破折號
                    line = re.sub(r'^-\s*', '', line)
                    if line:
                        recommendations.append(line)
            
            return recommendations[:5]  # 限制建議數量
        except:
            return ["Unable to parse recommendations"]

class QuantitativeEvaluator:
    """量化評估器 - 提供數值化的評估指標"""
    
    def __init__(self, vector_database: VectorDatabase):
        self.vector_db = vector_database
        self.topic_vectorizer = TopicVectorizer(vector_database)
    
    def evaluate_effectiveness(self, topics: List[str]) -> Dict[str, float]:
        """評估有效性 - 基於餘弦相似度"""
        try:
            # 計算主題與文檔的相似度
            similarity_results = self.topic_vectorizer.compute_topic_similarity(topics)
            
            effectiveness_metrics = {
                'mean_similarity': float(np.mean(similarity_results['mean_similarities'])),
                'max_similarity': float(np.max(similarity_results['mean_similarities'])),
                'min_similarity': float(np.min(similarity_results['mean_similarities'])),
                'std_similarity': float(np.std(similarity_results['mean_similarities'])),
                'coverage_score': self._compute_coverage_score(similarity_results['similarities'])
            }
            
            return effectiveness_metrics
            
        except Exception as e:
            logger.error(f"計算有效性失敗: {e}")
            return {
                'mean_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'std_similarity': 0.0,
                'coverage_score': 0.0
            }
    
    def evaluate_diversity(self, topics: List[str], top_k: int = 20) -> Dict[str, float]:
        """評估多樣性 - 基於詞彙重疊"""
        try:
            # 提取詞彙
            all_words = []
            topic_words = []
            
            for topic in topics:
                words = topic.lower().split()
                topic_words.append(set(words))
                all_words.extend(words)
            
            # 計算詞彙統計
            unique_words = set(all_words)
            total_words = len(all_words)
            
            # 計算詞彙重疊率
            overlap_scores = []
            for i in range(len(topic_words)):
                for j in range(i+1, len(topic_words)):
                    overlap = len(topic_words[i].intersection(topic_words[j]))
                    union = len(topic_words[i].union(topic_words[j]))
                    if union > 0:
                        overlap_scores.append(overlap / union)
            
            avg_overlap = np.mean(overlap_scores) if overlap_scores else 0
            diversity_score = 1 - avg_overlap  # 重疊越少，多樣性越高
            
            diversity_metrics = {
                'diversity_score': diversity_score,
                'unique_words_ratio': len(unique_words) / total_words if total_words > 0 else 0,
                'avg_overlap_ratio': avg_overlap,
                'total_unique_words': len(unique_words),
                'word_distribution_entropy': self._compute_word_entropy(all_words)
            }
            
            return diversity_metrics
            
        except Exception as e:
            logger.error(f"計算多樣性失敗: {e}")
            return {
                'diversity_score': 0.0,
                'unique_words_ratio': 0.0,
                'avg_overlap_ratio': 1.0,
                'total_unique_words': 0,
                'word_distribution_entropy': 0.0
            }
    
    def _compute_coverage_score(self, similarities: np.ndarray) -> float:
        """計算覆蓋分數"""
        # 基於每個文檔與最相似主題的相似度
        max_similarities = np.max(similarities, axis=0)
        return float(np.mean(max_similarities))
    
    def _compute_word_entropy(self, words: List[str]) -> float:
        """計算詞彙分佈熵"""
        from collections import Counter
        
        word_counts = Counter(words)
        total_words = len(words)
        
        entropy = 0
        for count in word_counts.values():
            prob = count / total_words
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _llm_evaluate_topics(self, topic_result: TopicResult, context: str) -> tuple:
        """使用LLM評估主題"""
        topics_str = "\n".join([f"{i+1}. {topic}: {desc}" 
                               for i, (topic, desc) in enumerate(zip(topic_result.topics, topic_result.topic_descriptions))])
        
        prompt = f"""你是一個專業的主題建模評估專家。請評估以下主題的質量。

主題列表：
{topics_str}

評估上下文：
{context[:500]}

請按照以下標準評估：
1. 相關性：主題是否與上下文相關
2. 清晰度：主題表達是否清晰明確
3. 多樣性：主題之間是否有良好的區分度
4. 具體性：主題是否具體而非過於抽象

請提供：
1. 整體評分（0-10分）
2. 詳細反饋
3. 3-5個改進建議

格式：
評分: [0-10的數字]
反饋: [詳細評估說明]
建議:
- [建議1]
- [建議2]
- [建議3]
"""
        
        try:
            response = self.llm_manager.generate(prompt, max_tokens=800, temperature=0.3)
            return self._parse_llm_evaluation(response)
        except Exception as e:
            logger.error(f"LLM評估失敗: {e}")
            return self._basic_evaluate_topics(topic_result, context)
    
    def _parse_llm_evaluation(self, response: str) -> tuple:
        """解析LLM評估結果"""
        try:
            # 提取評分
            score_match = re.search(r'評分[:：]\s*(\d+(?:\.\d+)?)', response)
            score = float(score_match.group(1)) if score_match else 7.0
            
            # 提取反饋
            feedback_match = re.search(r'反饋[:：]\s*(.*?)(?=建議[:：]|$)', response, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else "評估完成"
            
            # 提取建議
            suggestions_match = re.search(r'建議[:：]\s*(.*)', response, re.DOTALL)
            if suggestions_match:
                suggestions_text = suggestions_match.group(1)
                recommendations = [line.strip('- ').strip() for line in suggestions_text.split('\n') 
                                 if line.strip() and line.strip().startswith('-')]
            else:
                recommendations = ["建議改進主題的清晰度", "增強主題的具體性", "確保主題間的區分度"]
            
            return score, feedback, recommendations[:5]  # 限制建議數量
            
        except Exception as e:
            logger.error(f"解析LLM評估結果失敗: {e}")
            return 7.0, "評估解析失敗，使用默認評分", ["建議重新評估"]
    
    def _basic_evaluate_topics(self, topic_result: TopicResult, context: str) -> tuple:
        """基礎評估方法"""
        # 基於主題長度、多樣性等基本指標評估
        topics = topic_result.topics
        
        # 長度評估
        avg_length = np.mean([len(topic) for topic in topics])
        length_score = min(10, avg_length / 10) if avg_length > 0 else 5
        
        # 多樣性評估（基於詞彙重疊）
        diversity_score = self._calculate_diversity_score(topics)
        
        # 置信度評估
        confidence_score = np.mean(topic_result.confidence_scores) * 10 if topic_result.confidence_scores else 7
        
        # 綜合評分
        overall_score = (length_score + diversity_score + confidence_score) / 3
        
        feedback = f"基礎評估完成。平均長度: {avg_length:.1f}, 多樣性: {diversity_score:.1f}, 置信度: {confidence_score:.1f}"
        recommendations = [
            "考慮增加主題的具體性",
            "改善主題間的區分度",
            "提高主題描述的清晰度"
        ]
        
        return overall_score, feedback, recommendations
    
    def _calculate_diversity_score(self, topics: List[str]) -> float:
        """計算主題多樣性分數"""
        if len(topics) <= 1:
            return 5.0
        
        # 計算詞彙重疊度
        all_words = []
        topic_words = []
        
        for topic in topics:
            words = set(topic.lower().split())
            topic_words.append(words)
            all_words.extend(words)
        
        # 計算平均詞彙重疊度
        overlaps = []
        for i in range(len(topic_words)):
            for j in range(i + 1, len(topic_words)):
                if topic_words[i] and topic_words[j]:
                    overlap = len(topic_words[i].intersection(topic_words[j])) / len(topic_words[i].union(topic_words[j]))
                    overlaps.append(overlap)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0
        diversity_score = (1 - avg_overlap) * 10
        
        return max(0, min(10, diversity_score))
    
    def _evaluate_individual_topics_basic(self, topics: List[str], descriptions: List[str]) -> Dict[str, float]:
        """基礎個別主題評估"""
        individual_scores = {}
        
        for i, (topic, desc) in enumerate(zip(topics, descriptions)):
            # 基於長度和內容的簡單評分
            topic_score = min(10, len(topic) / 5) if len(topic) > 0 else 3
            desc_score = min(10, len(desc) / 10) if len(desc) > 0 else 3
            
            combined_score = (topic_score + desc_score) / 2
            individual_scores[topic] = combined_score
        
        return individual_scores
    
    def _compute_criteria_scores_basic(self, individual_scores: Dict[str, float]) -> Dict[str, float]:
        """基礎標準化評估"""
        if not individual_scores:
            return {'relevance': 5.0, 'diversity': 5.0, 'coherence': 5.0, 'text_quality': 5.0, 'confidence': 5.0}
        
        avg_score = np.mean(list(individual_scores.values()))
        
        return {
            'relevance': avg_score,
            'diversity': avg_score * 0.9,  # 稍微調低多樣性
            'coherence': avg_score * 1.1,  # 稍微調高一致性
            'text_quality': avg_score,
            'confidence': avg_score
        }
    
    def _create_default_evaluation(self, topic_result: TopicResult) -> EvaluationResult:
        """創建默認評估結果"""
        return EvaluationResult(
            overall_score=6.0,
            individual_scores={topic: 6.0 for topic in topic_result.topics},
            criteria_scores={'relevance': 6.0, 'diversity': 6.0, 'coherence': 6.0, 'text_quality': 6.0, 'confidence': 6.0},
            feedback="使用默認評估結果",
            recommendations=["建議檢查評估系統配置"],
            should_iterate=True,
            metadata={'evaluation_method': 'default', 'quality_threshold': self.quality_threshold},
            timestamp=datetime.now()
        )