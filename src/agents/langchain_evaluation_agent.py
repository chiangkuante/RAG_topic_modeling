"""
LangChain-based Topic Evaluation Agent
使用LangChain构建的主题评估代理
"""

from typing import List, Dict, Tuple, Optional, Any
import json
import re
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableSequence

from .langchain_reasoning_agent import TopicResult
from ..utils.langchain_llm_manager import LangChainLLMManager
from ..utils.vectorizer import VectorDatabase, TopicVectorizer

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

class EvaluationOutputParser(BaseOutputParser):
    """評估輸出解析器"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """解析評估輸出"""
        try:
            # 提取評分
            score_match = re.search(r'整體評分[:：]\s*(\d+(?:\.\d+)?)', text)
            overall_score = float(score_match.group(1)) if score_match else 7.0
            
            # 提取反饋
            feedback_match = re.search(r'詳細反饋[:：]\s*(.*?)(?=改進建議[:：]|$)', text, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else "評估完成"
            
            # 提取建議
            suggestions_match = re.search(r'改進建議[:：]\s*(.*)', text, re.DOTALL)
            recommendations = []
            if suggestions_match:
                suggestions_text = suggestions_match.group(1)
                recommendations = [
                    line.strip('- ').strip() 
                    for line in suggestions_text.split('\n') 
                    if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•'))
                ]
            
            # 提取個別評分
            individual_scores = {}
            individual_match = re.findall(r'(\d+)\.\s*([^|]+)\|\s*([^|]+)\|\s*(\d+(?:\.\d+)?)', text)
            for match in individual_match:
                topic_name = match[1].strip()
                score = float(match[3])
                individual_scores[topic_name] = score
            
            return {
                'overall_score': min(10.0, max(0.0, overall_score)),
                'feedback': feedback,
                'recommendations': recommendations[:5],  # 限制建議數量
                'individual_scores': individual_scores
            }
            
        except Exception as e:
            logger.error(f"評估解析失敗: {e}")
            return {
                'overall_score': 6.0,
                'feedback': f"解析錯誤: {str(e)}",
                'recommendations': ["建議重新評估"],
                'individual_scores': {}
            }

class LangChainTopicEvaluationAgent:
    """基於LangChain的主題評估代理"""
    
    def __init__(self, vector_database: VectorDatabase, quality_threshold: float = 7.0, 
                 llm_manager: Optional[LangChainLLMManager] = None):
        self.vector_db = vector_database
        self.topic_vectorizer = TopicVectorizer(vector_database)
        self.quality_threshold = quality_threshold
        self.llm_manager = llm_manager
        self.evaluation_parser = EvaluationOutputParser()
        
        # 設置提示模板
        self._setup_prompts()
    
    def _setup_prompts(self):
        """設置評估提示模板"""
        self.evaluation_prompt = PromptTemplate(
            input_variables=["topics", "context", "num_topics"],
            template="""你是一個專業的主題建模評估專家。請全面評估以下主題的質量。

待評估主題：
{topics}

評估上下文：
{context}

請按照以下標準進行評估：
1. 相關性 (0-10分): 主題是否與上下文文檔相關
2. 清晰度 (0-10分): 主題表達是否清晰明確  
3. 多樣性 (0-10分): 主題之間是否有良好的區分度
4. 具體性 (0-10分): 主題是否具體而非過於抽象
5. 覆蓋性 (0-10分): 主題是否全面覆蓋了文檔的主要內容

請提供以下評估結果：

整體評分: [0-10的數字]

個別主題評分:
1. [主題名稱] | [相關性評分] | [清晰度評分] | [綜合評分]
2. [主題名稱] | [相關性評分] | [清晰度評分] | [綜合評分]
(為每個主題提供評分)

詳細反饋:
[請詳細說明評估理由，包括優點和不足]

改進建議:
- [具體改進建議1]
- [具體改進建議2]  
- [具體改進建議3]
- [具體改進建議4]
- [具體改進建議5]

評估標準：
- 8-10分: 優秀，主題質量很高
- 6-7分: 良好，有改進空間
- 4-5分: 普通，需要顯著改進
- 0-3分: 較差，需要重新生成
"""
        )
        
        self.quality_prompt = PromptTemplate(
            input_variables=["topic", "description", "context"],
            template="""評估單個主題的質量：

主題: {topic}
描述: {description}
上下文: {context}

請從以下方面評分 (0-10分):
1. 相關性: 該主題與上下文的相關程度
2. 清晰度: 主題表達的清晰程度
3. 具體性: 主題的具體性和可操作性

格式：
相關性: [分數]
清晰度: [分數]  
具體性: [分數]
說明: [評分理由]
"""
        )
    
    def forward(self, topic_result: TopicResult, context: str = "") -> EvaluationResult:
        """主要評估流程"""
        
        try:
            if self.llm_manager and self.llm_manager.active_llm:
                # 使用LLM進行深度評估
                evaluation = self._llm_evaluate_topics(topic_result, context)
            else:
                # 使用基礎評估
                evaluation = self._basic_evaluate_topics(topic_result, context)
            
            # 計算標準化指標
            criteria_scores = self._compute_criteria_scores(evaluation, topic_result)
            
            # 決定是否繼續迭代
            should_iterate = evaluation['overall_score'] < self.quality_threshold
            
            result = EvaluationResult(
                overall_score=evaluation['overall_score'],
                individual_scores=evaluation.get('individual_scores', {}),
                criteria_scores=criteria_scores,
                feedback=evaluation['feedback'],
                recommendations=evaluation['recommendations'],
                should_iterate=should_iterate,
                metadata={
                    'evaluation_method': 'langchain_llm' if self.llm_manager else 'basic',
                    'quality_threshold': self.quality_threshold,
                    'num_topics_evaluated': len(topic_result.topics),
                    'avg_confidence': np.mean(topic_result.confidence_scores) if topic_result.confidence_scores else 0.0
                },
                timestamp=datetime.now()
            )
            
            logger.info(f"評估完成：整體分數 {result.overall_score:.2f}, 需要迭代: {should_iterate}")
            return result
            
        except Exception as e:
            logger.error(f"評估過程出錯: {e}")
            return self._create_default_evaluation(topic_result)
    
    def _llm_evaluate_topics(self, topic_result: TopicResult, context: str) -> Dict[str, Any]:
        """使用LLM進行主題評估"""
        try:
            # 準備主題字符串
            topics_str = "\n".join([
                f"{i+1}. {topic}: {desc} (信心度: {score:.2f})"
                for i, (topic, desc, score) in enumerate(zip(
                    topic_result.topics,
                    topic_result.topic_descriptions,
                    topic_result.confidence_scores
                ))
            ])
            
            # 創建評估鏈
            evaluation_chain = self.llm_manager.create_chain(
                self.evaluation_prompt,
                self.evaluation_parser
            )
            
            # 執行評估
            logger.info("開始LLM評估...")
            result = self.llm_manager.invoke_chain(evaluation_chain, {
                'topics': topics_str,
                'context': context[:1000],  # 限制上下文長度
                'num_topics': len(topic_result.topics)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"LLM評估失敗: {e}")
            return self._basic_evaluate_topics(topic_result, context)
    
    def _basic_evaluate_topics(self, topic_result: TopicResult, context: str) -> Dict[str, Any]:
        """基礎評估方法"""
        try:
            topics = topic_result.topics
            descriptions = topic_result.topic_descriptions
            confidence_scores = topic_result.confidence_scores
            
            # 長度評估
            avg_topic_length = np.mean([len(topic) for topic in topics])
            length_score = min(10, avg_topic_length / 8) if avg_topic_length > 0 else 3
            
            # 描述質量評估
            avg_desc_length = np.mean([len(desc) for desc in descriptions])
            desc_score = min(10, avg_desc_length / 20) if avg_desc_length > 0 else 3
            
            # 多樣性評估
            diversity_score = self._calculate_diversity_score(topics)
            
            # 置信度評估
            confidence_score = np.mean(confidence_scores) * 10 if confidence_scores else 5
            
            # 綜合評分
            overall_score = (length_score + desc_score + diversity_score + confidence_score) / 4
            
            # 個別評分
            individual_scores = {}
            for i, (topic, desc, conf) in enumerate(zip(topics, descriptions, confidence_scores)):
                topic_length_score = min(10, len(topic) / 8)
                desc_length_score = min(10, len(desc) / 20)
                individual_score = (topic_length_score + desc_length_score + conf * 10) / 3
                individual_scores[topic] = individual_score
            
            # 生成反饋和建議
            feedback = f"""基礎評估結果：
- 平均主題長度: {avg_topic_length:.1f} 字符
- 平均描述長度: {avg_desc_length:.1f} 字符  
- 多樣性評分: {diversity_score:.1f}/10
- 平均置信度: {np.mean(confidence_scores):.2f}
- 綜合評分: {overall_score:.2f}/10"""
            
            recommendations = [
                "增加主題的具體性和詳細程度" if avg_topic_length < 20 else "主題長度適中",
                "改善主題描述的豐富度" if avg_desc_length < 50 else "描述內容充實", 
                "提高主題間的區分度" if diversity_score < 6 else "主題多樣性良好",
                "增強生成的置信度" if np.mean(confidence_scores) < 0.7 else "置信度表現良好",
                "考慮優化整體主題質量"
            ]
            
            return {
                'overall_score': overall_score,
                'individual_scores': individual_scores,
                'feedback': feedback,
                'recommendations': recommendations[:4]
            }
            
        except Exception as e:
            logger.error(f"基礎評估失敗: {e}")
            return {
                'overall_score': 5.0,
                'individual_scores': {},
                'feedback': f"評估出錯: {str(e)}",
                'recommendations': ["建議檢查評估系統"]
            }
    
    def _calculate_diversity_score(self, topics: List[str]) -> float:
        """計算主題多樣性分數"""
        if len(topics) <= 1:
            return 5.0
        
        try:
            # 計算詞彙重疊度
            topic_words = []
            for topic in topics:
                words = set(topic.lower().split())
                topic_words.append(words)
            
            # 計算平均Jaccard相似度
            overlaps = []
            for i in range(len(topic_words)):
                for j in range(i + 1, len(topic_words)):
                    if topic_words[i] and topic_words[j]:
                        intersection = len(topic_words[i].intersection(topic_words[j]))
                        union = len(topic_words[i].union(topic_words[j]))
                        overlap = intersection / union if union > 0 else 0
                        overlaps.append(overlap)
            
            avg_overlap = np.mean(overlaps) if overlaps else 0
            diversity_score = (1 - avg_overlap) * 10
            
            return max(0, min(10, diversity_score))
            
        except Exception as e:
            logger.error(f"多樣性計算失敗: {e}")
            return 5.0
    
    def _compute_criteria_scores(self, evaluation: Dict[str, Any], topic_result: TopicResult) -> Dict[str, float]:
        """計算標準化評估指標"""
        try:
            individual_scores = evaluation.get('individual_scores', {})
            avg_individual = np.mean(list(individual_scores.values())) if individual_scores else evaluation['overall_score']
            
            # 基於多樣性的額外計算
            diversity = self._calculate_diversity_score(topic_result.topics)
            confidence = np.mean(topic_result.confidence_scores) * 10 if topic_result.confidence_scores else 6
            
            return {
                'relevance': min(10, avg_individual * 1.0),
                'clarity': min(10, avg_individual * 0.9),
                'diversity': min(10, diversity),
                'specificity': min(10, avg_individual * 0.8),
                'confidence': min(10, confidence),
                'coverage': min(10, evaluation['overall_score'] * 0.9)
            }
            
        except Exception as e:
            logger.error(f"標準化指標計算失敗: {e}")
            return {
                'relevance': 6.0, 'clarity': 6.0, 'diversity': 6.0,
                'specificity': 6.0, 'confidence': 6.0, 'coverage': 6.0
            }
    
    def _create_default_evaluation(self, topic_result: TopicResult) -> EvaluationResult:
        """創建默認評估結果"""
        return EvaluationResult(
            overall_score=6.0,
            individual_scores={topic: 6.0 for topic in topic_result.topics},
            criteria_scores={
                'relevance': 6.0, 'clarity': 6.0, 'diversity': 6.0,
                'specificity': 6.0, 'confidence': 6.0, 'coverage': 6.0
            },
            feedback="使用默認評估結果，建議檢查評估系統配置",
            recommendations=["檢查LLM配置", "驗證評估提示", "確認輸入數據"],
            should_iterate=True,
            metadata={
                'evaluation_method': 'default_fallback',
                'quality_threshold': self.quality_threshold
            },
            timestamp=datetime.now()
        )