import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from .retrieval_agent import TopicRetrievalAgent, RetrievalResult
from .reasoning_agent import TopicReasoningAgent, TopicResult
from .evaluation_agent import TopicEvaluationAgent, EvaluationResult
from ..utils.llm_api import LLMManager
from ..utils.vectorizer import VectorDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IterationState:
    """迭代狀態數據類"""
    iteration_count: int
    current_topics: TopicResult
    current_evaluation: EvaluationResult
    best_topics: Optional[TopicResult]
    best_score: float
    improvement_history: List[float]
    convergence_reached: bool
    stop_reason: str

class TopicModelingController:
    """主題建模迭代控制器"""
    
    def __init__(self, 
                 vector_database: VectorDatabase,
                 llm_manager: LLMManager,
                 max_iterations: int = 5,
                 quality_threshold: float = 7.0,
                 convergence_threshold: float = 0.1):
        
        self.vector_db = vector_database
        self.llm_manager = llm_manager
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.convergence_threshold = convergence_threshold
        
        # 初始化代理
        self.retrieval_agent = TopicRetrievalAgent(vector_database)
        self.reasoning_agent = TopicReasoningAgent(llm_manager)
        self.evaluation_agent = TopicEvaluationAgent(vector_database, quality_threshold, llm_manager)
        
        # 迭代狀態
        self.reset_state()
    
    def reset_state(self):
        """重置迭代狀態"""
        self.state = IterationState(
            iteration_count=0,
            current_topics=None,
            current_evaluation=None,
            best_topics=None,
            best_score=0.0,
            improvement_history=[],
            convergence_reached=False,
            stop_reason=""
        )
    
    def run_topic_modeling(self, 
                          query: str = "vaccine effectiveness and safety",
                          domain: str = "vaccine research") -> Dict[str, Any]:
        """執行完整的主題建模流程"""
        
        logger.info(f"開始主題建模迭代流程，最大迭代次數: {self.max_iterations}")
        self.reset_state()
        
        # 初始檢索
        try:
            retrieval_result = self.retrieval_agent.forward(query)
            if not retrieval_result.documents:
                return self._create_error_result("檢索失敗：未找到相關文檔")
        except Exception as e:
            return self._create_error_result(f"檢索錯誤：{e}")
        
        # 迭代改進
        while not self._should_stop():
            self.state.iteration_count += 1
            logger.info(f"開始第 {self.state.iteration_count} 輪迭代")
            
            try:
                # 生成或改進主題
                if self.state.current_topics is None:
                    # 初次生成
                    topic_result = self.reasoning_agent.forward(retrieval_result, domain)
                else:
                    # 基於評估反饋改進
                    feedback = self._generate_improvement_feedback()
                    topic_result = self.reasoning_agent.refine_topics(
                        self.state.current_topics, feedback
                    )
                
                # 評估主題質量
                context = self._prepare_evaluation_context(retrieval_result)
                evaluation_result = self.evaluation_agent.forward(topic_result, context)
                
                # 更新狀態
                self._update_state(topic_result, evaluation_result)
                
                # 記錄進度
                logger.info(f"第 {self.state.iteration_count} 輪完成，分數: {evaluation_result.overall_score:.2f}")
                
            except Exception as e:
                logger.error(f"第 {self.state.iteration_count} 輪執行失敗: {e}")
                break
        
        # 返回最終結果
        return self._create_final_result()
    
    def _should_stop(self) -> bool:
        """判斷是否應該停止迭代"""
        
        # 檢查最大迭代次數
        if self.state.iteration_count >= self.max_iterations:
            self.state.stop_reason = f"達到最大迭代次數 ({self.max_iterations})"
            return True
        
        # 檢查質量閾值
        if (self.state.current_evaluation and 
            self.state.current_evaluation.overall_score >= self.quality_threshold):
            self.state.stop_reason = f"達到質量閾值 ({self.quality_threshold})"
            return True
        
        # 檢查收斂性
        if self._check_convergence():
            self.state.stop_reason = "模型已收斂"
            self.state.convergence_reached = True
            return True
        
        return False
    
    def _check_convergence(self) -> bool:
        """檢查是否收斂"""
        if len(self.state.improvement_history) < 3:
            return False
        
        # 檢查最近3次的改進幅度
        recent_improvements = self.state.improvement_history[-3:]
        avg_improvement = sum(recent_improvements) / len(recent_improvements)
        
        return avg_improvement < self.convergence_threshold
    
    def _update_state(self, topic_result: TopicResult, evaluation_result: EvaluationResult):
        """更新迭代狀態"""
        
        # 計算改進幅度
        if self.state.current_evaluation:
            improvement = evaluation_result.overall_score - self.state.current_evaluation.overall_score
            self.state.improvement_history.append(improvement)
        
        # 更新當前結果
        self.state.current_topics = topic_result
        self.state.current_evaluation = evaluation_result
        
        # 更新最佳結果
        if evaluation_result.overall_score > self.state.best_score:
            self.state.best_topics = topic_result
            self.state.best_score = evaluation_result.overall_score
            logger.info(f"發現更好的結果，分數: {self.state.best_score:.2f}")
    
    def _generate_improvement_feedback(self) -> str:
        """生成改進反饋"""
        if not self.state.current_evaluation:
            return "請改進主題的相關性和清晰度"
        
        feedback_parts = [
            f"當前分數: {self.state.current_evaluation.overall_score:.2f}",
            f"目標分數: {self.quality_threshold}",
            f"迭代次數: {self.state.iteration_count}"
        ]
        
        # 添加具體建議
        if self.state.current_evaluation.recommendations:
            feedback_parts.append("建議改進:")
            feedback_parts.extend([f"- {rec}" for rec in self.state.current_evaluation.recommendations[:3]])
        
        # 添加評估反饋
        if self.state.current_evaluation.feedback:
            feedback_parts.append(f"詳細反饋: {self.state.current_evaluation.feedback}")
        
        return "\n".join(feedback_parts)
    
    def _prepare_evaluation_context(self, retrieval_result: RetrievalResult) -> str:
        """準備評估上下文"""
        context_parts = [
            f"查詢: {retrieval_result.query}",
            f"檢索到 {len(retrieval_result.documents)} 個相關文檔",
            f"平均相似度: {retrieval_result.metadata.get('avg_score', 0):.3f}"
        ]
        
        if retrieval_result.documents:
            context_parts.append("樣例文檔:")
            context_parts.extend([f"- {doc[:100]}..." for doc in retrieval_result.documents[:3]])
        
        return "\n".join(context_parts)
    
    def _create_final_result(self) -> Dict[str, Any]:
        """創建最終結果"""
        
        best_topics = self.state.best_topics or self.state.current_topics
        
        if not best_topics:
            return self._create_error_result("未能生成有效主題")
        
        result = {
            'success': True,
            'topics': best_topics.topics,
            'topic_descriptions': best_topics.topic_descriptions,
            'confidence_scores': best_topics.confidence_scores,
            'final_score': self.state.best_score,
            'iterations_completed': self.state.iteration_count,
            'stop_reason': self.state.stop_reason,
            'convergence_reached': self.state.convergence_reached,
            'improvement_history': self.state.improvement_history,
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'quality_threshold': self.quality_threshold,
                'max_iterations': self.max_iterations,
                'final_evaluation': self.state.current_evaluation.__dict__ if self.state.current_evaluation else None
            }
        }
        
        logger.info(f"主題建模完成: {len(best_topics.topics)} 個主題，最終分數: {self.state.best_score:.2f}")
        return result
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """創建錯誤結果"""
        logger.error(error_message)
        return {
            'success': False,
            'error': error_message,
            'topics': [],
            'topic_descriptions': [],
            'confidence_scores': [],
            'final_score': 0.0,
            'iterations_completed': self.state.iteration_count,
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'error': True
            }
        }

class BatchTopicModeling:
    """批量主題建模 - 支持多輪運行和可靠性評估"""
    
    def __init__(self, controller: TopicModelingController):
        self.controller = controller
    
    def run_multiple_rounds(self, 
                          num_rounds: int = 5,
                          query: str = "vaccine effectiveness and safety",
                          domain: str = "vaccine research") -> Dict[str, Any]:
        """運行多輪主題建模以評估可靠性"""
        
        logger.info(f"開始 {num_rounds} 輪批量主題建模")
        
        all_results = []
        all_topics = []
        all_scores = []
        
        for round_num in range(1, num_rounds + 1):
            logger.info(f"執行第 {round_num}/{num_rounds} 輪")
            
            try:
                # 重置控制器狀態
                self.controller.reset_state()
                
                # 執行單輪建模
                result = self.controller.run_topic_modeling(query, domain)
                
                if result['success']:
                    all_results.append(result)
                    all_topics.append(result['topics'])
                    all_scores.append(result['final_score'])
                    
                    logger.info(f"第 {round_num} 輪完成，分數: {result['final_score']:.2f}")
                else:
                    logger.error(f"第 {round_num} 輪失敗: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"第 {round_num} 輪執行異常: {e}")
        
        # 分析結果
        return self._analyze_batch_results(all_results, all_topics, all_scores)
    
    def _analyze_batch_results(self, 
                             results: List[Dict],
                             topics: List[List[str]],
                             scores: List[float]) -> Dict[str, Any]:
        """分析批量結果"""
        
        if not results:
            return {
                'success': False,
                'error': '所有輪次都失敗了',
                'reliability_score': 0.0
            }
        
        import numpy as np
        
        # 基本統計
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        # 主題一致性分析
        consistency_score = self._compute_topic_consistency(topics)
        
        # 可靠性評分
        reliability_score = self._compute_reliability_score(scores, consistency_score)
        
        # 選擇最佳結果
        best_idx = np.argmax(scores)
        best_result = results[best_idx]
        
        return {
            'success': True,
            'best_topics': best_result['topics'],
            'best_topic_descriptions': best_result['topic_descriptions'],
            'best_score': float(max(scores)),
            'average_score': float(avg_score),
            'score_std': float(std_score),
            'consistency_score': consistency_score,
            'reliability_score': reliability_score,
            'num_successful_rounds': len(results),
            'all_results': results,
            'analysis': {
                'score_range': [float(min(scores)), float(max(scores))],
                'score_variance': float(np.var(scores)),
                'most_consistent_topics': self._find_most_common_topics(topics)
            }
        }
    
    def _compute_topic_consistency(self, topics_list: List[List[str]]) -> float:
        """計算主題一致性"""
        if len(topics_list) < 2:
            return 1.0
        
        # 計算詞彙重疊度
        all_words = set()
        for topics in topics_list:
            for topic in topics:
                all_words.update(topic.lower().split())
        
        # 計算各輪次的詞彙重疊
        overlaps = []
        for i in range(len(topics_list)):
            for j in range(i+1, len(topics_list)):
                overlap = self._compute_word_overlap(topics_list[i], topics_list[j])
                overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _compute_word_overlap(self, topics1: List[str], topics2: List[str]) -> float:
        """計算兩組主題的詞彙重疊度"""
        words1 = set()
        words2 = set()
        
        for topic in topics1:
            words1.update(topic.lower().split())
        for topic in topics2:
            words2.update(topic.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_reliability_score(self, scores: List[float], consistency: float) -> float:
        """計算可靠性評分"""
        score_stability = 1.0 - (np.std(scores) / (np.mean(scores) + 1e-8))
        return (score_stability * 0.6 + consistency * 0.4)
    
    def _find_most_common_topics(self, topics_list: List[List[str]]) -> List[str]:
        """找出最常見的主題"""
        from collections import Counter
        
        all_topics = []
        for topics in topics_list:
            all_topics.extend(topics)
        
        # 基於詞彙相似度聚類
        topic_counter = Counter(all_topics)
        return [topic for topic, count in topic_counter.most_common(10)]