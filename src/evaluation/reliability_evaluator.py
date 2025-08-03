import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime
from collections import Counter, defaultdict
import json
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from .effectiveness_evaluator import EffectivenessEvaluator
from .diversity_evaluator import DiversityEvaluator
from ..agents.iteration_controller import BatchTopicModeling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReliabilityMetrics:
    """可靠性評估指標"""
    reliability_score: float
    consistency_score: float
    stability_score: float
    reproducibility_score: float
    confidence_interval: Tuple[float, float]
    variance_analysis: Dict[str, float]
    cross_run_similarity: float
    topic_frequency_analysis: Dict[str, Any]
    evaluation_rounds: int
    evaluation_timestamp: datetime

class ReliabilityEvaluator:
    """可靠性評估器 - 評估主題建模結果的穩定性和一致性"""
    
    def __init__(self, 
                 effectiveness_evaluator: EffectivenessEvaluator,
                 diversity_evaluator: DiversityEvaluator,
                 num_rounds: int = 5):
        
        self.effectiveness_evaluator = effectiveness_evaluator
        self.diversity_evaluator = diversity_evaluator
        self.num_rounds = num_rounds
    
    def evaluate_reliability(self, 
                           batch_results: Dict[str, Any],
                           detailed_analysis: bool = True) -> ReliabilityMetrics:
        """評估多輪運行結果的可靠性"""
        
        logger.info(f"開始可靠性評估，分析 {self.num_rounds} 輪結果")
        
        try:
            if not batch_results.get('success', False):
                return self._create_empty_metrics()
            
            # 提取多輪結果數據
            all_results = batch_results.get('all_results', [])
            if len(all_results) < 2:
                logger.warning("結果數量不足，無法進行可靠性分析")
                return self._create_empty_metrics()
            
            # 基本穩定性分析
            stability_metrics = self._analyze_score_stability(all_results)
            
            # 主題一致性分析
            consistency_metrics = self._analyze_topic_consistency(all_results)
            
            # 結果可重現性分析
            reproducibility_metrics = self._analyze_reproducibility(all_results)
            
            if detailed_analysis:
                # 詳細的主題頻率分析
                frequency_analysis = self._analyze_topic_frequency(all_results)
                
                # 跨輪相似度分析
                cross_similarity = self._analyze_cross_run_similarity(all_results)
            else:
                frequency_analysis = {}
                cross_similarity = 0.5
            
            # 計算綜合可靠性分數
            overall_reliability = self._compute_overall_reliability(
                stability_metrics, consistency_metrics, reproducibility_metrics
            )
            
            # 計算置信區間
            scores = [result['final_score'] for result in all_results]
            confidence_interval = self._compute_confidence_interval(scores)
            
            metrics = ReliabilityMetrics(
                reliability_score=overall_reliability,
                consistency_score=consistency_metrics['consistency_score'],
                stability_score=stability_metrics['stability_score'],
                reproducibility_score=reproducibility_metrics['reproducibility_score'],
                confidence_interval=confidence_interval,
                variance_analysis=stability_metrics['variance_analysis'],
                cross_run_similarity=cross_similarity,
                topic_frequency_analysis=frequency_analysis,
                evaluation_rounds=len(all_results),
                evaluation_timestamp=datetime.now()
            )
            
            logger.info(f"可靠性評估完成，整體分數: {overall_reliability:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"可靠性評估失敗: {e}")
            return self._create_empty_metrics()
    
    def _analyze_score_stability(self, results: List[Dict]) -> Dict[str, float]:
        """分析分數穩定性"""
        
        scores = [result['final_score'] for result in results]
        
        if not scores:
            return {'stability_score': 0.0, 'variance_analysis': {}}
        
        # 基本統計
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        cv = std_score / mean_score if mean_score > 0 else float('inf')  # 變異係數
        
        # 穩定性分數（變異係數越小，穩定性越高）
        stability_score = max(0, 1 - cv)
        
        # 方差分析
        variance_analysis = {
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'coefficient_of_variation': float(cv),
            'min_score': float(min(scores)),
            'max_score': float(max(scores)),
            'score_range': float(max(scores) - min(scores)),
            'median_score': float(np.median(scores))
        }
        
        return {
            'stability_score': stability_score,
            'variance_analysis': variance_analysis
        }
    
    def _analyze_topic_consistency(self, results: List[Dict]) -> Dict[str, float]:
        """分析主題一致性"""
        
        all_topics = [result['topics'] for result in results]
        
        if len(all_topics) < 2:
            return {'consistency_score': 1.0}
        
        # 計算詞彙重疊度
        word_overlaps = []
        semantic_similarities = []
        
        for i in range(len(all_topics)):
            for j in range(i + 1, len(all_topics)):
                # 詞彙重疊
                word_overlap = self._compute_word_overlap(all_topics[i], all_topics[j])
                word_overlaps.append(word_overlap)
                
                # 語義相似度
                semantic_sim = self._compute_semantic_similarity(all_topics[i], all_topics[j])
                semantic_similarities.append(semantic_sim)
        
        # 綜合一致性分數
        avg_word_overlap = np.mean(word_overlaps) if word_overlaps else 0
        avg_semantic_similarity = np.mean(semantic_similarities) if semantic_similarities else 0
        
        consistency_score = (avg_word_overlap * 0.4 + avg_semantic_similarity * 0.6)
        
        return {
            'consistency_score': consistency_score,
            'word_overlap_avg': avg_word_overlap,
            'semantic_similarity_avg': avg_semantic_similarity
        }
    
    def _analyze_reproducibility(self, results: List[Dict]) -> Dict[str, float]:
        """分析結果可重現性"""
        
        # 分析迭代次數的一致性
        iteration_counts = [result.get('iterations_completed', 0) for result in results]
        iteration_consistency = 1 - (np.std(iteration_counts) / (np.mean(iteration_counts) + 1))
        
        # 分析收斂性的一致性
        convergence_rates = []
        for result in results:
            if 'improvement_history' in result and result['improvement_history']:
                # 計算收斂速度（改進幅度遞減的程度）
                improvements = result['improvement_history']
                if len(improvements) > 1:
                    convergence_rate = abs(improvements[-1] - improvements[0]) / len(improvements)
                    convergence_rates.append(convergence_rate)
        
        convergence_consistency = 1 - (np.std(convergence_rates) / (np.mean(convergence_rates) + 1)) if convergence_rates else 0.5
        
        # 綜合可重現性分數
        reproducibility_score = (iteration_consistency * 0.5 + convergence_consistency * 0.5)
        
        return {
            'reproducibility_score': max(0, min(1, reproducibility_score)),
            'iteration_consistency': iteration_consistency,
            'convergence_consistency': convergence_consistency
        }
    
    def _analyze_topic_frequency(self, results: List[Dict]) -> Dict[str, Any]:
        """分析主題詞彙頻率"""
        
        # 收集所有主題的詞彙
        all_words = []
        topic_word_counts = defaultdict(int)
        
        for result in results:
            topics = result.get('topics', [])
            for topic in topics:
                words = topic.lower().split()
                all_words.extend(words)
                for word in words:
                    topic_word_counts[word] += 1
        
        total_words = len(all_words)
        unique_words = len(set(all_words))
        
        # 分析詞彙分佈
        word_frequency_distribution = Counter(topic_word_counts.values())
        most_common_words = Counter(all_words).most_common(20)
        
        # 計算詞彙穩定性（高頻詞的比例）
        high_frequency_words = sum(1 for count in topic_word_counts.values() if count >= len(results) * 0.6)
        word_stability = high_frequency_words / unique_words if unique_words > 0 else 0
        
        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'word_diversity_ratio': unique_words / total_words if total_words > 0 else 0,
            'word_stability': word_stability,
            'frequency_distribution': dict(word_frequency_distribution),
            'most_common_words': most_common_words,
            'high_frequency_words_count': high_frequency_words
        }
    
    def _analyze_cross_run_similarity(self, results: List[Dict]) -> float:
        """分析跨輪次相似度"""
        
        all_topics = [result['topics'] for result in results]
        
        if len(all_topics) < 2:
            return 1.0
        
        # 使用TF-IDF計算語義相似度
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # 將每輪的主題合併為一個文檔
            topic_documents = []
            for topics in all_topics:
                combined_text = ' '.join(topics)
                topic_documents.append(combined_text)
            
            # 向量化
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(topic_documents)
            
            # 計算相似度矩陣
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 計算平均相似度（排除對角線）
            mask = np.ones(similarity_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            
            avg_similarity = np.mean(similarity_matrix[mask])
            return float(avg_similarity)
            
        except Exception as e:
            logger.error(f"跨輪相似度計算失敗: {e}")
            return 0.5
    
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
    
    def _compute_semantic_similarity(self, topics1: List[str], topics2: List[str]) -> float:
        """計算兩組主題的語義相似度"""
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # 合併主題為文檔
            doc1 = ' '.join(topics1)
            doc2 = ' '.join(topics2)
            
            # 向量化
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
            
            # 計算相似度
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"語義相似度計算失敗: {e}")
            return 0.0
    
    def _compute_overall_reliability(self, 
                                   stability_metrics: Dict,
                                   consistency_metrics: Dict,
                                   reproducibility_metrics: Dict) -> float:
        """計算整體可靠性分數"""
        
        # 權重分配
        weights = {
            'stability': 0.4,
            'consistency': 0.4,
            'reproducibility': 0.2
        }
        
        # 組合分數
        overall_score = (
            stability_metrics['stability_score'] * weights['stability'] +
            consistency_metrics['consistency_score'] * weights['consistency'] +
            reproducibility_metrics['reproducibility_score'] * weights['reproducibility']
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _compute_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """計算置信區間"""
        
        if len(scores) < 2:
            return (0.0, 0.0)
        
        try:
            mean_score = np.mean(scores)
            sem = stats.sem(scores)  # 標準誤差
            
            # 使用t分佈計算置信區間
            t_value = stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
            margin_error = t_value * sem
            
            lower_bound = mean_score - margin_error
            upper_bound = mean_score + margin_error
            
            return (float(lower_bound), float(upper_bound))
            
        except Exception as e:
            logger.error(f"置信區間計算失敗: {e}")
            return (0.0, 0.0)
    
    def _create_empty_metrics(self) -> ReliabilityMetrics:
        """創建空的可靠性指標"""
        return ReliabilityMetrics(
            reliability_score=0.0,
            consistency_score=0.0,
            stability_score=0.0,
            reproducibility_score=0.0,
            confidence_interval=(0.0, 0.0),
            variance_analysis={},
            cross_run_similarity=0.0,
            topic_frequency_analysis={},
            evaluation_rounds=0,
            evaluation_timestamp=datetime.now()
        )
    
    def generate_reliability_report(self, metrics: ReliabilityMetrics) -> Dict[str, Any]:
        """生成可靠性評估報告"""
        
        # 評估等級
        reliability_grade = self._grade_reliability(metrics.reliability_score)
        
        # 分析結論
        analysis_conclusions = self._generate_analysis_conclusions(metrics)
        
        # 改進建議
        recommendations = self._generate_reliability_recommendations(metrics)
        
        report = {
            'summary': {
                'overall_reliability': reliability_grade,
                'reliability_score': metrics.reliability_score,
                'evaluation_rounds': metrics.evaluation_rounds,
                'confidence_interval': metrics.confidence_interval
            },
            'detailed_metrics': {
                'stability': {
                    'stability_score': metrics.stability_score,
                    'variance_analysis': metrics.variance_analysis
                },
                'consistency': {
                    'consistency_score': metrics.consistency_score,
                    'cross_run_similarity': metrics.cross_run_similarity
                },
                'reproducibility': {
                    'reproducibility_score': metrics.reproducibility_score
                }
            },
            'topic_analysis': {
                'frequency_analysis': metrics.topic_frequency_analysis,
                'word_stability': metrics.topic_frequency_analysis.get('word_stability', 0)
            },
            'analysis_conclusions': analysis_conclusions,
            'recommendations': recommendations,
            'metadata': {
                'evaluation_method': 'multi_round_consistency',
                'num_evaluation_rounds': metrics.evaluation_rounds,
                'timestamp': metrics.evaluation_timestamp.isoformat(),
                'evaluator_version': '1.0'
            }
        }
        
        return report
    
    def _grade_reliability(self, reliability_score: float) -> str:
        """評估可靠性等級"""
        if reliability_score >= 0.8:
            return "優秀"
        elif reliability_score >= 0.7:
            return "良好"
        elif reliability_score >= 0.6:
            return "中等"
        elif reliability_score >= 0.4:
            return "較差"
        else:
            return "很差"
    
    def _generate_analysis_conclusions(self, metrics: ReliabilityMetrics) -> List[str]:
        """生成分析結論"""
        conclusions = []
        
        # 穩定性結論
        if metrics.stability_score >= 0.8:
            conclusions.append("系統表現出優秀的分數穩定性")
        elif metrics.stability_score >= 0.6:
            conclusions.append("系統分數相對穩定，偶有波動")
        else:
            conclusions.append("系統分數波動較大，穩定性有待改善")
        
        # 一致性結論
        if metrics.consistency_score >= 0.7:
            conclusions.append("主題生成具有良好的一致性")
        elif metrics.consistency_score >= 0.5:
            conclusions.append("主題一致性中等，存在一定變化")
        else:
            conclusions.append("主題變化較大，一致性不足")
        
        # 可重現性結論
        if metrics.reproducibility_score >= 0.7:
            conclusions.append("結果具有良好的可重現性")
        else:
            conclusions.append("結果重現性有待提升")
        
        # 置信區間分析
        ci_lower, ci_upper = metrics.confidence_interval
        ci_width = ci_upper - ci_lower
        if ci_width < 1.0:
            conclusions.append("分數置信區間較窄，結果可信度高")
        elif ci_width < 2.0:
            conclusions.append("分數置信區間適中")
        else:
            conclusions.append("分數置信區間較寬，結果不確定性較高")
        
        return conclusions
    
    def _generate_reliability_recommendations(self, metrics: ReliabilityMetrics) -> List[str]:
        """生成可靠性改進建議"""
        recommendations = []
        
        if metrics.reliability_score < 0.6:
            recommendations.append("整體可靠性較低，建議檢查系統參數設置")
        
        if metrics.stability_score < 0.6:
            recommendations.append("分數穩定性不足，建議增加迭代次數或調整收斂條件")
        
        if metrics.consistency_score < 0.5:
            recommendations.append("主題一致性較差，建議固定隨機種子或優化主題生成策略")
        
        if metrics.reproducibility_score < 0.5:
            recommendations.append("可重現性不佳，建議標準化執行環境和參數")
        
        # 基於方差分析的建議
        if metrics.variance_analysis.get('coefficient_of_variation', 0) > 0.3:
            recommendations.append("分數變異係數較高，建議增加評估輪次")
        
        # 基於詞彙分析的建議
        word_stability = metrics.topic_frequency_analysis.get('word_stability', 0)
        if word_stability < 0.3:
            recommendations.append("詞彙穩定性較低，建議改善主題生成的一致性")
        
        if not recommendations:
            recommendations.append("可靠性表現良好，系統運行穩定")
        
        return recommendations

class ReliabilityComparisonAnalyzer:
    """可靠性比較分析器"""
    
    def __init__(self, reliability_evaluator: ReliabilityEvaluator):
        self.evaluator = reliability_evaluator
    
    def compare_reliability_across_settings(self, 
                                          multiple_batch_results: List[Dict],
                                          setting_names: List[str]) -> Dict[str, Any]:
        """比較不同設置下的可靠性"""
        
        if len(multiple_batch_results) != len(setting_names):
            raise ValueError("結果數量與設置名稱數量不匹配")
        
        # 評估各設置的可靠性
        reliability_results = []
        for i, batch_result in enumerate(multiple_batch_results):
            metrics = self.evaluator.evaluate_reliability(batch_result)
            reliability_results.append({
                'setting_name': setting_names[i],
                'metrics': metrics,
                'reliability_score': metrics.reliability_score
            })
        
        # 排序
        reliability_results.sort(key=lambda x: x['reliability_score'], reverse=True)
        
        # 比較分析
        comparison_analysis = self._analyze_reliability_differences(reliability_results)
        
        return {
            'ranking': reliability_results,
            'best_setting': reliability_results[0]['setting_name'] if reliability_results else None,
            'comparison_analysis': comparison_analysis
        }
    
    def _analyze_reliability_differences(self, results: List[Dict]) -> Dict[str, Any]:
        """分析可靠性差異"""
        
        if len(results) < 2:
            return {}
        
        scores = [r['reliability_score'] for r in results]
        
        analysis = {
            'score_range': max(scores) - min(scores),
            'performance_gap': scores[0] - scores[-1],  # 最佳與最差的差距
            'coefficient_of_variation': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
            'significant_differences': []
        }
        
        # 識別顯著差異
        for i in range(len(results) - 1):
            current_score = results[i]['reliability_score']
            next_score = results[i + 1]['reliability_score']
            
            if current_score - next_score > 0.1:  # 閾值可調整
                analysis['significant_differences'].append({
                    'better_setting': results[i]['setting_name'],
                    'worse_setting': results[i + 1]['setting_name'],
                    'score_difference': current_score - next_score
                })
        
        return analysis