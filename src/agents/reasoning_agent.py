from typing import List, Dict, Tuple, Optional, Any
import json
import re
import logging
from dataclasses import dataclass
from datetime import datetime

from .retrieval_agent import RetrievalResult
from ..utils.llm_api import LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TopicResult:
    """主題結果數據類"""
    topics: List[str]
    topic_descriptions: List[str]
    confidence_scores: List[float]
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime

class TopicReasoningAgent:
    """主題推理代理 - 負責基於檢索內容生成和優化主題"""
    
    def __init__(self, llm_manager: LLMManager, num_topics: int = 10):
        self.llm_manager = llm_manager
        self.num_topics = num_topics
        
    def forward(self, retrieval_result: RetrievalResult, domain: str = "vaccine research") -> TopicResult:
        """主要推理流程"""
        
        if self.llm_manager.active_llm is None:
            raise ValueError("請先設置活躍的LLM")
        
        # 準備上下文
        context = self._prepare_context(retrieval_result)
        
        # 構建主題生成提示
        prompt = self._build_topic_generation_prompt(context, domain)
        
        # 調用LLM生成主題
        logger.info("開始生成主題...")
        response = self.llm_manager.generate(prompt, max_tokens=1500, temperature=0.7)
        
        # 解析主題
        topics, descriptions, confidence_scores, reasoning = self._parse_llm_response(response)
        
        # 創建結果
        topic_result = TopicResult(
            topics=topics,
            topic_descriptions=descriptions,
            confidence_scores=confidence_scores,
            reasoning=reasoning,
            metadata={
                'generation_method': 'direct_llm_call',
                'retrieval_quality': retrieval_result.metadata.get('avg_score', 0),
                'num_source_docs': len(retrieval_result.documents),
                'domain': domain
            },
            timestamp=datetime.now()
        )
        
        logger.info(f"生成了{len(topics)}個主題")
        return topic_result
    
    def _build_topic_generation_prompt(self, context: str, domain: str) -> str:
        """構建主題生成提示"""
        prompt = f"""你是一個專業的主題建模專家。基於以下檢索到的文檔內容，請生成{self.num_topics}個相關主題。

領域：{domain}

檢索上下文：
{context}

請按照以下格式回答：

推理過程：
[請詳細說明你如何分析這些文檔並識別主要主題的過程]

主題列表：
1. [主題名稱] | [詳細描述] | [信心分數0.0-1.0]
2. [主題名稱] | [詳細描述] | [信心分數0.0-1.0]
...

要求：
1. 主題應該具體且有意義
2. 每個主題都要有清晰的描述
3. 信心分數反映主題在文檔中的支持程度
4. 主題之間應該有所區別，避免重複
5. 重點關注{domain}領域的核心概念
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> tuple:
        """解析LLM響應"""
        topics = []
        descriptions = []
        confidence_scores = []
        reasoning = ""
        
        try:
            # 分割響應
            parts = response.split("主題列表：")
            if len(parts) >= 2:
                reasoning = parts[0].replace("推理過程：", "").strip()
                topics_text = parts[1].strip()
                
                # 解析主题行
                for line in topics_text.split('\n'):
                    line = line.strip()
                    if not line or not any(line.startswith(str(i)) for i in range(1, 20)):
                        continue
                    
                    # 解析格式: "1. 主題名稱 | 描述 | 分數"
                    try:
                        # 移除序號
                        content = re.sub(r'^\d+\.\s*', '', line)
                        parts = content.split('|')
                        
                        if len(parts) >= 3:
                            topic = parts[0].strip()
                            description = parts[1].strip()
                            confidence = float(parts[2].strip())
                            
                            topics.append(topic)
                            descriptions.append(description)
                            confidence_scores.append(confidence)
                        elif len(parts) >= 2:
                            topic = parts[0].strip()
                            description = parts[1].strip()
                            
                            topics.append(topic)
                            descriptions.append(description)
                            confidence_scores.append(0.8)  # 默認分數
                        else:
                            topics.append(content)
                            descriptions.append(content)
                            confidence_scores.append(0.7)  # 默認分數
                            
                    except Exception as e:
                        logger.warning(f"解析主題行失敗: {line}, 錯誤: {e}")
                        continue
            
            # 確保至少有一些主題
            if not topics:
                logger.warning("未能解析到主題，使用備用解析")
                # 備用解析：簡單分行
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                for line in lines[-min(self.num_topics, len(lines)):]:
                    if len(line) > 10:  # 過濾太短的行
                        topics.append(line[:100])
                        descriptions.append(line)
                        confidence_scores.append(0.6)
            
            # 限制主題數量
            if len(topics) > self.num_topics:
                topics = topics[:self.num_topics]
                descriptions = descriptions[:self.num_topics]
                confidence_scores = confidence_scores[:self.num_topics]
            
            logger.info(f"成功解析{len(topics)}個主題")
            return topics, descriptions, confidence_scores, reasoning or response[:500]
            
        except Exception as e:
            logger.error(f"LLM響應解析失敗: {e}")
            # 返回默認結果
            default_topics = [f"主題 {i+1}" for i in range(self.num_topics)]
            default_descriptions = [f"基於檢索內容的主題 {i+1}" for i in range(self.num_topics)]
            default_confidence = [0.5] * self.num_topics
            return default_topics, default_descriptions, default_confidence, response[:500]
    
    def refine_topics(self, topic_result: TopicResult, feedback: str) -> TopicResult:
        """優化主題"""
        
        if self.llm_manager.active_llm is None:
            raise ValueError("請先設置活躍的LLM")
        
        # 構建主題優化提示
        prompt = self._build_topic_refinement_prompt(topic_result, feedback)
        
        # 調用LLM優化主題
        logger.info("開始優化主題...")
        response = self.llm_manager.generate(prompt, max_tokens=1500, temperature=0.7)
        
        # 解析優化後的主題
        refined_topics, descriptions, confidence_scores, improvements = self._parse_llm_response(response)
        
        # 創建新結果
        refined_result = TopicResult(
            topics=refined_topics,
            topic_descriptions=descriptions,
            confidence_scores=confidence_scores,
            reasoning=f"Original: {topic_result.reasoning}\nRefinement: {improvements}",
            metadata={
                **topic_result.metadata,
                'refinement_applied': True,
                'refinement_feedback': feedback
            },
            timestamp=datetime.now()
        )
        
        logger.info(f"優化了{len(refined_topics)}個主題")
        return refined_result
    
    def _build_topic_refinement_prompt(self, topic_result: TopicResult, feedback: str) -> str:
        """構建主題優化提示"""
        original_topics = "\n".join([f"{i+1}. {topic} | {desc}" 
                                   for i, (topic, desc) in enumerate(zip(topic_result.topics, topic_result.topic_descriptions))])
        
        prompt = f"""你是一個專業的主題建模專家。請根據以下反饋優化現有主題。

原始主題：
{original_topics}

原始推理過程：
{topic_result.reasoning[:500]}

優化反饋：
{feedback}

請按照以下格式提供優化後的主題：

推理過程：
[請說明你如何根據反饋改進這些主題]

主題列表：
1. [優化後主題名稱] | [詳細描述] | [信心分數0.0-1.0]
2. [優化後主題名稱] | [詳細描述] | [信心分數0.0-1.0]
...

要求：
1. 考慮反饋中的建議
2. 保持主題的相關性和具體性
3. 改進主題的清晰度和區分度
4. 調整信心分數以反映改進程度
5. 保持{len(topic_result.topics)}個主題
"""
        return prompt
    
    def _prepare_context(self, retrieval_result: RetrievalResult) -> str:
        """準備上下文信息"""
        if not retrieval_result.documents:
            return "No relevant documents found."
        
        # 選擇最相關的文檔
        top_docs = retrieval_result.documents[:5]
        
        context_parts = [
            f"Query: {retrieval_result.query}",
            f"Number of relevant documents: {len(retrieval_result.documents)}",
            "Most relevant documents:",
        ]
        
        for i, doc in enumerate(top_docs, 1):
            score = retrieval_result.scores[i-1] if i-1 < len(retrieval_result.scores) else 0
            context_parts.append(f"Document {i} (score: {score:.3f}): {doc[:200]}...")
        
        return "\n".join(context_parts)
    

class TopicValidator:
    """主題驗證器"""
    
    def __init__(self):
        self.min_topic_length = 3
        self.max_topic_length = 100
        
    def validate_topics(self, topics: List[str]) -> Dict[str, Any]:
        """驗證主題質量"""
        validation_results = {
            'valid_topics': [],
            'invalid_topics': [],
            'validation_score': 0.0,
            'issues': []
        }
        
        for i, topic in enumerate(topics):
            issues = []
            
            # 長度檢查
            if len(topic) < self.min_topic_length:
                issues.append(f"Topic {i+1} too short")
            elif len(topic) > self.max_topic_length:
                issues.append(f"Topic {i+1} too long")
            
            # 空值檢查
            if not topic.strip():
                issues.append(f"Topic {i+1} is empty")
            
            # 重複檢查
            if topic in validation_results['valid_topics']:
                issues.append(f"Topic {i+1} is duplicate")
            
            if issues:
                validation_results['invalid_topics'].append({
                    'index': i,
                    'topic': topic,
                    'issues': issues
                })
                validation_results['issues'].extend(issues)
            else:
                validation_results['valid_topics'].append(topic)
        
        # 計算驗證分數
        total_topics = len(topics)
        valid_topics = len(validation_results['valid_topics'])
        validation_results['validation_score'] = valid_topics / total_topics if total_topics > 0 else 0
        
        return validation_results

class TopicEnhancer:
    """主題增強器 - 基於多輪推理改進主題"""
    
    def __init__(self, reasoning_agent: TopicReasoningAgent):
        self.agent = reasoning_agent
        self.validator = TopicValidator()
        
    def enhance_topics_iteratively(self, retrieval_result: RetrievalResult, 
                                 max_iterations: int = 3) -> TopicResult:
        """迭代增強主題"""
        
        current_result = self.agent.forward(retrieval_result)
        
        for iteration in range(max_iterations):
            # 驗證當前主題
            validation = self.validator.validate_topics(current_result.topics)
            
            if validation['validation_score'] >= 0.9:  # 質量足夠好
                logger.info(f"主題質量達標，停止迭代（第{iteration+1}輪）")
                break
            
            # 準備反饋
            feedback = self._generate_feedback(validation, iteration)
            
            # 優化主題
            current_result = self.agent.refine_topics(current_result, feedback)
            
            logger.info(f"完成第{iteration+1}輪主題優化")
        
        return current_result
    
    def _generate_feedback(self, validation: Dict, iteration: int) -> str:
        """生成優化反饋"""
        feedback_parts = [
            f"Iteration {iteration + 1} feedback:",
            f"Validation score: {validation['validation_score']:.2f}",
        ]
        
        if validation['issues']:
            feedback_parts.append("Issues to address:")
            feedback_parts.extend([f"- {issue}" for issue in validation['issues'][:5]])
        
        feedback_parts.append("Please improve the topics to address these issues.")
        
        return "\n".join(feedback_parts)