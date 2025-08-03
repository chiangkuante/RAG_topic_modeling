"""
LangChain-based Topic Reasoning Agent
使用LangChain构建的主题推理代理
"""

from typing import List, Dict, Tuple, Optional, Any
import json
import re
import logging
from dataclasses import dataclass
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableSequence
from langchain.chains import SequentialChain

from .retrieval_agent import RetrievalResult
from ..utils.langchain_llm_manager import LangChainLLMManager

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

class TopicOutputParser(BaseOutputParser):
    """主題輸出解析器"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """解析LLM輸出為結構化主題數據"""
        try:
            # 分割推理過程和主題列表
            parts = text.split("主題列表：")
            reasoning = ""
            topics_text = text
            
            if len(parts) >= 2:
                reasoning = parts[0].replace("推理過程：", "").strip()
                topics_text = parts[1].strip()
            
            # 解析主題
            topics = []
            descriptions = []
            confidence_scores = []
            
            for line in topics_text.split('\n'):
                line = line.strip()
                if not line or not any(line.startswith(str(i)) for i in range(1, 21)):
                    continue
                
                try:
                    # 移除序號
                    content = re.sub(r'^\d+\.\s*', '', line)
                    parts = content.split('|')
                    
                    if len(parts) >= 3:
                        topic = parts[0].strip()
                        description = parts[1].strip()
                        confidence = float(parts[2].strip())
                    elif len(parts) >= 2:
                        topic = parts[0].strip()
                        description = parts[1].strip()
                        confidence = 0.8
                    else:
                        topic = content.strip()
                        description = content.strip()
                        confidence = 0.7
                    
                    if topic:  # 確保主題不為空
                        topics.append(topic)
                        descriptions.append(description)
                        confidence_scores.append(min(1.0, max(0.0, confidence)))
                        
                except Exception as e:
                    logger.warning(f"解析主題行失敗: {line}, 錯誤: {e}")
                    continue
            
            return {
                'topics': topics,
                'descriptions': descriptions,  
                'confidence_scores': confidence_scores,
                'reasoning': reasoning or text[:500]
            }
            
        except Exception as e:
            logger.error(f"主題解析失敗: {e}")
            return {
                'topics': ['解析失敗'],
                'descriptions': ['請重試生成主題'],
                'confidence_scores': [0.3],
                'reasoning': f"解析錯誤: {str(e)}"
            }

class LangChainTopicReasoningAgent:
    """基於LangChain的主題推理代理"""
    
    def __init__(self, llm_manager: LangChainLLMManager, num_topics: int = 10):
        self.llm_manager = llm_manager
        self.num_topics = num_topics
        self.topic_parser = TopicOutputParser()
        
        # 預定義提示模板
        self._setup_prompts()
    
    def _setup_prompts(self):
        """設置提示模板"""
        self.generation_prompt = PromptTemplate(
            input_variables=["context", "num_topics", "domain"],
            template="""你是一個專業的主題建模專家。基於以下檢索到的文檔內容，請生成{num_topics}個相關主題。

領域：{domain}

檢索上下文：
{context}

請按照以下格式回答：

推理過程：
[請詳細說明你如何分析這些文檔並識別主要主題的過程]

主題列表：
1. [主題名稱] | [詳細描述] | [信心分數0.0-1.0]
2. [主題名稱] | [詳細描述] | [信心分數0.0-1.0]
3. [主題名稱] | [詳細描述] | [信心分數0.0-1.0]

要求：
1. 主題應該具體且有意義
2. 每個主題都要有清晰的描述
3. 信心分數反映主題在文檔中的支持程度
4. 主題之間應該有所區別，避免重複
5. 重點關注{domain}領域的核心概念
"""
        )
        
        self.refinement_prompt = PromptTemplate(
            input_variables=["original_topics", "reasoning", "feedback"],
            template="""你是一個專業的主題建模專家。請根據以下反饋優化現有主題。

原始主題：
{original_topics}

原始推理過程：
{reasoning}

優化反饋：
{feedback}

請按照以下格式提供優化後的主題：

推理過程：
[請說明你如何根據反饋改進這些主題]

主題列表：
1. [優化後主題名稱] | [詳細描述] | [信心分數0.0-1.0]
2. [優化後主題名稱] | [詳細描述] | [信心分數0.0-1.0]
3. [優化後主題名稱] | [詳細描述] | [信心分數0.0-1.0]

要求：
1. 考慮反饋中的建議
2. 保持主題的相關性和具體性
3. 改進主題的清晰度和區分度
4. 調整信心分數以反映改進程度
"""
        )
    
    def forward(self, retrieval_result: RetrievalResult, domain: str = "vaccine research") -> TopicResult:
        """主要推理流程 - 生成初始主題"""
        
        try:
            # 準備上下文
            context = self._prepare_context(retrieval_result)
            
            # 創建生成鏈
            generation_chain = self.llm_manager.create_chain(
                self.generation_prompt, 
                self.topic_parser
            )
            
            # 執行生成
            logger.info("開始生成主題...")
            result = self.llm_manager.invoke_chain(generation_chain, {
                'context': context,
                'num_topics': self.num_topics,
                'domain': domain
            })
            
            # 創建結果
            topic_result = TopicResult(
                topics=result['topics'][:self.num_topics],
                topic_descriptions=result['descriptions'][:self.num_topics],
                confidence_scores=result['confidence_scores'][:self.num_topics],
                reasoning=result['reasoning'],
                metadata={
                    'generation_method': 'langchain_chain',
                    'retrieval_quality': retrieval_result.metadata.get('avg_score', 0),
                    'num_source_docs': len(retrieval_result.documents),
                    'domain': domain
                },
                timestamp=datetime.now()
            )
            
            logger.info(f"成功生成{len(topic_result.topics)}個主題")
            return topic_result
            
        except Exception as e:
            logger.error(f"主題生成失敗: {e}")
            return self._create_fallback_result(domain)
    
    def refine_topics(self, topic_result: TopicResult, feedback: str) -> TopicResult:
        """優化主題"""
        
        try:
            # 準備原始主題字符串
            original_topics_str = "\n".join([
                f"{i+1}. {topic} | {desc} | {score:.2f}" 
                for i, (topic, desc, score) in enumerate(zip(
                    topic_result.topics, 
                    topic_result.topic_descriptions,
                    topic_result.confidence_scores
                ))
            ])
            
            # 創建優化鏈
            refinement_chain = self.llm_manager.create_chain(
                self.refinement_prompt,
                self.topic_parser
            )
            
            # 執行優化
            logger.info("開始優化主題...")
            result = self.llm_manager.invoke_chain(refinement_chain, {
                'original_topics': original_topics_str,
                'reasoning': topic_result.reasoning[:500],
                'feedback': feedback
            })
            
            # 創建優化結果
            refined_result = TopicResult(
                topics=result['topics'][:self.num_topics],
                topic_descriptions=result['descriptions'][:self.num_topics],
                confidence_scores=result['confidence_scores'][:self.num_topics],
                reasoning=f"原始推理: {topic_result.reasoning[:200]}...\n\n優化推理: {result['reasoning']}",
                metadata={
                    **topic_result.metadata,
                    'refinement_applied': True,
                    'feedback': feedback,
                    'original_topics_count': len(topic_result.topics)
                },
                timestamp=datetime.now()
            )
            
            logger.info(f"成功優化{len(refined_result.topics)}個主題")
            return refined_result
            
        except Exception as e:
            logger.error(f"主題優化失敗: {e}")
            # 返回原始結果作為備選
            return topic_result
    
    def _prepare_context(self, retrieval_result: RetrievalResult) -> str:
        """準備上下文信息"""
        if not retrieval_result.documents:
            return "No relevant documents found."
        
        context_parts = [
            f"查詢: {retrieval_result.query}",
            f"找到 {len(retrieval_result.documents)} 個相關文檔",
            f"平均相似度: {retrieval_result.metadata.get('avg_score', 0):.3f}",
            "\n相關文檔片段:"
        ]
        
        # 添加文檔內容（限制長度）
        for i, doc in enumerate(retrieval_result.documents[:5]):  # 只取前5個
            score = retrieval_result.metadata.get('scores', [0.5] * len(retrieval_result.documents))[i]
            context_parts.append(f"文檔 {i+1} (相似度: {score:.3f}): {doc[:300]}...")
        
        return "\n".join(context_parts)
    
    def _create_fallback_result(self, domain: str) -> TopicResult:
        """創建備選結果"""
        fallback_topics = [
            f"{domain} - 主要概念 {i+1}" for i in range(self.num_topics)
        ]
        
        return TopicResult(
            topics=fallback_topics,
            topic_descriptions=[f"關於{domain}的主題 {i+1}" for i in range(self.num_topics)],
            confidence_scores=[0.5] * self.num_topics,
            reasoning="生成失敗，使用備選主題",
            metadata={
                'generation_method': 'fallback',
                'domain': domain
            },
            timestamp=datetime.now()
        )