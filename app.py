import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

# 將src目錄添加到路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.utils.data_processor import DataProcessor
from src.utils.vectorizer import VectorDatabase, TopicVectorizer
from src.utils.langchain_llm_manager import LangChainLLMManager, LLMProvider, LLMConfig
from src.agents.langchain_iteration_controller import LangChainTopicModelingController, LangChainBatchTopicModeling
from src.evaluation.effectiveness_evaluator import EffectivenessEvaluator
from src.evaluation.diversity_evaluator import DiversityEvaluator
from src.evaluation.reliability_evaluator import ReliabilityEvaluator
from src.ui.results_display import display_topic_modeling_results

# 頁面配置
st.set_page_config(
    page_title="疫苗主題建模RAG系統",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = VectorDatabase()
if 'llm_manager' not in st.session_state:
    st.session_state.llm_manager = LangChainLLMManager()
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'vector_index_built' not in st.session_state:
    st.session_state.vector_index_built = False
if 'modeling_results' not in st.session_state:
    st.session_state.modeling_results = None

def main():
    st.title("🧬 疫苗主題建模RAG系統")
    st.markdown("---")
    
    # 側邊欄配置
    with st.sidebar:
        st.header("⚙️ 系統配置")
        
        # LLM配置
        st.subheader("🤖 LLM配置")
        
        # 獲取支持的提供者
        supported_providers = st.session_state.llm_manager.get_supported_providers()
        
        provider = st.selectbox(
            "選擇LLM提供者",
            options=list(supported_providers.keys()),
            format_func=lambda x: x.upper()
        )
        
        model = st.selectbox(
            "選擇模型",
            options=supported_providers[provider]
        )
        
        api_key = st.text_input(
            f"輸入{provider.upper()} API Key",
            type="password",
            help=f"請輸入有效的{provider.upper()} API密鑰"
        )
        
        # LLM參數配置
        st.subheader("🎛️ 模型參數")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 100)
        
        # 測試連接按鈕
        if st.button("🔗 測試LLM連接"):
            if api_key:
                with st.spinner("測試連接中..."):
                    success = st.session_state.llm_manager.test_connection(provider, model, api_key)
                    if success:
                        st.success("✅ 連接成功！")
                        # 創建並設置LLM
                        llm = st.session_state.llm_manager.create_llm(
                            provider, model, api_key,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        # 創建配置對象
                        config = LLMConfig(
                            provider=LLMProvider(provider),
                            model=model,
                            api_key=api_key,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        st.session_state.llm_manager.set_active_llm(llm, config)
                        st.session_state.llm_configured = True
                    else:
                        st.error("❌ 連接失敗，請檢查API密鑰")
            else:
                st.warning("⚠️ 請先輸入API密鑰")
        
        st.markdown("---")
        
        # 數據配置
        st.subheader("📊 數據配置")
        
        # 數據上傳
        uploaded_file = st.file_uploader(
            "上傳CSV數據文件",
            type=['csv'],
            help="請上傳包含文本數據的CSV文件"
        )
        
        if uploaded_file is not None:
            # 保存上傳的文件
            data_path = "data/uploaded_data.csv"
            os.makedirs("data", exist_ok=True)
            with open(data_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ 文件上傳成功")
            
            # 預處理數據
            if st.button("🔄 預處理數據"):
                with st.spinner("預處理數據中..."):
                    try:
                        data = st.session_state.data_processor.load_csv(data_path)
                        processed_data = st.session_state.data_processor.preprocess_data()
                        st.session_state.processed_data = processed_data
                        
                        # 顯示統計信息
                        stats = st.session_state.data_processor.get_text_stats(processed_data)
                        st.success("✅ 數據預處理完成")
                        st.json(stats)
                        
                    except Exception as e:
                        st.error(f"❌ 預處理失敗：{e}")
        
        # 建立向量索引
        if st.session_state.processed_data is not None and not st.session_state.vector_index_built:
            if st.button("🔍 建立向量索引"):
                with st.spinner("建立向量索引中..."):
                    try:
                        texts = st.session_state.processed_data['processed_text'].tolist()
                        st.session_state.vector_db.build_index(texts, save_path="data/vector_index")
                        st.session_state.vector_index_built = True
                        st.success("✅ 向量索引建立成功")
                    except Exception as e:
                        st.error(f"❌ 索引建立失敗：{e}")
    
    # 主內容區域
    if st.session_state.processed_data is not None:
        # 顯示數據概覽
        st.header("📊 數據概覽")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("文檔總數", len(st.session_state.processed_data))
        
        with col2:
            avg_words = st.session_state.processed_data['processed_text'].str.split().str.len().mean()
            st.metric("平均詞數", f"{avg_words:.1f}")
        
        with col3:
            total_words = st.session_state.processed_data['processed_text'].str.split().str.len().sum()
            st.metric("總詞數", total_words)
        
        with col4:
            unique_words = len(set(' '.join(st.session_state.processed_data['processed_text']).split()))
            st.metric("獨特詞彙", unique_words)
        
        # 顯示數據樣本
        st.subheader("📝 數據樣本")
        sample_data = st.session_state.processed_data.head(10)
        st.dataframe(sample_data[['processed_text']], use_container_width=True)
        
        # 詞數分佈圖
        st.subheader("📈 詞數分佈")
        word_counts = st.session_state.processed_data['processed_text'].str.split().str.len()
        
        fig = px.histogram(
            x=word_counts,
            nbins=30,
            title="文檔詞數分佈",
            labels={'x': '詞數', 'y': '文檔數量'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # 主題建模執行區域
        if (st.session_state.vector_index_built and 
            hasattr(st.session_state, 'llm_configured') and 
            st.session_state.llm_configured):
            
            st.header("🎯 主題建模執行")
            
            # 參數配置
            with st.expander("⚙️ 建模參數設置", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    num_topics = st.slider("主題數量", 5, 20, 10)
                    max_iterations = st.slider("最大迭代次數", 1, 10, 5)
                
                with col2:
                    quality_threshold = st.slider("質量閾值", 5.0, 10.0, 7.0, 0.1)
                    num_rounds = st.slider("評估輪次", 1, 10, 5)
                
                with col3:
                    query = st.text_input("檢索查詢", "vaccine effectiveness and safety")
                    domain = st.text_input("領域描述", "vaccine research")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("✅ 系統已準備就緒，可以開始主題建模")
            
            with col2:
                start_modeling = st.button("▶️ 開始建模", type="primary")
            
            if start_modeling:
                run_topic_modeling(
                    num_topics=num_topics,
                    max_iterations=max_iterations,
                    quality_threshold=quality_threshold,
                    num_rounds=num_rounds,
                    query=query,
                    domain=domain
                )
        
        else:
            st.header("⚠️ 系統狀態")
            
            status_items = [
                ("數據預處理", st.session_state.processed_data is not None),
                ("向量索引", st.session_state.vector_index_built),
                ("LLM配置", hasattr(st.session_state, 'llm_configured') and st.session_state.llm_configured)
            ]
            
            for item, status in status_items:
                icon = "✅" if status else "❌"
                st.write(f"{icon} {item}")
    
    else:
        # 歡迎頁面
        st.header("👋 歡迎使用疫苗主題建模RAG系統")
        
        st.markdown("""
        ### 🚀 開始使用
        
        1. **配置LLM**: 在左側邊欄選擇LLM提供者並輸入API密鑰
        2. **上傳數據**: 上傳包含疫苗相關文本的CSV文件
        3. **預處理數據**: 清理和標準化文本數據
        4. **建立索引**: 創建向量數據庫索引
        5. **開始建模**: 執行主題建模和評估
        
        ### 📋 系統功能
        
        - **多LLM支持**: OpenAI、Gemini、Claude
        - **智能檢索**: 基於向量相似度的RAG檢索
        - **DSPy Agents**: 檢索、思考、評估三重代理
        - **多維評估**: 有效性、多樣性、可靠性評估
        - **視覺化展示**: 互動式結果展示和分析
        
        ### 🔧 技術特點
        
        - **向量化**: Sentence-BERT嵌入
        - **向量數據庫**: FAISS高效搜索
        - **迭代優化**: 自動質量評估和改進
        - **批量處理**: 支持大規模數據集
        """)
    
    # 顯示建模結果
    if st.session_state.modeling_results:
        st.markdown("---")
        display_topic_modeling_results(st.session_state.modeling_results)

def run_topic_modeling(num_topics: int = 10,
                      max_iterations: int = 5,
                      quality_threshold: float = 7.0,
                      num_rounds: int = 5,
                      query: str = "vaccine effectiveness and safety",
                      domain: str = "vaccine research"):
    """執行主題建模"""
    
    try:
        with st.spinner("🔄 正在執行主題建模..."):
            
            # 創建進度條
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 初始化控制器
            status_text.text("正在初始化控制器...")
            progress_bar.progress(10)
            
            controller = LangChainTopicModelingController(
                vector_database=st.session_state.vector_db,
                llm_manager=st.session_state.llm_manager,
                max_iterations=max_iterations,
                quality_threshold=quality_threshold
            )
            
            # 創建批量建模器
            status_text.text("正在準備批量建模...")
            progress_bar.progress(20)
            
            batch_modeling = LangChainBatchTopicModeling(controller)
            
            # 執行多輪建模
            status_text.text(f"正在執行 {num_rounds} 輪主題建模...")
            progress_bar.progress(30)
            
            batch_results = batch_modeling.run_multiple_rounds(
                num_rounds=num_rounds,
                query=query,
                domain=domain
            )
            
            progress_bar.progress(70)
            
            if batch_results.get('success', False):
                # 執行詳細評估
                status_text.text("正在執行詳細評估...")
                
                # 創建評估器
                effectiveness_evaluator = EffectivenessEvaluator(st.session_state.vector_db)
                diversity_evaluator = DiversityEvaluator()
                reliability_evaluator = ReliabilityEvaluator(
                    effectiveness_evaluator, diversity_evaluator, num_rounds
                )
                
                # 評估最佳結果
                best_topics = batch_results.get('best_topics', [])
                
                if best_topics:
                    # 有效性評估
                    effectiveness_metrics = effectiveness_evaluator.evaluate_topic_effectiveness(best_topics)
                    effectiveness_report = effectiveness_evaluator.generate_effectiveness_report(effectiveness_metrics, best_topics)
                    
                    # 多樣性評估
                    diversity_metrics = diversity_evaluator.evaluate_topic_diversity(best_topics)
                    diversity_report = diversity_evaluator.generate_diversity_report(diversity_metrics, best_topics)
                    
                    # 可靠性評估
                    reliability_metrics = reliability_evaluator.evaluate_reliability(batch_results)
                    reliability_report = reliability_evaluator.generate_reliability_report(reliability_metrics)
                    
                    # 整合結果
                    final_results = {
                        'success': True,
                        'topics': best_topics,
                        'topic_descriptions': batch_results.get('best_topic_descriptions', best_topics),
                        'confidence_scores': [0.8] * len(best_topics),  # 模擬信心分數
                        'final_score': batch_results.get('best_score', 0),
                        'iterations_completed': batch_results.get('num_successful_rounds', 0),
                        'stop_reason': '多輪評估完成',
                        'improvement_history': [],
                        'effectiveness_report': effectiveness_report,
                        'diversity_report': diversity_report,
                        'reliability_report': reliability_report,
                        'batch_results': batch_results
                    }
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 主題建模完成！")
                    
                    # 保存結果
                    st.session_state.modeling_results = final_results
                    
                    # 顯示成功消息
                    st.success(f"🎉 成功生成 {len(best_topics)} 個主題！")
                    
                    # 自動刷新頁面以顯示結果
                    st.rerun()
                
                else:
                    st.error("❌ 未能生成有效主題")
            
            else:
                error_msg = batch_results.get('error', '未知錯誤')
                st.error(f"❌ 主題建模失敗: {error_msg}")
    
    except Exception as e:
        st.error(f"❌ 執行過程中發生錯誤: {str(e)}")
        logger.error(f"主題建模執行錯誤: {e}")

if __name__ == "__main__":
    main()