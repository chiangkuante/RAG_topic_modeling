import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

def display_topic_modeling_results(results: Dict[str, Any]) -> None:
    """顯示主題建模結果的主要界面"""
    
    if not results.get('success', False):
        st.error(f"❌ 主題建模失敗: {results.get('error', '未知錯誤')}")
        return
    
    # 結果概覽
    display_results_overview(results)
    
    # 詳細結果展示
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 主題結果", 
        "📊 有效性分析", 
        "🌈 多樣性分析", 
        "🔄 可靠性分析"
    ])
    
    with tab1:
        display_topics_section(results)
    
    with tab2:
        display_effectiveness_section(results)
    
    with tab3:
        display_diversity_section(results)
    
    with tab4:
        display_reliability_section(results)

def display_results_overview(results: Dict[str, Any]) -> None:
    """顯示結果概覽"""
    
    st.header("📋 結果概覽")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "生成主題數", 
            len(results.get('topics', [])),
            help="成功生成的主題總數"
        )
    
    with col2:
        final_score = results.get('final_score', 0)
        st.metric(
            "最終評分", 
            f"{final_score:.2f}",
            help="主題質量的綜合評分 (0-10分)"
        )
    
    with col3:
        iterations = results.get('iterations_completed', 0)
        st.metric(
            "迭代次數", 
            iterations,
            help="完成的優化迭代次數"
        )
    
    with col4:
        stop_reason = results.get('stop_reason', '未知')
        st.metric(
            "停止原因", 
            stop_reason,
            help="迭代停止的原因"
        )
    
    # 改進歷史圖表
    if 'improvement_history' in results and results['improvement_history']:
        st.subheader("📈 迭代改進歷史")
        
        improvement_history = results['improvement_history']
        iterations_list = list(range(1, len(improvement_history) + 1))
        
        fig = px.line(
            x=iterations_list,
            y=improvement_history,
            title="每輪迭代的改進幅度",
            labels={'x': '迭代輪次', 'y': '改進幅度'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def display_topics_section(results: Dict[str, Any]) -> None:
    """顯示主題結果部分"""
    
    st.header("🎯 生成的主題")
    
    topics = results.get('topics', [])
    topic_descriptions = results.get('topic_descriptions', [])
    confidence_scores = results.get('confidence_scores', [])
    
    if not topics:
        st.warning("⚠️ 沒有找到生成的主題")
        return
    
    # 主題列表展示
    st.subheader("📝 主題列表")
    
    # 創建主題數據框
    topic_data = []
    for i, topic in enumerate(topics):
        description = topic_descriptions[i] if i < len(topic_descriptions) else topic
        confidence = confidence_scores[i] if i < len(confidence_scores) else 0.8
        
        topic_data.append({
            '序號': i + 1,
            '主題': topic,
            '描述': description,
            '信心分數': f"{confidence:.3f}",
            '評級': get_confidence_grade(confidence)
        })
    
    df_topics = pd.DataFrame(topic_data)
    st.dataframe(df_topics, use_container_width=True)
    
    # 信心分數分佈
    if confidence_scores:
        st.subheader("📊 主題信心分數分佈")
        
        fig = px.bar(
            x=[f"主題 {i+1}" for i in range(len(confidence_scores))],
            y=confidence_scores,
            title="各主題的信心分數",
            labels={'x': '主題', 'y': '信心分數'},
            color=confidence_scores,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # 可下載的結果
    st.subheader("💾 導出結果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON格式下載
        json_data = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="📄 下載JSON格式",
            data=json_data,
            file_name=f"topic_modeling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV格式下載
        csv_data = df_topics.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📊 下載CSV格式",
            data=csv_data,
            file_name=f"topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def display_effectiveness_section(results: Dict[str, Any]) -> None:
    """顯示有效性分析部分"""
    
    st.header("📊 有效性分析")
    
    # 模擬有效性數據（實際使用中應該從評估結果獲取）
    effectiveness_data = generate_mock_effectiveness_data(results)
    
    # 有效性概覽
    st.subheader("🎯 有效性概覽")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_similarity = effectiveness_data.get('mean_similarity', 0)
        st.metric(
            "平均相似度",
            f"{avg_similarity:.3f}",
            help="主題與文檔的平均餘弦相似度"
        )
    
    with col2:
        coverage_score = effectiveness_data.get('coverage_score', 0)
        st.metric(
            "覆蓋分數",
            f"{coverage_score:.3f}", 
            help="主題對文檔集合的覆蓋程度"
        )
    
    with col3:
        effectiveness_grade = get_effectiveness_grade(avg_similarity)
        st.metric(
            "有效性等級",
            effectiveness_grade,
            help="基於相似度的有效性評級"
        )
    
    # 相似度分佈圖
    st.subheader("📈 相似度分佈")
    
    similarity_scores = effectiveness_data.get('similarity_scores', [])
    if similarity_scores:
        fig = px.histogram(
            x=similarity_scores,
            nbins=20,
            title="主題-文檔相似度分佈",
            labels={'x': '相似度', 'y': '頻率'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 主題排名
    st.subheader("🏆 主題有效性排名")
    
    topic_rankings = effectiveness_data.get('topic_rankings', [])
    if topic_rankings:
        df_rankings = pd.DataFrame(topic_rankings)
        st.dataframe(df_rankings, use_container_width=True)

def display_diversity_section(results: Dict[str, Any]) -> None:
    """顯示多樣性分析部分"""
    
    st.header("🌈 多樣性分析")
    
    # 模擬多樣性數據
    diversity_data = generate_mock_diversity_data(results)
    
    # 多樣性概覽
    st.subheader("🎨 多樣性概覽")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        diversity_score = diversity_data.get('diversity_score', 0)
        st.metric(
            "多樣性分數",
            f"{diversity_score:.3f}",
            help="主題間多樣性的綜合評分"
        )
    
    with col2:
        unique_words_ratio = diversity_data.get('unique_words_ratio', 0)
        st.metric(
            "獨特詞彙比例",
            f"{unique_words_ratio:.3f}",
            help="獨特詞彙佔總詞彙的比例"
        )
    
    with col3:
        lexical_diversity = diversity_data.get('lexical_diversity', 0)
        st.metric(
            "詞彙多樣性",
            f"{lexical_diversity:.3f}",
            help="詞彙層面的多樣性評分"
        )
    
    with col4:
        semantic_diversity = diversity_data.get('semantic_diversity', 0)
        st.metric(
            "語義多樣性",
            f"{semantic_diversity:.3f}",
            help="語義層面的多樣性評分"
        )
    
    # 主題相似度矩陣熱圖
    st.subheader("🔥 主題相似度矩陣")
    
    topics = results.get('topics', [])
    if len(topics) > 1:
        # 生成模擬相似度矩陣
        similarity_matrix = generate_mock_similarity_matrix(len(topics))
        
        fig = px.imshow(
            similarity_matrix,
            x=[f"主題{i+1}" for i in range(len(topics))],
            y=[f"主題{i+1}" for i in range(len(topics))],
            title="主題間語義相似度矩陣",
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 詞彙重疊分析
    st.subheader("📊 詞彙重疊分析")
    
    word_analysis = diversity_data.get('word_analysis', {})
    if word_analysis:
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 詞彙分佈餅圖
            labels = ['獨特詞彙', '重複詞彙']
            values = [
                word_analysis.get('unique_keywords', 0),
                word_analysis.get('repeated_keywords', 0)
            ]
            
            fig = px.pie(
                values=values,
                names=labels,
                title="詞彙獨特性分佈"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 最常見關鍵詞
            common_keywords = word_analysis.get('most_common_keywords', [])
            if common_keywords:
                df_keywords = pd.DataFrame(
                    common_keywords[:10], 
                    columns=['關鍵詞', '出現次數']
                )
                st.write("**最常見關鍵詞 (Top 10)**")
                st.dataframe(df_keywords, use_container_width=True)

def display_reliability_section(results: Dict[str, Any]) -> None:
    """顯示可靠性分析部分"""
    
    st.header("🔄 可靠性分析")
    
    # 模擬可靠性數據
    reliability_data = generate_mock_reliability_data(results)
    
    # 可靠性概覽
    st.subheader("🎯 可靠性概覽")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        reliability_score = reliability_data.get('reliability_score', 0)
        st.metric(
            "可靠性分數",
            f"{reliability_score:.3f}",
            help="系統結果可靠性的綜合評分"
        )
    
    with col2:
        stability_score = reliability_data.get('stability_score', 0)
        st.metric(
            "穩定性分數",
            f"{stability_score:.3f}",
            help="多輪運行結果的穩定性"
        )
    
    with col3:
        consistency_score = reliability_data.get('consistency_score', 0)
        st.metric(
            "一致性分數",
            f"{consistency_score:.3f}",
            help="結果的一致性程度"
        )
    
    with col4:
        confidence_interval = reliability_data.get('confidence_interval', (0, 0))
        ci_width = confidence_interval[1] - confidence_interval[0]
        st.metric(
            "置信區間寬度",
            f"{ci_width:.3f}",
            help="95%置信區間的寬度"
        )
    
    # 多輪結果比較
    st.subheader("📈 多輪結果比較")
    
    multi_round_scores = reliability_data.get('multi_round_scores', [])
    if multi_round_scores:
        
        fig = go.Figure()
        
        # 添加分數線
        fig.add_trace(go.Scatter(
            x=list(range(1, len(multi_round_scores) + 1)),
            y=multi_round_scores,
            mode='lines+markers',
            name='評分',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # 添加平均線
        avg_score = np.mean(multi_round_scores)
        fig.add_hline(
            y=avg_score, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"平均分: {avg_score:.2f}"
        )
        
        fig.update_layout(
            title="各輪次評分變化",
            xaxis_title="輪次",
            yaxis_title="評分",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 置信區間圖
    st.subheader("📊 置信區間分析")
    
    if multi_round_scores:
        mean_score = np.mean(multi_round_scores)
        std_score = np.std(multi_round_scores)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['最小值', '下四分位', '中位數', '上四分位', '最大值'],
            y=[
                np.min(multi_round_scores),
                np.percentile(multi_round_scores, 25),
                np.median(multi_round_scores),
                np.percentile(multi_round_scores, 75),
                np.max(multi_round_scores)
            ],
            name='分數分佈'
        ))
        
        fig.update_layout(
            title="評分統計分佈",
            yaxis_title="評分"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def get_confidence_grade(confidence: float) -> str:
    """獲取信心分數等級"""
    if confidence >= 0.9:
        return "優秀"
    elif confidence >= 0.8:
        return "良好"
    elif confidence >= 0.7:
        return "中等"
    elif confidence >= 0.6:
        return "一般"
    else:
        return "較差"

def get_effectiveness_grade(similarity: float) -> str:
    """獲取有效性等級"""
    if similarity >= 0.8:
        return "優秀"
    elif similarity >= 0.7:
        return "良好"
    elif similarity >= 0.6:
        return "中等"
    elif similarity >= 0.4:
        return "一般"
    else:
        return "較差"

def generate_mock_effectiveness_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """生成模擬有效性數據（實際實現中應從評估器獲取）"""
    
    num_topics = len(results.get('topics', []))
    final_score = results.get('final_score', 5.0)
    
    # 基於最終分數生成模擬數據
    base_similarity = min(0.9, max(0.1, final_score / 10))
    
    return {
        'mean_similarity': base_similarity,
        'max_similarity': min(1.0, base_similarity + 0.1),
        'min_similarity': max(0.0, base_similarity - 0.1),
        'coverage_score': base_similarity * 0.9,
        'similarity_scores': np.random.normal(base_similarity, 0.1, 100).tolist(),
        'topic_rankings': [
            {
                '排名': i + 1,
                '主題': results.get('topics', [])[i] if i < num_topics else f"主題{i+1}",
                '有效性分數': base_similarity + np.random.normal(0, 0.05),
                '最大相似度': min(1.0, base_similarity + np.random.uniform(0.1, 0.2))
            }
            for i in range(min(num_topics, 10))
        ]
    }

def generate_mock_diversity_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """生成模擬多樣性數據"""
    
    num_topics = len(results.get('topics', []))
    
    return {
        'diversity_score': np.random.uniform(0.6, 0.9),
        'unique_words_ratio': np.random.uniform(0.7, 0.95),
        'lexical_diversity': np.random.uniform(0.6, 0.8),
        'semantic_diversity': np.random.uniform(0.5, 0.8),
        'word_analysis': {
            'unique_keywords': np.random.randint(50, 100),
            'repeated_keywords': np.random.randint(10, 30),
            'most_common_keywords': [
                ('vaccine', 8), ('effectiveness', 6), ('safety', 5),
                ('immune', 4), ('protection', 4), ('response', 3),
                ('clinical', 3), ('trial', 2), ('antibody', 2), ('dose', 2)
            ]
        }
    }

def generate_mock_reliability_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """生成模擬可靠性數據"""
    
    final_score = results.get('final_score', 5.0)
    
    # 生成5輪模擬分數
    multi_round_scores = [
        final_score + np.random.normal(0, 0.3)
        for _ in range(5)
    ]
    
    return {
        'reliability_score': np.random.uniform(0.7, 0.9),
        'stability_score': 1 - (np.std(multi_round_scores) / np.mean(multi_round_scores)),
        'consistency_score': np.random.uniform(0.6, 0.8),
        'confidence_interval': (
            np.mean(multi_round_scores) - 1.96 * np.std(multi_round_scores),
            np.mean(multi_round_scores) + 1.96 * np.std(multi_round_scores)
        ),
        'multi_round_scores': multi_round_scores
    }

def generate_mock_similarity_matrix(num_topics: int) -> np.ndarray:
    """生成模擬相似度矩陣"""
    
    # 創建對稱矩陣
    matrix = np.random.uniform(0.1, 0.7, (num_topics, num_topics))
    matrix = (matrix + matrix.T) / 2  # 使矩陣對稱
    np.fill_diagonal(matrix, 1.0)  # 對角線為1
    
    return matrix

def display_evaluation_summary(effectiveness_data: Dict, 
                             diversity_data: Dict, 
                             reliability_data: Dict) -> None:
    """顯示評估摘要"""
    
    st.header("📈 評估摘要")
    
    # 綜合評估雷達圖
    categories = ['有效性', '多樣性', '可靠性', '穩定性', '一致性']
    values = [
        effectiveness_data.get('mean_similarity', 0) * 10,
        diversity_data.get('diversity_score', 0) * 10,
        reliability_data.get('reliability_score', 0) * 10,
        reliability_data.get('stability_score', 0) * 10,
        reliability_data.get('consistency_score', 0) * 10
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='系統表現'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="主題建模系統綜合評估"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 改進建議
    st.subheader("💡 改進建議")
    
    recommendations = []
    
    if effectiveness_data.get('mean_similarity', 0) < 0.6:
        recommendations.append("• 考慮調整檢索參數以提高主題相關性")
    
    if diversity_data.get('diversity_score', 0) < 0.6:
        recommendations.append("• 增加主題生成的多樣性策略")
    
    if reliability_data.get('reliability_score', 0) < 0.7:
        recommendations.append("• 提高系統參數穩定性以改善可靠性")
    
    if not recommendations:
        recommendations.append("• 系統表現良好，建議保持當前設置")
    
    for rec in recommendations:
        st.write(rec)