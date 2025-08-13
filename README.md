# 🧬 疫苗主題建模RAG系統

基於RAG和多LLM的智能主題建模系統，專門用於分析疫苗相關文本數據。

## ✨ 系統特點

- **多LLM支持**: 支持OpenAI、Google Gemini、Anthropic Claude
- **智能代理架構**: 檢索、推理、評估三重代理系統
- **向量化檢索**: 使用Sentence-BERT和FAISS的高效RAG系統
- **多維評估**: 有效性、多樣性、可靠性三重評估體系
- **互動式界面**: Streamlit Web應用，支持實時進度監控
- **自動迭代**: 智能質量評估和自動改進機制

## 🏗️ 系統架構

```
數據層 (Data Layer)
├── vaxx.csv 數據集
├── 數據預處理模塊
└── 向量數據庫 (FAISS)

核心引擎層 (Core Engine Layer)
├── LLM API 集成模塊 (OpenAI/Gemini/Claude)
├── 代理框架集成
└── 向量化引擎 (Sentence-BERT)

業務邏輯層 (Business Logic Layer)
├── 檢索 Agent (Retrieval Agent)
├── 思考 Agent (Reasoning Agent)
├── 評估 Agent (Evaluation Agent)
└── 迭代控制器 (Iteration Controller)

評估層 (Evaluation Layer)
├── 有效性評估 (餘弦相似度)
├── 多樣性評估 (Top-K 獨特詞彙)
└── 可靠性評估 (五輪運算)

展示層 (Presentation Layer)
├── Streamlit Web 應用界面
├── 結果展示模塊
└── 系統狀態監控
```

## 🚀 快速開始

### 1. 環境準備

#### 方式一：使用UV（推薦）

```bash
# 克隆項目
git clone <repository-url>
cd RAG_topic_modeling

# 安裝UV（如果尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 自動安裝依賴
./dev-setup.sh
# 或手動執行
uv sync
```

#### 方式二：使用pip（傳統方式）

```bash
# 安裝依賴（備選方案）
pip install -r requirements.txt
```

### 2. 配置API密鑰

在系統界面中配置以下任一LLM提供者的API密鑰：

- **OpenAI**: 獲取API密鑰從 [OpenAI Platform](https://platform.openai.com/)
- **Google Gemini**: 獲取API密鑰從 [Google AI Studio](https://makersuite.google.com/)
- **Anthropic Claude**: 獲取API密鑰從 [Anthropic Console](https://console.anthropic.com/)

### 3. 啟動系統

#### 使用UV（推薦）

```bash
# 使用便捷腳本
./scripts/run-streamlit.sh

# 或使用UV運行
uv run streamlit run app.py
uv run python run.py
```

#### 使用傳統方式

```bash
# 激活虛擬環境後運行
python run.py
streamlit run app.py
```

### 4. 使用流程

1. **配置LLM**: 在左側邊欄選擇LLM提供者並輸入API密鑰
2. **上傳數據**: 上傳包含疫苗相關文本的CSV文件
3. **預處理數據**: 系統自動清理和標準化文本數據
4. **建立索引**: 創建向量數據庫索引以支持快速檢索
5. **開始建模**: 執行主題建模並實時查看結果

## 📊 功能模塊

### 數據處理
- 自動CSV文件讀取和解析
- 文本清理和標準化
- 停用詞移除和詞彙過濾
- 數據統計和可視化

### 向量化
- Sentence-BERT文本嵌入
- FAISS向量數據庫索引
- 餘弦相似度檢索
- 批量向量化處理

### 代理系統
- **檢索代理**: 智能文檔檢索和上下文構建
- **推理代理**: 基於LLM的主題生成和優化
- **評估代理**: 多維度質量評估和反饋

### 評估體系
- **有效性評估**: 主題與文檔的餘弦相似度分析
- **多樣性評估**: Top-K詞彙獨特性計算
- **可靠性評估**: 多輪結果一致性分析

## 🔧 配置選項

系統支持通過 `config.json` 文件進行詳細配置：

```json
{
  "data": {
    "sample_size": null,
    "min_word_length": 2,
    "max_documents": 10000
  },
  "vectorizer": {
    "model_name": "all-MiniLM-L6-v2",
    "dimension": 384,
    "batch_size": 32
  },
  "topic_modeling": {
    "num_topics": 10,
    "max_iterations": 5,
    "quality_threshold": 7.0,
    "num_rounds": 5
  }
}
```

## 📋 系統要求

### 硬件要求
- 內存: 至少 4GB RAM（推薦 8GB+）
- 存儲: 至少 2GB 可用空間
- 網絡: 穩定的互聯網連接（用於LLM API調用）

### 軟件要求
- Python 3.9+ （由於google-generativeai依賴要求）
- 支持的操作系統: Windows, macOS, Linux
- UV（推薦的包管理器）或 pip

### 主要依賴
- `langchain>=0.1.0` - LLM框架整合
- `streamlit>=1.28.0` - Web界面
- `sentence-transformers>=2.2.0` - 文本嵌入
- `faiss-cpu>=1.7.0` - 向量數據庫
- `pandas>=1.5.0` - 數據處理
- `plotly>=5.15.0` - 數據可視化

## 📝 使用示例

### API使用示例

```python
from src.utils.data_processor import DataProcessor
from src.utils.vectorizer import VectorDatabase
from src.utils.llm_api import LLMManager
from src.agents.retrieval_agent import TopicRetrievalAgent
from src.agents.reasoning_agent import TopicReasoningAgent

# 初始化組件
data_processor = DataProcessor()
vector_db = VectorDatabase()
llm_manager = LLMManager()

# 配置LLM
llm = llm_manager.create_llm("openai", "gpt-3.5-turbo", "your-api-key")
llm_manager.set_active_llm(llm)

# 處理數據
data = data_processor.load_csv("data/vaxx.csv")
processed_data = data_processor.preprocess_data()

# 建立向量索引
texts = processed_data['processed_text'].tolist()
vector_db.build_index(texts)

# 創建代理
retrieval_agent = TopicRetrievalAgent(vector_db)
reasoning_agent = TopicReasoningAgent(llm_manager)

# 執行主題建模
query = "vaccine effectiveness and safety"
retrieval_result = retrieval_agent.forward(query)
topic_result = reasoning_agent.forward(retrieval_result)

print("生成的主題:")
for i, topic in enumerate(topic_result.topics, 1):
    print(f"{i}. {topic}")
```

## 🤝 貢獻指南

歡迎提交Issue和Pull Request來改進系統！

### 開發設置

#### 使用UV（推薦）

```bash
# 安裝依賴
uv sync

# 運行測試
uv run python -m pytest tests/

# 代碼格式檢查
uv run black src/
uv run flake8 src/

# 添加開發依賴
uv add --dev pytest black flake8
```

#### 使用pip（傳統方式）

```bash
# 安裝開發依賴
pip install -r requirements.txt

# 運行測試
python -m pytest tests/

# 代碼格式檢查
black src/
flake8 src/
```

## 📄 許可證

本項目採用 MIT 許可證 - 詳見 [LICENSE](LICENSE) 文件。

## 🆘 故障排除

### 常見問題

1. **API密鑰錯誤**
   - 確保API密鑰正確且有效
   - 檢查API配額和限制

2. **內存不足**
   - 減小數據集大小或使用採樣
   - 調整batch_size參數

3. **依賴安裝失敗**
   - 使用UV: `uv sync`（自動管理虛擬環境）
   - 確保Python 3.9+版本
   - 傳統方式：使用虛擬環境和pip

4. **向量化慢**
   - 使用GPU版本的句嵌入模型
   - 調整batch_size參數

### 聯繫支持

如遇到問題，請：
1. 查看文檔和FAQ
2. 在GitHub上提交Issue
3. 提供詳細的錯誤信息和系統環境

---

## 📦 包管理

本項目已遷移至UV包管理器，提供：
- ⚡ 更快的依賴解析和安裝
- 🔒 可靠的lockfile支持（uv.lock）
- 🐍 自動虛擬環境管理
- 📝 現代化的pyproject.toml配置

### UV常用命令

```bash
uv sync          # 同步依賴
uv add <包名>     # 添加依賴
uv remove <包名>  # 移除依賴
uv run <命令>     # 運行命令
uv tree          # 查看依賴樹
```

詳細命令請參考：[UV_COMMANDS.md](UV_COMMANDS.md)

---

**開發團隊**: AI研究團隊  
**版本**: 1.0.0  
**最後更新**: 2025年  
**包管理**: UV (推薦) | pip (兼容)