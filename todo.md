# 主題建模系統開發任務清單

## 系統架構設計

### 整體架構
```
數據層 (Data Layer)
├── vaxx.csv 數據集
├── 數據預處理模塊
└── 向量數據庫

核心引擎層 (Core Engine Layer)
├── LLM API 集成模塊 (OpenAI/Gemini/Claude)
├── DSPy 框架集成
└── 向量化引擎

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
│   ├── API Key 輸入與配置頁面
│   ├── LLM 提供者選擇界面
│   ├── 主題建模執行控制面板
│   └── 實時進度監控
├── 結果展示模塊
│   ├── 互動式主題展示
│   ├── 評估指標視覺化 (圖表)
│   ├── Markdown 報告下載
│   └── 歷史結果比較
└── 系統狀態監控
    ├── 執行日誌顯示
    └── 系統性能監控
```

### 資料流程圖
```
1. Streamlit 界面啟動 → 2. 用戶配置 (API Key/LLM選擇)
           ↓                      ↓
3. 數據載入 → 4. 向量化 → 5. 建立向量數據庫
           ↓                      ↓
6. DSPy 檢索 Agent ← 7. 用戶觸發執行 (Streamlit)
      ↓
8. DSPy 思考 Agent → 9. 生成候選主題
      ↓                  ↓ (即時顯示進度)
10. DSPy 評估 Agent → 11. 質量評估
      ↓ (未達標)        ↓ (達標)
   回到步驟6        12. 輸出10個主題
   (更新進度條)         ↓ (Streamlit 展示)
                   13. 有效性評估
                        ↓ (圖表顯示)
                   14. 多樣性評估
                        ↓ (圖表顯示)
                   15. 重複五輪 (進度追蹤)
                        ↓
                   16. 可靠性評估
                        ↓
                   17. 生成報告 (下載/顯示)
```

## 開發任務清單

### 高優先級任務 (High Priority)

#### ✅ 已完成
- [x] 設計系統架構：數據預處理、向量化、RAG檢索、DSPy agent架構

#### 🔄 進行中
- [ ] **當前任務：系統架構設計已完成，更新為Streamlit前端架構**

#### ⏳ 待辦事項
- [ ] **建立數據預處理模塊**：讀取vaxx.csv並清理數據
  - 讀取CSV文件
  - 數據清理和標準化
  - 文本預處理（分詞、去停用詞等）

- [ ] **實作向量化模塊**：將數據集轉換為向量表示
  - 選擇嵌入模型（如 Sentence-BERT）
  - 批量向量化處理
  - 向量存儲和索引

- [ ] **建立LLM API集成模塊**：支持OpenAI、Gemini、Claude
  - API Key 管理
  - 統一的API接口抽象
  - 錯誤處理和重試機制

- [ ] **建立Streamlit Web界面**：用戶配置、API key輸入、LLM選擇
  - Streamlit 應用主頁面設計
  - 側邊欄配置面板 (API Key輸入)
  - LLM 提供者選擇下拉選單
  - 參數調整界面 (溫度、最大tokens等)
  - 配置驗證和錯誤提示
  - Session State 管理

### 中優先級任務 (Medium Priority)

- [ ] **實作DSPy檢索Agent**：負責從向量數據庫檢索相關內容
  - 相似度搜索實現
  - Top-K 檢索策略
  - 結果排序和過濾

- [ ] **實作DSPy思考Agent**：使用LLM進行主題建模推理
  - Prompt 工程設計
  - 主題生成邏輯
  - 結果格式化

- [ ] **實作DSPy評估Agent**：評估主題質量並決定是否重新迭代
  - 主題質量評分標準
  - 閾值設定機制
  - 評估報告生成

- [ ] **建立迭代控制機制**：當評估未達門檻時重新執行流程
  - 最大迭代次數限制
  - 迭代記錄和日誌
  - 停止條件判斷

- [ ] **實作有效性評估**：計算10個主題與數據集向量的餘弦相似度
  - 餘弦相似度計算
  - 批量相似度處理
  - 統計分析

- [ ] **實作多樣性評估**：使用top-k計算獨特詞彙比例
  - Top-K 詞彙提取
  - 重複率計算
  - 多樣性指標

- [ ] **實作可靠性評估**：運行五輪並計算結果穩定性
  - 多輪執行控制
  - 結果一致性分析
  - 穩定性指標計算

- [ ] **建立Streamlit結果展示模塊**：在Streamlit中展示五輪主題、有效性、多樣性、可靠性
  - 主題結果表格展示
  - 互動式圖表 (Plotly/Altair)
  - 評估指標儀表板
  - 進度條和實時狀態更新
  - Markdown 報告生成與下載
  - 結果比較功能 (多次執行對比)

### 低優先級任務 (Low Priority)

- [ ] **編寫配置文件和使用文檔**
  - README.md
  - 安裝指南
  - 使用範例

- [ ] **測試整個系統並優化性能**
  - 單元測試
  - 集成測試
  - 性能優化

## 技術棧選擇

### 核心技術
- **AI框架**: DSPy (用於構建AI系統)
- **前端框架**: Streamlit (Web界面)
- **語言**: Python 3.8+
- **向量數據庫**: FAISS 或 ChromaDB
- **嵌入模型**: Sentence-BERT 或 OpenAI Embeddings
- **圖表庫**: Plotly 或 Altair (互動式圖表)

### LLM API支持
- OpenAI GPT (gpt-3.5-turbo, gpt-4)
- Google Gemini (gemini-pro)
- Anthropic Claude (claude-3-sonnet, claude-3-haiku)

### 依賴套件
```python
# AI/ML 核心套件
dspy-ai>=2.4.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0  # 或 chromadb

# LLM API 套件
openai>=1.0.0
google-generativeai>=0.3.0
anthropic>=0.7.0

# Streamlit 前端套件
streamlit>=1.28.0
plotly>=5.15.0  # 或 altair>=5.0.0
streamlit-option-menu>=0.3.6
streamlit-aggrid>=0.3.4  # 表格展示
streamlit-echarts>=0.4.0  # 圖表展示
```

## 預期成果

1. **10個主題**: 每輪生成的主題列表
2. **有效性分數**: 基於餘弦相似度的主題相關性評估
3. **多樣性分數**: 基於Top-K獨特詞彙的主題多樣性評估
4. **可靠性分數**: 基於五輪結果一致性的系統穩定性評估
5. **完整報告**: Markdown格式的詳細分析報告

## 開發時程預估

- **第1週**: 完成高優先級任務（數據處理、向量化、API集成、Streamlit基礎界面）
- **第2週**: 完成中優先級任務（DSPy Agents、評估模塊）
- **第3週**: 完成Streamlit結果展示模塊和低優先級任務（文檔、測試）
- **第4週**: 系統整合測試、UI/UX優化和性能調優

## 注意事項

1. 確保API Key的安全存儲
2. 實現適當的錯誤處理和日誌記錄
3. 優化向量計算性能
4. 設計可擴展的評估指標體系
5. 保持代碼的模塊化和可維護性