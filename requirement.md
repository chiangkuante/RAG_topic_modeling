# 我想要建立一個主題建模系統 
## 使用的資料集是vaxx.csv 
## 使用代理架構
## 目標是能夠檢索資料集內容，並且用prompt詢問LLM，使用LLM主題建模出10個主題
## 我希望有三個agent，分別是檢索、思考、評估，如果評估的agent未達門檻，就再次迭代重新執行
## 執行完主題建模之後，再將10個主題與整個資料集的向量做餘弦相似度，計算出系統的有效性
## 我還要使用top-k來計算獨特詞彙比例，計算出系統的多樣性
## 最後系統要運算五輪，每次都要計算餘弦相似度，以便計算出可靠性
## LLM是使用api，讓使用者先填入api key，我要包括openai、gemini、claude
簡單的流程: 資料集->轉向量->檢索->思考->評估->給出10個主題->評估有效性、多樣性
                           ↑           ↓                        ↓
                           -------------                        |
                           ↑----------------------------------------------->運算五輪後給出可靠性

## 最終評估結果要使用markdown，內容包括:主題建模的五輪的10個主題、每輪的有效性、多樣性、最後的可靠性。

# 使用gh cli上傳到git@github.com:chiangkuante/vaxx_RAG_topic_modeling.git

## 套件管理更新 - Package Management Update

項目已從 pip 遷移到 UV 進行更好的依賴管理：
- 所有依賴現在通過 `pyproject.toml` 管理
- 使用 `uv sync` 安裝依賴
- 使用 `uv run <command>` 運行命令
- 虛擬環境自動在 `.venv/` 目錄創建

The project has been migrated from pip to UV for better dependency management:
- All dependencies are now managed through `pyproject.toml`
- Use `uv sync` to install dependencies  
- Use `uv run <command>` to run commands
- Virtual environment automatically created in `.venv/` directory