#!/usr/bin/env python3
"""
疫苗主題建模RAG系統啟動腳本
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """檢查依賴包是否安裝"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt 文件不存在")
        return False
    
    try:
        # 嘗試導入關鍵包
        import streamlit
        import langchain
        import pandas
        import numpy
        import sentence_transformers
        import faiss
        import openai
        import google.generativeai
        import anthropic
        import plotly
        
        print("✅ 所有必要依賴包已安裝")
        return True
        
    except ImportError as e:
        print(f"❌ 缺少依賴包: {e}")
        print("請運行: pip install -r requirements.txt")
        return False

def setup_directories():
    """創建必要的目錄"""
    directories = ["data", "output", "logs"]
    
    for dir_name in directories:
        dir_path = Path(__file__).parent / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ 目錄 {dir_name} 已準備就緒")

def run_streamlit():
    """啟動Streamlit應用"""
    app_file = Path(__file__).parent / "app.py"
    
    if not app_file.exists():
        print("❌ app.py 文件不存在")
        return False
    
    print("🚀 啟動Streamlit應用...")
    print("📝 應用將在瀏覽器中打開")
    print("🔗 默認地址: http://localhost:8501")
    print("⏹️  按 Ctrl+C 停止應用")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 應用已停止")
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        return False
    
    return True

def main():
    """主函數"""
    print("🧬 疫苗主題建模RAG系統")
    print("=" * 50)
    
    # 檢查依賴
    if not check_requirements():
        return
    
    # 設置目錄
    setup_directories()
    
    # 啟動應用
    if not run_streamlit():
        return
    
    print("✅ 系統啟動完成")

if __name__ == "__main__":
    main()