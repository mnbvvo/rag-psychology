# 启动脚本
# 确保已在虚拟环境中，且 Python >= 3.10（推荐 3.11 / 3.12）

# 安装依赖（请先激活 venv：.\.venv\Scripts\Activate.ps1）
echo "正在安装依赖..."
pip install -r requirements.txt

# 启动服务（仅绑本机 127.0.0.1，避免暴露到局域网）
echo "启动青少年心理RAG系统..."
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
