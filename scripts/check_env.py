"""运行配置自检：对话端点 + 向量端点各打一次，避免 10 分钟真实模型阶梯白跑。

用法（任意目录均可，脚本自己按位置定位 .env）：
    python scripts\\check_env.py

期望输出：
    chat  200
    embed 1024

常见结果对照：
    chat 401            → OPENAI_API_KEY 无效，换 key
    chat 403 ... quota  → 仍在「仅免费额度」模式：充值 + 关闭 free tier only
    chat <异常>          → OPENAI_API_BASE / CHAT_MODEL 配错
    embed 非 1024        → 向量模型与库列 vector(1024) 不匹配，不能用于本测试
    embed 401           → EMBEDDING_API_KEY 无效（与对话 key 可以不同）

退出码：0 = 全部符合预期；1 = 有问题（输出里有具体原因）。
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

try:
    from dotenv import load_dotenv
except ImportError:
    print("缺少 python-dotenv，请先 pip install python-dotenv（或用项目所用环境运行）")
    sys.exit(1)

load_dotenv(ENV_PATH)

REQUIRED = (
    "OPENAI_API_BASE",
    "CHAT_MODEL",
    "OPENAI_API_KEY",
    "EMBEDDING_API_BASE",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_KEY",
    "ENABLE_THINKING",
)


def post_json(url: str, payload: dict, api_key: str, timeout: int = 30, read_body: bool = True):
    """返回 (status, 解析后的 body 或 None)。HTTP 错误码也当正常返回，不抛。

    read_body=False 用于流式端点（SSE）：只为拿状态码，不读/不解析响应体；
    否则响应体是 `data: {...}` 文本而非 JSON，解析必然失败。
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not read_body:
                return resp.status, None
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        body = e.read()[:200].decode("utf-8", "ignore")
        return e.code, {"raw": body}
    except Exception as e:  # noqa: BLE001
        return 0, {"raw": f"{type(e).__name__}: {e}"}


def main() -> int:
    print(f"[check_env] .env = {ENV_PATH}")
    print(f"[check_env] ENABLE_THINKING = {os.getenv('ENABLE_THINKING') or '(空)'}"
          f"  → 默认 False（不注入思考参数）")
    missing = [k for k in REQUIRED if k != "ENABLE_THINKING" and not os.getenv(k)]
    if missing:
        print(f"[check_env] 以下必填项为空：{', '.join(missing)}")
        return 1

    chat_base = os.getenv("OPENAI_API_BASE").rstrip("/")
    embed_base = os.getenv("EMBEDDING_API_BASE").rstrip("/")

    chat_status, body = post_json(
        chat_base + "/chat/completions",
        {
            "model": os.getenv("CHAT_MODEL"),
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        os.getenv("OPENAI_API_KEY"),
        read_body=False,  # SSE 流式响应，不解析 body，只看状态码
    )
    if chat_status == 200:
        print("chat  200")
    else:
        detail = (body or {}).get("raw", "") if isinstance(body, dict) else ""
        print(f"chat  {chat_status}  {str(detail)[:160]}")

    embed_status, body = post_json(
        embed_base + "/embeddings",
        {"model": os.getenv("EMBEDDING_MODEL"), "input": "ping"},
        os.getenv("EMBEDDING_API_KEY"),
    )
    embed_dim = None
    if embed_status == 200 and isinstance(body, dict):
        try:
            embed_dim = len(body["data"][0]["embedding"])
        except Exception:  # noqa: BLE001
            embed_dim = None
    if embed_dim is not None:
        print(f"embed {embed_dim}")
    else:
        detail = (body or {}).get("raw", "") if isinstance(body, dict) else ""
        print(f"embed {embed_status}  {str(detail)[:160]}")

    print("----")
    if embed_dim == 1024:
        print("向量维度 OK（=1024，与 vector(1024) 匹配）")
    elif embed_dim is not None:
        print(f"向量维度 {embed_dim} ≠ 1024 → 与库列不匹配，不能用于本测试")
    if chat_status != 200:
        print("结论：对话端点异常，先按上面提示处理，再跑阶梯")
        return 1
    if embed_dim != 1024:
        print("结论：向量端点异常，先按上面提示处理，再跑阶梯")
        return 1
    print("结论：chat 200 + embed 1024，可以开跑")
    return 0


if __name__ == "__main__":
    sys.exit(main())
