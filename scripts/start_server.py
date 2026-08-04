"""
本地原型加固启动器
- 读取 config.settings 的 HOST / PORT / WEB_WORKERS / DEBUG
- DEBUG=true ：单进程 + 热重载（开发联调用）
- DEBUG=false（默认）：多 worker（WEB_WORKERS），充分利用多核，
  配合 Phase 0 全链路异步实现真并发

注意（Chroma 多 worker 读写）：
- 多 worker 下每个 worker 进程会独立加载 Chroma 客户端，读同一 chroma_db 目录。
  查询期只读，通常没问题；但写入/重建知识库请走单进程导入脚本
  （python scripts/import_cards.py），避免多进程并发写同一 SQLite 库。
"""
import uvicorn

from config.settings import settings


def main() -> None:
    if settings.DEBUG:
        # 开发模式：热重载必须单进程
        uvicorn.run(
            "api.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=True,
            workers=1,
        )
    else:
        # 生产/原型加固模式：多 worker 真并发
        uvicorn.run(
            "api.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=False,
            workers=settings.WEB_WORKERS,
        )


if __name__ == "__main__":
    main()
