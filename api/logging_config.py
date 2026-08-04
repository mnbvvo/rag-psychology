"""
结构化日志配置（标准库实现，无外部依赖）
在应用启动时调用 configure_logging()，统一输出 时间/级别/logger/消息。
便于本地排障；接入集中式日志（Loki / ELK）与全链路 tracing 属 Phase 1。
"""
import logging
import sys

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """配置根日志为结构化行格式，幂等（多次调用只生效一次）。"""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)

    # 收敛第三方库噪声，避免淹没业务日志
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _configured = True
