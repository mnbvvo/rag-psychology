"""AI 问答并发准入模块（总稿 Phase 1：memory 后端）。

结构：
- models.py          数据契约（Ticket/错误码/返回码）
- base.py            AdmissionBackend 协议
- memory_backend.py  Phase 1 asyncio 实现（单 worker）
- service.py         统一入口（同步与 SSE 共用同一 service 单例）
- metrics.py         指标与结构化日志

对应总稿文档：docs/并发能力方案总稿.md §4
"""
from modules.concurrency import models  # noqa: F401
from modules.concurrency.base import AdmissionBackend  # noqa: F401
from modules.concurrency.memory_backend import MemoryAdmissionBackend  # noqa: F401
from modules.concurrency.metrics import AdmissionMetrics  # noqa: F401
from modules.concurrency.service import admission  # noqa: F401
