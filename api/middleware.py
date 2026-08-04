"""
本地原型级限流中间件（无外部依赖）
- InMemoryRateLimiter：按客户端 IP 的固定窗口计数（单进程内生效）
- RateLimitMiddleware：仅对 POST /api/query 限流，超限返回 429；
  /health* 探针与 CORS 预检（OPTIONS）放行，避免误伤健康探测。

说明：多 worker 时每个 worker 进程各自独立计数（本地原型可接受）；
     若需跨进程统一配额，Phase 1 改为 Redis 共享限流。
"""
import time
import threading
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config.settings import settings


class InMemoryRateLimiter:
    """固定时间窗限流：单位时间窗内允许不超过 times 次请求。"""

    def __init__(self, times: int, seconds: int):
        self.times = max(1, times)
        self.seconds = max(1, seconds)
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            window = self._hits[key]
            # 清理已过期的时间戳
            self._hits[key] = [t for t in window if now - t < self.seconds]
            if len(self._hits[key]) >= self.times:
                return False
            self._hits[key].append(now)
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, times: int, seconds: int):
        super().__init__(app)
        self._limiter = InMemoryRateLimiter(times, seconds)

    async def dispatch(self, request: Request, call_next):
        # 仅对问答写接口限流；健康检查探针与 CORS 预检放行
        if request.url.path == "/api/query" and request.method == "POST":
            client_ip = request.client.host if request.client else "unknown"
            if not self._limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "rate_limited", "detail": "too_many_requests"},
                )
        return await call_next(request)
