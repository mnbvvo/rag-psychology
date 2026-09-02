"""后台任务队列：进程内实现架构图中「Queue → Worker → DB」的虚线路径。

总稿定位（§3.2/§3.6）：
- Phase 1 用进程内 asyncio.Queue + 线程池 Worker 执行同步落库（DB 写 + 长期记忆 embedding），
  把持久化从请求生命周期解耦（请求只负责生成回答并拿到 session_id）；
- Redis/Queue（Phase 2 多实例）引入时替换本模块的 Queue 实现即可，调用方（enqueue）不变；
- 每个任务在 Worker 中经 asyncio.to_thread 执行，使用独立 DB Session（SessionLocal），
  满足「Session 不跨任务共享」；任务函数内部自带 try/except，单任务失败不影响其他任务。

可靠性兜底：enqueue 返回 False（队列未启动/已满）时，调用方应回退到请求内同步执行。
"""
import asyncio
import threading
from typing import Any, Callable, List, Optional, Tuple

from config.settings import settings

_Task = Tuple[Callable, tuple, dict]


class BackgroundQueue:
    """进程内 FIFO 后台任务队列（线程池执行同步任务）。"""

    def __init__(self, num_workers: int = 1, maxsize: int = 512):
        self.num_workers = max(1, num_workers)
        self._queue: "asyncio.Queue[_Task]" = asyncio.Queue(maxsize=max(1, maxsize))
        self._tasks: List[asyncio.Task] = []
        self._started = False

    # ---------------- 生命周期（startup / shutdown） ----------------
    def start(self) -> None:
        """启动 Worker（须在运行中的事件循环内调用；重复调用幂等）。"""
        if self._started:
            return
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._worker(i)) for i in range(self.num_workers)]
        self._started = True
        print(f"[bg_queue] 已启动 {self.num_workers} 个后台持久化 Worker", flush=True)

    async def shutdown(self) -> None:
        """优雅停止：取消 Worker 任务（进行中的 to_thread 任务自然结束后被回收）。"""
        if not self._started:
            return
        self._started = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        print("[bg_queue] 后台 Worker 已停止", flush=True)

    # ---------------- 投递 ----------------
    async def enqueue(self, fn: Callable, *args, **kwargs) -> bool:
        """投递一个同步任务。返回 False 表示未入队（调用方应回退同步执行）。"""
        if not self._started:
            return False
        try:
            self._queue.put_nowait((fn, args, kwargs))
            return True
        except asyncio.QueueFull:
            # 背压：Worker 跟不上时显式失败，由调用方回退请求内同步，避免无限堆积
            print(f"[bg_queue][WARN] 队列已满（{self._queue.qsize()}），回退同步落库: {getattr(fn, '__name__', fn)}", flush=True)
            return False

    # ---------------- Worker ----------------
    async def _worker(self, idx: int) -> None:
        tid = threading.get_ident() % 100000
        while True:
            fn, args, kwargs = await self._queue.get()
            try:
                # 同步落库放到线程池，独立 DB Session（任务函数内自行开 Session）
                await asyncio.to_thread(fn, *args, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # 任务自身应已捕获；这里兜底，避免单任务异常拖垮 worker 循环
                print(f"[bg_queue][ERROR] worker[{tid}] 任务失败: {type(e).__name__}: {e}", flush=True)
            finally:
                self._queue.task_done()


# 进程级单例：startup 调用 bg_queue.start()，shutdown 调用 bg_queue.shutdown()
bg_queue = BackgroundQueue(
    num_workers=settings.AI_BG_WORKERS,
    maxsize=settings.AI_BG_QUEUE_SIZE,
)
