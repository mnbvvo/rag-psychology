"""认证接口：注册 / 登录 / 当前用户。

- 密码 bcrypt 哈希存储（modules.security），绝不存明文；
- 登录失败限流（内存级：连续失败锁定 15 分钟，防暴力破解）；
- JWT access token：HS256，默认 2h（settings.JWT_EXPIRE_MINUTES）。
"""
import re
import time
import uuid
from collections import defaultdict, deque

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from config.settings import settings
from db import crud
from db.models import User
from modules.security import hash_password, verify_password

from .deps import get_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])

# 登录失败记录：username -> deque[(ts)]，窗口内失败超限则锁定账号
_login_fails: dict[str, deque] = defaultdict(deque)
# IP 维度失败累计：同 IP 换用户名逐个试时累计超限 → 锁 IP
_login_ip_fails: dict[str, deque] = defaultdict(deque)
# 登录请求整体限流（频率）：IP -> deque（防止单 IP 爆破）
_login_ip: dict[str, deque] = defaultdict(deque)
# 注册请求限流：IP -> deque
_register_ip: dict[str, deque] = defaultdict(deque)

_last_sweep = 0.0  # 上次空桶清扫时间（限流桶键回收用）


def _prune(q: deque, window: float) -> None:
    while q and q[0] <= time.time() - window:
        q.popleft()


def _sweep_empty_buckets() -> None:
    """限流桶键回收：defaultdict 的键只 prune 不删除，公网暴露下键数会随来源 IP
    无限增长（每个 IP 残留一个空 deque）→ 每 10 分钟清扫一次空桶，防内存缓慢泄漏。"""
    global _last_sweep
    now = time.time()
    if now - _last_sweep < 600.0:
        return
    if len(_login_fails) + len(_login_ip_fails) + len(_login_ip) + len(_register_ip) < 500:
        return
    _last_sweep = now
    for store in (_login_fails, _login_ip_fails, _login_ip, _register_ip):
        for key in [k for k, v in store.items() if not v]:
            del store[key]


def _is_locked(username: str) -> bool:
    _prune(_login_fails[username], settings.LOGIN_LOCK_SECONDS)
    return len(_login_fails[username]) >= settings.LOGIN_MAX_FAILS


def _ip_locked(ip: str) -> bool:
    _prune(_login_ip_fails[ip], settings.LOGIN_LOCK_SECONDS)
    return len(_login_ip_fails[ip]) >= settings.LOGIN_IP_FAIL_MAX


def _record_fail(username: str, ip: str) -> None:
    """认证失败计数（username 存在与否**同代价**，防止响应差异泄露存在性）。

    - username 桶：超过 LOGIN_USER_BUCKET_MAX 后不再为陌生 username 建桶
      （攻击者用随机不存在用户名灌桶只占 5 次/名，防桶集合无限膨胀）；已存在的
      桶继续累计。
    - IP 桶：无条件累计（分布式换用户名探测的兜底维度）。
    """
    now = time.time()
    if username in _login_fails or len(_login_fails) < settings.LOGIN_USER_BUCKET_MAX:
        _login_fails[username].append(now)
    _login_ip_fails[ip].append(now)


def _clear_fails(username: str, ip: str) -> None:
    _login_fails[username].clear()
    _login_ip_fails[ip].clear()


_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")


def _make_token(user: User) -> dict:
    now = int(time.time())
    payload = {
        "sub": user.id,
        "role": user.role,
        "username": user.username,
        "iat": now,
        "exp": now + settings.JWT_EXPIRE_MINUTES * 60,
        "jti": uuid.uuid4().hex,
    }
    token = pyjwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": settings.JWT_EXPIRE_MINUTES * 60,
        "user": _user_view(user),
    }


def _user_view(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
    }


class RegisterBody(BaseModel):
    # 长度等业务规则在 register 内显式校验并返回 400（Pydantic min_length 会先返回 422，
    # 与"参数类错误 → 400"的接口约定不一致）
    username: str = Field(..., max_length=32, description="登录名（3-32 位字母/数字/下划线）")
    password: str = Field(..., max_length=128, description="密码（至少 8 位）")
    display_name: str | None = Field(None, max_length=64, description="显示名，默认取 username")


class LoginBody(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


@router.post("/register", status_code=201)
def register(body: RegisterBody, request: Request):
    """注册普通用户（role=user）。用户名冲突 → 409；格式/强度不合规 → 400。

    带单 IP 注册限流（默认 100 次/60s，env REGISTER_IP_MAX_REQUESTS 可调）：
    register 此前无任何限流，每次注册触发一次 bcrypt cost=12 哈希，可被批量
    注册刷 CPU 与 DB。
    """
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    _prune(_register_ip[ip], settings.RATE_LIMIT_SECONDS)
    if len(_register_ip[ip]) >= settings.REGISTER_IP_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="注册过于频繁，请稍后再试")
    _register_ip[ip].append(now)

    username = body.username.strip()
    if not _USERNAME_RE.match(username):
        raise HTTPException(status_code=400, detail="用户名须为 3-32 位字母/数字/下划线")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="密码长度至少 8 位")
    with crud.get_db() as db:
        if crud.get_user_by_username(db, username) is not None:
            raise HTTPException(status_code=409, detail="用户名已被占用")
        user = crud.create_user(
            db,
            username=username,
            password_hash=hash_password(body.password),
            display_name=(body.display_name or username).strip()[:64] or username,
            role="user",
            is_active=True,
        )
        _sweep_empty_buckets()
        return {"id": user.id, "username": user.username, "display_name": user.display_name, "role": user.role}


@router.post("/login")
def login(body: LoginBody, request: Request):
    """登录：校验用户名密码 → 签发 JWT。失败统一 401 文案（不泄露用户名是否存在）。

    锁定语义（2026-09-07 修正）：认证失败对「存在与否的用户名」同代价计数，
    不存在用户连续失败同样触发 429 —— 消除「只对存在用户 429」的存在性探针；
    另加 IP 维度累计失败锁（换用户名逐个试的兜底）。旧实现只在用户存在时计数，
    反而让攻击者能对已知用户名（如 admin）盲打 5 次错误密码直接锁号 15 分钟
    （认证 DoS）——本改动不消除该 tradeoff（彻底防需验证码/风控），但单 IP
    攻击 30 次即锁 IP，明显抬高成本。
    """
    username = body.username.strip()
    # IP 级请求频率限流（简单内存，多进程部署需共享存储）
    ip = request.client.host if request.client else "unknown"
    _prune(_login_ip[ip], settings.RATE_LIMIT_SECONDS)
    if len(_login_ip[ip]) >= settings.LOGIN_IP_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="登录尝试过于频繁，请稍后再试")
    _login_ip[ip].append(time.time())

    # IP 累计失败锁（换用户名逐个试的兜底）→ username 锁（存在与否同价）
    if _ip_locked(ip):
        raise HTTPException(status_code=429, detail="登录尝试过于频繁，请稍后再试")
    if _is_locked(username):
        raise HTTPException(status_code=429, detail="失败次数过多，请稍后再试")

    with crud.get_db() as db:
        user = crud.get_user_by_username(db, username)
        ok = user is not None and user.is_active and verify_password(body.password, user.password_hash)
        if ok:
            _clear_fails(username, ip)
            _sweep_empty_buckets()
            return _make_token(user)
    # 失败（含用户不存在）：username/IP 双维度同价计数。若用户不存在也计数，
    # 攻击者拿随机名灌桶会占内存 —— 由 _record_fail 的桶数量上限 LOGIN_USER_BUCKET_MAX 兜底。
    _record_fail(username, ip)
    _sweep_empty_buckets()
    raise HTTPException(status_code=401, detail="用户名或密码错误")


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    """当前登录用户信息（token 有效性校验）。"""
    return _user_view(user)
