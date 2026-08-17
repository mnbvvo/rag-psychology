"""认证与授权依赖（FastAPI 依赖注入，三层防线的 L1 认证 + L2 授权）。

- get_current_user：解析 Bearer JWT → 校验签名/过期 → 加载用户 → 401 语义；
- require_admin：RBAC 角色校验 → 403 语义（垂直越权防线）。

铁律：user_id 永远从 token 解析（payload["sub"]），客户端传入的一律不信任。
"""
import jwt as pyjwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config.settings import settings
from db import crud
from db.models import User

# auto_error=False：缺 Authorization 头时不直接抛 401，由本依赖统一给出中文语义
oauth2_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    cred: HTTPAuthorizationCredentials | None = Depends(oauth2_scheme),
) -> User:
    """认证层 L1：无 token / 无效 / 过期 → 401；用户被禁用 → 401。"""
    if cred is None:
        raise HTTPException(status_code=401, detail="未认证，请先登录")
    try:
        payload = pyjwt.decode(
            cred.credentials,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的登录凭证")

    with crud.get_db() as db:
        user = crud.get_user(db, payload.get("sub"))
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="账号已被禁用")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """授权层 L2：普通用户访问管理员接口 → 403（垂直越权）。"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user
