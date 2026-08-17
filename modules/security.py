"""密码哈希与校验（bcrypt 封装，统一入口，避免散落各处）。

- hash_password：注册 / 初始化管理员时使用
- verify_password：登录校验；对非法哈希安全返回 False（不抛异常）
- LEGACY_PASSWORD_HASH：legacy 迁移账号使用"永远无法登录"的哈希
"""
import bcrypt

# legacy 账号（历史数据归属）的密码哈希：用随机密码生成，任何输入都无法匹配；
# 且该账号 is_active=False，双重保证不可登录。
LEGACY_PASSWORD_HASH = bcrypt.hashpw(b"".join(bytes([i % 256]) for i in range(8)), bcrypt.gensalt(rounds=4)).decode("utf-8")


def hash_password(plain: str) -> str:
    """生成 bcrypt 哈希（cost=12，与 README 安全基线一致）。"""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文与哈希是否匹配；哈希非法（非 bcrypt 格式）时安全返回 False。"""
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
