"""清理测试账号与数据（幂等，安全）。

只删除测试脚本产生的账号前缀（见 TEST_PREFIXES），
不影响 admin / legacy 及任何真实业务账号。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import delete, select  # noqa: E402

from db import SessionLocal  # noqa: E402
from db.models import CompareHistory, CrisisAudit, Message, Prompt, Session, User  # noqa: E402

# 覆盖所有测试脚本用到的账号前缀：
#   鉴权：user_a_/user_b_/lock_/test_/pta_/ptb_
#   数据隔离：isa_/isb_
#   并发：ccA_/ccB_/ccQ_
#   诊断：diag_
TEST_PREFIXES = (
    "user_a_", "user_b_", "lock_", "test_", "pta_", "ptb_",
    "isa_", "isb_", "ccA_", "ccB_", "ccQ_", "diag_",
)


def main():
    with SessionLocal() as db, db.begin():
        users = db.execute(select(User)).scalars().all()
        targets = [u for u in users if u.username.startswith(TEST_PREFIXES)]
        for u in targets:
            sids = select(Session.id).where(Session.user_id == u.id)
            db.execute(delete(Message).where(Message.session_id.in_(sids)))
            db.execute(delete(Session).where(Session.user_id == u.id))
            db.execute(delete(Prompt).where(Prompt.user_id == u.id))
            db.execute(delete(CompareHistory).where(CompareHistory.user_id == u.id))
            db.execute(delete(CrisisAudit).where(CrisisAudit.user_id == u.id))
            db.delete(u)
        print(f"已清理测试账号 {len(targets)} 个：{[u.username for u in targets] or '(无)'}")
    # 事务已提交，再查询确认最终状态
    with SessionLocal() as db:
        print(f"当前用户：{[u.username for u in db.execute(select(User)).scalars().all()]}")


if __name__ == "__main__":
    main()
