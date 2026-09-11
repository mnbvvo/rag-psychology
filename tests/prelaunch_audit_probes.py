"""2026-09-10 audit probes. Synthetic users, mocked LLM, rolled-back DB writes.

Run from project root with the project's Python. No real LLM calls or user-data output.
This records CURRENT defects; it is not a regression suite asserting desired behavior.
"""
import asyncio
import contextlib
import importlib
import json
import sys
import types
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Avoid modules/__init__.py's eager PGVector setup during isolated probes.
pkg = types.ModuleType('modules')
pkg.__path__ = [str(ROOT / 'modules')]
sys.modules['modules'] = pkg
from config.settings import settings
from db import crud, crud_async, SessionLocal
from db.models import User, Message
from modules.security import LEGACY_PASSWORD_HASH
from modules import gateway
from sqlalchemy import select, text
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api import auth

results = {}
fake = User(id='audit_synthetic_mp_user', username=crud_async.mp_username('audit_synthetic_mp_user'),
            password_hash=LEGACY_PASSWORD_HASH, is_active=True, role='user', display_name='audit')
app = FastAPI()
app.include_router(auth.router)
with patch.object(crud, 'get_db', lambda: contextlib.nullcontext(None)), \
     patch.object(crud, 'get_user_by_username', return_value=fake):
    with TestClient(app) as client:
        resp = client.post('/api/auth/login', json={
            'username': fake.username, 'password': ''.join(chr(i) for i in range(8))})
        results['mirror_login'] = {'http_status': resp.status_code,
                                  'token_issued': bool(resp.json().get('access_token'))}

# Both requests pass ownership precheck while a client-chosen session does not exist;
# then their persistence interleaves. All synthetic writes are rolled back.
sid = uuid.uuid4().hex
with SessionLocal() as db:
    db.execute(text("SET LOCAL statement_timeout = '3s'"))
    db.execute(text("SET LOCAL lock_timeout = '1s'"))
    prechecks = [crud.session_belongs_to(db, sid, uid) for uid in ('audit_A', 'audit_B')]
    crud.append_turn(db, sid, 'A-question', 'A-answer', user_id='audit_A')
    db.flush()
    crud.append_turn(db, sid, 'B-question', 'B-answer', user_id='audit_B')
    db.flush()
    contents = db.execute(select(Message.content).where(Message.session_id == sid)).scalars().all()
    results['session_race'] = {'both_initial_prechecks_pass': all(prechecks),
        'owner': crud.ensure_session(db, sid).user_id,
        'other_user_message_written': 'B-question' in contents, 'message_count': len(contents)}
    db.rollback()
with SessionLocal() as db:
    results['session_race']['rollback_verified'] = db.get(__import__('db.models', fromlist=['Session']).Session, sid) is None

before = gateway.get_persist_metrics()['critical_failures']
with patch.object(crud, 'append_turn', side_effect=RuntimeError('synthetic DB failure')), \
     patch.object(crud, 'get_db', lambda: contextlib.nullcontext(None)), patch.object(gateway.time, 'sleep'):
    returned = gateway.flush_db_turn_sync('audit', 'q', 'a', None, 'audit', is_crisis_response=True)
results['critical_persistence_failure'] = {'returned_normally': returned is None,
    'counter_incremented': gateway.get_persist_metrics()['critical_failures'] == before + 1}

from modules.safety_checker import SafetyChecker
checker = SafetyChecker()
with patch.object(checker, 'semantic_check', return_value={'is_crisis': True, 'level': 'high', 'distance': 0.01}):
    classified = checker.check_full('我想自杀，朋友劝我也没有用')
results['safety_downgrade'] = {'semantic_high_final_level': classified['level'],
                             'safety_global_enabled': settings.SAFETY_ENABLED}

# Test real mp route and real persistence orchestration with a synthetic answer.
pkg.rag_system = types.SimpleNamespace()
from api import mp
from modules.bg_queue import BackgroundQueue
from modules.concurrency.memory_backend import MemoryAdmissionBackend
from modules.concurrency.service import AdmissionService
from modules.concurrency.metrics import AdmissionMetrics
from modules.memory import memory_service
async def probes():
    service = AdmissionService(MemoryAdmissionBackend(), AdmissionMetrics())
    with patch.object(mp, '_ensure_mp_user', new=AsyncMock()), \
         patch.object(mp, '_load_profile_text', new=AsyncMock(return_value='')), \
         patch.object(mp, 'admission', service), \
         patch.object(mp, 'rag_system', types.SimpleNamespace(aquery=AsyncMock(return_value={'answer':'synthetic answer','question':'q'}))), \
         patch.object(settings, 'MEMORY_ENABLED', False), \
         patch.object(crud, 'append_turn', side_effect=RuntimeError('synthetic DB failure')), \
         patch.object(crud, 'get_db', lambda: contextlib.nullcontext(None)), patch.object(gateway.time, 'sleep'):
        response = await mp.mp_query(mp.MpQueryBody(userId='audit', query='q'), db=None)
        results['mp_response_after_db_failure'] = {'success_body': isinstance(response, dict) and 'answer' in response,
                                                 'persistence_error_exposed': 'error' in response}
    queue = BackgroundQueue()
    with patch.object(memory_service, 'embed', side_effect=RuntimeError('synthetic embedding failure')):
        queue.start()
        await queue.enqueue(gateway.flush_memory_sync, 'audit', 'q', 'a')
        await queue._queue.join()
        results['memory_failure_metrics'] = {'completed': queue.completed_count, 'failed': queue.failed_count}
        await queue.shutdown()
    disabled = User(id='audit', username='mp_synthetic', is_active=False)
    results['disabled_mirror'] = {'guard_accepts': crud_async._guard_mp_mirror(disabled) is disabled}
    # Scale 30 DB connections / 60 admission places down to 2 / 3. Only SELECT 1;
    # actual mp route, FastAPI yield dependency, and SQLAlchemy pool behavior.
    import httpx
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from api.deps import get_db_session
    eng = create_async_engine(settings.DB_URL, pool_size=2, max_overflow=0, pool_timeout=0.2)
    sessions = async_sessionmaker(eng)
    async def dependency():
        async with sessions() as db:
            yield db
            await db.commit()
    async def check_user(db, *args, **kwargs):
        await db.execute(text('SELECT 1'))
    release_llm = asyncio.Event()
    async def generate(**kwargs):
        await release_llm.wait()
        return {'question': 'q', 'answer': 'synthetic'}
    small_service = AdmissionService(MemoryAdmissionBackend(max_active=1, max_queue=2,
                                    queue_wait_timeout_seconds=5), AdmissionMetrics())
    mini = FastAPI()
    mini.include_router(mp.router)
    mini.dependency_overrides[get_db_session] = dependency
    mini.dependency_overrides[mp.require_mp_key] = lambda: None
    with patch.object(mp, '_ensure_mp_user', check_user), \
         patch.object(mp, '_load_profile_text', new=AsyncMock(return_value='')), \
         patch.object(mp, 'admission', small_service), \
         patch.object(mp, 'enqueue_persist', new=AsyncMock()), \
         patch.object(mp, 'rag_system', types.SimpleNamespace(aquery=generate)):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mini, raise_app_exceptions=False), base_url='http://audit') as client:
            pending = [asyncio.create_task(client.post('/api/mp/query', json={'userId':str(i),'query':'q'})) for i in range(2)]
            try:
                for _ in range(200):
                    if len(small_service.backend._queue) == 1:
                        break
                    await asyncio.sleep(0.01)
                third = await client.post('/api/mp/query', json={'userId':'third','query':'q'})
                snap = await small_service.snapshot()
                results['pool_before_admission'] = {'third_http_status':third.status_code,
                    'active':snap.active, 'queued':snap.queued, 'queue_capacity':snap.max_queue,
                    'checked_out_connections':eng.pool.checkedout()}
            finally:
                release_llm.set()
                await asyncio.gather(*pending)
    await eng.dispose()
asyncio.run(probes())

# Health endpoint still returns HTTP 200 when the DB probe raises.
main = importlib.import_module('api.main')
import db as db_module
with patch.object(db_module, 'SessionLocal', side_effect=RuntimeError('synthetic DB unavailable')), \
     patch.object(settings, 'HEALTH_PROBE_EMBEDDING', False):
    health = TestClient(main.app).get('/api/health')
    results['health_db_down'] = {'http_status':health.status_code, 'status':health.json()['status']}

# Username cap is checked after _is_locked has already allocated the new bucket.
auth._login_fails.clear()
with patch.object(settings, 'LOGIN_USER_BUCKET_MAX', 2):
    for i in range(5):
        auth._is_locked(f'audit_absent_{i}')
        auth._record_fail(f'audit_absent_{i}', 'audit_ip')
    results['login_bucket_cap'] = {'configured_cap':2, 'actual_buckets':len(auth._login_fails)}

from sqlalchemy.engine import make_url
parsed = make_url('postgresql+psycopg://audit:synthetic@password@127.0.0.1:5432/audit')
results['db_password_url_encoding'] = {'host_parsed_correctly':parsed.host == '127.0.0.1'}

# Match POSIX asyncio API without importing the Windows-only module entrypoint.
source = (ROOT/'api/main.py').read_text(encoding='utf-8')
initial = source[:source.index('import sys')]
policy = asyncio.WindowsSelectorEventLoopPolicy
try:
    del asyncio.WindowsSelectorEventLoopPolicy
    try:
        exec(compile(initial, 'api/main.py', 'exec'), {})
        results['linux_entrypoint'] = {'exception': None}
    except AttributeError as exc:
        results['linux_entrypoint'] = {'exception': type(exc).__name__}
finally:
    asyncio.WindowsSelectorEventLoopPolicy = policy

out = ROOT/'tests/results/prelaunch-audit-probes.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(results, ensure_ascii=False, indent=2))
