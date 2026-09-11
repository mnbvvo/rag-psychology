# 心理知识问答 RAG 系统（青少年 / 家庭方向）

面向青少年及家庭心理知识问答的本地检索增强生成（RAG）系统。知识以结构化「卡片」形式入库，检索后由大模型基于卡片生成回答，内置**分级危机干预检测**（关键词硬门控 + 语义锚点 + 回答侧复查）、**长期记忆**（向量检索式）与 **JWT 多租户数据隔离**。

技术栈：FastAPI + LangChain + PostgreSQL 16（关系库 + pgvector 向量库，统一单一后端）+ OpenAI 兼容接口（默认 DashScope / 通义千问兼容模式）。

---

## 功能总览

| 能力 | 说明 | 开关 |
|---|---|---|
| 对话联调 | 多会话 + SSE 流式问答 + 来源/耗时展示 + 导出 | — |
| RAG 检索 | pgvector 向量召回 ∪ BM25 关键词召回 → 本地重排精排 | `RAG_ENABLED` |
| 危机检测 | L0 关键词硬门控 + L1 语义锚点距离 + 回答侧复查，命中落 `crisis_audit` 审计 | `SAFETY_ENABLED` |
| 长期记忆 | 每轮问答落 `user_chat_history`（双向量），提问时检索相似历史注入上下文 | `MEMORY_ENABLED` |
| 多租户 | JWT 鉴权 + 会话/审计按用户隔离 + admin 垂直权限 | — |

> ⚠️ 两个关键开关默认**关闭**（`config/settings.py` 静态值）：`RAG_ENABLED=False`（纯 LLM 对话，不检索）、`SAFETY_ENABLED=False`（整条安全链路跳过）。需要完整 RAG / 安全能力时在 `config/settings.py` 改为 `True` 并重启。

---

## 快速开始（新环境完整流程）

> 按序执行即可在本机跑通。本机开发环境：conda `juliy`（Python 3.13）+ PostgreSQL 16 + pgvector 0.8.2。

### 0. 前置依赖

- Python 3.12+（新环境装包慢可用国内镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`）
- PostgreSQL（含 pgvector 扩展，见步骤 2）

### 1. 安装依赖

```powershell
pip install -r requirements.txt
# 本地重排（bge-reranker-v2-m3）依赖 torch，体积大，需单独装：
#   CPU：  pip install torch
#   GPU：  pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. 准备 PostgreSQL 数据库

```powershell
psql -U postgres
CREATE DATABASE rag_psychology;
\c rag_psychology
CREATE EXTENSION IF NOT EXISTS vector;      # pgvector（必装）
CREATE EXTENSION IF NOT EXISTS pg_trgm;     # 可选（混合检索辅助）
```

> 业务表、长期记忆检索函数（`fn_search_chat_history`）与 HNSW 索引都由服务首启
> `init_db()` **自动幂等创建**（含老库轻量迁移），无需手工执行任何 SQL。

### 3. 配置 `.env`

```powershell
Copy-Item .env.example .env     # Linux: cp .env.example .env
```

必填三项：**`OPENAI_API_KEY`**（LLM 密钥）、**`JWT_SECRET`**（≥32 字节随机串，
`python -c "import secrets;print(secrets.token_urlsafe(48))"`）、**`PG_PASSWORD`**。
其余为可选调参（限流/并发/小程序通道 `MP_API_KEY` 等），见 `.env.example` 注释与下方「配置」。

### 4. 启动

```powershell
python api/main.py                     # 默认 127.0.0.1:8000
PORT=8001 python api/main.py           # 8000 被占用时
```

- ⚠️ 请用 `python api/main.py` 启动：该入口自持 Selector 事件循环（Windows 下
  psycopg async 必须），并做单实例守卫。**不要**用 `python -m uvicorn --reload`：
  Windows 会踩 Proactor 事件循环坑，且 reload 的 master+worker 拓扑与并发准入的
  单实例锁互斥。
- 首启自动：建表 + 迁移 + 长期记忆函数/HNSW 索引 + 引导账号（`legacy` +
  管理员 `admin` / `admin123456`，生产用 `.env` 的 `INIT_ADMIN_*` 覆盖）。

### 5. 导入知识库（可选但推荐，让 RAG 有料可检）

```powershell
python scripts/import_cards.py "你的知识库.jsonl" --reset   # 格式见「知识库构建」
```

### 6. 验证

- Web 前端：`http://127.0.0.1:8000`（先注册/登录）
- 健康检查：`http://127.0.0.1:8000/api/health` → `deps.db` 应为 `ok`
- Swagger：`http://127.0.0.1:8000/docs`
- 启动日志会打印实际生效配置（向量库后端 / RAG / 安全检查 / 重排状态，DB 连接串已脱敏）

---

## 配置

复制 `.env.example` 为 `.env` 并填入密钥（**.env 含密钥，勿提交**，已被 `.gitignore` 忽略）。

**配置分层原则**：`.env` 只放「密钥 + 随部署环境变化的值」；检索 / 生成 / 安全 / 服务等调参都是 `config/settings.py` 里的静态常量（含中文注释），不填也能跑。

`.env` 通常只要这几行：

```dotenv
OPENAI_API_KEY=你的API密钥
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1  # OpenAI 兼容接口
CHAT_MODEL=deepseek-v4-flash-0731       # 对话模型
DB_BACKEND=postgres                     # 关系库
PG_PASSWORD=你的数据库密码              # 本机 PostgreSQL 密码
JWT_SECRET=你的强随机密钥               # ≥32 字节，见下方说明
```

> 数据库必填：本系统使用 **PostgreSQL**（关系库 + pgvector 向量库同库，默认库名 `rag_psychology`）。`PG_HOST/PG_PORT/PG_USER/PG_DB` 按需覆盖；`VECTOR_BACKEND` 固定 `pgvector`。缺 `JWT_SECRET` 或强度不足（<32 字节）时服务**拒绝启动**（fail-closed）。

关键静态开关（`config/settings.py`，改后重启）：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `RAG_ENABLED` | `False` | `True` 走完整 RAG（安全检测 + 混合检索 + 重排 + 生成）；`False` 纯 LLM 对话 |
| `SAFETY_ENABLED` | `False` | 安全链路总开关（L0 关键词 + L1 语义 + 回答侧复查）；`False` 全部跳过 |
| `SAFETY_CHECK_ENABLED` | `True` | L0 关键词快检细分开关（仅 `SAFETY_ENABLED` 之下生效） |
| `SEMANTIC_CHECK_ENABLED` | `True` | L1 语义锚点细分开关（仅 `SAFETY_ENABLED` 之下生效） |
| `MEMORY_ENABLED` | `True` | 长期记忆（向量检索式历史注入） |
| `RERANK_ENABLED` | `True` | 本地重排（bge-reranker-v2-m3）；仅 `RAG_ENABLED` 时生效 |

> `rag_enabled` / `safety_enabled` 可在单个请求里按次覆盖（`None`=用全局配置）。RAG/安全关闭时，启动不会加载重排模型与语义锚点（避免白费资源），请求级覆盖会触发首次懒加载。

---

## 系统流程

**启动**：`settings.validate()` → `init_db()`（建表 + 轻量迁移 + 长期记忆函数 `fn_search_chat_history` 与 HNSW 索引启动自愈 + 引导 legacy/admin 账号）→ 自动命名旧会话 → 条件化预热（仅 RAG 开时加载重排/BM25，仅安全开时 embed 语义锚点）。

**一次问答**：前端提交 → JWT 鉴权 + 会话越权校验 + IP 限流 → `prepare`（L0 关键词 → L1 语义锚点距离 → 向量 ∪ BM25 召回 → 重排精排）→ `generate`（代码常量提示词 + 长期记忆双通道注入 → LLM 同步/流式）→ 持久化（sessions/messages + 危机命中写 crisis_audit + 每轮写 user_chat_history）→ SSE/JSON 返回。

---

## 项目结构

```text
api/
  main.py        FastAPI 接口（/api/query、/api/query/stream、/api/sessions、/api/auth/*、/api/admin/*）+ 启动钩子
  auth.py        注册 / 登录 / me（JWT 签发、登录失败锁定）
  deps.py        get_current_user（JWT→用户）、require_admin（RBAC）
modules/
  __init__.py    rag_system 编排：prepare（安全+检索）/ query（完整同步链路）
  rag_core.py    LLM 封装、消息组装、generate / stream_generate
  vector_store.py  pgvector 向量检索、embedding 缓存计时
  safety_checker.py L0 关键词 + L1 语义 + 回答侧复查
  crisis_detector.py L1 高危意图锚点距离检测
  hybrid_search.py BM25 关键词召回（jieba 分词）
  reranker.py    本地重排（bge-reranker-v2-m3）
  memory.py      长期记忆（双向量落库 + 相似历史检索）
  prompt_store.py 系统提示词常量 ACTIVE_PROMPT + 组装 system prompt
  security.py    bcrypt 密码哈希封装
db/
  models.py      5 张业务表 ORM
  crud.py        各表读写（get_db 上下文管理器）
  crud_async.py  异步读写（AsyncSession，请求级）
  __init__.py    引擎 / init_db 幂等建表 / legacy+admin 引导
config/          settings.py 调参；crisis_keywords.json（L0 关键词）；high_risk_intents.json（L1 种子）
frontend/        纯静态单页（对话联调，FastAPI 托管，无需构建）
scripts/
  import_cards.py                 知识卡片 JSONL 导入
  concurrency_pressure_test.py    并发压测（boundary / steady / spike 三模式）
  backup_pg.py                    PostgreSQL 全量备份
  calibrate_crisis_thresholds.py  L1 阈值标定
  diagnose_retrieval.py           检索质量诊断
data/            运行时数据（.gitignore 忽略）：crisis_prototypes.json（L1 锚点缓存）等
requirements.txt Python 依赖
```

---

## 接口

> 除 `/api/auth/register`、`/api/auth/login`、`/api/health` 外**全部接口需要登录**：请求头携带 `Authorization: Bearer <token>`。
> - 无 token / 无效 / 过期 → `401`；角色不足（普通用户访问 `/api/admin/*`）→ `403`；
> - 访问他人资源（会话）→ `403`（水平越权防护）；请求体携带的 `user_id` 一律忽略，身份以 token 为准（篡改无效）。
> - 例外：`/api/mp/*`（微信小程序通道）不依赖 JWT，身份来自请求体 `userId`，见下节「微信小程序通道」。

### 微信小程序通道（/api/mp/*，第二身份通道）

供「已有微信小程序」把本系统当 AI 对话后端接入：小程序自己负责注册登录，仅把既有用户 id 与前端生成的会话 id 传给本服务，**后端不自建账号**（首次出现自动建不可登录的 users 镜像行，会话/记忆/危机审计/家庭档案均挂该 id）。注册与资料页在小程序完成，本服务只收档案。

⚠️ **信任边界（务必遵守）**：mp 通道的身份 = 请求体 `userId`（模式 a 内网直传，`MP_TRUST_MODE=a`）。任何能访问本服务的人可伪造 userId 越权读取他人会话/档案/危机审计 → **只应部署在本机/内网/网关白名单后，切勿直接暴露公网**。公网直连需先实现模式 b（微信 code2session 换 token，配置预留未实现）。

纵深防御：在 `.env` 配置 **`MP_API_KEY`**（非空）后，`/api/mp/*` 全部端点（含 `GET /api/mp/profile`）强制要求 `X-API-Key` 请求头，防止同网段进程/误暴露时直接调用与枚举档案；留空则保持免 key 联调。另：`/api/mp/*` 全命名空间已纳入单 IP 限流中间件（与 `/api/query` 同桶）。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/mp/register` | 注册时写家庭档案：`{userId, profile}`；建镜像号 + 档案 upsert（幂等可重放） |
| POST | `/api/mp/update` | 资料更新：同 `register` 的 body 与 upsert 语义 |
| GET | `/api/mp/profile?userId=xxx` | 回读整份档案（camelCase，资料页回显用） |
| POST | `/api/mp/query` | 非流式对话：`{userId, sessionId, query}` → `{answer, sessionId, sources, safetyNote, isCrisisResponse}` |
| POST | `/api/mp/query/stream` | SSE 流式对话（body 同上）：`queue → started → sources → token×N → done`；高危直达 `done`；异常发 `error` |

`profile` 结构：

```json
{
  "userNickname": "用户1834",
  "birthday": "1988-05-12",
  "parent_role": "爸爸",
  "children": [
    { "childId": "D972CEEE-B", "childNickname": "为人父", "childBirthDate": "2026-08-04", "gender": "女" }
  ]
}
```

- 落库表：`user_family_profile`（家长主表）+ `user_children`（孩子子表，整份提交全量替换）；
- 对话时按 userId 读取档案，经 `modules/family_profile.py` **加工后**注入 system prompt：生日换算为「约 X 岁 / X 个月」等相对表述，昵称/角色/孩子性别保留，原始生日不落 prompt，并锚定「档案是对话用户（家长）的背景、不是助手身份」防止角色混淆；
- 微信小程序消费 SSE：基础库 **≥2.20.2**，`wx.request({ enableChunked: true, responseType: "text" })` + `requestTask.onChunkReceived()` 按 `event:`/`data:` 帧拆包（chunk 为 ArrayBuffer 需 utf-8 解码并缓冲拼帧），收到 `done` 后 `requestTask.abort()`；
- 会话历史：只发 `sessionId` + 本轮 `query`，服务端按会话从 DB 注入最近 N 轮原文（与 Web 端同机制，跨设备一致）；
- 说明：`userId` 上限 64 字符；`sessionId` 建议 ≤36（会话主键列宽）；mp 流式暂不提供独立取消接口（断连自动清理在途请求）。

### 认证

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/auth/register` | 注册：`{username, password, display_name?}`；用户名 3-32 位字母/数字/下划线、密码 ≥8 位；冲突 409、不合规 400 |
| POST | `/api/auth/login` | 登录 → `{access_token, expires_in, user}`；失败锁定双维度：username 5 次锁 15 分钟 + IP 累计 30 次锁 IP（**存在与否同价计数**，不泄露用户名存在性）；失败统一 401 / 429（429） |
| GET | `/api/auth/me` | 当前用户信息（token 有效性校验） |

初始管理员：首次启动 `users` 表为空时自动创建（`INIT_ADMIN_USERNAME` / `INIT_ADMIN_PASSWORD`，默认 `admin` / `admin123456`，**生产务必在 .env 覆盖**）。历史无归属数据启动时归入不可登录的 `legacy` 账号。

### 问答

**POST `/api/query`**（同步）

```json
{ "question": "孩子总是情绪低落、没兴趣，家长该怎么做？" }
```

返回 `{answer, sources[], safety_note, is_crisis_response, safety_check, timings, session_id}`。命中中/低危时附关怀提示；命中高危时 `answer` 直接为危机干预提示（不再走常规回答）。

**POST `/api/query/stream`**（SSE，对话页使用）

事件顺序：`sources → token×N → done`；高危直接 `done`；异常发 `error`。`done` 含完整回答 / safety_note / timings / session_id。

请求体可选字段：`messages`（多轮历史，role 兼容 human/ai/user/assistant）、`session_id`、`persist`（默认 true，false=不落库）、`rag_enabled` / `safety_enabled`（按次覆盖全局开关）。

### 会话

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/sessions` | 列出最近会话（含 message_count） |
| GET | `/api/sessions/{id}/messages` | 取会话全部消息（按时间序） |
| POST | `/api/sessions` | 新建空会话 |
| PATCH | `/api/sessions/{id}` | 重命名会话 |
| DELETE | `/api/sessions/{id}` | 删除会话（级联删消息） |

### 管理员

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/admin/users` | 用户列表（普通用户 403） |
| GET | `/api/admin/crisis-audit` | 危机审计（合规留痕，仅管理员） |

### 健康检查

```powershell
curl http://127.0.0.1:8000/api/health
# => {"status":"healthy","version":"1.0.0","deps":{"db":"ok","embedding":null},"persist":{...}}
```

> `deps.db`：每次请求 `SELECT 1` 探活（失败 → `status=degraded`，仍返回 200 兼容探活脚本）；`deps.embedding`：默认 `null`，设 `HEALTH_PROBE_EMBEDDING=true` 后每次真实 embed 一次 "ping"（计费，默认关）。所有响应携带 `X-Request-ID` 便于跨日志关联单次请求。只确认服务已启动，不校验知识库是否已导入；返回 healthy 但问答无结果时，先执行导入。

> 部署配置（.env 可覆盖，无需改代码）：`HOST`（默认 127.0.0.1）、`PORT`、`CORS_ORIGINS`（逗号分隔，覆盖默认的 `http://{HOST}:{PORT}`/localhost）。

---

## 前端（对话联调）

纯静态单页（`frontend/`，FastAPI 直接托管，无需构建）：多会话（新建/重命名/删除/导出）、SSE 流式问答（token 增量渲染 + 来源 chips + 分阶段耗时栏）、危机安全提示与高危标记、多轮历史与长期记忆。数据全部持久化在服务端，前端无浏览器缓存。登录后即进入对话页。

> **Postman / 外部调用**：直接 `POST http://127.0.0.1:8000/api/query`，`Content-Type: application/json`，请求体 `{"question": "..."}` 即可，无需走前端页面。

## 系统提示词（代码内常量，用户不可修改）

系统提示词**直接定义在代码里**（`modules/prompt_store.py` 的 `ACTIVE_PROMPT` 常量），不存数据库：

- **所有用户共用同一条**，每次问答零额外查询（无 DB 依赖）；
- 用户侧**没有任何修改入口**：前端无编辑 UI、后端无写接口、请求体不接受 `prompt_id` / `system_prompt_override`；
- **修改方式**：改 `ACTIVE_PROMPT` 常量 → 重启服务生效（仅开发者/管理员操作）；
- `LOW_RELEVANCE_NOTE`（防编造说明）仅在「RAG 开启且检索为空」时自动追加，纯对话模式不追加。

---

## 知识库构建

主链路仅支持 JSONL 卡片导入（不依赖 `unstructured` / `pypdf` / `docx2txt`）：

```powershell
python scripts/import_cards.py "data/your_knowledge_base.jsonl" --reset
```

- `--reset`：先清空再导入（避免重复）；`--review-status approved`：只导指定审核状态（**安全敏感场景建议只导 approved**）；`--batch-size 50`：分批写入。

JSONL 每行一条记录（`card_json` 结构化字段 + 审核字段，见项目内示例）。导入时 `card_json` 拼成可读文本向量化，结构化字段写入 metadata（`card_id` / `source_id` / `chunk_id` / `title` / `domains` / `audiences` / `age_stages` / `risk_level` / `evidence_level` / `review_status` / `age_group` 等）。

> ℹ️ **`age_group` 仅作数据留档**：由 `age_stages` 中文标签归一化（婴儿/幼儿/儿童→child，少年/中小学→early_teen，青少年→teen，青年/高中→late_teen），当前**不参与检索过滤与语气适配**（知识库未按年龄充分分类，硬过滤会误杀卡片）。

---

## 检索与生成链路

### 向量库（PostgreSQL + pgvector）

- 向量由 `EMBEDDING_MODEL` 生成，文档来自 JSONL 结构化卡片（一卡一文档）；
- **pgvector**：向量存 PostgreSQL 的 `langchain_pg_embedding` 表（与关系库同库），余弦相似度检索；
- 重建：先 `--reset` 再重新导入。

> ⚠️ 向量与 `EMBEDDING_MODEL` 绑定：更换 embedding 模型后旧向量**静默失效**（不报错但检索质量崩坏），必须 `--reset` 重新导入。

### 混合检索 + 本地重排

检索默认启用**混合召回**：pgvector 向量召回（`FETCH_K=10`）∪ BM25 关键词召回（`HYBRID_KEYWORD_K=5`，jieba 分词），去重后交给本地 Cross-Encoder 精排取 top3（`bge-reranker-v2-m3`）。重排失败自动回退原排序，不影响可用性。模型首次使用懒加载（约 5-10 秒），前端耗时栏显示「重排」耗时。

模型下载（放到 `data/rerank_models/bge-reranker-v2-m3`，已被 .gitignore 忽略）：

```powershell
# 方式一：ModelScope（国内快）
pip install -U modelscope
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir "data/rerank_models/bge-reranker-v2-m3"

# 方式二：HuggingFace 镜像
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir "data/rerank_models/bge-reranker-v2-m3"
```

依赖：`sentence-transformers` + torch（有 NVIDIA GPU 建议装 CUDA 版：`pip install torch --index-url https://download.pytorch.org/whl/cu124`）。相关调参见 `config/settings.py`「本地重排序」一节：`RERANK_MIN_SCORE`（分数下限护栏，默认 0 不启用；实测相关约 0.4+、无关 <0.15，建议 0.2~0.3 起试）。

### 长期记忆

`MEMORY_ENABLED` 开启时，每轮问答写入 `user_chat_history` 并打**双向量**（query 向量 + query+answer 拼接的 qa 向量，检索主用）；下次提问先用当前问题向量检索该用户相似历史（`MEMORY_TOP_K=5`，`MEMORY_MIN_SIMILARITY=0.3`）注入 system prompt。另有 `MEMORY_RECENT_ROUNDS=6` 的本会话最近 N 轮原文直插，解决指代消解（"那个方法""刚才说的"）。检索成本恒定，不随历史总量线性增长。

> 检索函数 `fn_search_chat_history` 与 2 个 HNSW 索引 + user_id 索引由 `init_db()` 启动时**幂等自愈创建**（`db/__init__.py::_ensure_user_chat_history_sql`），维度取 `settings.VECTOR_DIMENSION`，新库无需手工执行 SQL。历史 DDL `scripts/user_chat_history.sql` 已恢复，仅作参考/手工兜底（曾被 858a30f「4.1精简版」误删导致新库记忆静默失效）。

---

## 安全与危机干预

`SAFETY_ENABLED=True` 时，每次问答前做**两级检测**：

**L0 关键词快检**（`check_text`，毫秒级）：关键词与等级定义在 `config/crisis_keywords.json`，命中判定 high/medium/low/none。命中 high 但属**求助型提问**（孩子/朋友等称谓 + 怎么办/帮助等动作词，如"孩子有自伤倾向怎么办"）自动降级 medium，避免把家长/老师的求助误当危机实施者拦截。

**L1 语义检测**（`semantic_check`，高危意图锚点距离）：隐喻/隐晦表达（如"想用脑袋和房梁比赛"=上吊）字面无关键词必漏。L1 把种子集（`config/high_risk_intents.json`，标准意图句 + 隐喻变体，覆盖自杀/自伤/伤害他人/被伤害四簇）embed 后作为**锚点集合**（簇内保留每条句子向量，非均值原型），用户问题与所有锚点算余弦距离、取最近：

- 距离 ≤ `CRISIS_INTERCEPT_DIST`（默认 0.25）→ 高危拦截；
- 距离 ≤ `CRISIS_GRAY_DIST`（默认 0.36）→ 疑似，附关怀 + 转介（不拦截）；
- 其余 → 放行。

embedding 复用检索阶段那次 API 调用（进程内 LRU 缓存），**不增加额外成本**。锚点向量缓存到 `data/crisis_prototypes.json`，种子文件变更自动重建。

**回答侧复查**（`review_answer`）：LLM 回答再跑一遍 L0 复查，命中高危时在末尾**追加**安全提醒（不替换原文），以 `detect_method=answer_check` 记入审计。

**响应与审计**：高危直接返回危机干预提示与热线；中/低危正常回答，末尾附 `safety_note`。所有命中（含灰区）写入 `crisis_audit`，记录 `detect_method`（keyword/semantic）与 `confidence`（语义距离）。

**阈值标定**：`python scripts/calibrate_crisis_thresholds.py --top 5` 对种子集正例 + 内置负例计算距离分布，输出建议阈值（实测正例 0.095~0.321、负例 0.339+，两组完全分离）。新增隐喻表达请回填 `high_risk_intents.json` 的 `variants`，召回率随积累单调上升。

> ⚠️ 这是**原型级**防护：关键词 + 语义原型无法 100% 识别所有隐晦危机表达（灰区误报/漏报依然存在），不能替代专业心理危机干预。真实场景需引入 LLM 精判、人工审核与升级机制。

---

## 数据库

### 两层持久化（分工明确）

- **向量库**：只负责语义检索（pgvector，与关系库同库）；
- **关系库**：只负责结构化留痕——用户 / 会话 / 消息 / 危机审计 / 长期记忆（SQLAlchemy ORM，统一走 PostgreSQL）。系统提示词**不存数据库**，见「系统提示词」章节。

### 表结构（业务表 7 张）

| 表 | 关键字段 | 说明 |
|---|---|---|
| `users` | `id`、`username`(唯一)、`password_hash`(bcrypt)、`role`(user/admin)、`is_active` | 登录账号 + RBAC；小程序通道为外部 userId 的镜像行（不可登录，username=`mp_<sha1>`），`id` 列宽 64 |
| `sessions` | `id`、`user_id`(索引)、`title`、`created_at`、`updated_at` | 一次完整对话；messages 级联删除 |
| `messages` | `id`(自增)、`session_id`(FK, CASCADE, 索引)、`role`(human/ai)、`content` | 单条消息 |
| `crisis_audit` | `user_id`、`session_id`、`crisis_level`、`keywords_found`(JSON)、`question`、`response`、`detect_method`、`confidence` | 危机命中审计（合规留痕） |
| `user_chat_history` | `user_id`(索引)、`query`、`answer`、`embedding`(Vector)、`qa_embedding`(Vector) | 长期记忆（双向量） |
| `user_family_profile` | `user_id`(PK)、`user_nickname`、`birthday`(Date)、`parent_role`、`created_at`、`updated_at` | 小程序通道家庭档案主表（1 用户 1 行） |
| `user_children` | `user_id`(FK, CASCADE) + `child_id`(复合 PK)、`child_nickname`、`child_birth_date`(Date)、`gender` | 家庭档案的孩子子表（register/update 整份全量替换） |

> 约定：库内 `Message.role` 只存 `human`/`ai`；前端显示用 `user`/`assistant`，映射在加载/导入边界处理。`QueryRequest.messages[].role` 兼容四种取值，最后一条须为用户问题。`crisis_audit.keywords_found` 在库中为 JSON 编码字符串。老库 `users.id` / `sessions.user_id` / `crisis_audit.user_id` 由启动迁移自动扩到 `VARCHAR(64)`（幂等，PG only）。

### 常用运维（PostgreSQL）

```powershell
# 查看各表行数
psql -U postgres -d rag_psychology -c "\dt" -c "SELECT 'sessions' t, COUNT(*) FROM sessions UNION ALL SELECT 'messages', COUNT(*) FROM messages;"

# 全量备份（含 pgvector 向量，输出到 backups/）
python scripts/backup_pg.py

# 清空全部数据（保留表结构；重启后自动重建 legacy/admin 账号，
# 向量库需重新导入知识卡片）
psql -U postgres -d rag_psychology -c "TRUNCATE users, sessions, messages, crisis_audit, user_chat_history, langchain_pg_embedding, langchain_pg_collection RESTART IDENTITY CASCADE;"
```

> 切换 embedding 模型不影响关系库（只与向量库绑定），更换后旧向量静默失效，需 `--reset` 重新导入。

---

## 测试与压测

**并发压测**（`scripts/concurrency_pressure_test.py`，boundary / steady / spike 三模式，真实调 LLM 注意 API quota）：

```powershell
# 边界：单轮 N 个请求验证 20 活跃 + 40 排队准入（61 → 恰 1 个 AI_QUEUE_FULL）
python scripts/concurrency_pressure_test.py boundary --level 61

# 稳态：固定并发持续施压（如 20 并发 600 秒）
python scripts/concurrency_pressure_test.py steady --concurrency 20 --duration 600

# 尖峰：多轮同步突发（每轮 level 个请求，验证每轮拒绝数）
python scripts/concurrency_pressure_test.py spike --level 80 --rounds 10
```

压测命令会按目标并发数自动注册独立测试账号（`--user-prefix` / `--password`），并按 `--endpoint query|stream|mixed` 与 `--persist/--no-persist` 区分链路。运行结束输出 JSON 报告与 PASS/FAIL 判定（`--output` 指定路径）。鉴权 / 越权 / 数据隔离由 `api/deps.py` 的 JWT/RBAC 依赖 + `db/crud_async.py` 的归属校验在代码层保证（见「接口」章节）。

**检索质量诊断**：`python scripts/diagnose_retrieval.py`——把知识库全量文档直接交给重排器打分，定位"检索相似度低"的根源（重排模型异常 vs 知识库缺内容）。

---

## 常见问题

1. **缺模块**：`pip install -r requirements.txt`。主链路无需 `unstructured`。
2. **问答无结果**：依次检查——知识库是否导入成功（`SELECT COUNT(*) FROM langchain_pg_embedding;`）；`OPENAI_API_KEY` / `OPENAI_API_BASE` 是否有效；`RAG_ENABLED` 是否为 `True`（默认纯对话模式不检索）；开了 `RERANK_MIN_SCORE` 后结果变少则调低或设 0。
3. **数据重复**：`import_cards.py` 对重复 `card_id` 报错；重复运行会重复写入，建议 `--reset` 重建。
4. **RAG/安全默认关闭**：`config/settings.py` 里 `RAG_ENABLED` / `SAFETY_ENABLED` 默认 `False`，需要完整能力时改为 `True` 重启（启动时才会加载重排模型与语义锚点）。
5. **端口被占用**：本机 8000 常被占用，用 `--port 8001` 启动（前端与接口同端口）。

---

## 免责声明

本项目适合做本地知识问答原型。若用于真实心理咨询场景，安全检测、危机升级通道、内容审核与人工复核必须再做专业化加固；本系统不构成任何专业医疗或心理诊断建议。
