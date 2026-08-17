# 心理知识问答 RAG 项目（青少年 / 家庭方向）

本地运行的检索增强生成（RAG）系统，面向青少年及家庭心理知识问答。知识以结构化「卡片」形式入库，检索后由大模型基于卡片内容生成回答，并内置关键词级危机干预检测。

技术栈：FastAPI + LangChain + Chroma（本地持久化向量库）+ OpenAI 兼容接口（默认使用通义千问 / DashScope 兼容模式）。

## 快速开始

```powershell
# 1. 创建并激活 conda 环境（建议 Python 3.10+）
conda create -n rag python=3.11 -y
conda activate rag
pip install -r requirements.txt

# 2. 配置（复制后填入 API Key）
Copy-Item .env.example .env

# 3. 导入知识库（把 JSONL 换成你自己的文件）
python scripts/import_cards.py "你自己的文件" --reset

# 4. 启动
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

启动后访问 `http://127.0.0.1:8000`。

## 项目结构

```text
api/main.py              FastAPI 接口（/api/health、/api/query、/api/system-prompt、/api/sessions、/api/compare-history）
config/                 settings.py、危机关键词 crisis_keywords.json、系统提示词 json
db/                     SQLAlchemy 关系库：models（5 张表）、crud（读写）、__init__（引擎/建表）
modules/                RAG 核心、向量库封装、安全检测、提示词存储 prompt_store（SQLite 后端）
frontend/               纯静态前端页面（系统提示词管理，由 FastAPI 托管）
scripts/import_cards.py 离线 JSONL 卡片导入
data/                  运行时生成的数据目录（已被 .gitignore 忽略）：
                         - data/chroma/             本地向量库（Chroma）
                         - data/rag_psychology.sqlite3  关系库（SQLite）
test_rag.py             端到端自测脚本
requirements.txt        Python 依赖
```

## 配置

复制 `.env.example` 为 `.env` 并填入密钥与必要的环境项（**.env 含密钥，勿提交**，已被 .gitignore 忽略）。

**配置分层原则**：`.env` 只放「密钥 + 随部署环境变化、需要覆盖默认值的值」；所有检索 / 生成 / 安全 / 服务等调参都有合理默认值，写在 `config/settings.py`（含中文注释），不填也能跑。

`.env` 通常只要三行（其余一律走默认值）：

```dotenv
OPENAI_API_KEY=你的API密钥
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1  # OpenAI 兼容接口
CHAT_MODEL=qwen3.6-flash       # 对话模型
```

## 构建知识库

主链路仅支持 JSONL 卡片导入（见 `scripts/import_cards.py`），不依赖 `unstructured` / `pypdf` / `docx2txt`；如需扩展 PDF / Markdown 导入请自行补充依赖与加载逻辑。

```powershell
python scripts/import_cards.py "data/your_knowledge_base.jsonl" --reset
```

参数：

- `cards_path`（位置参数）：JSONL 路径，每行一条记录。
- `--reset`：先删集合再导入（避免重复）。
- `--review-status approved`：只导指定审核状态（**安全敏感场景建议只导 `approved`**）。
- `--batch-size 50`：分批写入，默认 50。

导入把每条记录转成一个文档，写入的 metadata 含 `card_id` / `source_id` / `chunk_id` / `title` / `domains` / `audiences` / `age_stages` / `risk_level` / `evidence_level` / `review_status` / `age_group` / `source` / `filename`，供来源展示与后续扩展使用。

> ℹ️ **`age_group` 仅作数据留档**：该字段在导入时由 `age_stages` 中文标签归一化得到，保留在卡片元数据中供后续扩展；**当前检索与回答不使用年龄过滤/语气适配**（知识库未按年龄充分分类，硬过滤会误杀卡片）。

## 年龄分层（数据留档）

导入时按 `age_stages` 映射到分桶（`age_group`），仅写入元数据：

| age_stages 含 | age_group |
|---|---|
| 婴儿 / 幼儿 / 儿童 / 0-2 / 3-6 | `child` |
| 少年 / 中小学 / 小学生 / 初中生 / 中学生 | `early_teen` |
| 青少年 | `teen` |
| 青年 / 高中生 / 职高 / 大学 | `late_teen` |

无法识别的标签映射为空。该字段只随卡片留存，不参与检索过滤与回答生成。

## 启动

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

- 接口：`http://127.0.0.1:8000`
- Swagger：`http://127.0.0.1:8000/docs`
- 前端：打开 `http://127.0.0.1:8000` 后先进入登录页（注册 / 登录），登录后使用提示词工作台。

## 登录系统测试

`python scripts/test_auth.py`（TestClient 直测，无需起服务；自动关闭重排/语义预热）。
覆盖：无 token 401、水平越权 403（A 访问 B 的会话/提示词/对比记录）、垂直越权 403
（普通用户访问 `/api/admin/*`）、请求体篡改 `user_id` 403、A 写入数据 B 查不到、
提示词越权窃取 403、登录失败锁定 429。共 42 项断言。

## 接口

> 自 v1.1.0 起，除 `/api/auth/register`、`/api/auth/login`、`/api/health` 外**全部接口需要登录**：
> 请求头携带 `Authorization: Bearer <token>`（登录返回的 `access_token`）。
> - 无 token / 无效 / 过期 → `401`
> - 角色不足（普通用户访问 `/api/admin/*`）→ `403`
> - 访问他人资源（会话 / 提示词 / 对比记录）→ `403`（水平越权防护）
> - 请求体携带的 `user_id` 一律忽略，身份以 token 为准（篡改无效）

### 认证接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/auth/register` | 注册普通用户：`{username, password, display_name?}`；用户名 3-32 位字母/数字/下划线、密码 ≥8 位；冲突 409、不合规 400 |
| POST | `/api/auth/login` | 登录：`{username, password}` → `{access_token, expires_in, user}`；连续失败 5 次锁定 15 分钟（429） |
| GET | `/api/auth/me` | 当前用户信息（token 有效性校验） |

初始管理员：首次启动 `users` 表为空时自动创建，用户名/密码来自环境变量
`INIT_ADMIN_USERNAME`（默认 `admin`）/ `INIT_ADMIN_PASSWORD`（默认 `admin123456`，**本地原型专用，生产务必改**）。
历史数据（无归属的会话/审计/提示词）在启动时自动归入不可登录的 `legacy` 账号，保证数据隔离。

管理员接口：`GET /api/admin/users`（用户列表）、`GET /api/admin/crisis-audit`（危机审计查看），普通用户访问返回 403。

> 认证配置见 `config/settings.py`「认证与授权」一节：`JWT_SECRET`（生产必须用
> `openssl rand -hex 32` 生成并写入 `.env`）、`JWT_EXPIRE_MINUTES`（默认 120）。
> 密码使用 bcrypt（cost 12）哈希存储，库中不存明文。

### GET /api/health

```powershell
curl http://127.0.0.1:8000/api/health
# => {"status":"healthy","version":"1.0.0"}
```

> 只确认服务已启动，不校验知识库是否已导入。返回 healthy 但问答无结果时，请先执行导入。

### POST /api/query

请求体：

```json
{ "question": "孩子总是情绪低落、没兴趣，家长该怎么做？" }
```

- `question`（必填）。

返回：

```json
{
  "answer": "……",
  "sources": [{ "index": 1, "card_id": "KB-000001", "title": "……", "source_id": "SRC-202607-0001", "risk_level": "normal" }],
  "safety_note": null,
  "is_crisis_response": false,
  "safety_check": null
}
```

- 命中中/低危关键词时，`safety_note` 附带关怀与求助资源，`safety_check` 含等级与命中关键词明细。
- 命中**高危**时，`is_crisis_response=true` 且 `answer` 直接为危机干预提示，不再走常规检索。

## 前端页面与系统提示词管理

启动服务后（`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`），直接访问根路径即可打开前端页面：

```powershell
# 打开浏览器访问
Start-Process http://127.0.0.1:8000/
```

前端为纯静态页面（`frontend/`，由 FastAPI 直接托管，无需额外构建），是一个 **PromptLab 风格的提示词工作台**（设计/交互参考 `xinli` 项目），包含三个页面：

1. **提示词管理**：不区分儿童 / 少年 / 青少年 / 青年。`config/system_prompt.json` 现在是一个**提示词库**（`prompts[]`），支持新增、重命名、删除、编辑、设置激活提示词；所有提示词都在左侧列表展示，不会互相覆盖。中间是编辑器，右侧是实时渲染预览。点击「保存并同步」（`Ctrl/Cmd + S`）把当前提示词写入文件，「设为激活」指定 RAG 默认使用的提示词，「还原默认」一键复位。
2. **对话联调**：多会话（新建 / 重命名 / 删除 / 导出），直接调用后端 `/api/query` 进行 RAG 问答。右侧 Inspector 可选择使用哪条提示词；默认使用「激活提示词」。
3. **提示词对比**：输入同一问题，A/B 双栏可分别选择不同提示词（A 可选出厂默认库，B 可选当前编辑库）跑 RAG，对照答案与引用来源，并保存对比历史（可点击恢复）。

相关接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/system-prompt` | 返回 `{ current, default }` 两套配置（均含 `prompts[]` 与 `activeId`） |
| PUT | `/api/system-prompt` | 支持 `prompts` / `activeId` / `add` / `update` / `deleteId` 多种操作 |
| POST | `/api/system-prompt/reset` | 还原为出厂默认库 |
| POST | `/api/query` | RAG 问答；可选 `system_prompt_override`（不落盘覆盖）或 `prompt_id`（使用库中指定提示词） |

> **Postman / 外部调用**：直接 `POST http://127.0.0.1:8000/api/query`，`Content-Type: application/json`，请求体 `{"question": "..."}` 即可。前端只在浏览器访问根路径 `/` 时出现，不影响接口调用，无需单独端口。

## 安全与危机干预

`modules/safety_checker.py` 在每次问答前做**两级检测**（`SAFETY_CHECK_ENABLED` 总开关）：

**L0 关键词快检**（`check_text`，毫秒级）：关键词与等级定义在 `config/crisis_keywords.json`，命中后判定 `high` / `medium` / `low` / `none`。命中 high 但属**求助型提问**（孩子/朋友等称谓 + 怎么办/帮助等动作词，如"孩子有自伤倾向怎么办"）时自动降级为 medium，避免把家长/老师的求助误当成危机实施者拦截。

**L1 语义检测**（`semantic_check`，高危意图锚点距离）：隐喻/隐晦表达（如"想用脑袋和房梁比赛"=上吊）字面无关键词，靠关键词必然漏报。L1 把种子集（`config/high_risk_intents.json`，标准意图句 + 隐喻变体，覆盖自杀/自伤/伤害他人/被伤害四簇）embed 后作为**锚点集合**（簇内保留每条句子向量，非均值原型），用户问题与所有锚点算余弦距离、取最近：

- 距离 ≤ `CRISIS_INTERCEPT_DIST`（默认 0.25）→ 高危拦截；
- 距离 ≤ `CRISIS_GRAY_DIST`（默认 0.36）→ 疑似，附关怀 + 转介（不拦截）；
- 其余 → 放行。

embedding 复用检索阶段的那次 API 调用（`EMBED_CACHE_SIZE` 进程内 LRU 缓存），**不增加额外 API 成本**。锚点向量缓存到 `data/crisis_prototypes.json`，种子文件变更自动重建；启动时后台预热，未就绪时问答自动回退关键词（不阻塞）。

**回答侧复查**（`review_answer`）：LLM 生成的回答也会跑一遍 L0 关键词复查，命中高危时在末尾**追加**安全提醒（不替换原文——正常科普回答常含敏感词），并以 `detect_method=answer_check` 记入审计。

**响应**：高危直接返回危机干预提示与热线（不再走常规回答）；中/低危正常检索回答，末尾附 `safety_note` 关怀提示与求助热线。热线号码来自 `config/crisis_keywords.json` 的 `hotlines` 字段。所有命中（含"疑似但未拦截"的灰区）写入 `crisis_audit` 表，并记录 `detect_method`（keyword/semantic）与 `confidence`（语义距离），审计可追溯检测来源。

**阈值标定**：`python scripts/calibrate_crisis_thresholds.py --top 5` 对种子集正例 + 内置负例计算距离分布，输出建议阈值（实测正例 0.095~0.321、负例 0.339+，两组完全分离）。新增的隐喻表达请回填 `high_risk_intents.json` 的 `variants`，召回率随积累单调上升。

> ⚠️ 这是**原型级**防护：关键词 + 语义原型无法 100% 识别所有隐晦危机表达（灰区误报/漏报依然存在），不能替代专业心理危机干预。真实场景需引入 LLM 精判、人工审核与升级机制。

## 知识库数据格式

JSONL 每行一条记录：

```json
{
  "card_id": "KB-000001",
  "source_id": "SRC-202607-0001",
  "chunk_id": "SRC-202607-0001-C001",
  "card_json": {
    "title": "0-2岁婴儿情绪教养：情感回应与安抚技巧",
    "domains": ["情绪困扰", "亲子沟通"],
    "audiences": ["家长"],
    "age_stages": ["儿童"],
    "scenario": "……", "applicable_conditions": "……",
    "clarifying_questions": ["……"], "possible_explanations": ["……"],
    "actions": [{ "step": "……", "frequency": "……", "duration": "……", "observe": "……", "safety_note": null }],
    "do_not_use_when": ["……"], "referral_conditions": ["……"],
    "risk_level": "normal", "evidence_level": "experiential"
  },
  "review_status": "pending_review", "review_reason": "",
  "created_at": "2026-07-22T07:33:51+00:00", "updated_at": "2026-07-22T07:33:51+00:00"
}
```

导入脚本把 `card_json` 字段拼成可读文本做向量化，结构化字段写入 metadata。

## 本地重排序

检索默认启用本地 Cross-Encoder 重排（`bge-reranker-v2-m3`）：召回候选后按「问题 × 文档」逐对打分精排取 top3，替代仅按向量分数截断的假重排。模型加载失败/异常时**自动回退**到原排序逻辑，不影响检索可用性。服务启动时后台预热模型（约 5-10 秒，不阻塞启动），首次问答不卡顿；前端耗时栏会显示「重排」耗时。

模型下载（放到 `data/rerank_models/bge-reranker-v2-m3`，该目录已被 `.gitignore` 忽略）：

```powershell
# 方式一：ModelScope（国内速度快）
pip install -U modelscope
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir "data/rerank_models/bge-reranker-v2-m3"

# 方式二：HuggingFace 镜像
pip install -U huggingface_hub
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir "data/rerank_models/bge-reranker-v2-m3"
```

依赖：`sentence-transformers` + torch（有 NVIDIA GPU 建议装 CUDA 版：`pip install torch --index-url https://download.pytorch.org/whl/cu124`，重排从秒级降到几十毫秒）。相关配置见 `config/settings.py` 的「本地重排序」一节：`RERANK_ENABLED` / `RERANK_MODEL` / `RERANK_DEVICE` / `RERANK_BATCH_SIZE` / `RERANK_MAX_LENGTH` / `RERANK_MIN_SCORE`（可选最低分数护栏，默认 0=不启用；bge 分数 0~1，本项目实测相关约 0.4+、无关 <0.15，建议 0.2~0.3 起试；低于阈值的候选被丢弃，全部丢弃时回答会提示"没有足够信息"防编造）。

## 本地向量库

- 向量由 `EMBEDDING_MODEL` 生成，文档来自 JSONL 结构化卡片（一卡一文档），持久化在 `data/chroma/`（统一收在 data/ 下），重启后仍可用。
- 重建：先 `--reset` 再重新导入。

> ⚠️ 向量与 `EMBEDDING_MODEL` 绑定：更换 embedding 模型后旧向量会**静默失效**（不报错但检索质量崩坏），必须 `--reset` 重新导入。

## 关系型数据库（SQLite）

本项目有**两层持久化**，分工明确：

- **Chroma（向量库）**：只负责语义检索——把知识卡片向量化、按相似度召回相关片段。
- **SQLite（关系库）**：只负责结构化留痕——会话 / 消息 / 危机审计 / 提示词库 / 对比历史。文件为 `data/rag_psychology.sqlite3`（单文件、零部署）。

两者互不替代：RAG 检索走 Chroma，对话记录与配置走 SQLite。未来若要上多 worker / 生产环境，把 `settings.DB_URL` 改成 `mysql+pymysql://user:pwd@host/db` 即可，**业务代码无需改动**（SQLAlchemy 已做抽象）。

### 配置

| 配置项 | 位置 | 默认值 | 说明 |
|---|---|---|---|
| `DB_PATH` | `config/settings.py` | `data/rag_psychology.sqlite3` | 库文件相对项目根的路径 |
| `DB_URL` | `config/settings.py` | `sqlite:///<DB_PATH>` | SQLAlchemy 连接串 |

> 文件后缀曾为 `.db`，后统一改为 `.sqlite3`（两者格式完全相同，仅命名习惯）。`.gitignore` 已忽略整个 `data/` 目录（同时覆盖 `data/chroma/` 向量库与 `*.db`/`*.sqlite3` 关系库），两库均不会被提交；备份请用 `data/` 整体拷贝。

### 代码结构（`db/` 包）

```text
db/
├── __init__.py   引擎创建 + init_db() 幂等建表；get_db() 在 crud 中
├── models.py     5 张表的 ORM 定义（见下表）
└── crud.py       get_db() 上下文管理器 + 各表读写函数
modules/
└── prompt_store.py  提示词库读写（SQLite 后端，替代原 system_prompt.json 数据源）
```

- `db/__init__.py`：对 SQLite 关闭 `check_same_thread`（FastAPI 用线程池跑同步 DB 调用）；`init_db()` 用 `create_all` 建表，幂等——启动时会自动补建缺失的表，旧库升级时**无需手动迁移**。
- `db/crud.py`：`get_db()` 是上下文管理器，退出自动 commit、异常回滚、始终关闭连接。

### 数据表（5 张）
sessions：聊天会话（浏览器一个标签页 = 1 次对话）
messages：会话里面一条一条的用户 / AI 消息
crisis_audit：心理风险命中日志（合规留痕，重中之重，心理产品强制审计）
prompts：系统提示词模板，支持切换 RAG 默认 Prompt
compare_history：AB 测试记录，用来对比两套大模型返回结果
| 表 | 字段 | 说明 |
|---|---|---|
| `sessions` | `id`(PK, String36, 缺省 `uuid4().hex`)、`title`(String255, 默认"新会话")、`created_at`、`updated_at` | 一次完整对话；`messages` 级联删除 |
| `messages` | `id`(PK, int 自增)、`session_id`(FK→sessions.id, `ON DELETE CASCADE`, 索引)、`role`(String20: `human`/`ai`)、`content`(Text)、`created_at` | 单条消息 |
| `crisis_audit` | `id`、`session_id`(可空, 索引)、`crisis_level`(high/medium/low)、`keywords_found`(Text, JSON)、`question`、`response`(可空)、`is_crisis_response`(bool)、`created_at`(索引) | 危机命中审计留痕（合规可追溯） |
| `prompts` | `id`(PK, String36)、`name`(String255)、`content`(Text)、`is_active`(bool)、`created_at`、`updated_at` | 提示词库 |
| `compare_history` | `id`(PK, int 自增)、`input`(Text)、`result_a`(Text, JSON, 可空)、`result_b`(Text, JSON, 可空)、`created_at`(索引) | 提示词对比历史 |

> 约定：`Message.role` 在库中存 `human` / `ai`；前端显示用 `user` / `assistant`，映射在恢复 / 导入边界处理。`QueryRequest` 接受 `human`/`ai`/`user`/`assistant` 四种 role。`crisis_audit.keywords_found` 与 `compare_history.result_a/result_b` 在库中均为 **JSON 编码的字符串**。

### 启动行为

`api/main.py` 的 `startup_event` 依次执行：

1. `settings.validate()`：校验 `OPENAI_API_KEY` 已配置。
2. `init_db()`：确保目录存在 + `create_all` 建表（缺失即补，幂等）。
3. `ensure_prompts_seeded()`（`modules/prompt_store.py`）：
   - 若 `prompts` 表为空，则**优先从 `config/system_prompt.json` 迁移**，否则用出厂默认 `config/system_prompt.default.json` seed 出 4 条；
   - 多进程锁（`threading.Lock`）保护，重复启动不会重复 seed。

### 接口一览（关系库相关）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/query` | RAG 问答；`persist=true`（默认）时把本轮写入 `sessions`/`messages` 与（若命中）`crisis_audit` |
| `GET` | `/api/sessions` | 列出最近会话（含 `message_count`） |
| `GET` | `/api/sessions/{id}/messages` | 取某会话全部消息（按时间序） |
| `POST` | `/api/sessions` | 新建空会话，返回服务端生成的 `id` |
| `PATCH` | `/api/sessions/{id}` | 重命名会话 |
| `DELETE` | `/api/sessions/{id}` | 删除会话（级联删消息） |
| `GET` | `/api/compare-history` | 列出对比历史（含 A/B 完整结果） |
| `POST` | `/api/compare-history` | 新增一条对比历史（`a` / `b` 为完整结果对象） |
| `DELETE` | `/api/compare-history/{id}` | 删除一条对比历史 |
| `GET` | `/api/system-prompt` | 返回 `{ current, default }` 两套提示词配置 |
| `PUT` | `/api/system-prompt` | 更新提示词库（`prompts` / `activeId` / `add` / `update` / `deleteId`） |
| `POST` | `/api/system-prompt/reset` | 还原为出厂默认库（清空 `prompts` 表重新 seed） |

### `persist`

`/api/query` 的 `persist` 参数（默认 `true`）：

- **对话联调页**保留默认 `true`，正常把问答写入 `sessions`/`messages`。
- **提示词对比页**调用时传 `persist=false`，使临时对比问答**只进 `compare_history`，不污染会话表**。

> 早期 bug：对比页没传 `persist`，而 `/api/query` 对未带 `session_id` 的请求会生成随机 id 并**无条件落库**，导致每次对比都泄漏一个孤儿会话、窜进对话联调的历史列表。修复为显式 `persist` 开关后解决；不持久化时响应 `session_id` 返回 `None`（前端对话页从不读响应里的 `session_id`，安全）。

### 角色字段约定（再次强调）

- 库内 `Message.role` 只存 `human` / `ai`；
- 前端对话展示用 `user` / `assistant`，映射在加载 / 导入时处理；
- `QueryRequest.messages[].role` 兼容四种取值（`human`/`ai`/`user`/`assistant`），最后一条须为用户问题。

### 首次使用

首次启动服务时，无需手动建表或导入数据：

- `init_db()` 在启动事件中自动创建 `data/rag_psychology.sqlite3` 及全部 5 张表（`create_all` 幂等，旧库升级只补缺失的表，不破坏已有数据）；
- `ensure_prompts_seeded()` 在 `prompts` 表为空时自动 seed 出 4 条提示词（优先用 `config/system_prompt.json`，否则用出厂默认 `config/system_prompt.default.json`）。

启动后即可在前端直接开始对话、管理提示词、保存对比历史，无需额外初始化步骤。详细流程见上文「启动行为」。

### 常用运维

```powershell
# 查看库内容
sqlite3 data/rag_psychology.sqlite3 ".tables"
sqlite3 data/rag_psychology.sqlite3 "SELECT id, title FROM sessions;"

# 备份（直接拷文件，格式与扩展名无关）
Copy-Item data/rag_psychology.sqlite3 data/rag_psychology.backup.sqlite3

# 重置提示词为出厂默认（清空 prompts 表并重新 seed）
# POST /api/system-prompt/reset

# 清空全部结构化数据：删除库文件，重启服务会自动重建空库（含提示词 seed）
Remove-Item data/rag_psychology.sqlite3
```

> 切换 embedding 模型**不影响** SQLite；它只与 Chroma 向量绑定。SQLite 的提示词 / 会话数据可独立于知识库长期留存。

## 测试

导入知识库且 `.env` 配置有效后：

```powershell
python test_rag.py
```

脚本验证配置、发起常规问答与危机语句测试并打印安全明细（会真实调用模型与嵌入接口，需有效 API Key）。

## 常见问题

1. **缺模块**：`pip install -r requirements.txt`。主链路无需 `unstructured`。
2. **导入后无结果**：依次检查——知识库是否导入成功；`OPENAI_API_KEY` / `OPENAI_API_BASE` 是否有效；问题与卡片是否匹配；开了 `MIN_RELEVANCE_SCORE` 后结果变少则调低或设 `0`。
3. **数据重复**：`import_cards.py` 对文件内重复 `card_id` 报错；重复运行会重复写入，建议 `--reset` 重建或先清空 `data/chroma/`。

## 备注

本项目适合做本地知识问答原型。若用于真实心理咨询场景，安全检测、危机升级通道、内容审核与人工复核必须再做专业化加固；本系统不构成任何专业医疗或心理诊断建议。
