# 心理知识问答 RAG 项目（青少年 / 家庭方向）

一个**本地**运行的检索增强生成（RAG）系统，面向青少年及家庭心理知识问答。知识以结构化"卡片"形式入库，检索后由大模型基于卡片内容生成回答，并内置关键词级危机干预检测。

> 范围说明：当前知识卡片覆盖**婴儿至青年**阶段（含家长指导），并非严格限定 6–18 岁。系统的安全检测仅为原型级防护，详见文末"安全与危机干预"与"备注"。

技术栈：FastAPI + LangChain + Chroma（本地持久化向量库）+ OpenAI 兼容接口（本项目默认使用通义千问 / DashScope 兼容模式）。

## 项目结构

```text
api/                 FastAPI 接口（/api/health、/api/query）
config/             配置（settings.py）与危机关键词（crisis_keywords.json）
modules/            RAG 核心、向量库封装、安全检测
scripts/            离线导入脚本（import_cards.py）
data/               示例数据（samples/）
chroma_db/          本地持久化向量库（运行时生成，已被 .gitignore 忽略）
test_rag.py         端到端自测脚本
requirements.txt    Python 依赖
start.ps1           Windows 启动脚本
```

## 运行前准备

建议使用 **Python 3.11 或 3.12**（最低 3.10，脚本用到 3.10+ 语法），并在独立虚拟环境中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

主链路（JSONL 卡片导入）不需要 `unstructured` 等重型依赖。若要自行扩展 PDF / Markdown 导入，可参考 `modules/document_processor.py`（需额外安装 `unstructured`、`pypdf`、`docx2txt`），当前主链路未接线该模块。

## 配置

复制示例环境变量文件（**不要提交 `.env`**，里面含密钥，已被 `.gitignore` 忽略）：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，典型内容如下（本项目实际使用的通义千问 / DashScope 兼容配置）：

```dotenv
# OpenAI 兼容接口（此处为通义千问 DashScope 兼容模式）
OPENAI_API_KEY=你的API密钥
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模型
CHAT_MODEL=qwen3.6-flash
EMBEDDING_MODEL=text-embedding-v3

# RAG / 向量库
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=psychology_knowledge

# 检索策略
RETRIEVAL_TOP_K=5          # 召回后用于重排的候选数
RERANK_TOP_K=3             # 最终喂给模型的文档数（先多召回，再截断）
FETCH_K=10                 # 相似度检索的召回候选数
SEARCH_TYPE=similarity     # similarity(按相关性分数重排) 或 mmr(最大边际相关，兼顾多样性)
MMR_LAMBDA=0.5             # mmr 模式下多样性权重，0=最多样，1=最相关
MIN_RELEVANCE_SCORE=0.0    # 相关性分数下限，0 表示不启用；建议 0.2~0.35
CHAT_TEMPERATURE=0.3       # 事实/建议类问答，温度偏低以减少幻觉

# 安全
CRISIS_KEYWORDS_FILE=./config/crisis_keywords.json
SAFETY_CHECK_ENABLED=true

# 服务（仅绑本机，避免暴露到局域网）
HOST=127.0.0.1
PORT=8000
DEBUG=false
```

配置项说明：

- `OPENAI_API_KEY` / `OPENAI_API_BASE`：用于 Chat 与 Embedding 的 OpenAI 兼容接口。
- `CHAT_MODEL` / `EMBEDDING_MODEL`：对话与向量化模型。
- `CHROMA_PERSIST_DIR` / `COLLECTION_NAME`：本地数据库目录与集合名。
- `RETRIEVAL_TOP_K` / `RERANK_TOP_K` / `FETCH_K`：检索先多召回（`FETCH_K`），按相关性降序重排后只取 `RERANK_TOP_K` 条喂给模型。
- `MIN_RELEVANCE_SCORE`：低于该分数的候选会被剔除；设为 `0` 不启用（启用后若回答变少，可调低该值）。
- `SEARCH_TYPE=mmr`：开启最大边际相关，减少重复/雷同内容。
- `CHAT_TEMPERATURE`：生成温度，默认 `0.3`。
- `SAFETY_CHECK_ENABLED` / `CRISIS_KEYWORDS_FILE`：是否启用安全检测及其关键词配置。
- `HOST`：默认 `127.0.0.1`，**请勿改成 `0.0.0.0`** 暴露到局域网。
- `CHUNK_SIZE` / `CHUNK_OVERLAP`：仅对"可选"的 PDF/MD 文档处理器生效；JSONL 卡片导入为一卡一文档，不使用这两项。

## 如何构建数据库

推荐使用离线 JSONL 卡片文件导入：

```text
scripts/import_cards.py
```

导入命令示例（路径换成你自己的 `output_cards.jsonl`）：

```powershell
python scripts/import_cards.py "data/your_knowledge_base.jsonl" --reset
```

参数说明：

- `cards_path`（位置参数）：JSONL 文件路径，每行一条记录。
- `--reset`：先删除当前 Chroma 集合，再重新导入（避免重复）。
- `--review-status approved`：只导入指定审核状态的数据（**正式/安全敏感场景建议只导 `approved`**）。
- `--batch-size 50`：分批写入，默认 50。

导入逻辑把每条 JSONL 记录转换成一个文档，写入的 metadata 包括：

- `card_id`、`source_id`、`chunk_id`
- `title`
- `domains`、`audiences`、`age_stages`
- `risk_level`、`evidence_level`、`review_status`
- `age_group`：由 `age_stages` 中文标签自动归一化得到（见下节），用于年龄过滤
- `source`、`filename`：来源文件信息

> ⚠️ **年龄过滤依赖 `age_group` 元数据**：该字段是在最新版导入脚本中加入的。若你之前导入过旧数据，请务必加 `--reset` 重新导入，否则按 `age_group` 过滤时会漏掉未带该字段的旧卡片。

## 年龄分层说明

导入时按卡片的 `age_stages` 标签映射到以下分桶（`age_group`）：

| age_stages 含 | age_group |
|---|---|
| 婴儿 / 幼儿 / 儿童 / 0-2 / 3-6 | `child` |
| 少年 / 中小学 / 小学生 / 初中生 / 中学生 | `early_teen` |
| 青少年 | `teen` |
| 青年 / 高中生 / 职高 / 大学 | `late_teen` |

无法识别的标签映射为空（不参与年龄过滤）。API 的 `age_group` 参数取值即上表四个值；传入后检索按对应元数据过滤，且回答语气适配该年龄段。不传则不过滤、语气默认按 `teen`。

## 如何启动

Windows 下：

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

或使用启动脚本（其内部锁定 `127.0.0.1`）：

```powershell
.\start.ps1
```

启动后：

- 接口地址：`http://127.0.0.1:8000`
- Swagger 文档：`http://127.0.0.1:8000/docs`

## 接口说明

系统提供以下两个接口。

### 1. 健康检查

`GET /api/health`

```powershell
curl http://127.0.0.1:8000/api/health
```

返回：

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

> 注意：`/api/health` 只确认服务已启动，不校验知识库是否已导入。若返回 healthy 但问答无结果，请确认已执行导入。

### 2. 问答接口

`POST /api/query`

请求体：

```json
{
  "question": "孩子总是情绪低落、没兴趣，家长该怎么做？",
  "age_group": "teen"
}
```

字段说明：

- `question`（必填）：用户问题。
- `age_group`（可选）：`child` / `early_teen` / `teen` / `late_teen`，详见"年龄分层说明"。

返回示例（无危机时）：

```json
{
  "answer": "......",
  "sources": [
    {
      "index": 1,
      "card_id": "KB-000001",
      "title": "0-2岁婴儿情绪教养：情感回应与安抚技巧",
      "source_id": "SRC-202607-0001",
      "risk_level": "normal"
    }
  ],
  "safety_note": null,
  "is_crisis_response": false,
  "safety_check": null
}
```

返回示例（命中危机关键词、中低危时）：

```json
{
  "answer": "......",
  "sources": [  ],
  "safety_note": "⚠️ **安全提示**\n......专业支持资源：\n• 全国心理援助热线: 400-161-9995\n......",
  "is_crisis_response": false,
  "safety_check": {
    "is_crisis": true,
    "level": "medium",
    "keywords_found": [{ "keyword": "被霸凌", "level": "medium" }],
    "safety_response": { "level": "medium", "message": "......", "hotlines": {  }, "should_intervene": true }
  }
}
```

字段说明：

- `answer`：基于检索卡片生成的回答。
- `sources`：引用来源，含 `card_id` / `title` / `source_id` / `risk_level`。
- `safety_note`：中/低危时附带的关怀与求助资源提示；无危机时为 `null`。
- `is_crisis_response`：是否为高危危机的直接响应（此时 `answer` 即为危机干预提示）。
- `safety_check`：危机检测的完整明细（是否危机、等级、命中关键词、热线等）；无危机时为 `null`。

## Postman 调用方式

- 健康检查：`GET http://127.0.0.1:8000/api/health`
- 提问：`POST http://127.0.0.1:8000/api/query`，Headers `Content-Type: application/json`，Body raw JSON：

```json
{
  "question": "考试前焦虑、失眠怎么办？"
}
```

## 安全与危机干预

`modules/safety_checker.py` 在每次问答前做关键词检测（可在 `.env` 用 `SAFETY_CHECK_ENABLED` 关闭）：

- 命中关键词后判定等级 `high` / `medium` / `low` / `none`，关键词与等级定义在 `config/crisis_keywords.json`。
- **高危（high）**：直接返回危机干预提示与热线（如 110/120 求助指引），不再走常规检索回答。
- **中/低危（medium/low）**：正常检索回答，并在末尾附上 `safety_note` 关怀提示与求助热线。
- 热线号码来自 `config/crisis_keywords.json` 的 `hotlines` 字段，可按需补充。

> ⚠️ 这是**原型级**防护：基于关键词，存在漏判与误判，不能替代专业心理危机干预。若用于真实场景，需引入更可靠的风险识别、人工审核与升级机制。

## 知识库数据格式

JSONL 文件每行一条记录，典型结构：

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
    "scenario": "......",
    "applicable_conditions": "......",
    "clarifying_questions": ["......"],
    "possible_explanations": ["......"],
    "actions": [
      {
        "step": "......",
        "frequency": "......",
        "duration": "......",
        "observe": "......",
        "safety_note": null
      }
    ],
    "do_not_use_when": ["......"],
    "referral_conditions": ["......"],
    "risk_level": "normal",
    "evidence_level": "experiential"
  },
  "review_status": "pending_review",
  "review_reason": "",
  "created_at": "2026-07-22T07:33:51+00:00",
  "updated_at": "2026-07-22T07:33:51+00:00"
}
```

导入脚本会把 `card_json` 的字段拼成一段可读文本做向量化，并把结构化字段写入 metadata 供过滤与来源展示。

## 本地数据库如何工作

项目使用 Chroma 作为本地持久化向量数据库：

- 向量由 OpenAI 兼容接口（`EMBEDDING_MODEL`）生成。
- 文档内容来自 JSONL 的结构化卡片（一卡一文档）。
- 数据持久化在 `chroma_db/`，重启服务后仍可继续使用。
- 重建数据库：先删除旧集合（`--reset`）再重新导入 JSONL。

> ⚠️ 向量与 `EMBEDDING_MODEL` 绑定：若更换 embedding 模型，旧向量会**静默失效**（不报错但检索质量崩坏）。更换模型后必须 `--reset` 重新导入。

## 运行测试

导入知识库且 `.env` 配置有效后，可跑端到端自测：

```powershell
python test_rag.py
```

脚本会验证配置、发起若干常规问答与危机语句测试，并打印命中关键词等安全明细。需保证已导入数据且 API Key 有效（测试会真实调用模型与嵌入接口）。

## 常见问题

### 1. 启动时报缺少模块

```powershell
python -m pip install -r requirements.txt
```

主链路无需 `unstructured`。若要扩展 PDF/MD 导入再单独补装相关依赖。

### 2. 导入后问答结果不对 / 没结果

依次检查：

- 知识库是否已成功导入（导入脚本有成功的写入日志）。
- `OPENAI_API_KEY` 与 `OPENAI_API_BASE` 是否有效（可访问、模型名正确）。
- 问题是否和知识卡片内容匹配。
- 若使用了 `age_group` 过滤但无结果：是否用最新脚本并 `--reset` 重新导入（旧数据无 `age_group` 字段）。
- 若开启了 `MIN_RELEVANCE_SCORE` 后结果变少：适当调低该值，或设为 `0` 关闭。

### 3. 数据重复

`import_cards.py` 会对文件内重复 `card_id` 报错。重复运行导入会重复写入，建议用 `--reset` 重建，或先清空 `chroma_db/`。

### 4. 不想暴露到局域网

启动时只绑定本机（默认 `.env` 的 `HOST=127.0.0.1`）：

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. 更换了 embedding 模型

必须 `--reset` 重新导入，否则旧向量与新模型维度/分布不匹配，检索质量会明显下降且无报错。

## 备注

本项目适合做本地知识问答原型。若用于真实心理咨询场景，安全检测、危机升级通道、内容审核与人工复核必须再做专业化加固；本系统不构成任何专业医疗或心理诊断建议。
