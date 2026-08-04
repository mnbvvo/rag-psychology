# 心理知识问答 RAG 项目（青少年 / 家庭方向）

本地运行的检索增强生成（RAG）系统，面向青少年及家庭心理知识问答。知识以结构化「卡片」形式入库，检索后由大模型基于卡片内容生成回答，并内置关键词级危机干预检测。

> 范围说明：知识卡片覆盖**婴儿至青年**阶段（含家长指导），并非严格限定 6–18 岁。安全检测仅为原型级防护，详见文末「安全与危机干预」与「备注」。

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
python scripts/import_cards.py "C:\Users\Thunderobot\Desktop\knowledge_base_automation\out\output_cards.jsonl" --reset

# 4. 启动
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

启动后访问 `http://127.0.0.1:8000/docs`。

## 项目结构

```text
api/main.py              FastAPI 接口（/api/health、/api/query、/api/system-prompt）
config/                 settings.py、危机关键词 crisis_keywords.json、系统提示词 json
modules/                RAG 核心、向量库封装、安全检测、提示词存储 prompt_store
frontend/               纯静态前端页面（系统提示词管理，由 FastAPI 托管）
scripts/import_cards.py 离线 JSONL 卡片导入
data/samples/           示例数据
chroma_db/              本地向量库（运行时生成，已被 .gitignore 忽略）
test_rag.py             端到端自测脚本
requirements.txt        Python 依赖
```

## 配置

复制 `.env.example` 为 `.env` 并填入 `OPENAI_API_KEY` 等（**.env 含密钥，勿提交**，已被 .gitignore 忽略）。所有检索/生成参数均有合理默认值，不填也能跑；完整变量见 `.env.example`。

```dotenv
OPENAI_API_KEY=你的API密钥
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1  # OpenAI 兼容接口

CHAT_MODEL=qwen3.6-flash
EMBEDDING_MODEL=text-embedding-v3

CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=psychology_knowledge

RETRIEVAL_TOP_K=5          # 召回候选数
RERANK_TOP_K=3             # 最终喂给模型的文档数
FETCH_K=10                 # 相似度检索召回候选数
SEARCH_TYPE=similarity     # similarity 或 mmr（最大边际相关，兼顾多样性）
MMR_LAMBDA=0.5             # mmr 模式下多样性权重，0=最多样，1=最相关
MIN_RELEVANCE_SCORE=0.0    # 相关性下限，0=不启用（建议 0.2~0.35）
CHAT_TEMPERATURE=0.3       # 事实/建议类问答，温度偏低以减少幻觉

CRISIS_KEYWORDS_FILE=./config/crisis_keywords.json
SAFETY_CHECK_ENABLED=true

HOST=127.0.0.1
PORT=8000
DEBUG=false
```

> ⚠️ **安全提醒**：`HOST` 默认 `127.0.0.1`（仅本机）。**切勿改成 `0.0.0.0`**，避免暴露到局域网。

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

导入把每条记录转成一个文档，写入的 metadata 含 `card_id` / `source_id` / `chunk_id` / `title` / `domains` / `audiences` / `age_stages` / `risk_level` / `evidence_level` / `review_status` / `age_group` / `source` / `filename`，供过滤与来源展示。

> ⚠️ **年龄过滤依赖 `age_group` 元数据**：该字段在导入时由 `age_stages` 中文标签归一化得到（见下节）。若你之前导入过旧数据，请务必 `--reset` 重新导入，否则按 `age_group` 过滤会漏掉旧卡片。

## 年龄分层

导入时按 `age_stages` 映射到分桶（`age_group`）：

| age_stages 含 | age_group |
|---|---|
| 婴儿 / 幼儿 / 儿童 / 0-2 / 3-6 | `child` |
| 少年 / 中小学 / 小学生 / 初中生 / 中学生 | `early_teen` |
| 青少年 | `teen` |
| 青年 / 高中生 / 职高 / 大学 | `late_teen` |

无法识别的标签映射为空（不参与年龄过滤）。API 的 `age_group` 参数取上表四个值，传入后检索按对应元数据过滤，且回答语气适配该年龄段；不传则不过滤、语气默认 `teen`。

## 启动

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

- 接口：`http://127.0.0.1:8000`
- Swagger：`http://127.0.0.1:8000/docs`

## 接口

### GET /api/health

```powershell
curl http://127.0.0.1:8000/api/health
# => {"status":"healthy","version":"1.0.0"}
```

> 只确认服务已启动，不校验知识库是否已导入。返回 healthy 但问答无结果时，请先执行导入。

### POST /api/query

请求体：

```json
{ "question": "孩子总是情绪低落、没兴趣，家长该怎么做？", "age_group": "teen" }
```

- `question`（必填）、`age_group`（可选：`child` / `early_teen` / `teen` / `late_teen`）。

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

> 系统提示词原本硬编码在 `modules/rag_core.py`，现已外置为可编辑文件，由 `modules/prompt_store.py` 统一加载/组装。`config/system_prompt.default.json` 为出厂默认库（已提交，作对比基线，请勿手改其语义）；`config/system_prompt.json` 为用户态库（已被 `.gitignore` 忽略）。旧版单字段 `system_prompt` 结构会在首次读取时自动迁移为 `prompts[]`。前端所有状态（提示词选择、会话、对比历史）持久化在浏览器 `localStorage`。

相关接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/system-prompt` | 返回 `{ current, default }` 两套配置（均含 `prompts[]` 与 `activeId`） |
| PUT | `/api/system-prompt` | 支持 `prompts` / `activeId` / `add` / `update` / `deleteId` 多种操作 |
| POST | `/api/system-prompt/reset` | 还原为出厂默认库 |
| POST | `/api/query` | RAG 问答；可选 `system_prompt_override`（不落盘覆盖）或 `prompt_id`（使用库中指定提示词） |

> **Postman / 外部调用**：直接 `POST http://127.0.0.1:8000/api/query`，`Content-Type: application/json`，请求体 `{"question": "..."}` 即可。前端只在浏览器访问根路径 `/` 时出现，不影响接口调用，无需单独端口。

## 安全与危机干预

`modules/safety_checker.py` 在每次问答前做关键词检测（可用 `.env` 的 `SAFETY_CHECK_ENABLED` 关闭）：

- 命中后判定 `high` / `medium` / `low` / `none`，关键词与等级定义在 `config/crisis_keywords.json`。
- **高危**：直接返回危机干预提示与热线（如 110/120），不再走常规回答。
- **中/低危**：正常检索回答，末尾附 `safety_note` 关怀提示与求助热线。
- 热线号码来自 `config/crisis_keywords.json` 的 `hotlines` 字段，可按需补充。

> ⚠️ 这是**原型级**防护：基于关键词，存在漏判与误判，不能替代专业心理危机干预。真实场景需引入更可靠的风险识别、人工审核与升级机制。

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

## 本地向量库

- 向量由 `EMBEDDING_MODEL` 生成，文档来自 JSONL 结构化卡片（一卡一文档），持久化在 `chroma_db/`，重启后仍可用。
- 重建：先 `--reset` 再重新导入。

> ⚠️ 向量与 `EMBEDDING_MODEL` 绑定：更换 embedding 模型后旧向量会**静默失效**（不报错但检索质量崩坏），必须 `--reset` 重新导入。

## 测试

导入知识库且 `.env` 配置有效后：

```powershell
python test_rag.py
```

脚本验证配置、发起常规问答与危机语句测试并打印安全明细（会真实调用模型与嵌入接口，需有效 API Key）。

## 常见问题

1. **缺模块**：`pip install -r requirements.txt`。主链路无需 `unstructured`。
2. **导入后无结果**：依次检查——知识库是否导入成功；`OPENAI_API_KEY` / `OPENAI_API_BASE` 是否有效；问题与卡片是否匹配；用了 `age_group` 过滤却无结果需 `--reset` 重导；开了 `MIN_RELEVANCE_SCORE` 后结果变少则调低或设 `0`。
3. **数据重复**：`import_cards.py` 对文件内重复 `card_id` 报错；重复运行会重复写入，建议 `--reset` 重建或先清空 `chroma_db/`。

## 备注

本项目适合做本地知识问答原型。若用于真实心理咨询场景，安全检测、危机升级通道、内容审核与人工复核必须再做专业化加固；本系统不构成任何专业医疗或心理诊断建议。
