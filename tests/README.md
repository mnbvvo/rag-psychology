# rag-psychology 系统测试套件

面向登录系统（v1.1.0+）的专项测试目录。**已覆盖 7 个测试点，全部通过**：

| 测试点 | 脚本 | 断言/结果 | 结果 |
|---|---|---|---|
| 测试点 1：鉴权与越权 | `test_auth_security.py` | 42 项断言 | 42/42 通过 |
| 测试点 2：数据隔离 | `test_data_isolation.py` | 25 项断言 | 25/25 通过 |
| 测试点 3：并发安全 | `test_concurrency.py` | 18 项断言 | 进程内 18/18、真实服务 14/14 |
| 测试点 4：性能压测（Mock） | `test_performance.py` | 12 组阶梯加压 + 边界探测 | 0 失败；硬边界 512 并发 |
| 测试点 5：真实链路压测 | `test_performance.py --live` | 1-150 并发阶梯 | 0 失败；吞吐由模型 QPS 决定 |
| 测试点 6：数据备份与恢复 | `scripts/backup_pg.py` + `test_backup_restore.py` | 删库→恢复→逐表核对 | 11/11；完整 RTO 1.5s、一致性 8/8 |
| 测试点 7：超时与慢响应 | `test_timeout_slow.py` | mock LLM 挂起 60s 验证超时/兜底/重试/线程池 | 9/9 通过 |

## 目录结构

```text
tests/
├── README.md                       本说明（测试方法 / 用例矩阵 / 运行方式）
├── test_auth_security.py           测试点 1：鉴权与越权（无 token 401 / 越权 403 / 篡改 403）
├── test_data_isolation.py          测试点 2：数据隔离（接口 403 + 内容标记 + 数据层归属 + 并发）
├── test_concurrency.py             测试点 3：并发安全（并发登录/创建/写/问答，不丢/不串/不损坏）
├── test_performance.py             测试点 4+5：性能压测（默认 Mock 系统上限；--live 真实链路）
├── test_backup_restore.py          测试点 6：备份恢复演练（删库→恢复→核对，RPO/RTO）
├── test_timeout_slow.py             测试点 7：超时与慢响应（mock LLM 挂起，超时/兜底/重试/线程池）
├── cleanup_test_users.py           清理测试产生的账号与数据（防污染真实库）
├── postman/
│   ├── rag-psychology-auth.collection.json     Postman Collection（可直接导入）
│   └── rag-psychology-auth.environment.json    Postman 环境变量
├── results/                        自动化测试 JSON 报告
├── 鉴权与越权测试报告.md            测试点 1 报告（42/42）
├── 数据隔离测试报告.md              测试点 2 报告（25/25）
├── 并发安全测试报告.md              测试点 3 报告（18/18 / 14/14）
├── 性能压测测试报告.md              测试点 4 报告（Mock：0 失败，硬边界 512）
├── 真实链路压测测试报告.md          测试点 5 报告（真实 LLM：吞吐由模型决定）
├── 数据备份与恢复测试报告.md        测试点 6 报告（完整 RTO 1.5s，一致性 8/8）
└── 超时与慢响应测试报告.md          测试点 7 报告（9/9）
```

---

## 测试点 1：鉴权与越权（42 项断言）

### 测试方案（与需求一致）

用 Postman（或等价脚本）注册 A/B 两账号各拿 token，遍历所有受保护接口，分别用：

1. **无 token** → 断言 `401`
2. **A 的 token 访问 B 的资源**（水平越权）→ 断言 `403`
3. **普通 token 访问管理员接口**（垂直越权）→ 断言 `403`
4. **请求体篡改 user_id** → 断言 `403`（身份一律以 token 为准）
5. **数据隔离**：A 写入的数据 B 查不到（列表接口过滤）

### 受保护接口清单（全部要求登录）

| 接口 | 方法 | 认证 | 归属校验 |
|---|---|---|---|
| `/api/auth/me` | GET | ✅ | token 自身 |
| `/api/query` | POST | ✅ | session_id / prompt_id / user_id 篡改 |
| `/api/query/stream` | POST | ✅ | 同上 |
| `/api/sessions` | GET | ✅ | 列表只返回本人 |
| `/api/sessions` | POST | ✅ | 创建归属本人 |
| `/api/sessions/{id}` | PATCH / DELETE | ✅ | 非本人 → 403 |
| `/api/sessions/{id}/messages` | GET | ✅ | 非本人 → 403 |
| `/api/system-prompt` | GET / PUT | ✅ | 提示词按用户隔离，操作他人 id → 403 |
| `/api/system-prompt/reset` | POST | ✅ | 只重置本人 |
| `/api/compare-history` | GET / POST / DELETE | ✅ | 非本人记录 → 403 |
| `/api/admin/users` | GET | ✅ + admin 角色 | 普通用户 → 403 |
| `/api/admin/crisis-audit` | GET | ✅ + admin 角色 | 普通用户 → 403 |

### 用例矩阵（42 项断言）

| 分组 | 用例 | 断言 |
|---|---|---|
| 注册 | 注册 A / B | 201 |
| 注册 | 重复注册 A | 409 |
| 注册 | 弱密码 / 非法用户名 | 400 |
| 登录 | 登录 A / B / admin | 200 + 拿到 token |
| 登录 | 错误密码 | 401 |
| 无 token | 遍历 7 个受保护接口 | 401 |
| 水平越权 | B 读/改/删 A 的会话 | 403 |
| 水平越权 | B 删 A 的对比记录 | 403 |
| 水平越权 | B 更新/删除/激活 A 的提示词 | 403 |
| 水平越权 | B 用 A 的 session_id / prompt_id 调 `/api/query` | 403（LLM 前拦截） |
| 垂直越权 | 普通用户访问 `/api/admin/*` | 403 |
| 垂直越权 | 管理员访问 `/api/admin/*` | 200 |
| 篡改 user_id | A 请求体带 B 的 user_id | 403 |
| 数据隔离 | B 的会话/提示词/对比列表不含 A 数据 | 不包含 |
| 数据隔离 | A 的会话消息 B 读不到 | 403 |
| 数据隔离 | A 读自己的会话消息 | 200 |
| 登录锁定 | 连续失败 5 次 | 429 |

### 运行方式

#### 方式一：Postman（图形化，逐条可看）

1. 启动服务：`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`
2. Postman → Import → 选择 `tests/postman/rag-psychology-auth.collection.json`
3. Postman → 右上角环境选择器 → Manage Environments → Import → `rag-psychology-auth.environment.json`，并选中该环境
4. 打开 Collection → **按顺序 Run**（或用 Collection Runner：选中整个 Collection → Run，逐请求断言自动校验）
5. 每次运行会在环境变量中自动生成新的随机用户名（A/B），并自动保存 token、会话/提示词 id 供后续请求引用

#### 方式二：Newman（命令行跑 Postman Collection）

```bash
npx newman run tests/postman/rag-psychology-auth.collection.json \
  -e tests/postman/rag-psychology-auth.environment.json \
  --reporters cli,json --reporter-json-export results/auth-postman.json
```

#### 方式三：Python 自动化脚本（推荐，等价 Postman 行为）

```bash
# 默认打 http://127.0.0.1:8000（服务需已启动）
python tests/test_auth_security.py

# 指定地址 / 输出 JSON 报告 / 测试后自动清理测试账号
python tests/test_auth_security.py --url http://127.0.0.1:8000 --report tests/results/auth.json

# 进程内直测（无需起服务，自动关闭重排/语义预热）——适合 CI
python tests/test_auth_security.py --inprocess

# 关闭自动清理（保留测试账号便于人工复检）
python tests/test_auth_security.py --no-cleanup
```

> 管理员密码：脚本默认取 `settings.INIT_ADMIN_PASSWORD`（`.env` 默认 `admin123456`）。
> 若 admin 密码已手动改过（本项目当前为 `123456789`），请显式传入：
> `python tests/test_auth_security.py --admin-password 123456789`

### 预期结果

```text
通过 42 / 42
全部通过 ✓
```

---

## 测试点 2：数据隔离（25 项断言）

### 测试方案（与需求一致）

> 测试数据库读写是否会读取其他人的隐私：多账户登录，同时尝试在提示词中输入其他用户的 id 尝试窃取其他用户隐私。

### 四个层次验证（`tests/test_data_isolation.py`）

| 层次 | 验证内容 | 断言 |
|---|---|---|
| ① 接口层 | B 用 A 的 session_id / prompt_id / compare_id 操作（读/改/删/激活/完整替换列表/调 query） | 403 |
| ② 提示词窃取专项 | B 通过 update / deleteId / activeId / 完整替换列表 / `/api/query` 引用 A 的 prompt_id | 403 |
| ③ 内容层 | B 的**响应文本**不含 A 的私有标记 `PRIV_A_xxx`（防"能访问但泄露内容"） | 不含标记 |
| ④ 并发层 | A、B 同时（线程池 8 并发）创建会话并读列表 | 零交集 |
| ⑤ 数据层 | 直查当前数据库：B 名下 prompts / messages / compare_history 中均无 A 的标记内容；对照 A 名下确有 | 计数为 0 / >0 |

### 运行

```bash
# 打真实服务（默认 http://127.0.0.1:8000）
python tests/test_data_isolation.py

# 进程内直测（无需起服务，适合 CI）
python tests/test_data_isolation.py --inprocess

# 输出 JSON 报告
python tests/test_data_isolation.py --report tests/results/data-isolation.json
```

**验证结果**：进程内与真实服务模式均 **25/25 全部通过**。

---

## 测试点 3：并发安全（18 项断言）

### 测试方案（与需求一致）

> 用 mock 对同一资源并发操作；模拟多用户同时登录与问答时，验证进程内共享状态在并发下**不丢更新、不串号、不损坏**。

### 用例（`tests/test_concurrency.py`）

| 分组 | 用例 | 并发方式 | 断言 |
|---|---|---|---|
| A. 并发登录 | 8 用户同时登录，每个 token 调 me | 8 线程 | 用户名与账号一一对应（不串号） |
| B. 并发创建会话 | A/B 各并发建 5 个 | 16 线程 | 总数不丢、列表零交集 |
| C. 并发写提示词 | A 并发新增 5 条 | 8 线程 | 全部保留（不丢更新） |
| D. 并发越权写 | B 并发 10 次 update/delete A 的提示词 | 16 线程 | 全部 403 且 A 数据不损坏 |
| E. 并发问答 | A/B 并发 `/api/query`（**mock LLM/embedding**） | 8 线程 | 会话消息/审计归属不串 |
| F. 数据库层 | 并发后按 user 计数 | — | 与预期一致、无跨用户行 |

### 运行

```bash
# 推荐（并发问答需同进程 mock）
python tests/test_concurrency.py --inprocess        # 18/18

# 真实服务：并发问答（E）自动跳过
python tests/test_concurrency.py                    # 14/14 + 1 SKIP
```

> 说明：E 用例用 `unittest.mock` 替换 embedding 与 LLM（固定向量 + 固定回答），使并发问答完全本地化、稳定可复现；mock 需与 API 进程同进程，故仅 `--inprocess` 支持。

**验证结果**：进程内 18/18、真实服务 14/14 全部通过。

---

## 测试点 4：性能压测（Mock · 系统自身上限）

### 测试方案

**Mock embedding/LLM**（剔除外部噪音），同进程 uvicorn 使 mock 生效，测**单进程系统自身上限**：阶梯并发找拐点 + 倍增加压找边界。

### 运行

```bash
# ============ 阶梯模式（找拐点） ============
# 默认 1/10/50/100 并发，每级 6s，测 3 个接口
python tests/test_performance.py

# 自定义并发阶梯 / 时长 / 端口（可加更多档位细化曲线）
python tests/test_performance.py --levels 1,5,10,20,50,100,200 --duration 6 --port 8029

# 只测特定接口的密集阶梯（如只关心完整问答）
python tests/test_performance.py --levels 1,10,50,100 --duration 10

# 导出 JSON 报告（原始数据）
python tests/test_performance.py --report tests/results/performance.json

# ============ 边界探测模式（找系统上限） ============
# 并发按 1→2→4→…倍增直到失败：找硬边界（错误率>0）与软边界（P99 超阈值）
python tests/test_performance.py --boundary --duration 5

# 提高探测上限 / 调整软边界阈值（P99 ms）
python tests/test_performance.py --boundary --boundary-max 2048 --boundary-p99 8000

# 边界探测导出报告
python tests/test_performance.py --boundary --report tests/results/performance-boundary.json
```

### 结果摘要（2026-08-19，详见 `性能压测测试报告.md`）

- **0 失败**：100 并发下系统稳定、无错误响应；
- **单用户体验好**：1 并发时完整问答 ~38ms、流式 TTFT ~41ms；
- **拐点**：并发 ≥ 50 延迟显著劣化（system-prompt 14.8ms → 2271ms，153×）；
- **硬边界（边界探测）**：512 并发首次出现失败（208/976，~21%），连接/数据库连接池耗尽；
- **可靠上限**：256 并发内 0 失败（P99 5.3s）；
- **吞吐封顶**（单进程）：问答 ~26 RPS、流式 ~20 RPS，峰值 ~195 RPS（512 并发）；
- **瓶颈**：prompt_store 全局锁串行（system-prompt 读取）+ 单 worker asyncio（流式劣化最快）+ pgvector 无 HNSW 索引 + 连接池过小；
- **调优建议**（P0→P2）：pgvector HNSW 索引、prompt seed 状态内存缓存、连接池调大、多 worker + Redis 共享状态。

---

## 测试点 5：真实链路并发压测（`--live`）

### 测试方案

embedding / LLM 走**真实 API**（注意 token 费用），测**端到端**表现：LLM 生成时延占比、流式 TTFT、真实吞吐（受模型 QPS 限制）。与测试点 4（Mock 系统上限）互补：4 回答"系统自己能扛多少"，5 回答"真实用户并发下的端到端体验"。

### 运行

```bash
# 默认 1/2/5/10 并发（小阶梯控制费用），每级 6s
python tests/test_performance.py --live --duration 6

# 自定义并发阶梯（请求越多费用越高）
python tests/test_performance.py --live --levels 1,2,5,10,20 --duration 10

# 导出 JSON 报告
python tests/test_performance.py --live --report tests/results/performance-live.json
```

### 结果摘要（2026-08-20 思考模式全量重测，详见 `真实链路压测测试报告.md`）

> 配置：**所有模型开启思考模式**（`ENABLE_THINKING=True`），同步接口经同步流式实现思考，与流式行为一致。

- **0 失败**：1~**150** 并发全部成功（无 429、无超时）；
- **同步/流式时延一致**：低并发 4.2-5.2s（消除了之前流式比同步慢 3 倍的配置不一致）；
- **思考模式成本**：单请求 +65%（~2.8s → ~4.6s）、吞吐 -40%（模型输出 token 翻倍）；
- **吞吐平台期**：~19 RPS（150 并发；非思考为 ~28）；
- **容量规划（思考模式）**：**单进程建议上限 20 并发**（P99 < 6s）；50+ 需多 worker + Redis 或评估关闭思考；100 并发 P99 17s、150 并发 P99 25-31.5s 不可接受。

## 测试报告

| 报告 | 对应测试点 | 结论 |
|---|---|---|
| `鉴权与越权测试报告.md` | 测试点 1 | 42/42 通过 |
| `数据隔离测试报告.md` | 测试点 2 | 25/25 通过 |
| `并发安全测试报告.md` | 测试点 3 | 18/18（进程内）通过 |
| `性能压测测试报告.md` | 测试点 4 | 0 失败，拐点 50 并发，硬边界 512 |
| `真实链路压测测试报告.md` | 测试点 5 | 0 失败，吞吐由模型 QPS 决定 |
| `数据备份与恢复测试报告.md` | 测试点 6 | 完整 RTO 1.5s，一致性 8/8 |
| `超时与慢响应测试报告.md` | 测试点 7 | 9/9 通过 |

## 测试点 6：数据备份与恢复

### 测试方案

配置定时/全量备份 → 在测试环境**故意删库/清空表** → 用备份恢复 → 对比恢复前后逐表一致性；明确 RPO（备份后丢多少）与 RTO（恢复耗时）。

### 运行

```bash
# 1) 建立备份（每天定时跑此命令即可）
python scripts/backup_pg.py                 # 全量备份到 backups/（含 pgvector 向量表）

# 2) 灾难恢复演练（真的会 DROP 全部表再恢复，请确保已有备份）
python tests/test_backup_restore.py         # 备份→写 RPO 数据→删库→恢复→核对
```

### 结果摘要（2026-08-20，11/11 通过）

- **一致性**：删库→恢复后 8 张表（含 pgvector 向量表）逐表行数与备份快照完全一致；
- **RPO**：备份完成后新写入的数据全部丢失（符合预期；丢多少取决于备份频率）；
- **完整 RTO**：1.5s（数据恢复 0.4s + 服务重启 0.7s，1.7 MB 全库）；
- **应用**：恢复后 init_db 幂等、管理员账号可用，服务可直接启动；
- 备份能力：`backups/` 目录下 dump + 行数快照，Windows 任务计划程序可定时执行。

## 测试点 7：超时与慢响应

### 测试方案

用 mock 把 LLM 响应延迟调大（挂起 60s > 超时阈值），断言：① 请求在超时窗口内主动放弃；② 返回友好兜底；③ 慢请求不长期占用线程拖垮其他用户；④ 超时后重试正常。

### 运行（仅 --inprocess，mock 需同进程）

```bash
python tests/test_timeout_slow.py --inprocess        # 9/9
```

### 结果摘要（2026-08-21，9/9 通过）

- **超时主动放弃**：上游挂起 60s > 超时 8s → 请求 ~16s 内主动放弃（httpx read-timeout 真实生效），不无限挂起；
- **兜底**：返回 500 + "内部处理失败，请稍后重试。"（不泄露内部细节；建议后续升级为 LLM 超时专用 503 文案）；
- **重试**：SDK `max_retries` 自动重试，首次超时后重试成功（8.5s）；
- **不拖垮线程池**：8 个慢请求并发完成（15s vs 串行 120s），`run_in_threadpool` 架构下互不阻塞；
- **恢复能力**：上游恢复后链路立即正常（1.2s）。

## 测试账号处理

- 测试脚本默认自动清理产生的测试账号（`user_a_*`/`user_b_*`/`lock_*`/`test_*`/`isa_*`/`isb_*`/`cc*_*`/`perf_*` 等前缀）；
- 残留账号可随时执行：`python tests/cleanup_test_users.py`（不影响 admin / legacy）；
- 管理员账号：`admin` / 密码见 `.env` 的 `INIT_ADMIN_PASSWORD`（本项目当前为 `123456789`）。

## 后续扩展

- 测试点 8：SSE 断线重连（流式中断 → 不崩、落盘、重连上下文不丢、多路互不干扰）
- 测试点 9：危机检测（关键词 / 语义锚点 / 求助型降级 / 审计留痕）
- 测试点 10：多轮对话历史隔离（跨会话上下文是否会串入他人历史）
