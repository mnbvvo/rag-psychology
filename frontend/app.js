// ============================================================
// 心理 RAG · 对话联调（SPA）
// 单页：多会话 + SSE 流式 RAG 问答
// 后端接本项目 FastAPI：/api/query/stream（SSE）、/api/sessions
// 提示词为全局激活项（prompts 表），由管理员在数据库直接配置，用户不可修改
// ============================================================
const defaultState = {
  sessions: [],                    // 全部来自服务端 SQLite（含 messages）
  activeSessionId: null,           // 当前打开的会话 id
};

// 所有业务数据（会话 / 消息）均持久化在服务端 SQLite，
// 前端不再使用浏览器缓存（localStorage）。state 仅作为当前页面运行时的内存镜像。
// token 例外：登录凭证必须存 localStorage（httpOnly cookie 方案需后端配 CORS 与 CSRF，本地原型不做）。
let state = structuredClone(defaultState);
let toastTimer;
let authMode = "login"; // login | register

// ---------------- 认证（JWT Bearer） ----------------
const _USERNAME_RE = /^[a-zA-Z0-9_]{3,32}$/;
const ICON_USER = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-7 8-7s8 3 8 7"/></svg>`;
const ICON_LOCK = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>`;
const ICON_SPARK = `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M12 2l1.6 5.4L19 9l-5.4 1.6L12 16l-1.6-5.4L5 9l5.4-1.6L12 2zM19 14l.9 2.6L22.5 18l-2.6.9L19 21.5l-.9-2.6L15.5 18l2.6-1.4L19 14z"/></svg>`;
const LOGO_SVG = `<svg viewBox="0 0 48 48" width="34" height="34" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <linearGradient id="heartFill" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0.98"/>
      <stop offset="1" stop-color="#ffffff" stop-opacity="0.78"/>
    </linearGradient>
  </defs>
  <path d="M24 40.5s-12.5-7.2-12.5-17.1c0-4.6 3.5-8.4 8-8.4 2.5 0 4.8 1.3 6 3.4 1.2-2.1 3.5-3.4 6-3.4 4.5 0 8 3.8 8 8.4 0 9.9-12.5 17.1-12.5 17.1z" fill="url(#heartFill)"/>
  <circle cx="24" cy="22" r="2.4" fill="#1D9E75"/>
  <circle cx="24" cy="22" r="4.5" fill="none" stroke="#1D9E75" stroke-width="1" opacity="0.45"/>
</svg>`;

function getToken() { return localStorage.getItem("rag_token") || ""; }
function setToken(t) { localStorage.setItem("rag_token", t); }
function clearToken() { localStorage.removeItem("rag_token"); state.currentUser = null; }

// 统一请求封装：自动带 Authorization，401 → 清 token 回登录页，403 → 抛"无权限"
async function apiFetch(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401) {
    clearToken();
    // 已在登录页则不重复渲染；否则回到登录视图（会话过期）
    if (!document.querySelector(".auth-card")) renderAuthView();
    const err = new Error("登录已过期，请重新登录");
    err.status = 401;
    throw err;
  }
  if (res.status === 403) {
    let detail = "无权限执行该操作";
    try { const j = await res.clone().json(); if (j.detail) detail = j.detail; } catch { /* noop */ }
    const err = new Error(detail);
    err.status = 403;
    throw err;
  }
  return res;
}

// 字段级错误提示（a11y：role=alert + aria-invalid）
function setFieldError(field, msg) {
  const errEl = document.querySelector(`#err-${field}`);
  const inputEl = document.querySelector(`#auth-${field}`);
  if (errEl) errEl.textContent = msg || "";
  if (inputEl) inputEl.setAttribute("aria-invalid", msg ? "true" : "false");
}

// 密码强度（注册模式）：长度/字母混合/数字/特殊 四维度，1~4 段
function updatePasswordStrength(pw) {
  if (authMode !== "register") return;
  const el = document.querySelector(".pw-strength");
  if (!el) return;
  let score = 0;
  if (pw.length >= 8) score++;
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) score++;
  if (/\d/.test(pw)) score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;
  el.dataset.level = String(score);
}

// 登录 / 注册视图（未认证时唯一界面）
function renderAuthView() {
  const root = document.querySelector("#app");
  const isRegister = authMode === "register";
  root.innerHTML = `
    <div class="auth-shell">
      <div class="auth-bg" aria-hidden="true">
        <span class="blob blob-a"></span>
        <span class="blob blob-b"></span>
        <span class="blob blob-c"></span>
      </div>
      <div class="auth-card" data-mode="${authMode}">
        <div class="auth-logo">${LOGO_SVG}</div>
        <h1 class="auth-title">心理 RAG</h1>
        <p class="auth-sub">${isRegister ? "创建账号，开始使用对话联调" : "青少年心理知识问答 · 温暖、专业、值得信任"}</p>
        <form id="auth-form" novalidate>
          <div class="auth-field">
            <label class="form-label" for="auth-username">用户名</label>
            <div class="input-wrap">
              <span class="input-icon">${ICON_USER}</span>
              <input id="auth-username" name="username" type="text" placeholder="3-32 位字母/数字/下划线" autocomplete="username" autocapitalize="off" spellcheck="false" required />
            </div>
            <p class="form-error" id="err-username" role="alert" aria-live="polite"></p>
          </div>
          <div class="auth-field">
            <label class="form-label" for="auth-password">密码</label>
            <div class="input-wrap">
              <span class="input-icon">${ICON_LOCK}</span>
              <input id="auth-password" name="password" type="password" placeholder="${isRegister ? "至少 8 位，建议字母+数字" : "请输入密码"}" autocomplete="${isRegister ? "new-password" : "current-password"}" required />
            </div>
            ${isRegister ? '<div class="pw-strength" data-level="0" aria-hidden="true"><span></span><span></span><span></span><span></span></div>' : ""}
            <p class="form-error" id="err-password" role="alert" aria-live="polite"></p>
          </div>
          ${isRegister ? `
          <div class="auth-field">
            <label class="form-label" for="auth-display">显示名 <span class="form-label-hint">（可选）</span></label>
            <div class="input-wrap">
              <span class="input-icon input-icon-spark">${ICON_SPARK}</span>
              <input id="auth-display" name="display_name" type="text" placeholder="留空则使用用户名" maxlength="64" />
            </div>
          </div>` : ""}
          <button class="auth-submit" type="submit" id="auth-submit">
            <span class="btn-label">${isRegister ? "创建账号" : "登 录"}</span>
            <span class="btn-spinner" aria-hidden="true"></span>
          </button>
        </form>
        <div class="auth-divider"><span>或</span></div>
        <button class="auth-toggle" id="auth-toggle" type="button">${isRegister ? "已有账号？立即登录" : "还没有账号？免费注册"}</button>
        <p class="auth-foot">登录即表示同意《服务协议》与《隐私政策》（占位文案）</p>
      </div>
      <div id="toast" class="toast hidden"></div>
    </div>`;
  document.querySelector("#auth-toggle").addEventListener("click", () => { authMode = isRegister ? "login" : "register"; renderAuthView(); });
  document.querySelector("#auth-form").addEventListener("submit", handleAuthSubmit);
  const pwInput = document.querySelector("#auth-password");
  if (pwInput) pwInput.addEventListener("input", () => updatePasswordStrength(pwInput.value));
  document.querySelector("#auth-username").focus();
}

async function handleAuthSubmit(e) {
  e.preventDefault();
  const isRegister = authMode === "register";
  const username = document.querySelector("#auth-username").value.trim();
  const password = document.querySelector("#auth-password").value;
  const display = isRegister ? (document.querySelector("#auth-display")?.value.trim() || username) : undefined;

  // 清空字段错误
  setFieldError("username", "");
  setFieldError("password", "");

  // 客户端预校验：内联错误（比 toast 更接近用户视线）
  if (!username || !password) {
    if (!username) setFieldError("username", "请输入用户名");
    if (!password) setFieldError("password", "请输入密码");
    showToast("请填写用户名和密码", "error");
    return;
  }
  if (!_USERNAME_RE.test(username)) {
    setFieldError("username", "用户名须为 3-32 位字母/数字/下划线");
    return;
  }
  if (isRegister && password.length < 8) {
    setFieldError("password", "密码长度至少 8 位");
    return;
  }

  const btn = document.querySelector("#auth-submit");
  btn.disabled = true;
  btn.dataset.loading = "true";
  btn.setAttribute("aria-busy", "true");

  try {
    if (isRegister) {
      await api("POST", "/api/auth/register", { username, password, display_name: display });
      authMode = "login";
      renderAuthView();
      showToast("注册成功，请登录", "success");
      return;
    }
    const data = await api("POST", "/api/auth/login", { username, password });
    setToken(data.access_token);
    state.currentUser = data.user;
    showToast(`欢迎回来，${data.user.display_name || data.user.username}`, "success");
    // 登录已成功：数据加载失败不阻断进入工作台（渲染壳兜底），也不显示"认证失败"
    try {
      await initApp();
    } catch (loadErr) {
      console.error("[initApp] 数据加载失败（已登录，进入工作台）:", loadErr);
      render();
      checkConnection();
    }
  } catch (err) {
    // 服务端 401/403 统一内联到密码字段下方（登录失败 / 账号问题）
    if (err.status === 401) {
      setFieldError("password", err.message || "用户名或密码错误");
    } else {
      setFieldError("password", err.message || "认证失败，请稍后重试");
    }
  } finally {
    const liveBtn = document.querySelector("#auth-submit");
    if (liveBtn) {
      liveBtn.disabled = false;
      delete liveBtn.dataset.loading;
      liveBtn.removeAttribute("aria-busy");
    }
  }
}

function logout() {
  clearToken();
  showToast("已退出登录", "info");
  renderAuthView();
}

// ---------------- 持久化指示（数据已统一存于服务端，此处仅更新 UI 提示） ----------------
function saveState() {
  updateSaveState();
}

function updateSaveState() {
  const el = document.querySelector("#save-state");
  if (el) el.textContent = "已同步至服务器";
}

// ---------------- 工具 ----------------
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[c]));
}

function showToast(message, type = "info") {
  clearTimeout(toastTimer);
  let root = document.querySelector("#toast");
  // 健壮性：登录页等未渲染 #toast 的场景自动创建，避免 root 为 null 抛 TypeError
  if (!root) {
    root = document.createElement("div");
    root.id = "toast";
    root.className = "toast hidden";
    document.body.appendChild(root);
  }
  root.textContent = message;
  root.className = `toast ${type}`;
  toastTimer = setTimeout(() => (root.className = "toast hidden"), 3200);
}

// 自定义确认弹窗（替代原生 confirm，样式与项目统一）。
// 返回 Promise<boolean>：点击「确认」或按 Enter 为 true；「取消」/遮罩点击/Esc 为 false。
function confirmDialog({ title = "确认操作", message = "", confirmText = "确认", danger = false }) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    overlay.innerHTML = `
      <div class="modal-card" role="alertdialog" aria-modal="true" aria-label="${escapeHtml(title)}">
        <div class="modal-title">${escapeHtml(title)}</div>
        <div class="modal-message">${escapeHtml(message)}</div>
        <div class="modal-actions">
          <button class="ghost-btn modal-cancel" type="button">取消</button>
          <button class="${danger ? "danger-btn" : "primary-btn"} modal-ok" type="button">${escapeHtml(confirmText)}</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);

    const close = (result) => {
      overlay.remove();
      document.removeEventListener("keydown", onKey, true);
      resolve(result);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.preventDefault(); close(false); }
      else if (e.key === "Enter") { e.preventDefault(); close(true); }
    };
    overlay.querySelector(".modal-cancel").addEventListener("click", () => close(false));
    overlay.querySelector(".modal-ok").addEventListener("click", () => close(true));
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(false); });
    document.addEventListener("keydown", onKey, true);
    overlay.querySelector(".modal-ok").focus();
  });
}

function currentSession() {
  return state.sessions.find((s) => s.id === state.activeSessionId) || state.sessions[0];
}

function formatMs(value) {
  const n = typeof value === "number" ? value : parseFloat(value);
  return Number.isFinite(n) ? `${Math.round(n)}ms` : "-";
}

function renderTimings(timings, elapsed) {
  if (!timings && elapsed == null) return "";
  const parts = [];
  if (timings) {
    // RAG 关闭时前端明确提示（检索/嵌入等阶段无意义）
    if (timings.rag_enabled === false) {
      parts.push(`RAG 关`);
    }
    // 安全检测关闭时提示（仅联调/实验场景）
    if (timings.safety_enabled === false) {
      parts.push(`安全 关`);
    }
    parts.push(`嵌入 ${formatMs(timings.embed)}`);
    parts.push(`检索 ${formatMs(timings.retrieve)}`);
    if (timings.hybrid != null) parts.push(`混合 ${formatMs(timings.hybrid)}`);
    if (timings.rerank != null) parts.push(`重排 ${formatMs(timings.rerank)}`);
    // llm / total 只在生成完成后才有值：缺失时跳过，避免显示占位的 "-"
    if (timings.llm != null) parts.push(`生成 ${formatMs(timings.llm)}`);
    if (timings.total != null) {
      parts.push(`后端总 ${formatMs(timings.total)}`);
      if (elapsed != null) parts.push(`总时间 ${formatMs(elapsed)}`);
    } else if (elapsed != null) {
      parts.push(`总时间 ${formatMs(elapsed)}`);
    }
  } else if (elapsed != null) {
    parts.push(`总时间 ${formatMs(elapsed)}`);
  }
  return `<div class="timing-bar">${parts.map((p) => `<span class="timing-pill">${p}</span>`).join("")}</div>`;
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await apiFetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* noop */ }
  if (!res.ok) throw new Error(data?.detail || `请求失败（${res.status}）`);
  return data;
}

async function checkConnection() {
  const el = document.querySelector("#conn");
  if (!el) return;
  try {
    const r = await fetch("/api/health");
    if (r.ok) { el.className = "conn online"; el.querySelector(".label").textContent = "已连接"; return; }
  } catch { /* noop */ }
  el.className = "conn offline";
  el.querySelector(".label").textContent = "未连接";
}

// ---------------- 渲染框架（单页：对话联调） ----------------
function render() {
  const root = document.querySelector("#app");
  if (!root.querySelector(".topbar")) renderShell();
  renderChat();
  updateSaveState();
}

function renderShell() {
  document.querySelector("#app").innerHTML = `
    <div class="app-shell">
      <header class="topbar">
        <div class="brand"><div class="brand-mark">✦</div><div><div class="brand-name">心理 RAG</div><div class="brand-sub">对话联调</div></div></div>
        <div class="top-actions">
          <span class="user-chip" id="user-chip">${escapeHtml(state.currentUser?.display_name || state.currentUser?.username || "")}</span>
          <button class="ghost-btn" id="logout-btn" type="button" title="退出登录">退出</button>
          <span id="conn" class="conn"><span class="dot"></span><span class="label">连接中…</span></span>
          <span id="save-state" class="save-state">已同步至服务器</span>
        </div>
      </header>
      <main class="workspace">
        <section id="page-chat" class="page active"></section>
      </main>
      <div id="toast" class="toast hidden"></div>
    </div>`;
  document.querySelector("#logout-btn")?.addEventListener("click", logout);
  checkConnection();
}

// ============================================================
// 对话联调（多会话 + SSE 流式 RAG 问答）
// ============================================================
function renderChat() {
  const page = document.querySelector("#page-chat");

  // 没有任何会话：引导新建（数据在服务端，不能凭空造本地会话）
  if (!state.sessions.length) {
    page.innerHTML = `
      <div class="welcome">
        <div class="bot-avatar">✦</div>
        <h2>还没有对话</h2>
        <p>点击下面按钮开始一段新的对话，所有内容都会保存在服务端数据库。</p>
        <button class="primary-btn" id="empty-new">＋ 新建对话</button>
      </div>`;
    page.querySelector("#empty-new").addEventListener("click", createSession);
    return;
  }

  const session = currentSession();
  const messages = session.messages || [];
  // 保留输入框中尚未发送的内容，避免异步回答返回时整页重渲染将其清空
  const prevInput = page.querySelector("#chat-input");
  const draft = prevInput ? prevInput.value : "";
  page.innerHTML = `
    <div class="chat-layout">
      <aside class="sidebar">
        <div class="section-title"><span>对话</span><button class="icon-btn" id="new-session" title="新建对话">＋</button></div>
        <div class="session-list">${state.sessions.map((s) => renderSessionItem(s, session.id)).join("")}</div>
      </aside>
      <section class="chat-main">
        <div class="chat-toolbar">
          <strong>${escapeHtml(session.name)}</strong>
          <span class="model-chip"><i class="model-dot"></i>RAG 问答</span>
          <span class="heading-spacer"></span>
          <button class="ghost-btn" id="rename-session">重命名</button>
          <button class="ghost-btn" id="export-session">导出</button>
        </div>
        <div class="message-list" id="message-list" aria-live="polite">${messages.length ? messages.map(renderMessage).join("") : renderWelcome()}</div>
        <form class="composer" id="chat-form">
          <textarea id="chat-input" rows="1" placeholder="输入问题，按 Enter 发送，Shift + Enter 换行"></textarea>
          <div class="composer-actions">
            <button class="stop-btn is-hidden" id="stop-btn" type="button" title="停止生成（取消排队或中断回答）" aria-label="停止生成">■</button>
            <button class="send-btn" id="send-btn" type="submit" title="发送" aria-label="发送">↑</button>
          </div>
        </form>
      </section>
      <aside class="inspector">
        <div class="section-title"><span>本次对话配置</span></div>
        <div class="inspector-body">
          <div class="info-block">
            <h3>使用提示词</h3>
            <div class="prompt-preview">系统默认提示词<br/>（由管理员在数据库中配置，用户不可修改）</div>
          </div>
          <div class="info-block">
            <h3>请求状态</h3>
            <div class="info-row"><span>后端连接</span><strong id="insp-conn">检测中</strong></div>
          </div>
        </div>
      </aside>
    </div>`;

  const restoredInput = page.querySelector("#chat-input");
  if (restoredInput && draft) {
    restoredInput.value = draft;
    // 恢复草稿时同步自动增高，避免高度与内容不匹配
    restoredInput.style.height = "auto";
    restoredInput.style.height = `${Math.min(restoredInput.scrollHeight, 160)}px`;
  }

  page.querySelectorAll("[data-session]").forEach((b) => b.addEventListener("click", (e) => { if (e.target.dataset.deleteSession) return; selectSession(b.dataset.session); }));
  // 删除会话：鼠标点击 + 键盘（Enter/Space）均可达（删除按钮为 span，需 role=button + tabindex）
  const handleSessionDeleteKey = (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); deleteSession(e.currentTarget.dataset.deleteSession); }
  };
  page.querySelectorAll("[data-delete-session]").forEach((b) => {
    b.addEventListener("click", (e) => { e.stopPropagation(); deleteSession(b.dataset.deleteSession); });
    b.addEventListener("keydown", handleSessionDeleteKey);
  });
  page.querySelector("#new-session").addEventListener("click", () => createSession());
  page.querySelector("#rename-session").addEventListener("click", () => { const name = prompt("输入新的对话名称", session.name); if (name?.trim()) renameSession(session.id, name.trim()); });
  page.querySelector("#export-session").addEventListener("click", exportSession);
  const form = page.querySelector("#chat-form");
  const chatInput = page.querySelector("#chat-input");
  // 输入自动增高（上限与 CSS max-height 160px 一致），发送后由 submit 重置高度
  const autoGrow = () => {
    chatInput.style.height = "auto";
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 160)}px`;
  };
  chatInput.addEventListener("input", autoGrow);
  form.addEventListener("submit", async (e) => { e.preventDefault(); const content = chatInput.value.trim(); if (!content) return; chatInput.value = ""; chatInput.style.height = "auto"; await sendChat(content); });
  chatInput.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  // 欢迎页快捷问题：点击后直接填入输入框并发送
  page.querySelectorAll(".suggestion").forEach((b) => b.addEventListener("click", () => {
    const input = page.querySelector("#chat-input");
    if (!input) return;
    input.value = b.textContent;
    input.focus();
    page.querySelector("#chat-form")?.requestSubmit();
  }));
  const connEl = page.querySelector("#insp-conn");
  if (connEl) connEl.textContent = document.querySelector("#conn")?.classList.contains("online") ? "已连接" : "未连接";
  const list = page.querySelector("#message-list");
  list.scrollTop = list.scrollHeight;
}

function renderSessionItem(s, activeId) {
  return `
    <button class="session ${s.id === activeId ? "active" : ""}" data-session="${s.id}">
      <span>◌</span>
      <span class="session-copy"><span class="session-name">${escapeHtml(s.name)}</span><span class="session-time">${new Date(s.createdAt).toLocaleDateString("zh-CN")}</span></span>
      <span class="session-delete" data-delete-session="${s.id}" tabindex="0" role="button" aria-label="删除会话 ${escapeHtml(s.name)}">×</span>
    </button>`;
}

function renderWelcome() {
  return `<div class="welcome"><div class="bot-avatar">✦</div><h2>开始一段新的对话</h2><p>这里调用后端 RAG 问答接口，回答基于知识库与系统默认提示词生成。</p><div class="suggestions"><button class="suggestion">孩子总说睡不着，怎么沟通？</button><button class="suggestion">考试前焦虑怎么办？</button><button class="suggestion">如何判断是否需要专业帮助？</button></div></div>`;
}

function renderSourceChips(sources) {
  if (!sources || !sources.length) return "";
  const chips = sources.map((s) => `<span class="source-chip">[${s.index}] ${escapeHtml(s.title || "来源")}</span>`).join("");
  return `<div class="sources">${chips}</div>`;
}

// 轻量 Markdown 符号清洗：本项目是纯文本渲染，去掉 **加粗**/__下划线__/*斜体*/`代码`/行首标题号，
// 避免满屏 * 号影响观感（危机话术模板与 LLM 输出都可能带这些符号）
function cleanMarkdown(s) {
  if (!s) return s;
  return String(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "$1")
    .replace(/__([^_\n]+)__/g, "$1")
    .replace(/(^|\n)#{1,6}\s*/g, "$1")
    .replace(/\*([^*\n]+)\*/g, "$1")
    .replace(/`([^`\n]+)`/g, "$1");
}

function renderMessage(m) {
  const sources = renderSourceChips(m.sources);
  const timings = m.role === "assistant" ? renderTimings(m.timings, m.elapsed) : "";
  // 流式消息：保留固定 id，供 token 增量更新时定位；内容为空时显示占位文字
  const streamingMsgId = m.streaming ? ' id="streaming-msg"' : "";
  const streamingBubbleId = m.streaming ? ' id="streaming-bubble"' : "";
  const body = cleanMarkdown(m.streaming && !m.content ? "正在生成…" : m.content);
  return `<div class="message ${m.role}"${streamingMsgId}><div><div class="message-meta">${m.role === "user" ? "你" : "心理 RAG"}</div><div class="message-bubble"${streamingBubbleId}>${escapeHtml(body)}${sources}${timings}</div></div></div>`;
}

// ---- 流式渲染辅助：只操作当前占位气泡 DOM，不整页重绘 ----
function appendStreamingBubble() {
  const list = document.querySelector("#message-list");
  if (!list) return null;
  const div = document.createElement("div");
  div.className = "message assistant";
  // 文本独立放进 .streaming-text，来源 chips / 耗时栏是它的兄弟节点（token 更新只改
  // 文本节点，不误清来源）。排队状态行 .queue-status 独立展示，放行后隐藏。
  div.innerHTML = '<div><div class="message-meta">心理 RAG</div><div class="message-bubble"><span class="queue-status is-hidden"></span><span class="streaming-text">正在生成…</span></div></div>';
  list.appendChild(div);
  list.scrollTop = list.scrollHeight;
  return { bubble: div.querySelector(".message-bubble"), textEl: div.querySelector(".streaming-text"), statusEl: div.querySelector(".queue-status") };
}

// 排队/状态提示（总稿 §4.4：queue 事件 → “前方还有 N 个请求”）
function showQueueStatus(els, text) {
  if (!els) return;
  els.statusEl.textContent = text;
  els.statusEl.classList.remove("is-hidden");
}

function hideQueueStatus(els) {
  if (!els) return;
  els.statusEl.classList.add("is-hidden");
  els.statusEl.textContent = "";
}

function updateStreamingBubble(m, els) {
  if (!els) return;
  els.textEl.textContent = cleanMarkdown(m.content) || "正在生成…";
  // 仅在用户接近底部时跟随滚动，避免打扰上翻阅读
  const list = document.querySelector("#message-list");
  if (list && list.scrollHeight - list.scrollTop - list.clientHeight < 100) {
    list.scrollTop = list.scrollHeight;
  }
}

function finalizeStreamingBubble(m, els) {
  if (!els) return;
  // 兜底：任何"只有 done 没有 token"的路径（如高危拦截），把最终答案写回文本节点
  els.textEl.textContent = cleanMarkdown(m.content) || "正在生成…";
  // 主路径：答案流式完成后，在此展示来源文档（先答案、后依据），再补耗时栏
  if (m.sources && m.sources.length && !els.bubble.querySelector(".sources")) {
    els.bubble.insertAdjacentHTML("beforeend", renderSourceChips(m.sources));
  }
  if (m.timings) els.bubble.insertAdjacentHTML("beforeend", renderTimings(m.timings, m.elapsed));
  // 中/低危关怀提示（safety_note）与高危危机标记
  if (m.safetyNote && !els.bubble.querySelector(".safety-note")) {
    els.bubble.insertAdjacentHTML("beforeend", `<div class="safety-note">${escapeHtml(cleanMarkdown(m.safetyNote))}</div>`);
  }
  if (m.isCrisis) {
    const msg = els.bubble.closest(".message");
    if (msg) msg.classList.add("crisis");
  }
  // 收尾新增了来源/耗时内容，接近底部时跟随滚动
  const list = document.querySelector("#message-list");
  if (list && list.scrollHeight - list.scrollTop - list.clientHeight < 100) {
    list.scrollTop = list.scrollHeight;
  }
}

// 准入类错误码 → 用户可读文案（总稿 §4.3.3/§4.4 映射表）
function admissionErrorText(data) {
  if (!data) return "";
  switch (data.code) {
    case "AI_QUEUE_FULL": return "当前排队已满，请稍后重试";
    case "AI_QUEUE_TIMEOUT": return "排队等待时间较长，请重新发起";
    case "AI_REQUEST_IN_PROGRESS": return "你已有问题正在处理，请稍候";
    case "AI_REQUEST_CANCELLED": return "请求已取消";
    default: return "";
  }
}

async function sendChat(content) {
  const session = currentSession();
  // 中断该会话上一个未完成的流（重复发送/切换时）
  if (session._streamAbort) {
    const oldPlaceholder = session.messages.find((m) => m.streaming);
    session._streamAbort.abort();
    session._streamAbort = null;
    // 立即摘掉旧占位的流式标记：避免随后 renderChat 重建时渲染出多余流式节点，
    // 也保证旧流收尾不会再把来源/耗时写进重建后的新气泡
    if (oldPlaceholder) oldPlaceholder.streaming = false;
  }
  session.messages.push({ role: "user", content });
  // 首次提问：立即用问题自动命名（清洗空白 + 限长），并在请求中带给后端持久化
  const isFirstTurn = session.messages.length === 1;
  if (isFirstTurn) session.name = content.replace(/\s+/g, " ").slice(0, 30);
  saveState(); renderChat();
  const btn = document.querySelector("#send-btn");
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>'; }
  // 显示“停止”按钮（排队中/生成中均可取消/中断）
  const stopBtn = document.querySelector("#stop-btn");
  if (stopBtn) stopBtn.classList.remove("is-hidden");

  // 占位 assistant 消息：SSE 期间增量填充，不整页重绘
  const placeholder = { role: "assistant", content: "", sources: [], timings: null, streaming: true };
  session.messages.push(placeholder);
  const streamEls = appendStreamingBubble();

  const ac = new AbortController();
  session._streamAbort = ac;
  const started = performance.now();
  let requestId = null;      // 准入 request_id（queue/started/done 事件带回，用于取消接口）
  try {
    // 停止/取消：中断本地流 + 尽力通知服务端释放占位（幂等，失败不影响本地）
    if (stopBtn) {
      stopBtn.onclick = () => {
        if (ac.signal.aborted) return;
        ac.abort();
        if (requestId) {
          apiFetch(`/api/query/requests/${encodeURIComponent(requestId)}`, { method: "DELETE" }).catch(() => { /* 尽力而为 */ });
        }
        // 无内容时以“已停止”占位；有部分内容时在末尾追加停止标记
        hideQueueStatus(streamEls);
        if (!placeholder.content.trim()) {
          placeholder.content = "（已停止）";
        } else {
          placeholder.content += "\n\n（已停止）";
        }
        updateStreamingBubble(placeholder, streamEls);
      };
    }
    // 多轮记忆（2026-09-04）：短期窗口由服务端按 session_id 从 messages 表组装
    // （最近 MEMORY_RECENT_ROUNDS 轮，见 prepare/_build_messages），前端只发本轮
    // 问题 —— 请求体 O(1)，会话隔离与跨设备/刷新一致性由服务端保证。
    const body = { question: content, session_id: session.id };
    if (isFirstTurn) body.title = session.name;

    const resp = await apiFetch("/api/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
    if (!resp.ok) {
      let detail = `请求失败（${resp.status}）`;
      try { const j = await resp.json(); detail = admissionErrorText(j) || j.detail || detail; } catch { /* noop */ }
      throw new Error(detail);
    }

    // 读取 SSE 流：按空行切分事件，逐事件处理
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split("\n\n");
      buf = events.pop();
      for (const evt of events) {
        if (!evt.trim()) continue;
        const lines = evt.split("\n");
        const evtName = (lines.find((l) => l.startsWith("event: ")) || "event: message").slice(7).trim();
        const dataLine = lines.find((l) => l.startsWith("data: "));
        if (!dataLine) continue;
        let data;
        try { data = JSON.parse(dataLine.slice(6)); } catch { continue; }
        // 未知事件一律忽略（总稿 FR-BE-04：旧前端兼容性）
        if (evtName === "queue") {
          // 排队中：展示实时位置（position 为 0 基下标 = 前方请求数）
          requestId = data.request_id || requestId;
          if (typeof data.position === "number") {
            showQueueStatus(streamEls, `排队中 · 前方还有 ${data.position} 个请求`);
          }
        } else if (evtName === "started") {
          requestId = data.request_id || requestId;
          hideQueueStatus(streamEls);
        } else if (evtName === "sources") {
          // 只暂存来源数据，不立即渲染：等答案流式生成完成、收尾时再显示，
          // 呈现顺序为「先看答案 → 再展示依据文档与耗时」
          placeholder.sources = data.sources || [];
        } else if (evtName === "token") {
          hideQueueStatus(streamEls);
          placeholder.content += data.text || "";
          updateStreamingBubble(placeholder, streamEls);
        } else if (evtName === "done") {
          hideQueueStatus(streamEls);
          if (data.answer != null) placeholder.content = data.answer;
          placeholder.timings = data.timings || null;
          if (data.safety_note) placeholder.safetyNote = data.safety_note;
          if (data.is_crisis_response) placeholder.isCrisis = true;
          // 高危危机拦截等场景后端只发 done 不发 token，这里必须回写 DOM，
          // 否则界面停留在"正在生成…"（数据有答案、界面没显示）
          updateStreamingBubble(placeholder, streamEls);
        } else if (evtName === "error") {
          hideQueueStatus(streamEls);
          throw new Error(admissionErrorText(data) || data.detail || "生成失败");
        }
      }
    }
    placeholder.elapsed = Math.round(performance.now() - started);
    showToast(`回答已生成 · ${placeholder.elapsed}ms`, "success");
  } catch (e) {
    if (e.name === "AbortError") {
      // 主动中断：用户停止/切换会话/删除/新发送。用户停止时标记已展示在 stop 回调里；
      // 其余场景保留已生成部分，不做额外文案
    } else {
      placeholder.content = placeholder.content || `请求失败：${e.message}`;
      showToast(e.message, "error");
    }
  } finally {
    placeholder.streaming = false;
    // 仅当自己仍是当前流时才清空句柄：并发场景下旧流 finally 不能误清新流的 abort 引用
    if (session._streamAbort === ac) session._streamAbort = null;
    if (stopBtn) stopBtn.classList.add("is-hidden");
    finalizeStreamingBubble(placeholder, streamEls);
    // 恢复发送按钮（流式版不 renderChat，按钮不会自动重建，需要手动恢复）
    const liveBtn = document.querySelector("#send-btn");
    if (liveBtn) { liveBtn.disabled = false; liveBtn.innerHTML = "↑"; }
    saveState();
  }
  // 发送后自动聚焦输入框，便于连续追问
  const chatInputNow = document.querySelector("#chat-input");
  if (chatInputNow) chatInputNow.focus();
}

function exportSession() {
  const session = currentSession();
  const text = session.messages.map((m) => `${m.role === "user" ? "问" : "答"}：${m.content}`).join("\n\n");
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `对话-${session.name}-${new Date().toISOString().slice(0, 10)}.txt`;
  a.click(); URL.revokeObjectURL(url);
  showToast("对话已导出", "success");
}

// ---------------- 会话：服务端 SQLite 为唯一数据源 ----------------
async function loadSessions() {
  const list = await api("GET", "/api/sessions");
  state.sessions = (Array.isArray(list) ? list : []).map((s) => ({
    id: s.id,
    name: s.title || "新会话",
    createdAt: s.updated_at || s.created_at ? new Date(s.updated_at || s.created_at).getTime() : Date.now(),
    messages: [],
    loaded: false,
  }));
  if (!state.sessions.find((s) => s.id === state.activeSessionId)) {
    state.activeSessionId = state.sessions.length ? state.sessions[0].id : null;
  }
}

async function loadSessionMessages(id) {
  try {
    const msgs = await api("GET", `/api/sessions/${encodeURIComponent(id)}/messages`);
    // 后端按语义存 human/ai；前端显示用 user/assistant，这里做一次对账映射
    const roleMap = { human: "user", ai: "assistant" };
    const mapped = (Array.isArray(msgs) ? msgs : []).map((m) => ({ role: roleMap[m.role] || m.role, content: m.content }));
    const sess = state.sessions.find((s) => s.id === id);
    if (sess) {
      sess.messages = mapped;
      sess.loaded = true;
      // 未命名会话：用最早一条用户消息自动取名（与后端命名规则一致，打开旧会话即时可见）
      if (!sess.name || sess.name === "新会话" || sess.name === "新的对话") {
        const firstUser = mapped.find((m) => m.role === "user");
        if (firstUser) sess.name = firstUser.content.replace(/\s+/g, " ").slice(0, 30);
      }
    }
  } catch { /* 忽略：保持空消息，不阻断界面 */ }
}

// 保证始终有一个可用会话（首屏 / 删除到空时自动在服务端新建）
async function ensureActiveSession() {
  if (!state.sessions.length) {
    const data = await api("POST", "/api/sessions", { name: "新的对话" });
    state.sessions.push({ id: data.id, name: data.name || "新的对话", createdAt: Date.now(), messages: [], loaded: true });
  }
  if (!state.sessions.find((s) => s.id === state.activeSessionId)) {
    state.activeSessionId = state.sessions[0].id;
  }
  const active = state.sessions.find((s) => s.id === state.activeSessionId);
  if (active && !active.loaded) await loadSessionMessages(active.id);
}

async function selectSession(id) {
  // 切换会话前中断旧会话正在进行的流式生成
  const cur = currentSession();
  if (cur && cur._streamAbort) { cur._streamAbort.abort(); cur._streamAbort = null; }
  state.activeSessionId = id;
  const sess = state.sessions.find((s) => s.id === id);
  if (sess && !sess.loaded) await loadSessionMessages(id);
  renderChat();
}

async function createSession() {
  try {
    const data = await api("POST", "/api/sessions", { name: "新的对话" });
    const sess = { id: data.id, name: data.name || "新的对话", createdAt: Date.now(), messages: [], loaded: true };
    state.sessions.unshift(sess);
    state.activeSessionId = sess.id;
    renderChat();
    showToast("已新建对话", "success");
  } catch (e) {
    showToast(`新建失败：${e.message}`, "error");
  }
}

async function renameSession(id, name) {
  try {
    await api("PATCH", `/api/sessions/${encodeURIComponent(id)}`, { name });
    const sess = state.sessions.find((s) => s.id === id);
    if (sess) sess.name = name;
    renderChat();
  } catch (e) {
    showToast(`重命名失败：${e.message}`, "error");
  }
}

async function deleteSession(id) {
  const sess = state.sessions.find((s) => s.id === id);
  if (!sess) return;
  const ok = await confirmDialog({ title: "删除对话", message: `确定删除对话「${sess.name}」吗？删除后不可恢复。`, confirmText: "删除", danger: true });
  if (!ok) return;
  // 删除前中断该会话正在进行的流式生成
  if (sess._streamAbort) { sess._streamAbort.abort(); sess._streamAbort = null; }
  try {
    await api("DELETE", `/api/sessions/${encodeURIComponent(id)}`);
    state.sessions = state.sessions.filter((s) => s.id !== id);
    if (state.activeSessionId === id) {
      state.activeSessionId = state.sessions.length ? state.sessions[0].id : null;
    }
    // 始终至少保留一个会话（与旧行为一致）
    if (!state.sessions.length) {
      await createSession();
      return;
    }
    renderChat();
  } catch (e) {
    showToast(`删除失败：${e.message}`, "error");
  }
}

// ---------------- 启动（认证门禁：未登录只显示登录页） ----------------
async function initApp() {
  await loadSessions();          // 加载会话列表（服务端 SQLite，仅当前用户）
  await ensureActiveSession();   // 保证有一个可用会话并载入其消息
  render();
  checkConnection();
}

(async function init() {
  const token = getToken();
  if (!token) { renderAuthView(); return; }
  try {
    // 校验 token 有效性（无效/过期 → 401 → api 抛错）
    state.currentUser = await api("GET", "/api/auth/me");
  } catch {
    clearToken();
    renderAuthView();
    return;
  }
  await initApp();
})();
