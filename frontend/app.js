// ============================================================
// 心理 RAG · 对话联调（SPA）
// 单页：多会话 + SSE 流式 RAG 问答
// 后端接本项目 FastAPI：/api/query/stream（SSE）、/api/sessions、/api/auth/*
// 提示词为全局激活项（prompts 表），由管理员在数据库直接配置，用户不可修改
//
// 2026-09-10 前端审查修复要点（对应《rag-psychology 前端检查报告》）
//   1) 账号生命周期编号 state.epoch + 统一认证清理：旧账号的响应不再污染新页面
//   2) 会话切换：立即渲染加载态并禁用发送，提交绑定明确的会话 id
//   3) 弹窗：不再全局拦截 Enter；危险操作默认聚焦「取消」；Tab 焦点圈定 + 关闭后焦点恢复
//   4) 生成状态以会话 id 为键集中管理：重绘后重新绑定流式节点；旧流不得改动新控件
//   5) 启动/历史加载区分「加载中 / 失败 / 成功但为空」，并提供重试
//   6) SSE 以 done 判定完成：EOF 无 done、中断、部分失败都在消息上显式标未完成并给重试入口
//   7) 消息元数据（来源/耗时/关怀提示/危机/未完成）由同一个渲染函数输出，重绘不丢失
//   8) 草稿按会话隔离（内存 Map，不落 localStorage）
//   9) 认证接口错误按服务端 detail 落位，不再把「用户名或密码错误」改写成「登录已过期」
// ============================================================
import { escapeHtml, renderAnswer, toPlainText, createSseParser, authErrorField } from "./lib/format.js";
import { ApiError, api, apiFetch, checkHealth, clearToken, configureApi, getToken, setToken } from "./lib/api.js";

const SESSION_LIST_LIMIT = 200; // 后端 GET /api/sessions 上限 200（默认 50）
const DRAFT_MAX_HEIGHT = 160;   // 输入框自动增高上限，与 CSS max-height 一致

const USERNAME_RE = /^[a-zA-Z0-9_]{3,32}$/;
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

const STALE = Symbol("stale"); // 账号生命周期已变化的哨兵值

// ============================================================
// 运行时状态
// ============================================================
// 业务数据（会话 / 消息）以服务端为唯一数据源，前端 state 只是当前页面的内存镜像。
// 唯一落盘的是登录凭证（httpOnly cookie 方案需后端配 CORS 与 CSRF，本地原型不做）。
const state = {
  view: "auth",          // auth | app
  currentUser: null,
  sessions: [],          // { id, name, createdAt, messages, loaded, status, error }
  activeSessionId: null,
  epoch: 0,              // 账号生命周期编号：登录/退出递增，所有异步响应回写前必须校验
};

let boot = { status: "loading", error: "" }; // 工作台启动状态：loading | ready | error
let authMode = "login";                      // login | register
let toastTimer;
let connState = "unknown";                   // online | offline | unknown（单一数据源，多处展示）
let lastRenderedSessionId = null;            // 用于判断是否需要在重绘后保留滚动位置
let historyPanelCleanup = null;              // 小屏历史浮层的监听清理函数

const drafts = new Map();    // 会话 id → 未发送草稿（仅内存）
const streams = new Map();   // 会话 id → 活动流句柄（生成状态集中管理）
const dialogs = new Set();   // 打开的弹窗销毁器（认证切换时统一清理）

// ============================================================
// 小工具
// ============================================================
function getSession(id) {
  return state.sessions.find((s) => s.id === id) || null;
}

function currentSession() {
  return getSession(state.activeSessionId) || state.sessions[0] || null;
}

function autoName(content) {
  return String(content || "").replace(/\s+/g, " ").trim().slice(0, 30);
}

function isNearBottom(list, threshold = 100) {
  if (!list) return true;
  return list.scrollHeight - list.scrollTop - list.clientHeight < threshold;
}

const scheduleFrame = typeof requestAnimationFrame === "function"
  ? (cb) => requestAnimationFrame(cb)
  : (cb) => setTimeout(cb, 16);

function announce(text) {
  const el = document.querySelector("#sr-status");
  if (el) el.textContent = text;
}

function focusInput() {
  const input = document.querySelector("#chat-input");
  if (input && !input.disabled) input.focus();
}

// 属性选择器取值的转义（会话 id 由服务端生成；这里只处理引号/反斜杠，不依赖 CSS.escape）
function attrValue(value) {
  return String(value).replace(/["\\]/g, "\\$&");
}

// ============================================================
// 认证（JWT Bearer）
// ============================================================
function resetWorkspaceState() {
  // 清空业务状态：会话 / 草稿 / 流 / 启动状态（不清 token，供登录成功路径复用）
  cancelAllStreams();
  closeAllDialogs();
  state.sessions = [];
  state.activeSessionId = null;
  drafts.clear();
  boot = { status: "loading", error: "" };
  lastRenderedSessionId = null;
}

// 统一的认证清理：退出登录、会话过期、切换账号都走这里
function resetAuthState() {
  state.epoch += 1;        // 旧账号的一切在途响应自此作废
  resetWorkspaceState();
  state.currentUser = null;
  state.view = "auth";
  clearToken();
}

// 会话过期（业务请求 401）：清干净再回登录页，避免上一般号的会话泄漏到下一个账号
function handleSessionExpired() {
  if (state.view === "auth") return;
  resetAuthState();
  renderAuthView();
  showToast("登录已过期，请重新登录", "error");
}

function logout() {
  resetAuthState();
  renderAuthView();
  showToast("已退出登录", "info");
}

// 字段级错误提示（a11y：role=alert + aria-invalid）
function setFieldError(field, msg) {
  const errEl = document.querySelector(`#err-${field}`);
  const inputEl = document.querySelector(`#auth-${field}`);
  if (errEl) errEl.textContent = msg || "";
  if (inputEl) inputEl.setAttribute("aria-invalid", msg ? "true" : "false");
}

// 表单级错误（无法归属到某个字段时，例如登录尝试过于频繁）
function setFormError(msg) {
  const el = document.querySelector("#err-auth");
  if (el) el.textContent = msg || "";
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
  if (!root) return;
  state.view = "auth";
  const isRegister = authMode === "register";
  root.innerHTML = `
    <div class="auth-shell">
      <a class="home-link auth-home" href="/">← 返回首页</a>
      <section class="auth-intro" aria-label="欢迎"><span class="intro-kicker">青少年心理问答</span><h2>给心事，<br>一点说出来的空间。</h2><p>关于成长、学习和家庭的小困惑，<br>可以从你愿意分享的那一点开始。</p><div class="intro-plant" aria-hidden="true"><span></span><span></span><span></span></div><p class="intro-caption">不必一次说清楚，我们慢慢聊。</p></section>
      <div class="auth-bg" aria-hidden="true">
        <span class="blob blob-a"></span>
        <span class="blob blob-b"></span>
        <span class="blob blob-c"></span>
      </div>
      <div class="auth-card" data-mode="${authMode}">
        <div class="auth-logo">${LOGO_SVG}</div>
        <h1 class="auth-title">${isRegister ? "创建你的账号" : "欢迎回来"}</h1>
        <p class="auth-sub">${isRegister ? "注册后，开始你的第一段对话" : "登录后，继续你的对话与记录"}</p>
        <form id="auth-form" novalidate>
          <div class="auth-field">
            <label class="form-label" for="auth-username">用户名</label>
            <div class="input-wrap">
              <span class="input-icon" aria-hidden="true">${ICON_USER}</span>
              <input id="auth-username" name="username" type="text" placeholder="3-32 位字母/数字/下划线" autocomplete="username" autocapitalize="off" spellcheck="false" aria-describedby="err-username" required />
            </div>
            <p class="form-error" id="err-username" role="alert" aria-live="polite"></p>
          </div>
          <div class="auth-field">
            <label class="form-label" for="auth-password">密码</label>
            <div class="input-wrap">
              <span class="input-icon" aria-hidden="true">${ICON_LOCK}</span>
              <input id="auth-password" name="password" type="password" placeholder="${isRegister ? "至少 8 位，建议字母+数字" : "请输入密码"}" autocomplete="${isRegister ? "new-password" : "current-password"}" aria-describedby="err-password" required />
            </div>
            ${isRegister ? '<div class="pw-strength" data-level="0" aria-hidden="true"><span></span><span></span><span></span><span></span></div>' : ""}
            <p class="form-error" id="err-password" role="alert" aria-live="polite"></p>
          </div>
          ${isRegister ? `
          <div class="auth-field">
            <label class="form-label" for="auth-display">显示名 <span class="form-label-hint">（可选）</span></label>
            <div class="input-wrap">
              <span class="input-icon input-icon-spark" aria-hidden="true">${ICON_SPARK}</span>
              <input id="auth-display" name="display_name" type="text" placeholder="留空则使用用户名" maxlength="64" />
            </div>
          </div>` : ""}
          <p class="form-error form-error-block" id="err-auth" role="alert" aria-live="polite"></p>
          <button class="auth-submit" type="submit" id="auth-submit">
            <span class="btn-label">${isRegister ? "创建账号" : "登 录"}</span>
            <span class="btn-spinner" aria-hidden="true"></span>
          </button>
        </form>
        <div class="auth-divider"><span>或</span></div>
        <button class="auth-toggle" id="auth-toggle" type="button">${isRegister ? "已有账号？立即登录" : "还没有账号？创建账号"}</button>
        <p class="auth-foot">对话记录保存在服务器，可在登录后查看。<br>AI 提供心理科普与支持，不能替代专业诊疗。</p>
      </div>
      <div id="toast" class="toast hidden" role="status" aria-live="polite"></div>
    </div>`;
  document.querySelector("#auth-toggle")?.addEventListener("click", () => {
    authMode = isRegister ? "login" : "register";
    renderAuthView();
  });
  document.querySelector("#auth-form")?.addEventListener("submit", handleAuthSubmit);
  const pwInput = document.querySelector("#auth-password");
  if (pwInput) pwInput.addEventListener("input", () => updatePasswordStrength(pwInput.value));
  document.querySelector("#auth-username")?.focus();
}

async function handleAuthSubmit(e) {
  e.preventDefault();
  const epoch = state.epoch;
  const isRegister = authMode === "register";
  const username = document.querySelector("#auth-username")?.value.trim() || "";
  const password = document.querySelector("#auth-password")?.value || "";
  const display = isRegister ? (document.querySelector("#auth-display")?.value.trim() || username) : undefined;

  setFieldError("username", "");
  setFieldError("password", "");
  setFormError("");

  // 客户端预校验：内联错误（比 toast 更接近用户视线）
  if (!username || !password) {
    if (!username) setFieldError("username", "请输入用户名");
    if (!password) setFieldError("password", "请输入密码");
    showToast("请填写用户名和密码", "error");
    return;
  }
  if (!USERNAME_RE.test(username)) {
    setFieldError("username", "用户名须为 3-32 位字母/数字/下划线");
    return;
  }
  if (isRegister && password.length < 8) {
    setFieldError("password", "密码长度至少 8 位");
    return;
  }

  const btn = document.querySelector("#auth-submit");
  if (btn) {
    btn.disabled = true;
    btn.dataset.loading = "true";
    btn.setAttribute("aria-busy", "true");
  }

  try {
    if (isRegister) {
      await api("POST", "/api/auth/register", { username, password, display_name: display }, { auth: "none" });
      if (epoch !== state.epoch) return;
      authMode = "login";
      renderAuthView();
      showToast("注册成功，请登录", "success");
      return;
    }
    // 认证接口用 auth:"none"：401 是「用户名或密码错误」，不能被改写成「登录已过期」
    const data = await api("POST", "/api/auth/login", { username, password }, { auth: "none" });
    if (epoch !== state.epoch) return; // 期间已退出/切换账号：丢弃这次登录结果

    setToken(data.access_token);
    resetWorkspaceState();  // 切换账号：清掉上一账号的会话/草稿/流
    state.epoch += 1;       // 新账号生命周期开始
    state.currentUser = data.user;
    const newEpoch = state.epoch;
    // 登录已成功：数据加载失败进入可重试的错误页，不阻断进入工作台，也不显示「认证失败」
    await bootWorkspace(newEpoch);
    showToast(`欢迎回来，${data.user.display_name || data.user.username}`, "success");
  } catch (err) {
    if (epoch !== state.epoch) return;
    if (err?.name === "AbortError") return;
    const msg = err?.message || "认证失败，请稍后重试";
    if (!document.querySelector(".auth-card")) return; // 视图已离开登录页
    const field = authErrorField(msg);
    if (field) setFieldError(field, msg);
    else setFormError(msg);
  } finally {
    const liveBtn = document.querySelector("#auth-submit");
    if (liveBtn) {
      liveBtn.disabled = false;
      delete liveBtn.dataset.loading;
      liveBtn.removeAttribute("aria-busy");
    }
  }
}

// ============================================================
// 连接状态（单一数据源：顶栏 + 侧栏信息区共用）
// ============================================================
function paintConnection() {
  const el = document.querySelector("#conn");
  if (el) {
    const label = el.querySelector(".label");
    if (connState === "online") { el.className = "conn online"; if (label) label.textContent = "已连接"; }
    else if (connState === "offline") { el.className = "conn offline"; if (label) label.textContent = "未连接（点击重试）"; }
    else { el.className = "conn"; if (label) label.textContent = "检测中…"; }
  }
  const insp = document.querySelector("#insp-conn");
  if (insp) {
    insp.textContent = connState === "online" ? "已连接" : connState === "offline" ? "未连接" : "检测中";
    insp.dataset.state = connState;
  }
}

async function checkConnection() {
  connState = "unknown";
  paintConnection();
  const ok = await checkHealth();
  connState = ok ? "online" : "offline";
  paintConnection();
}

// ============================================================
// 提示（toast）
// ============================================================
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
  root.setAttribute("role", "status");
  root.setAttribute("aria-live", type === "error" ? "assertive" : "polite");
  toastTimer = setTimeout(() => (root.className = "toast hidden"), 3200);
}

// ============================================================
// 自定义确认弹窗（替代原生 confirm，样式与项目统一）
// 返回 Promise<boolean>：点击「确认」为 true；「取消」/遮罩/Esc 为 false。
// 键盘策略：不做全局 Enter 拦截（否则焦点在「取消」上按 Enter 也会确认），
// 交给原生按钮处理键盘激活；危险操作默认聚焦「取消」，关闭后焦点回到触发元素。
// ============================================================
function confirmDialog({ title = "确认操作", message = "", confirmText = "确认", cancelText = "取消", danger = false, trigger = null }) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    overlay.innerHTML = `
      <div class="modal-card" role="alertdialog" aria-modal="true" aria-labelledby="modal-title" aria-describedby="modal-message">
        <div class="modal-title" id="modal-title">${escapeHtml(title)}</div>
        <div class="modal-message" id="modal-message">${escapeHtml(message)}</div>
        <div class="modal-actions">
          <button class="ghost-btn modal-cancel" type="button">${escapeHtml(cancelText)}</button>
          <button class="${danger ? "danger-btn" : "primary-btn"} modal-ok" type="button">${escapeHtml(confirmText)}</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);

    const cancelBtn = overlay.querySelector(".modal-cancel");
    const okBtn = overlay.querySelector(".modal-ok");
    // 关闭后焦点回到触发元素（显式传入优先，避免依赖各浏览器不一致的 activeElement 行为）
    const prevFocus = trigger || document.activeElement;
    let settled = false;

    const destroy = (result) => {
      if (settled) return;
      settled = true;
      overlay.remove();
      document.removeEventListener("keydown", onKey, true);
      dialogs.delete(destroy);
      if (prevFocus && typeof prevFocus.focus === "function" && document.contains(prevFocus)) {
        try { prevFocus.focus(); } catch { /* noop */ }
      }
      resolve(result);
    };

    const onKey = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        destroy(false);
        return;
      }
      if (e.key === "Tab") {
        // 焦点圈定在弹窗内（两个按钮间循环）
        const ring = [cancelBtn, okBtn];
        const idx = ring.indexOf(document.activeElement);
        const next = e.shiftKey
          ? (idx <= 0 ? ring.length - 1 : idx - 1)
          : (idx === ring.length - 1 || idx === -1 ? 0 : idx + 1);
        e.preventDefault();
        ring[next].focus();
      }
      // Enter / Space 不做拦截：原生按钮会以「当前聚焦元素」为语义触发
    };

    cancelBtn.addEventListener("click", () => destroy(false));
    okBtn.addEventListener("click", () => destroy(true));
    overlay.addEventListener("click", (e) => { if (e.target === overlay) destroy(false); });
    document.addEventListener("keydown", onKey, true);
    dialogs.add(destroy);
    (danger ? cancelBtn : okBtn).focus();
  });
}

function closeAllDialogs() {
  for (const destroy of [...dialogs]) destroy(false);
  dialogs.clear();
}

// ============================================================
// 渲染：框架
// ============================================================
function render() {
  if (state.view === "auth") {
    renderAuthView();
    return;
  }
  if (!document.querySelector(".topbar")) renderShell();
  renderChat();
}

function renderShell() {
  document.querySelector("#app").innerHTML = `
    <div class="app-shell">
      <header class="topbar">
        <div class="brand"><div class="brand-mark" aria-hidden="true">✦</div><div><div class="brand-name">青少年心理问答</div><div class="brand-sub">给心事一点空间</div></div></div>
        <div class="top-actions"><a class="home-link" href="/">返回首页</a>
          <span class="user-chip" id="user-chip">${escapeHtml(state.currentUser?.display_name || state.currentUser?.username || "")}</span>
          <button class="ghost-btn" id="logout-btn" type="button" title="退出登录">退出</button>
          <button class="conn" id="conn" type="button" title="点击重新检测连接"><span class="dot" aria-hidden="true"></span><span class="label">连接中…</span></button>
          <span id="save-state" class="save-state">已同步至服务器</span>
        </div>
      </header>
      <main class="workspace">
        <section id="page-chat" class="page active"></section>
      </main>
      <div id="toast" class="toast hidden" role="status" aria-live="polite"></div>
      <p id="sr-status" class="sr-only" role="status" aria-live="polite"></p>
    </div>`;
  document.querySelector("#logout-btn")?.addEventListener("click", logout);
  document.querySelector("#conn")?.addEventListener("click", () => checkConnection());
  paintConnection();
}

// ============================================================
// 渲染：对话页
// ============================================================
function renderChat() {
  const page = document.querySelector("#page-chat");
  if (!page) return;

  // 启动阶段：先渲染骨架，再区分加载成功/失败
  if (boot.status === "error") {
    page.innerHTML = `
      <div class="state-panel error" role="alert">
        <div class="state-icon" aria-hidden="true">!</div>
        <p class="state-title">数据加载失败</p>
        <p class="state-detail">${escapeHtml(boot.error || "请稍后重试")}</p>
        <button class="primary-btn" id="retry-boot" type="button">重新加载</button>
      </div>`;
    page.querySelector("#retry-boot")?.addEventListener("click", () => bootWorkspace(state.epoch));
    lastRenderedSessionId = null;
    paintConnection();
    return;
  }
  if (boot.status === "loading" && !state.sessions.length) {
    page.innerHTML = `
      <div class="state-panel" role="status" aria-live="polite">
        <span class="state-spinner" aria-hidden="true"></span>
        <p class="state-title">正在加载你的对话…</p>
      </div>`;
    lastRenderedSessionId = null;
    paintConnection();
    return;
  }

  const session = currentSession();
  if (!session) {
    page.innerHTML = `
      <div class="welcome">
        <div class="bot-avatar" aria-hidden="true">✦</div>
        <h2>你好，今天想聊些什么？</h2>
        <p>从一件小事开始也可以。创建对话后，你可以随时回来继续聊。</p>
        <button class="primary-btn" id="empty-new" type="button">＋ 新建对话</button>
      </div>`;
    page.querySelector("#empty-new")?.addEventListener("click", () => createSession());
    lastRenderedSessionId = null;
    paintConnection();
    return;
  }

  // 重绘前记录滚动位置：同一个会话的重绘尽量保留阅读位置（换会话则滚到底）
  const prevList = page.querySelector("#message-list");
  const prevScrollTop = prevList ? prevList.scrollTop : null;
  const keepScroll = lastRenderedSessionId === session.id && prevScrollTop != null;

  historyPanelCleanup?.();
  historyPanelCleanup = null;
  page.innerHTML = chatLayoutHtml(session);
  bindChat(page, session);
  lastRenderedSessionId = session.id;

  const list = page.querySelector("#message-list");
  if (list) {
    list.scrollTop = keepScroll ? prevScrollTop : list.scrollHeight;
    updateJumpButton(list);
  }

  paintConnection();
  rebindVisibleStream();   // 重绘前若正在流式生成：把 DOM 引用接回新节点，避免写入游离节点
  updateComposerState();
}

function chatLayoutHtml(session) {
  return `
    <div class="chat-layout">
      <aside class="sidebar" id="chat-history" aria-label="对话记录">
        <div class="section-title"><span>对话</span><button class="icon-btn" id="new-session" type="button" title="新建对话" aria-label="新建对话">＋</button></div>
        <div class="session-list" role="list">${state.sessions.map((s) => sessionRowHtml(s, session.id)).join("")}</div>
        <p class="session-scope">最多显示最近 ${SESSION_LIST_LIMIT} 个对话</p>
      </aside>
      <section class="chat-main" aria-label="当前对话">
        <div class="chat-toolbar"><button class="ghost-btn history-toggle" id="history-toggle" type="button" aria-expanded="false" aria-controls="chat-history">对话记录</button>
          <strong id="chat-title">${escapeHtml(session.name)}</strong>
          <span class="model-chip"><i class="model-dot" aria-hidden="true"></i>心理支持</span>
          <span class="heading-spacer"></span>
          <button class="ghost-btn" id="rename-session" type="button">重命名</button>
          <button class="ghost-btn" id="export-session" type="button">导出</button>
        </div>
        <div class="message-list" id="message-list" role="log">${messageListHtml(session)}</div>
        <button class="jump-latest is-hidden" id="jump-latest" type="button">↓ 回到最新</button>
        <form class="composer" id="chat-form">
          <textarea id="chat-input" rows="1" aria-label="想说的话" aria-describedby="composer-note" placeholder="写下你想聊的事，不用急着组织好语言…"></textarea>
          <div class="composer-actions">
            <button class="stop-btn is-hidden" id="stop-btn" type="button" title="停止生成（取消排队或中断回答）" aria-label="停止生成">■</button>
            <button class="send-btn" id="send-btn" type="submit" title="发送" aria-label="发送">↑</button>
          </div>
        </form>
        <p class="composer-note" id="composer-note">AI 回答仅供参考，不替代专业诊疗。<span>Enter 发送 · Shift + Enter 换行</span></p>
      </section>
      <aside class="inspector" aria-label="说明">
        <div class="section-title"><span>慢慢聊，没关系</span></div>
        <div class="inspector-body">
          <div class="info-block">
            <h3>从你的感受开始</h3>
            <div class="prompt-preview">你可以说说发生了什么，以及它带给你的感受。不需要一次说完，也可以随时停止回答。</div>
          </div>
          <div class="info-block">
            <h3>连接状态</h3>
            <div class="info-row"><span>问答服务</span><strong id="insp-conn" data-state="unknown">检测中</strong></div>
            <button class="ghost-btn info-retry" id="insp-conn-retry" type="button">重新检测</button>
          </div>
        </div>
      </aside>
    </div>`;
}

// 会话项：选择与删除拆成两个同级原生按钮（不再在 button 内嵌 role=button 的 span）
function sessionRowHtml(s, activeId) {
  const active = s.id === activeId;
  return `
    <div class="session-row" role="listitem">
      <button class="session ${active ? "active" : ""}" type="button" data-session="${escapeHtml(s.id)}" ${active ? 'aria-current="true"' : ""}>
        <span aria-hidden="true">◌</span>
        <span class="session-copy"><span class="session-name">${escapeHtml(s.name)}</span><span class="session-time">${new Date(s.createdAt).toLocaleDateString("zh-CN")}</span></span>
      </button>
      <button class="session-delete" type="button" data-delete-session="${escapeHtml(s.id)}" aria-label="删除会话 ${escapeHtml(s.name)}" title="删除对话">×</button>
    </div>`;
}

function messageListHtml(session) {
  if (session.status === "loading" || session.status === "idle") {
    return `
      <div class="state-panel inline" role="status" aria-live="polite">
        <span class="state-spinner" aria-hidden="true"></span>
        <p class="state-title">正在加载对话记录…</p>
      </div>`;
  }
  if (session.status === "error") {
    return `
      <div class="state-panel inline error" role="alert">
        <p class="state-title">对话记录加载失败</p>
        <p class="state-detail">${escapeHtml(session.error || "请稍后重试")}</p>
        <button class="ghost-btn" id="retry-load" type="button">重试</button>
      </div>`;
  }
  if (!session.messages.length) return renderWelcome();
  return session.messages.map((m) => messageHtml(m)).join("");
}

function renderWelcome() {
  return `<div class="welcome"><div class="bot-avatar" aria-hidden="true">✦</div><span class="welcome-kicker">在这里，慢慢说</span><h2>今天，什么事让你挂心？</h2><p>可以是学习的压力、人际关系，或一个说不清的感受。<br>从你愿意分享的地方开始。</p><div class="suggestions"><button class="suggestion" type="button">考试前总是很紧张，怎么办？</button><button class="suggestion" type="button">和朋友闹矛盾了，怎么开口？</button><button class="suggestion" type="button">孩子总说睡不着，怎么沟通？</button><button class="suggestion" type="button">如何判断是否需要专业帮助？</button></div></div>`;
}

function renderSourceChips(sources) {
  if (!sources || !sources.length) return "";
  const chips = sources.map((s, i) => {
    const idx = s.index ?? i + 1;
    const title = s.title || "来源";
    const meta = [s.card_id ? `卡片 ${s.card_id}` : "", s.risk_level ? `风险等级 ${s.risk_level}` : ""].filter(Boolean).join(" · ");
    return `<span class="source-chip"${meta ? ` title="${escapeHtml(`${title} · ${meta}`)}"` : ""}>[${escapeHtml(String(idx))}] ${escapeHtml(String(title))}</span>`;
  }).join("");
  return `<div class="sources">${chips}</div>`;
}

function formatMs(value) {
  const n = typeof value === "number" ? value : parseFloat(value);
  return Number.isFinite(n) ? `${Math.round(n)}ms` : "-";
}

// 技术耗时默认收进可展开详情（先看回答与来源，需要时再看阶段耗时）
function renderTimings(timings, elapsed) {
  if (!timings && elapsed == null) return "";
  const parts = [];
  if (timings) {
    if (timings.rag_enabled === false) parts.push("RAG 关");
    if (timings.safety_enabled === false) parts.push("安全 关");
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
  if (!parts.length) return "";
  return `<details class="timing-details"><summary>技术耗时 · ${parts.length} 项</summary><div class="timing-bar">${parts.map((p) => `<span class="timing-pill">${escapeHtml(p)}</span>`).join("")}</div></details>`;
}

function renderMessageState(m) {
  if (m.error) return `<div class="message-state error">生成失败：${escapeHtml(m.errorMessage || "请稍后重试")}</div>`;
  if (m.stopped) return `<div class="message-state">已停止生成，内容可能不完整。</div>`;
  if (m.incomplete) return `<div class="message-state">回答未完整结束（连接中断），内容可能不完整。</div>`;
  return "";
}

// ============================================================
// 消息渲染：初始渲染 / 流式收尾 / 重绘 共用同一份逻辑
// （报告问题 7：关怀提示与危机样式曾在重绘后丢失，这里统一到一个函数）
// ============================================================
function messageHtml(m) {
  const isUser = m.role === "user";
  const cls = ["message", isUser ? "user" : "assistant"];
  if (m.isCrisis) cls.push("crisis");
  if (m.streaming) cls.push("streaming");
  if (m.incomplete || m.stopped) cls.push("incomplete");
  if (m.error) cls.push("error");

  const ids = m.streaming ? ' id="streaming-msg"' : "";
  const bubbleId = m.streaming ? ' id="streaming-bubble"' : "";

  // 用户原文保持原样（仅转义）；回答走受限富文本（先整体转义，再注入本函数产生的标签）
  let bodyHtml;
  if (isUser) bodyHtml = escapeHtml(m.content || "");
  else if (m.streaming && !m.content) bodyHtml = "正在生成…";
  else bodyHtml = renderAnswer(m.content || "") || "（无内容）";

  const inner = isUser
    ? bodyHtml
    : `${m.streaming ? '<span class="queue-status is-hidden"></span>' : ""}${m.streaming ? `<span class="streaming-text">${bodyHtml}</span>` : bodyHtml}`;

  const sourcesHtml = isUser ? "" : renderSourceChips(m.sources);
  const timingsHtml = isUser ? "" : renderTimings(m.timings, m.elapsed);
  const noteHtml = m.safetyNote ? `<div class="safety-note">${escapeHtml(toPlainText(m.safetyNote))}</div>` : "";
  const stateHtml = isUser ? "" : renderMessageState(m);
  const canRetry = !isUser && !m.streaming && (m.incomplete || m.error || m.stopped || m.canRetry);
  const retryHtml = canRetry ? `<div class="message-foot"><button class="ghost-btn retry-btn" type="button" data-retry="1">重新生成</button></div>` : "";

  return `<div class="${cls.join(" ")}"${ids}><div><div class="message-meta">${isUser ? "你" : "心理问答"}</div><div class="message-bubble"${bubbleId}>${inner}${sourcesHtml}${timingsHtml}${noteHtml}</div>${stateHtml}${retryHtml}</div></div>`;
}

// ============================================================
// 渲染：交互绑定
// ============================================================
function bindChat(page, session) {
  page.querySelectorAll("[data-session]").forEach((b) => {
    b.addEventListener("click", () => selectSession(b.dataset.session));
  });
  page.querySelectorAll("[data-delete-session]").forEach((b) => {
    b.addEventListener("click", () => deleteSession(b.dataset.deleteSession, b));
  });
  page.querySelector("#new-session")?.addEventListener("click", () => createSession());
  page.querySelector("#rename-session")?.addEventListener("click", () => renameCurrent(session.id));
  page.querySelector("#export-session")?.addEventListener("click", () => exportSession(session.id));
  page.querySelector("#retry-load")?.addEventListener("click", () => activateSession(session.id));
  page.querySelectorAll("[data-retry]").forEach((b) => {
    b.addEventListener("click", () => retryLastUserMessage(session.id));
  });

  // 小屏历史浮层：Esc 关闭 + 点击外部关闭
  const toggle = page.querySelector("#history-toggle");
  const layout = page.querySelector(".chat-layout");
  toggle?.addEventListener("click", () => {
    const open = layout.classList.toggle("history-open");
    toggle.setAttribute("aria-expanded", String(open));
    if (open) bindHistoryPanelDismiss(layout, toggle);
    else { historyPanelCleanup?.(); historyPanelCleanup = null; }
  });

  // 输入区：草稿按会话隔离（内存 Map），每次输入即记录，切换后按目标会话恢复
  const form = page.querySelector("#chat-form");
  const chatInput = page.querySelector("#chat-input");
  if (chatInput) {
    const autoGrow = () => {
      chatInput.style.height = "auto";
      chatInput.style.height = `${Math.min(chatInput.scrollHeight, DRAFT_MAX_HEIGHT)}px`;
    };
    const draft = drafts.get(session.id);
    if (draft) { chatInput.value = draft; autoGrow(); }
    chatInput.addEventListener("input", () => {
      drafts.set(session.id, chatInput.value);
      autoGrow();
    });
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
        e.preventDefault();
        form?.requestSubmit();
      }
    });
  }
  form?.addEventListener("submit", (e) => {
    e.preventDefault();
    submitComposer(session.id);
  });

  // 欢迎页快捷问题：填入输入框后直接发送
  page.querySelectorAll(".suggestion").forEach((b) => b.addEventListener("click", () => {
    const input = page.querySelector("#chat-input");
    if (!input) return;
    input.value = b.textContent;
    drafts.set(session.id, input.value);
    input.focus();
    page.querySelector("#chat-form")?.requestSubmit();
  }));

  page.querySelector("#stop-btn")?.addEventListener("click", () => stopStream(state.activeSessionId));
  page.querySelector("#insp-conn-retry")?.addEventListener("click", () => checkConnection());

  const list = page.querySelector("#message-list");
  list?.addEventListener("scroll", () => updateJumpButton(list));
  page.querySelector("#jump-latest")?.addEventListener("click", () => {
    if (list) { list.scrollTop = list.scrollHeight; updateJumpButton(list); }
  });
}

function bindHistoryPanelDismiss(layout, toggle) {
  historyPanelCleanup?.();
  const sidebar = layout.querySelector(".sidebar");
  const close = () => {
    layout.classList.remove("history-open");
    toggle.setAttribute("aria-expanded", "false");
    cleanup();
  };
  const onKey = (e) => {
    if (e.key === "Escape") { e.preventDefault(); toggle.focus(); close(); }
  };
  const onClick = (e) => {
    // 点击开关本身交给开关自己的处理器（否则会被「先关后开」抵消）
    if (e.target === toggle || toggle.contains(e.target)) return;
    close();
  };
  const cleanup = () => {
    document.removeEventListener("keydown", onKey, true);
    document.removeEventListener("click", onClick, true);
    historyPanelCleanup = null;
  };
  document.addEventListener("keydown", onKey, true);
  document.addEventListener("click", onClick, true);
  historyPanelCleanup = cleanup;
}

function updateJumpButton(list = document.querySelector("#message-list")) {
  const btn = document.querySelector("#jump-latest");
  if (!btn || !list) return;
  btn.classList.toggle("is-hidden", isNearBottom(list, 60));
}

// ============================================================
// 提交（submit 层做忙碌检查，并绑定渲染时的会话 id）
// ============================================================
function submitComposer(renderedSessionId) {
  // 页面可能已被重绘：只有仍然渲染着同一个会话时才继续，避免发到别的会话
  if (renderedSessionId !== state.activeSessionId) return;
  const session = getSession(renderedSessionId);
  if (!session) return;
  if (session.status === "loading" || session.status === "idle") {
    showToast("对话正在加载，请稍候", "info");
    return;
  }
  if (streams.has(renderedSessionId)) {
    showToast("上一条回复还在生成中，请先停止或等待完成", "info");
    return;
  }
  const input = document.querySelector("#chat-input");
  const content = input ? input.value.trim() : "";
  if (!content) return;
  input.value = "";
  input.style.height = "auto";
  drafts.delete(renderedSessionId);
  sendChat(content, { sessionId: renderedSessionId });
}

// ============================================================
// 流式问答
// ============================================================
function scheduleStreamSync(handle) {
  if (handle.rafId) return;
  handle.rafId = scheduleFrame(() => {
    handle.rafId = 0;
    syncStreamText(handle);
  });
}

// 重绘后重新绑定流式节点：把内存里的内容写回新 DOM，避免继续写入游离节点
function acquireStreamEls(handle) {
  if (!handle || handle.sessionId !== state.activeSessionId) {
    if (handle) handle.els = null;
    return null;
  }
  const bubble = document.querySelector("#streaming-bubble");
  if (!bubble) { handle.els = null; return null; }
  handle.els = {
    bubble,
    textEl: bubble.querySelector(".streaming-text"),
    statusEl: bubble.querySelector(".queue-status"),
  };
  if (handle.els.textEl) {
    handle.els.textEl.textContent = toPlainText(handle.placeholder.content) || "正在生成…";
  }
  if (handle.queueText && handle.els.statusEl) {
    handle.els.statusEl.textContent = handle.queueText;
    handle.els.statusEl.classList.remove("is-hidden");
  }
  return handle.els;
}

function rebindVisibleStream() {
  const handle = streams.get(state.activeSessionId);
  if (!handle) return false;
  acquireStreamEls(handle);
  return true;
}

// 仅更新当前可见会话的流式气泡；先判断是否贴底，再决定是否跟随滚动
function syncStreamText(handle) {
  if (!handle || handle.sessionId !== state.activeSessionId) return;
  const els = handle.els || acquireStreamEls(handle);
  if (!els || !els.textEl) return;
  const list = document.querySelector("#message-list");
  const stick = isNearBottom(list);
  els.textEl.textContent = toPlainText(handle.placeholder.content) || "正在生成…";
  if (list) {
    if (stick) list.scrollTop = list.scrollHeight;
    else updateJumpButton(list);
  }
}

function showQueueStatus(handle, text) {
  handle.queueText = text;
  const els = handle.els || acquireStreamEls(handle);
  if (!els || !els.statusEl) return;
  els.statusEl.textContent = text;
  els.statusEl.classList.remove("is-hidden");
}

function hideQueueStatus(handle) {
  handle.queueText = "";
  const els = handle.els;
  if (!els || !els.statusEl) return;
  els.statusEl.textContent = "";
  els.statusEl.classList.add("is-hidden");
}

// 准入类错误码 → 用户可读文案
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

async function sendChat(content, { sessionId = state.activeSessionId, isRetry = false } = {}) {
  const epoch = state.epoch;
  const session = getSession(sessionId);
  if (!session) return;
  if (session.status === "loading" || session.status === "idle") return;
  if (streams.has(sessionId)) {
    showToast("上一条回复还在生成中", "info");
    return;
  }

  let titleForServer = null;
  if (!isRetry) {
    session.messages.push({ role: "user", content });
    // 首次提问：立即用问题自动命名（清洗空白 + 限长），并在请求中带给后端持久化
    if (session.messages.length === 1) {
      session.name = autoName(content);
      titleForServer = session.name;
    }
  }
  const placeholder = { role: "assistant", content: "", sources: [], timings: null, streaming: true };
  session.messages.push(placeholder);

  const handle = {
    sessionId,
    ac: new AbortController(),
    placeholder,
    requestId: null,
    els: null,
    queueText: "",
    rafId: 0,
    cancelReason: null,
    question: content,
    titleForServer,
  };
  streams.set(sessionId, handle);

  if (sessionId === state.activeSessionId) {
    renderChat();          // 用户消息 + 占位气泡一次性进入 DOM
    rebindVisibleStream(); // 把句柄接到新渲染出的流式节点上
  }
  updateComposerState();
  announce("正在生成回答");

  await runStream(handle);

  // 收尾：状态判定 + 若仍可见则整页重绘（与统一渲染逻辑保持一致）
  if (epoch !== state.epoch) return;
  if (streams.get(sessionId) === handle) streams.delete(sessionId);
  if (sessionId === state.activeSessionId) renderChat();
  else updateComposerState();
  focusInput();
}

async function runStream(handle) {
  const { sessionId, ac, placeholder } = handle;
  const startedAt = performance.now();
  let sawDone = false;
  let failure = null;
  let aborted = false;

  const onEvent = ({ event, data }) => {
    switch (event) {
      case "queue":
        // 排队中：展示实时位置（position 为 0 基下标 = 前方请求数）
        handle.requestId = data.request_id || handle.requestId;
        if (typeof data.position === "number") showQueueStatus(handle, `排队中 · 前方还有 ${data.position} 个请求`);
        break;
      case "started":
        handle.requestId = data.request_id || handle.requestId;
        hideQueueStatus(handle);
        break;
      case "sources":
        // 只暂存来源，等回答收尾后再显示：先看答案 → 再展示依据文档与耗时
        placeholder.sources = Array.isArray(data.sources) ? data.sources : [];
        break;
      case "token":
        hideQueueStatus(handle);
        placeholder.content += data.text || "";
        scheduleStreamSync(handle);
        break;
      case "done":
        sawDone = true;
        hideQueueStatus(handle);
        if (typeof data.answer === "string" && data.answer) placeholder.content = data.answer;
        if (data.timings) placeholder.timings = data.timings;
        if (data.safety_note) placeholder.safetyNote = data.safety_note;
        if (data.is_crisis_response) placeholder.isCrisis = true;
        placeholder.elapsed = Math.round(performance.now() - startedAt);
        // 高危危机拦截只发 done 不发 token，这里必须回写，否则界面停在「正在生成…」
        syncStreamText(handle);
        break;
      case "error":
        hideQueueStatus(handle);
        throw new ApiError(admissionErrorText(data) || data.detail || "生成失败", {
          code: data.code || null,
          detail: data.detail || "",
        });
      default:
        break; // 未知事件忽略（后端向后兼容约定）
    }
  };

  try {
    const body = { question: handle.question, session_id: sessionId };
    if (handle.titleForServer) body.title = handle.titleForServer;
    const resp = await apiFetch("/api/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
    if (!resp.body) throw new ApiError("当前环境不支持流式响应", { status: 0 });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    const parser = createSseParser();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const frame of parser.push(decoder.decode(value, { stream: true }))) onEvent(frame);
    }
    for (const frame of parser.end()) onEvent(frame);
  } catch (e) {
    if (e?.name === "AbortError") aborted = true;
    else failure = failure || e?.message || "生成失败";
  }

  // ---- 收尾：显式区分 完成 / 用户停止 / 中断 / 失败 / 无 done 的 EOF ----
  placeholder.streaming = false;
  if (handle.rafId) { try { cancelAnimationFrame(handle.rafId); } catch { /* noop */ } handle.rafId = 0; }

  const reason = handle.cancelReason;
  if (aborted) {
    placeholder.incomplete = true;
    placeholder.stopped = reason === "user";
    placeholder.errorMessage = reason === "user" ? "已停止生成" : "生成已被中断";
  } else if (failure) {
    if (placeholder.content) {
      // 部分内容 + 出错：保留内容但显式标为未完成，不要伪装成完整回答
      placeholder.incomplete = true;
      placeholder.errorMessage = failure;
    } else {
      placeholder.error = true;
      placeholder.errorMessage = failure;
      placeholder.content = "";
    }
    showToast(failure, "error");
  } else if (!sawDone) {
    // 读到 EOF 但从未收到业务 done：不是成功
    placeholder.incomplete = true;
    placeholder.errorMessage = "连接提前结束";
    showToast("连接提前结束，回答可能不完整", "error");
  } else {
    showToast(`回答已生成 · ${placeholder.elapsed ?? Math.round(performance.now() - startedAt)}ms`, "success");
    announce("回答已生成");
  }
}

function cancelStream(sessionId, { reason = "switch" } = {}) {
  const handle = streams.get(sessionId);
  if (!handle) return false;
  handle.cancelReason = reason;
  try { handle.ac.abort(); } catch { /* noop */ }
  // 通知服务端释放准入占位（幂等，失败不影响本地）
  if (handle.requestId) {
    apiFetch(`/api/query/requests/${encodeURIComponent(handle.requestId)}`, { method: "DELETE" }).catch(() => { /* 尽力而为 */ });
  }
  // 立即标记，保证紧接着的重绘不会把已停止的气泡画成「生成中」
  const ph = handle.placeholder;
  if (ph && ph.streaming) {
    ph.streaming = false;
    ph.incomplete = true;
    ph.stopped = reason === "user";
    ph.errorMessage = reason === "user" ? "已停止生成" : "生成已被中断";
  }
  return true;
}

function cancelAllStreams() {
  for (const sessionId of [...streams.keys()]) cancelStream(sessionId, { reason: "switch" });
  streams.clear();
}

function stopStream(sessionId = state.activeSessionId) {
  if (!sessionId || !streams.has(sessionId)) return;
  cancelStream(sessionId, { reason: "user" });
  announce("已停止生成");
  if (sessionId === state.activeSessionId) renderChat();
}

// 未完成/失败的回答：从最后一条用户消息重新生成
function retryLastUserMessage(sessionId) {
  const session = getSession(sessionId);
  if (!session) return;
  if (streams.has(sessionId)) { showToast("正在生成中，请先停止", "info"); return; }
  let idx = -1;
  for (let i = session.messages.length - 1; i >= 0; i--) {
    if (session.messages[i].role === "user") { idx = i; break; }
  }
  if (idx < 0) return;
  const question = session.messages[idx].content;
  session.messages = session.messages.slice(0, idx + 1);
  sendChat(question, { sessionId, isRetry: true });
}

// 只有当前会话的活动流可以改动当前页面控件
function updateComposerState() {
  const sessionId = state.activeSessionId;
  const session = getSession(sessionId);
  const handle = streams.get(sessionId);
  const loading = session?.status === "loading" || session?.status === "idle";
  const sendBtn = document.querySelector("#send-btn");
  const stopBtn = document.querySelector("#stop-btn");
  const input = document.querySelector("#chat-input");
  if (sendBtn) {
    sendBtn.disabled = !!handle || loading;
    sendBtn.innerHTML = handle ? '<span class="spinner"></span>' : "↑";
    sendBtn.setAttribute("aria-busy", handle ? "true" : "false");
  }
  if (stopBtn) stopBtn.classList.toggle("is-hidden", !handle);
  if (input) {
    input.disabled = loading;
    input.placeholder = loading
      ? "正在加载对话记录…"
      : "写下你想聊的事，不用急着组织好语言…";
  }
}

// ============================================================
// 会话数据（服务端为唯一数据源）
// ============================================================
async function loadSessions(epoch) {
  const list = await api("GET", `/api/sessions?limit=${SESSION_LIST_LIMIT}`);
  if (epoch !== state.epoch) return STALE;
  const prev = new Map(state.sessions.map((s) => [s.id, s]));
  state.sessions = (Array.isArray(list) ? list : []).map((s) => {
    const old = prev.get(s.id);
    const loaded = !!(old && old.loaded);
    return {
      id: s.id,
      name: s.title || "新的对话",
      createdAt: (s.updated_at || s.created_at) ? new Date(s.updated_at || s.created_at).getTime() : Date.now(),
      messages: loaded ? old.messages : [],
      loaded,
      status: loaded ? "ready" : "idle",
      error: "",
    };
  });
  if (!getSession(state.activeSessionId)) {
    state.activeSessionId = state.sessions.length ? state.sessions[0].id : null;
  }
}

async function loadSessionMessages(session, epoch) {
  try {
    const msgs = await api("GET", `/api/sessions/${encodeURIComponent(session.id)}/messages`);
    if (epoch !== state.epoch) return false;
    // 后端按语义存 human/ai；前端显示用 user/assistant，这里做一次对账映射
    const roleMap = { human: "user", ai: "assistant" };
    session.messages = (Array.isArray(msgs) ? msgs : []).map((m) => ({
      role: roleMap[m.role] || m.role,
      content: m.content,
      createdAt: m.created_at,
    }));
    session.loaded = true;
    session.status = "ready";
    session.error = "";
    // 未命名会话：用最早一条用户消息自动取名（与后端命名规则一致）
    if (!session.name || session.name === "新会话" || session.name === "新的对话") {
      const firstUser = session.messages.find((m) => m.role === "user");
      if (firstUser) session.name = autoName(firstUser.content);
    }
    return true;
  } catch (e) {
    if (epoch !== state.epoch) return false;
    // 加载失败必须显式暴露（不再静默吞掉，用户无法分辨「没有历史」与「加载失败」）
    session.status = "error";
    session.error = e?.message || "对话记录加载失败";
    return false;
  }
}

// 统一的会话激活逻辑：切换 / 删除回退 / 刷新 / 重试 都走这里
async function activateSession(id, { epoch = state.epoch, renderNow = true } = {}) {
  const session = getSession(id);
  if (!session) return;
  state.activeSessionId = id;
  if (!session.loaded && session.status !== "loading") {
    session.status = "loading";
    session.error = "";
    if (renderNow) renderChat();   // 立即展示加载态并禁用发送
    await loadSessionMessages(session, epoch);
    if (epoch !== state.epoch) return;
  }
  if (renderNow) renderChat();
}

function selectSession(id) {
  if (!id || id === state.activeSessionId) return;
  // 切换会话前中断旧会话正在进行的生成（统一取消入口）
  cancelStream(state.activeSessionId, { reason: "switch" });
  activateSession(id);
}

async function ensureActiveSession(epoch) {
  if (epoch !== state.epoch) return STALE;
  if (!state.sessions.length) {
    await createSession({ epoch, announce: false, cancelCurrent: false });
    return;
  }
  if (!getSession(state.activeSessionId)) state.activeSessionId = state.sessions[0].id;
  const active = getSession(state.activeSessionId);
  if (active && !active.loaded) {
    active.status = "loading";
    render();
    await loadSessionMessages(active, epoch);
  }
}

async function createSession({ epoch = state.epoch, announce: notify = true, cancelCurrent = true } = {}) {
  if (cancelCurrent) cancelStream(state.activeSessionId, { reason: "switch" });
  try {
    const data = await api("POST", "/api/sessions", { name: "新的对话" });
    if (epoch !== state.epoch) return;
    const session = { id: data.id, name: data.name || "新的对话", createdAt: Date.now(), messages: [], loaded: true, status: "ready", error: "" };
    state.sessions.unshift(session);
    state.activeSessionId = session.id;
    renderChat();
    if (notify) { showToast("已新建对话", "success"); focusInput(); }
  } catch (e) {
    if (epoch !== state.epoch) return;
    showToast(`新建失败：${e.message}`, "error");
  }
}

async function renameCurrent(id) {
  const session = getSession(id);
  if (!session) return;
  const name = askSessionName(session.name);
  if (!name) return;
  await renameSession(id, name);
}

// 原生 prompt 在部分嵌入环境（无 dialog 支持的 webview / 测试环境）不可用，
// 这里兜底：取不到名字时保持原样，不要让重命名把整个页面抛崩
function askSessionName(currentName) {
  try {
    const value = window.prompt("输入新的对话名称", currentName);
    return value == null ? null : String(value).trim() || null;
  } catch {
    showToast("当前环境不支持重命名输入框，请在浏览器中操作", "error");
    return null;
  }
}

async function renameSession(id, name) {
  const epoch = state.epoch;
  const session = getSession(id);
  if (!session) return;
  try {
    const data = await api("PATCH", `/api/sessions/${encodeURIComponent(id)}`, { name });
    if (epoch !== state.epoch) return;
    session.name = data?.name || name;
    // 只更新标题与侧栏文案：不再整页重绘（重绘会销毁流式气泡、丢失滚动位置）
    updateSessionTitleInDom(session);
  } catch (e) {
    if (epoch !== state.epoch) return;
    showToast(`重命名失败：${e.message}`, "error");
  }
}

function updateSessionTitleInDom(session) {
  const sel = attrValue(session.id);
  const nameEl = document.querySelector(`[data-session="${sel}"] .session-name`);
  if (nameEl) nameEl.textContent = session.name;
  const del = document.querySelector(`[data-delete-session="${sel}"]`);
  if (del) del.setAttribute("aria-label", `删除会话 ${session.name}`);
  if (session.id === state.activeSessionId) {
    const title = document.querySelector("#chat-title");
    if (title) title.textContent = session.name;
  }
}

async function deleteSession(id, trigger = null) {
  const session = getSession(id);
  if (!session) return;
  const ok = await confirmDialog({
    title: "删除对话",
    message: `确定删除对话「${session.name}」吗？删除后不可恢复。`,
    confirmText: "删除",
    danger: true, // 危险操作默认聚焦「取消」
    trigger,
  });
  if (!ok) return;

  const epoch = state.epoch;
  cancelStream(id, { reason: "delete" });
  try {
    await api("DELETE", `/api/sessions/${encodeURIComponent(id)}`);
    if (epoch !== state.epoch) return;
    const wasActive = state.activeSessionId === id;
    state.sessions = state.sessions.filter((s) => s.id !== id);
    drafts.delete(id);
    // 始终至少保留一个会话（与旧行为一致）
    if (!state.sessions.length) {
      await createSession({ epoch, announce: false, cancelCurrent: false });
      return;
    }
    if (wasActive) {
      // 删除当前会话后回退到剩余会话：复用统一激活逻辑（会加载它的历史，不再显示空欢迎页）
      await activateSession(state.sessions[0].id, { epoch });
    } else {
      renderChat();
    }
  } catch (e) {
    if (epoch !== state.epoch) return;
    showToast(`删除失败：${e.message}`, "error");
  }
}

function exportSession(id) {
  const session = getSession(id);
  if (!session) return;
  const text = session.messages
    .filter((m) => (m.content || "").trim())
    .map((m) => {
      const who = m.role === "user" ? "问" : "答";
      const flag = (m.incomplete || m.error) ? "（未完成）" : "";
      return `${who}${flag}：${m.content}`;
    })
    .join("\n\n");
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const safeName = String(session.name || "对话").replace(/[\\/:*?"<>|]/g, "_");
  a.href = url;
  a.download = `对话-${safeName}-${new Date().toISOString().slice(0, 10)}.txt`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("对话已导出", "success");
}

// ============================================================
// 启动
// ============================================================
async function bootWorkspace(epoch = state.epoch) {
  state.view = "app";
  boot = { status: "loading", error: "" };
  render();  // 先渲染骨架（顶栏 + 加载态），避免加载失败时留下空白页
  try {
    await loadSessions(epoch);
    if (epoch !== state.epoch) return;
    await ensureActiveSession(epoch);
    if (epoch !== state.epoch) return;
    boot = { status: "ready", error: "" };
    render();
  } catch (e) {
    if (epoch !== state.epoch) return;
    boot = { status: "error", error: e?.message || "数据加载失败，请稍后重试" };
    render();
  }
  checkConnection();
}

// 返回首页会重新加载同一个应用：有未发送草稿时先确认，避免无意丢失
function hasUnsavedDraft() {
  const live = document.querySelector("#chat-input");
  if (live && live.value.trim()) return true;
  for (const value of drafts.values()) if (value && value.trim()) return true;
  return false;
}

function bindHomeGuard() {
  document.addEventListener("click", async (e) => {
    const link = e.target?.closest?.(".home-link");
    if (!link) return;
    if (!hasUnsavedDraft()) return;
    e.preventDefault();
    const ok = await confirmDialog({
      title: "离开当前页面？",
      message: "还有没发送的内容，离开后会丢失。",
      confirmText: "离开",
      danger: true,
      trigger: link,
    });
    if (ok) {
      drafts.clear();
      window.location.assign(link.getAttribute("href") || "/");
    }
  });
}

(async function init() {
  configureApi({ onSessionExpired: handleSessionExpired });
  bindHomeGuard();

  const token = getToken();
  if (!token) {
    state.view = "auth";
    renderAuthView();
    return;
  }
  const epoch = state.epoch;
  try {
    // 校验 token 有效性（登录/注册之外的场景：401 直接回登录页，不弹「登录已过期」）
    const me = await api("GET", "/api/auth/me", undefined, { auth: "none" });
    if (epoch !== state.epoch) return;
    state.currentUser = me;
  } catch (e) {
    if (epoch !== state.epoch) return;
    resetAuthState();
    renderAuthView();
    if (e?.status && e.status !== 401) showToast(e.message || "无法校验登录状态", "error");
    return;
  }
  await bootWorkspace(epoch);
})();
