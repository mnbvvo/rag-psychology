// ============================================================
// 心理 RAG · 提示词工作台  (SPA, 设计/功能参考 xinli Prompt Lab)
// 三页：提示词管理 / 对话联调 / 提示词对比
// 后端接本项目 FastAPI：/api/system-prompt（提示词库）+ /api/query（RAG 问答）
// ============================================================
const defaultState = {
  activePage: "prompt",            // prompt | chat | compare
  prompts: [],                     // 提示词库 { id, name, content }
  activePromptId: "",              // 当前激活提示词 id（RAG 默认使用）
  selectedPromptId: "",            // 提示词管理页当前选中编辑的 id
  defaultPrompts: [],              // 出厂默认提示词库（只读参考）
  sessions: [],                    // 全部来自服务端 SQLite（含 messages）
  activeSessionId: null,           // 当前打开的会话 id
  chatPromptId: "",                // 当前对话使用的提示词 id（空=激活提示词）
  compareInput: "孩子最近总说睡不着，作为家长该怎么和他温和地沟通？",
  compareSelections: { a: "", b: "" }, // 对比页 A/B 分别使用的提示词 id
  compareHistory: [],              // 全部来自服务端 SQLite
  currentCompare: null,
};

// 所有业务数据（提示词 / 会话 / 消息 / 对比历史）均持久化在服务端 SQLite，
// 前端不再使用浏览器缓存（localStorage）。state 仅作为当前页面运行时的内存镜像。
let state = structuredClone(defaultState);
let toastTimer;
let comparing = false;

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
  const root = document.querySelector("#toast");
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
    parts.push(`安全 ${formatMs(timings.safety)}`);
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

function activePrompt() {
  return state.prompts.find((p) => p.id === state.activePromptId) || state.prompts[0] || { id: "", name: "", content: "" };
}

function selectedPrompt() {
  return state.prompts.find((p) => p.id === state.selectedPromptId) || state.prompts[0] || { id: "", name: "", content: "" };
}

function getPromptById(id) {
  return state.prompts.find((p) => p.id === id) || state.defaultPrompts.find((p) => p.id === id) || { id: "", name: "", content: "" };
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* noop */ }
  if (!res.ok) throw new Error(data?.detail || `请求失败（${res.status}）`);
  return data;
}

// ---------------- 后端数据加载 ----------------
async function loadPromptConfig() {
  try {
    const data = await api("GET", "/api/system-prompt");
    const current = data.current || {};
    const defaults = data.default || {};
    state.prompts = Array.isArray(current.prompts) ? current.prompts : [];
    state.defaultPrompts = Array.isArray(defaults.prompts) ? defaults.prompts : [];
    state.activePromptId = current.activeId || (state.prompts[0]?.id || "");
    if (!state.selectedPromptId || !state.prompts.find((p) => p.id === state.selectedPromptId)) {
      state.selectedPromptId = state.activePromptId || state.prompts[0]?.id || "";
    }
    // 对比/对话选择兜底
    const anyPromptId = state.prompts[0]?.id || state.defaultPrompts[0]?.id || "";
    if (!state.compareSelections.a || !getPromptById(state.compareSelections.a).id) state.compareSelections.a = state.defaultPrompts[0]?.id || anyPromptId;
    if (!state.compareSelections.b || !getPromptById(state.compareSelections.b).id) state.compareSelections.b = state.activePromptId || anyPromptId;
    // chatPromptId 为空表示"跟随激活提示词"；不要填成 active id，否则下拉会重复显示激活项
    if (state.chatPromptId && (state.chatPromptId === state.activePromptId || !state.prompts.find((p) => p.id === state.chatPromptId))) state.chatPromptId = "";
  } catch (e) {
    showToast(`读取提示词库失败：${e.message}`, "error");
  }
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

// ---------------- 渲染框架 ----------------
function render() {
  const root = document.querySelector("#app");
  if (!root.querySelector(".topbar")) renderShell();
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${state.activePage}`));
  document.querySelectorAll("[data-page]").forEach((b) => b.classList.toggle("active", b.dataset.page === state.activePage));
  if (state.activePage === "prompt") renderPromptManager();
  if (state.activePage === "chat") renderChat();
  if (state.activePage === "compare") renderCompare();
  updateSaveState();
}

function renderShell() {
  document.querySelector("#app").innerHTML = `
    <div class="app-shell">
      <header class="topbar">
        <div class="brand"><div class="brand-mark">✦</div><div><div class="brand-name">心理 RAG</div><div class="brand-sub">提示词工作台</div></div></div>
        <nav class="topnav" aria-label="主导航">
          <button class="nav-btn" data-page="prompt">提示词管理</button>
          <button class="nav-btn" data-page="chat">对话联调</button>
          <button class="nav-btn" data-page="compare">提示词对比</button>
        </nav>
        <div class="top-actions">
          <span id="conn" class="conn"><span class="dot"></span><span class="label">连接中…</span></span>
          <span id="save-state" class="save-state">已同步至服务器</span>
        </div>
      </header>
      <main class="workspace">
        <section id="page-prompt" class="page"></section>
        <section id="page-chat" class="page"></section>
        <section id="page-compare" class="page"></section>
      </main>
      <div id="toast" class="toast hidden"></div>
    </div>`;
  document.querySelectorAll("[data-page]").forEach((b) => b.addEventListener("click", () => { state.activePage = b.dataset.page; saveState(); render(); }));
  checkConnection();
}

// ============================================================
// 页 1：提示词管理（提示词库 + 编辑器 + 预览）
// ============================================================
function renderPreview(content) {
  const base = (content || "").trim();
  return `${base}\n\n参考资料:\n{context}`;
}

function renderPromptManager() {
  const page = document.querySelector("#page-prompt");
  const prompt = selectedPrompt();
  const activeId = state.activePromptId;
  page.innerHTML = `
    <div class="prompt-manager">
      <aside class="prompt-library">
        <div class="library-header">
          <h2>提示词库</h2>
          <button class="add-prompt-btn" id="add-prompt">＋ 新建提示词</button>
        </div>
        <div class="prompt-list" id="prompt-list">
          ${state.prompts.map((p) => renderPromptItem(p, activeId)).join("")}
        </div>
      </aside>
      <section class="prompt-editor-area">
        <div class="editor-toolbar">
          <input id="prompt-name" value="${escapeHtml(prompt.name)}" placeholder="提示词名称" ${state.prompts.length ? "" : "disabled"} />
          <span class="spacer"></span>
          <span class="status" id="editor-status">${prompt.id === activeId ? "当前激活 · RAG 默认使用" : "未激活"}</span>
          <button class="ghost-btn" id="set-active-btn" ${prompt.id === activeId || !prompt.id ? "disabled" : ""}>设为激活</button>
          <button class="primary-btn" id="save-prompt-btn" ${!prompt.id ? "disabled" : ""}>保存并同步</button>
        </div>
        <div class="editor-body">
          <div class="editor-main">
            <div class="editor-header">✎ 编辑内容</div>
            <textarea id="prompt-content" placeholder="输入系统提示词内容…" ${!prompt.id ? "disabled" : ""}>${escapeHtml(prompt.content)}</textarea>
            <div class="editor-foot">
              <span>按 Ctrl/Cmd + S 快速保存</span>
              <span id="prompt-chars">${prompt.content?.length || 0} 字</span>
            </div>
          </div>
          <aside class="preview-panel">
            <div class="preview-card">
              <header>组合后提示词预览</header>
              <div class="preview-content" id="preview-box">${prompt.content ? escapeHtml(renderPreview(prompt.content)) : '<span class="placeholder">在左侧编辑器输入内容以查看预览</span>'}</div>
            </div>
            <div class="preview-info">
              <strong>提示</strong><br/>
              这里展示系统提示词与参考资料占位符拼接后的最终效果，即实际发送给模型的内容。激活的提示词会被 RAG 问答默认使用。
            </div>
          </aside>
        </div>
      </section>
    </div>`;

  bindPromptManagerEvents(prompt);
}

function renderPromptItem(p, activeId) {
  const isActive = p.id === activeId;
  return `
    <div class="prompt-item ${p.id === state.selectedPromptId ? "active" : ""}">
      <button class="prompt-item-main" data-prompt-id="${p.id}">
        <span class="item-dot"></span>
        <span class="item-main">
          <div class="item-name" id="prompt-name-${p.id}">${escapeHtml(p.name)}${isActive ? '<span class="badge-active">激活</span>' : ""}</div>
          <div class="item-meta">${p.content?.length || 0} 字 · ${isActive ? "RAG 默认" : "未激活"}</div>
        </span>
      </button>
      <button class="item-delete" data-delete="${p.id}" title="删除">×</button>
    </div>`;
}

// 切换选中提示词时的局部刷新：只更新列表选中态与编辑器内容，不重建整页（消除抖动）
function refreshPromptEditor() {
  const page = document.querySelector("#page-prompt");
  const prompt = selectedPrompt();
  const activeId = state.activePromptId;

  // 列表选中态
  page.querySelectorAll(".prompt-item").forEach((el) => {
    const btn = el.querySelector("[data-prompt-id]");
    el.classList.toggle("active", !!btn && btn.dataset.promptId === prompt.id);
  });

  const nameInput = page.querySelector("#prompt-name");
  const contentInput = page.querySelector("#prompt-content");
  const previewBox = page.querySelector("#preview-box");
  const statusEl = page.querySelector("#editor-status");
  const setActiveBtn = page.querySelector("#set-active-btn");
  const charCount = page.querySelector("#prompt-chars");

  if (nameInput) nameInput.value = prompt.name;
  if (contentInput) contentInput.value = prompt.content;
  if (previewBox) previewBox.innerHTML = prompt.content ? escapeHtml(renderPreview(prompt.content)) : '<span class="placeholder">在左侧编辑器输入内容以查看预览</span>';
  if (statusEl) statusEl.textContent = prompt.id === activeId ? "当前激活 · RAG 默认使用" : "未激活";
  if (setActiveBtn) setActiveBtn.disabled = prompt.id === activeId || !prompt.id;
  if (charCount) charCount.textContent = `${prompt.content?.length || 0} 字`;
}

// 提示词列表事件绑定（选中切换 / 删除）：列表被局部重建后需重新调用。
// 注意：新建按钮(#add-prompt)在列表容器之外，由 renderPromptManager 整页渲染时绑定一次，避免重复监听。
function bindPromptListEvents(page) {
  page.querySelectorAll("[data-prompt-id]").forEach((b) => {
    b.addEventListener("click", () => {
      state.selectedPromptId = b.dataset.promptId;
      saveState();
      // 局部刷新选中项与编辑器内容，避免整页重建导致抖动
      refreshPromptEditor();
    });
  });

  page.querySelectorAll("[data-delete]").forEach((b) => {
    b.addEventListener("click", async (e) => {
      e.stopPropagation();
      const p = getPromptById(b.dataset.delete);
      if (state.prompts.length <= 1) return showToast("至少保留一条提示词", "error");
      const ok = await confirmDialog({ title: "删除提示词", message: `确定删除提示词「${p.name}」吗？删除后不可恢复。`, confirmText: "删除", danger: true });
      if (!ok) return;
      await updatePrompt({ deleteId: p.id });
    });
  });
}

// 提示词页局部刷新：重建列表（激活 badge / 新增项 / 字数）+ 刷新编辑器，不重建整页（消除抖动）
function refreshPromptPage() {
  const page = document.querySelector("#page-prompt");
  if (!page) return;
  const list = page.querySelector("#prompt-list");
  if (list) {
    list.innerHTML = state.prompts.map((p) => renderPromptItem(p, state.activePromptId)).join("");
    bindPromptListEvents(page);
  }
  refreshPromptEditor();
}

function bindPromptManagerEvents(prompt) {
  const page = document.querySelector("#page-prompt");

  bindPromptListEvents(page);

  // 新建按钮在列表容器外，仅在整页渲染时绑定一次（避免局部刷新重复监听）
  const addBtn = page.querySelector("#add-prompt");
  if (addBtn) addBtn.addEventListener("click", addPrompt);

  const setActiveBtn = page.querySelector("#set-active-btn");
  if (setActiveBtn) setActiveBtn.addEventListener("click", () => setActivePrompt(selectedPrompt().id));

  const saveBtn = page.querySelector("#save-prompt-btn");
  if (saveBtn) saveBtn.addEventListener("click", saveSelectedPrompt);

  const nameInput = page.querySelector("#prompt-name");
  const contentInput = page.querySelector("#prompt-content");
  const previewBox = page.querySelector("#preview-box");
  const charCount = page.querySelector("#prompt-chars");

  if (nameInput) {
    // 实时同步名字到 state，避免按 Ctrl+S 时还没触发 change 导致保存旧名字。
    // 用 selectedPrompt() 实时取当前选中项（选中切换后闭包里的 prompt 已过期）
    nameInput.addEventListener("input", () => {
      const p = selectedPrompt();
      if (!p.id) return;
      p.name = nameInput.value;
      saveState();
      // 只刷新列表项文字，不重绘整个编辑器，避免输入焦点丢失
      const listNameEl = document.querySelector(`#prompt-name-${p.id}`);
      if (listNameEl) listNameEl.textContent = nameInput.value || "未命名";
    });
  }

  if (contentInput) {
    contentInput.addEventListener("input", () => {
      const p = selectedPrompt();
      if (!p.id) return;
      p.content = contentInput.value;
      saveState();
      if (previewBox) previewBox.textContent = renderPreview(p.content);
      if (charCount) charCount.textContent = `${p.content.length} 字`;
    });
  }
}

async function updatePrompt(payload) {
  try {
    const data = await api("PUT", "/api/system-prompt", payload);
    state.prompts = data.config.prompts;
    state.activePromptId = data.config.activeId;
    // 保持选中的提示词仍存在
    const stillExists = state.prompts.find((p) => p.id === state.selectedPromptId);
    if (!stillExists) state.selectedPromptId = state.activePromptId || state.prompts[0]?.id || "";
    saveState();
    // 提示词页：局部刷新列表与编辑器，避免整页重建抖动；
    // 仅当选中项已不存在（如删除了正在编辑的那条）时才整页渲染兜底
    if (state.activePage === "prompt" && stillExists) {
      refreshPromptPage();
    } else {
      render();
    }
    showToast("提示词库已同步", "success");
  } catch (e) {
    showToast(`同步失败：${e.message}`, "error");
  }
}

async function addPrompt() {
  try {
    const data = await api("PUT", "/api/system-prompt", {
      add: { name: "新提示词", content: "你是专业的青少年心理咨询师。" },
    });
    state.prompts = data.config.prompts;
    state.activePromptId = data.config.activeId;
    state.selectedPromptId = data.config.activeId; // 新增后自动选中并激活
    saveState();
    // 新增走局部刷新（新列表项插入 + 编辑器切换），不整页重建
    if (state.activePage === "prompt") {
      refreshPromptPage();
    } else {
      render();
    }
    showToast("已新增提示词", "success");
  } catch (e) {
    showToast(`新增失败：${e.message}`, "error");
  }
}

async function setActivePrompt(id) {
  await updatePrompt({ activeId: id });
}

async function saveSelectedPrompt() {
  const prompt = selectedPrompt();
  if (!prompt.id) return;
  if (!prompt.content.trim()) return showToast("提示词内容不能为空", "error");
  // 兜底：从 DOM 读取当前名字，防止 state 因 change 未触发而过期
  const nameInput = document.querySelector("#prompt-name");
  const finalName = nameInput ? nameInput.value.trim() : prompt.name;
  prompt.name = finalName;
  prompt.content = document.querySelector("#prompt-content")?.value ?? prompt.content;
  saveState();
  await updatePrompt({ update: { id: prompt.id, name: finalName, content: prompt.content } });
}

// ============================================================
// 页 2：对话联调（多会话 + RAG 问答 + 提示词选择）
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
  const chatPrompt = state.chatPromptId ? getPromptById(state.chatPromptId) : activePrompt();
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
        <form class="composer" id="chat-form"><textarea id="chat-input" rows="1" placeholder="输入问题，按 Enter 发送，Shift + Enter 换行"></textarea><button class="send-btn" id="send-btn" type="submit" title="发送" aria-label="发送">↑</button></form>
      </section>
      <aside class="inspector">
        <div class="section-title"><span>本次对话配置</span></div>
        <div class="inspector-body">
          <div class="info-block">
            <h3>使用提示词</h3>
            <select class="prompt-select" id="chat-prompt-select">${renderPromptOptions(state.chatPromptId, true)}</select>
            <div class="prompt-preview" id="chat-prompt-preview">${escapeHtml(chatPrompt.content)}</div>
          </div>
          <div class="info-block">
            <h3>请求状态</h3>
            <div class="info-row"><span>后端连接</span><strong id="insp-conn">检测中</strong></div>
            <div class="info-row"><span>提示词来源</span><strong id="chat-prompt-src">${state.chatPromptId ? "手动选择" : "默认激活"}</strong></div>
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
  // 切换提示词：只局部更新下拉预览与来源文字，不整页重建（避免消息列表滚到底、页面抖动）
  page.querySelector("#chat-prompt-select").addEventListener("change", (e) => {
    state.chatPromptId = e.target.value;
    saveState();
    const chatPrompt = state.chatPromptId ? getPromptById(state.chatPromptId) : activePrompt();
    const preview = page.querySelector("#chat-prompt-preview");
    if (preview) preview.textContent = chatPrompt.content || "";
    const src = page.querySelector("#chat-prompt-src");
    if (src) src.textContent = state.chatPromptId ? "手动选择" : "默认激活";
  });
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

function renderPromptOptions(selectedId, includeDefault) {
  const options = [];
  const active = activePrompt();
  const isActiveSelected = selectedId === active.id;
  // "默认激活"占位项代表当前激活提示词，避免与列表里的激活项重复
  if (includeDefault) options.push(`<option value="" ${isActiveSelected ? "selected" : ""}>默认激活：${escapeHtml(active.name)}</option>`);
  state.prompts.forEach((p) => {
    if (includeDefault && p.id === active.id) return; // 已在"默认激活"占位项中体现
    const sel = p.id === selectedId ? "selected" : "";
    options.push(`<option value="${p.id}" ${sel}>${escapeHtml(p.name)}${p.id === state.activePromptId ? "（激活）" : ""}</option>`);
  });
  return options.join("");
}

function renderWelcome() {
  return `<div class="welcome"><div class="bot-avatar">✦</div><h2>用提示词库测试 RAG 问答</h2><p>这里调用后端 /api/query，可在右侧选择使用哪条提示词。右侧「使用提示词」下拉框选择后，该对话会沿用此提示词。</p><div class="suggestions"><button class="suggestion">孩子总说睡不着，怎么沟通？</button><button class="suggestion">考试前焦虑怎么办？</button><button class="suggestion">如何判断是否需要专业帮助？</button></div></div>`;
}

function renderSourceChips(sources) {
  if (!sources || !sources.length) return "";
  const chips = sources.map((s) => `<span class="source-chip">[${s.index}] ${escapeHtml(s.title || "来源")}</span>`).join("");
  return `<div class="sources">${chips}</div>`;
}

function renderMessage(m) {
  const sources = renderSourceChips(m.sources);
  const timings = m.role === "assistant" ? renderTimings(m.timings, m.elapsed) : "";
  // 流式消息：保留固定 id，供 token 增量更新时定位；内容为空时显示占位文字
  const streamingMsgId = m.streaming ? ' id="streaming-msg"' : "";
  const streamingBubbleId = m.streaming ? ' id="streaming-bubble"' : "";
  const body = m.streaming && !m.content ? "正在生成…" : m.content;
  return `<div class="message ${m.role}"${streamingMsgId}><div><div class="message-meta">${m.role === "user" ? "你" : "心理 RAG"}</div><div class="message-bubble"${streamingBubbleId}>${escapeHtml(body)}${sources}${timings}</div></div></div>`;
}

// ---- 流式渲染辅助：只操作当前占位气泡 DOM，不整页重绘 ----
function appendStreamingBubble() {
  const list = document.querySelector("#message-list");
  if (!list) return null;
  const div = document.createElement("div");
  div.className = "message assistant";
  // 文本独立放进 .streaming-text，来源 chips / 耗时栏是它的兄弟节点（token 更新只改
  // 文本节点，不误清来源）。返回气泡 DOM 引用由调用方闭包持有：整页重建/并发流存在时
  // 不再依赖全局 querySelector，避免拿到旧气泡造成答案与来源交叉污染。
  div.innerHTML = '<div><div class="message-meta">心理 RAG</div><div class="message-bubble"><span class="streaming-text">正在生成…</span></div></div>';
  list.appendChild(div);
  list.scrollTop = list.scrollHeight;
  return { bubble: div.querySelector(".message-bubble"), textEl: div.querySelector(".streaming-text") };
}

function updateStreamingBubble(m, els) {
  if (!els) return;
  els.textEl.textContent = m.content || "正在生成…";
  // 仅在用户接近底部时跟随滚动，避免打扰上翻阅读
  const list = document.querySelector("#message-list");
  if (list && list.scrollHeight - list.scrollTop - list.clientHeight < 100) {
    list.scrollTop = list.scrollHeight;
  }
}

function finalizeStreamingBubble(m, els) {
  if (!els) return;
  // 主路径：答案流式完成后，在此展示来源文档（先答案、后依据），再补耗时栏
  if (m.sources && m.sources.length && !els.bubble.querySelector(".sources")) {
    els.bubble.insertAdjacentHTML("beforeend", renderSourceChips(m.sources));
  }
  if (m.timings) els.bubble.insertAdjacentHTML("beforeend", renderTimings(m.timings, m.elapsed));
  // 收尾新增了来源/耗时内容，接近底部时跟随滚动
  const list = document.querySelector("#message-list");
  if (list && list.scrollHeight - list.scrollTop - list.clientHeight < 100) {
    list.scrollTop = list.scrollHeight;
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

  // 占位 assistant 消息：SSE 期间增量填充，不整页重绘
  const placeholder = { role: "assistant", content: "", sources: [], timings: null, streaming: true };
  session.messages.push(placeholder);
  const streamEls = appendStreamingBubble();

  const ac = new AbortController();
  session._streamAbort = ac;
  const started = performance.now();
  try {
    // 多轮记忆：历史 = 除占位外的全部消息（占位尚未有内容，不应发给后端）
    const history = session.messages
      .filter((m) => m !== placeholder)
      .map((m) => ({ role: m.role, content: m.content }));
    const body = { messages: history, session_id: session.id };
    if (state.chatPromptId) body.prompt_id = state.chatPromptId;
    if (isFirstTurn) body.title = session.name;

    const resp = await fetch("/api/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
    if (!resp.ok) {
      let detail = `请求失败（${resp.status}）`;
      try { const j = await resp.json(); detail = j.detail || detail; } catch { /* noop */ }
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
        if (evtName === "sources") {
          // 只暂存来源数据，不立即渲染：等答案流式生成完成、收尾时再显示，
          // 呈现顺序为「先看答案 → 再展示依据文档与耗时」
          placeholder.sources = data.sources || [];
        } else if (evtName === "token") {
          placeholder.content += data.text || "";
          updateStreamingBubble(placeholder, streamEls);
        } else if (evtName === "done") {
          if (data.answer != null) placeholder.content = data.answer;
          placeholder.timings = data.timings || null;
        } else if (evtName === "error") {
          throw new Error(data.detail || "生成失败");
        }
      }
    }
    placeholder.elapsed = Math.round(performance.now() - started);
    showToast(`回答已生成 · ${placeholder.elapsed}ms`, "success");
  } catch (e) {
    if (e.name === "AbortError") {
      // 用户主动取消（切换会话/删除/新发送），保留已生成部分
    } else {
      placeholder.content = placeholder.content || `请求失败：${e.message}`;
      showToast(e.message, "error");
    }
  } finally {
    placeholder.streaming = false;
    // 仅当自己仍是当前流时才清空句柄：并发场景下旧流 finally 不能误清新流的 abort 引用
    if (session._streamAbort === ac) session._streamAbort = null;
    finalizeStreamingBubble(placeholder, streamEls);
    // 恢复发送按钮（流式版不 renderChat，按钮不会自动重建，需要手动恢复）
    const btn = document.querySelector("#send-btn");
    if (btn) { btn.disabled = false; btn.innerHTML = "↑"; }
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
  // 与提示词管理/对比历史一致的自定义确认弹窗，避免误触即删
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

// ============================================================
// 页 3：提示词对比（同一问题：A/B 选择不同提示词跑 RAG）
// ============================================================
function renderCompare() {
  const page = document.querySelector("#page-compare");
  page.innerHTML = `
    <div class="compare-layout">
      <aside class="compare-sidebar">
        <div class="field">
          <label for="compare-input">测试问题</label>
          <textarea id="compare-input" rows="4" placeholder="输入一个问题，分别用两套提示词跑 RAG 对比">${escapeHtml(state.compareInput)}</textarea>
        </div>
        <div class="run-actions">
          <button class="primary-btn" id="run-compare">运行对比</button>
        </div>
        <div class="history-section">
          <div class="history-heading"><strong>对比历史</strong></div>
          <div class="history-list" id="compare-history">
            ${state.compareHistory.length ? renderCompareHistory() : '<div class="empty-small">还没有对比记录。</div>'}
          </div>
        </div>
      </aside>
      <section class="compare-main">
        <div class="compare-toolbar">
          <strong>双栏对比结果</strong>
          <span class="heading-spacer"></span>
          <button class="ghost-btn" id="clear-compare">清空结果</button>
        </div>
        <div class="compare-grid">
          ${renderCompareCard("a", "提示词 A", state.compareSelections.a)}
          ${renderCompareCard("b", "提示词 B", state.compareSelections.b)}
        </div>
      </section>
    </div>`;

  page.querySelector("#compare-input").addEventListener("input", (e) => { state.compareInput = e.target.value; state.currentCompare = null; saveState(); });
  page.querySelector("#run-compare").addEventListener("click", runCompare);
  page.querySelector("#clear-compare").addEventListener("click", clearCompare);
  page.querySelectorAll(".compare-prompt-select").forEach((sel) => {
    // 切换 A/B 提示词：select 原生即显示选中项，无需整页重建（避免抖动）
    sel.addEventListener("change", (e) => { state.compareSelections[e.target.dataset.side] = e.target.value; saveState(); });
  });
  bindCompareHistory();

  // 保持按钮禁用态 + 恢复上次/当前对比结果
  const runBtn = page.querySelector("#run-compare");
  if (runBtn && comparing) { runBtn.disabled = true; runBtn.innerHTML = '<span class="spinner"></span> 生成中…'; }
  if (state.currentCompare && state.currentCompare.input === state.compareInput) {
    ["a", "b"].forEach((side) => {
      const result = state.currentCompare[side];
      if (result) fillCompareCard(side, result);
      else if (comparing) setCompareCardLoading(side);
    });
  }
}

function renderCompareCard(side, title, selectedId) {
  // A/B 两侧均可从「当前库 + 出厂默认库」中任选；出厂默认里与当前库 id 重复的不再展示，避免"两个默认"
  const currentIds = new Set(state.prompts.map((p) => p.id));
  const currentOptions = state.prompts.map((p) => `<option value="${p.id}" ${p.id === selectedId ? "selected" : ""}>${escapeHtml(p.name)}${p.id === state.activePromptId ? "（激活）" : ""}</option>`).join("");
  const factoryPrompts = state.defaultPrompts.filter((p) => !currentIds.has(p.id));
  const defaultOptions = factoryPrompts.length
    ? `<optgroup label="出厂默认">${factoryPrompts.map((p) => `<option value="${p.id}" ${p.id === selectedId ? "selected" : ""}>${escapeHtml(p.name)}</option>`).join("")}</optgroup>`
    : "";
  const options = `${currentOptions}${defaultOptions}`;
  const tagColor = side === "a" ? "a" : "b";
  return `
    <div class="compare-card">
      <header><span class="tag ${tagColor}">${side.toUpperCase()}</span><span class="title">${escapeHtml(title)}</span><select class="compare-prompt-select" data-side="${side}">${options}</select></header>
      <div class="answer placeholder" id="answer-${side}">点击「运行对比」生成</div>
      <div class="card-foot"><span>来源 <strong id="src-${side}">-</strong></span><span id="meta-${side}">待运行</span></div>
    </div>`;
}

function renderCompareHistory() {
  if (!state.compareHistory.length) return '<div class="empty-small">还没有对比记录。</div>';
  return state.compareHistory.map((r) => `
    <div class="history-row">
      <button class="history-item" data-history-id="${r.id}">
        <span class="history-time">${new Date(r.createdAt).toLocaleString("zh-CN")}</span>
        <span class="history-input">${escapeHtml(r.input)}</span>
        <span class="history-models">A/B 对比</span>
        <span class="history-del" data-del-history="${r.id}" tabindex="0" role="button" title="删除记录" aria-label="删除记录">×</span>
      </button>
    </div>`).join("");
}

function bindCompareHistory() {
  document.querySelectorAll("[data-history-id]").forEach((b) => b.addEventListener("click", () => loadCompareRecord(b.dataset.historyId)));
  // 删除记录：鼠标点击 + 键盘（Enter/Space）均可达（按钮为 span，需 role=button + tabindex）
  document.querySelectorAll("[data-del-history]").forEach((b) => {
    b.addEventListener("click", (e) => { e.stopPropagation(); deleteCompareRecord(b.dataset.delHistory); });
    b.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); deleteCompareRecord(b.dataset.delHistory); }
    });
  });
}

async function deleteCompareRecord(id) {
  const r = state.compareHistory.find((x) => String(x.id) === String(id));
  if (!r) return;
  const ok = await confirmDialog({ title: "删除记录", message: "确定删除这条对比记录吗？删除后不可恢复。", confirmText: "删除", danger: true });
  if (!ok) return;
  try {
    await api("DELETE", `/api/compare-history/${encodeURIComponent(id)}`);
    state.compareHistory = state.compareHistory.filter((x) => String(x.id) !== String(id));
    saveState();
    // 局部更新：只重建对比历史列表，不动输入框与结果卡片（避免抖动）
    const hist = document.querySelector("#compare-history");
    if (hist) {
      hist.innerHTML = renderCompareHistory();
      bindCompareHistory();
    }
    showToast("已删除对比记录", "success");
  } catch (e) {
    showToast(`删除失败：${e.message}`, "error");
  }
}

async function loadCompareHistory() {
  try {
    const list = await api("GET", "/api/compare-history");
    state.compareHistory = Array.isArray(list) ? list : [];
  } catch {
    state.compareHistory = [];
  }
}

function loadCompareRecord(id) {
  const r = state.compareHistory.find((x) => String(x.id) === String(id));
  if (!r) return;
  state.compareInput = r.input;
  state.currentCompare = null;
  saveState();
  // 局部更新：只改问题输入框与两侧结果卡片，不重建整页（避免抖动）
  const inputEl = document.querySelector("#compare-input");
  if (inputEl) inputEl.value = r.input;
  fillCompareCard("a", r.a);
  fillCompareCard("b", r.b);
}

function setCompareCardLoading(side) {
  const el = document.querySelector(`#answer-${side}`);
  if (el) { el.className = "answer placeholder"; el.innerHTML = '<div class="thinking"><span></span><span></span><span></span></div>'; }
  const meta = document.querySelector(`#meta-${side}`);
  if (meta) meta.textContent = "运行中…";
}

function fillCompareCard(side, block) {
  const answerEl = document.querySelector(`#answer-${side}`);
  const metaEl = document.querySelector(`#meta-${side}`);
  const srcEl = document.querySelector(`#src-${side}`);
  if (!answerEl) return;
  if (!block) { answerEl.className = "answer placeholder"; answerEl.textContent = "无记录"; metaEl.textContent = "—"; srcEl.textContent = "-"; return; }
  answerEl.className = "answer";
  answerEl.innerHTML = escapeHtml(block.answer) + renderSources(block.sources);
  if (block.ok === false) {
    metaEl.textContent = "失败";
  } else {
    metaEl.innerHTML = renderTimings(block.timings, block.elapsed);
  }
  srcEl.textContent = `${block.sources?.length || 0} 条`;
}

function renderSources(sources) {
  if (!sources || !sources.length) return "";
  return `<ul class="sources-list">${sources.map((s) => `<li>[${s.index}] ${escapeHtml(s.title || "来源")}</li>`).join("")}</ul>`;
}

function clearCompare() {
  state.currentCompare = null;
  saveState();
  renderCompare();
}

async function runCompare() {
  const input = state.compareInput.trim();
  if (!input) return showToast("请先填写测试问题", "error");
  const promptA = getPromptById(state.compareSelections.a);
  const promptB = getPromptById(state.compareSelections.b);
  if (!promptA.id || !promptB.id) return showToast("请先为 A/B 选择提示词", "error");

  comparing = true;
  const btn = document.querySelector("#run-compare");
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> 生成中…'; }
  state.currentCompare = { input, a: null, b: null };
  saveState();
  ["a", "b"].forEach(setCompareCardLoading);

  const jobs = [
    ["a", promptA.content],
    ["b", promptB.content],
  ].map(async ([side, promptContent]) => {
    const started = performance.now();
    const result = { side };
    try {
      const body = { question: input, system_prompt_override: promptContent, persist: false };
      const data = await api("POST", "/api/query", body);
      result.answer = data.answer || "（无返回内容）";
      result.sources = data.sources || [];
      result.timings = data.timings || null;
      result.ok = true;
    } catch (e) {
      result.answer = `请求失败：${e.message}`;
      result.sources = [];
      result.ok = false;
    }
    result.elapsed = Math.round(performance.now() - started);
    state.currentCompare[side] = result;
    saveState();
    fillCompareCard(side, result);
    return result;
  });

  const results = await Promise.all(jobs);
  const map = {};
  results.forEach((r) => { map[r.side] = r; });
  // 持久化到服务端 SQLite（对比历史）
  try {
    const record = await api("POST", "/api/compare-history", { input, a: map.a, b: map.b });
    state.compareHistory.unshift(record);
  } catch (e) {
    showToast(`对比历史保存失败：${e.message}`, "error");
  }
  const hist = document.querySelector("#compare-history");
  if (hist) hist.innerHTML = renderCompareHistory();
  bindCompareHistory();

  comparing = false;
  const liveBtn = document.querySelector("#run-compare");
  if (liveBtn) { liveBtn.disabled = false; liveBtn.textContent = "运行对比"; }
  showToast("对比完成", "success");
}

// ---------------- 全局快捷键 ----------------
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
    if (state.activePage === "prompt") { e.preventDefault(); saveSelectedPrompt(); }
  }
});

// ---------------- 启动 ----------------
(async function init() {
  await loadPromptConfig();      // 加载提示词库（服务端 SQLite）
  await loadSessions();          // 加载会话列表（服务端 SQLite）
  await ensureActiveSession();   // 保证有一个可用会话并载入其消息
  await loadCompareHistory();    // 加载对比历史（服务端 SQLite）
  render();
  checkConnection();
})();
