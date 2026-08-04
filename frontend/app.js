// ============================================================
// 心理 RAG · 提示词工作台  (SPA, 设计/功能参考 xinli Prompt Lab)
// 三页：提示词管理 / 对话联调 / 提示词对比
// 后端接本项目 FastAPI：/api/system-prompt（提示词库）+ /api/query（RAG 问答）
// ============================================================
const STORAGE_KEY = "rpsy-prompt-lab-v2";

const defaultState = {
  activePage: "prompt",            // prompt | chat | compare
  prompts: [],                     // 提示词库 { id, name, content }
  activePromptId: "",              // 当前激活提示词 id（RAG 默认使用）
  selectedPromptId: "",            // 提示词管理页当前选中编辑的 id
  defaultPrompts: [],              // 出厂默认提示词库（只读参考）
  sessions: [{ id: "welcome", name: "新的对话", createdAt: Date.now(), messages: [] }],
  activeSessionId: "welcome",
  chatPromptId: "",                // 当前对话使用的提示词 id（空=激活提示词）
  compareInput: "孩子最近总说睡不着，作为家长该怎么和他温和地沟通？",
  compareSelections: { a: "", b: "" }, // 对比页 A/B 分别使用的提示词 id
  compareHistory: [],
  currentCompare: null,
};

let state = loadState();
let toastTimer;
let comparing = false;

// ---------------- 持久化 ----------------
function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    const merged = { ...defaultState, ...saved };
    merged.sessions = Array.isArray(saved.sessions) && saved.sessions.length
      ? saved.sessions
      : structuredClone(defaultState.sessions);
    if (!merged.sessions.find((s) => s.id === merged.activeSessionId)) {
      merged.activeSessionId = merged.sessions[0].id;
    }
    return merged;
  } catch {
    return structuredClone(defaultState);
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  updateSaveState();
}

function updateSaveState() {
  const el = document.querySelector("#save-state");
  if (el) el.textContent = `已保存 ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
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

function currentSession() {
  return state.sessions.find((s) => s.id === state.activeSessionId) || state.sessions[0];
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
    if (!state.compareSelections.a) state.compareSelections.a = state.defaultPrompts[0]?.id || "";
    if (!state.compareSelections.b) state.compareSelections.b = state.activePromptId || "";
    if (!state.chatPromptId) state.chatPromptId = state.activePromptId || "";
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
          <span id="save-state" class="save-state">已保存</span>
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
              <header>实时渲染预览</header>
              <div class="preview-content" id="preview-box">${prompt.content ? escapeHtml(renderPreview(prompt.content)) : '<span class="placeholder">在左侧编辑器输入内容以查看预览</span>'}</div>
            </div>
            <div class="preview-info">
              <strong>提示</strong><br/>
              激活的提示词会被 RAG 问答默认使用。在「对话联调」和「提示词对比」中也可以临时切换其他提示词。
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
    <button class="prompt-item ${p.id === state.selectedPromptId ? "active" : ""}" data-prompt-id="${p.id}">
      <span class="item-dot"></span>
      <span class="item-main">
        <div class="item-name" id="prompt-name-${p.id}">${escapeHtml(p.name)}${isActive ? '<span class="badge-active">激活</span>' : ""}</div>
        <div class="item-meta">${p.content?.length || 0} 字 · ${isActive ? "RAG 默认" : "未激活"}</div>
      </span>
      <span class="item-actions">
        <button class="rename" data-rename="${p.id}" title="重命名">✎</button>
        <button class="delete" data-delete="${p.id}" title="删除">×</button>
      </span>
    </button>`;
}

function bindPromptManagerEvents(prompt) {
  const page = document.querySelector("#page-prompt");

  page.querySelectorAll("[data-prompt-id]").forEach((b) => {
    b.addEventListener("click", (e) => {
      if (e.target.dataset.rename || e.target.dataset.delete) return;
      state.selectedPromptId = b.dataset.promptId;
      saveState();
      renderPromptManager();
    });
  });

  page.querySelectorAll("[data-rename]").forEach((b) => {
    b.addEventListener("click", async (e) => {
      e.stopPropagation();
      const p = getPromptById(b.dataset.rename);
      const name = prompt("重命名提示词", p.name);
      if (name === null) return;
      const trimmed = name.trim();
      if (!trimmed) return showToast("名称不能为空", "error");
      await updatePrompt({ update: { id: p.id, name: trimmed } });
    });
  });

  page.querySelectorAll("[data-delete]").forEach((b) => {
    b.addEventListener("click", async (e) => {
      e.stopPropagation();
      const p = getPromptById(b.dataset.delete);
      if (state.prompts.length <= 1) return showToast("至少保留一条提示词", "error");
      if (!confirm(`确定删除提示词「${p.name}」吗？`)) return;
      await updatePrompt({ deleteId: p.id });
    });
  });

  const addBtn = page.querySelector("#add-prompt");
  if (addBtn) addBtn.addEventListener("click", addPrompt);

  const setActiveBtn = page.querySelector("#set-active-btn");
  if (setActiveBtn) setActiveBtn.addEventListener("click", () => setActivePrompt(prompt.id));

  const saveBtn = page.querySelector("#save-prompt-btn");
  if (saveBtn) saveBtn.addEventListener("click", saveSelectedPrompt);

  const nameInput = page.querySelector("#prompt-name");
  const contentInput = page.querySelector("#prompt-content");
  const previewBox = page.querySelector("#preview-box");
  const charCount = page.querySelector("#prompt-chars");

  if (nameInput) {
    // 实时同步名字到 state，避免按 Ctrl+S 时还没触发 change 导致保存旧名字
    nameInput.addEventListener("input", () => {
      if (!prompt.id) return;
      prompt.name = nameInput.value;
      saveState();
      // 只刷新列表项文字，不重绘整个编辑器，避免输入焦点丢失
      const listNameEl = document.querySelector(`#prompt-name-${prompt.id}`);
      if (listNameEl) listNameEl.textContent = nameInput.value || "未命名";
    });
  }

  if (contentInput) {
    contentInput.addEventListener("input", () => {
      if (!prompt.id) return;
      prompt.content = contentInput.value;
      saveState();
      if (previewBox) previewBox.textContent = renderPreview(prompt.content);
      if (charCount) charCount.textContent = `${prompt.content.length} 字`;
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
    render();
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
    render();
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
  const session = currentSession();
  const messages = session.messages || [];
  const chatPrompt = getPromptById(state.chatPromptId);
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
        <div class="message-list" id="message-list">${messages.length ? messages.map(renderMessage).join("") : renderWelcome()}</div>
        <form class="composer" id="chat-form"><textarea id="chat-input" rows="1" placeholder="输入问题，按 Enter 发送，Shift + Enter 换行"></textarea><button class="send-btn" id="send-btn" type="submit" title="发送">↑</button></form>
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
            <div class="info-row"><span>提示词来源</span><strong>${state.chatPromptId ? "手动选择" : "默认激活"}</strong></div>
          </div>
        </div>
      </aside>
    </div>`;

  page.querySelectorAll("[data-session]").forEach((b) => b.addEventListener("click", (e) => { if (e.target.dataset.deleteSession) return; state.activeSessionId = b.dataset.session; saveState(); renderChat(); }));
  page.querySelectorAll("[data-delete-session]").forEach((b) => b.addEventListener("click", (e) => { e.stopPropagation(); if (state.sessions.length === 1) return showToast("至少保留一个对话", "error"); state.sessions = state.sessions.filter((s) => s.id !== b.dataset.deleteSession); state.activeSessionId = state.sessions[0].id; saveState(); renderChat(); }));
  page.querySelector("#new-session").addEventListener("click", () => { const id = `session-${Date.now()}`; state.sessions.unshift({ id, name: "新的对话", createdAt: Date.now(), messages: [] }); state.activeSessionId = id; saveState(); renderChat(); });
  page.querySelector("#rename-session").addEventListener("click", () => { const name = prompt("输入新的对话名称", session.name); if (name?.trim()) { session.name = name.trim(); saveState(); renderChat(); } });
  page.querySelector("#export-session").addEventListener("click", exportSession);
  const form = page.querySelector("#chat-form");
  form.addEventListener("submit", async (e) => { e.preventDefault(); const input = page.querySelector("#chat-input"); const content = input.value.trim(); if (!content) return; await sendChat(content); });
  page.querySelector("#chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); page.querySelector("#chat-form").requestSubmit(); } });
  page.querySelector("#chat-prompt-select").addEventListener("change", (e) => { state.chatPromptId = e.target.value; saveState(); renderChat(); });
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
      <span class="session-delete" data-delete-session="${s.id}">×</span>
    </button>`;
}

function renderPromptOptions(selectedId, includeDefault) {
  const options = [];
  if (includeDefault) options.push(`<option value="">默认激活：${escapeHtml(activePrompt().name)}</option>`);
  state.prompts.forEach((p) => {
    const sel = p.id === selectedId ? "selected" : "";
    options.push(`<option value="${p.id}" ${sel}>${escapeHtml(p.name)}${p.id === state.activePromptId ? "（激活）" : ""}</option>`);
  });
  return options.join("");
}

function renderWelcome() {
  return `<div class="welcome"><div class="bot-avatar">✦</div><h2>用提示词库测试 RAG 问答</h2><p>这里调用后端 /api/query，可在右侧选择使用哪条提示词。右侧「使用提示词」下拉框选择后，该对话会沿用此提示词。</p><div class="suggestions"><button class="suggestion">孩子总说睡不着，怎么沟通？</button><button class="suggestion">考试前焦虑怎么办？</button><button class="suggestion">如何判断是否需要专业帮助？</button></div></div>`;
}

function renderMessage(m) {
  const sources = m.sources && m.sources.length
    ? `<div class="sources">${m.sources.map((s) => `<span class="source-chip">[${s.index}] ${escapeHtml(s.title || "来源")}</span>`).join("")}</div>`
    : "";
  return `<div class="message ${m.role}"><div><div class="message-meta">${m.role === "user" ? "你" : "心理 RAG"}</div><div class="message-bubble">${escapeHtml(m.content)}${sources}</div></div></div>`;
}

async function sendChat(content) {
  const session = currentSession();
  session.messages.push({ role: "user", content });
  session.name = session.messages.length === 1 ? content.slice(0, 22) : session.name;
  saveState(); renderChat();
  const btn = document.querySelector("#send-btn");
  btn.disabled = true; btn.textContent = "…";
  try {
    const body = { question: content };
    if (state.chatPromptId) body.prompt_id = state.chatPromptId;
    const data = await api("POST", "/api/query", body);
    session.messages.push({ role: "assistant", content: data.answer || "（无返回内容）", sources: data.sources || [] });
    showToast("回答已生成", "success");
  } catch (e) {
    session.messages.push({ role: "assistant", content: `请求失败：${e.message}` });
    showToast(e.message, "error");
  }
  saveState(); renderChat();
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
          <textarea id="compare-input" placeholder="输入一个问题，分别用两套提示词跑 RAG 对比">${escapeHtml(state.compareInput)}</textarea>
        </div>
        <div class="run-actions">
          <button class="primary-btn" id="run-compare">运行对比</button>
        </div>
        <div class="history-section">
          <div class="history-heading"><strong>对比历史</strong><span class="muted-label">点击恢复</span></div>
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
          ${renderCompareCard("a", "出厂默认", state.compareSelections.a)}
          ${renderCompareCard("b", "当前编辑", state.compareSelections.b)}
        </div>
      </section>
    </div>`;

  page.querySelector("#compare-input").addEventListener("input", (e) => { state.compareInput = e.target.value; saveState(); });
  page.querySelector("#run-compare").addEventListener("click", runCompare);
  page.querySelector("#clear-compare").addEventListener("click", clearCompare);
  page.querySelectorAll(".compare-prompt-select").forEach((sel) => {
    sel.addEventListener("change", (e) => { state.compareSelections[e.target.dataset.side] = e.target.value; saveState(); renderCompare(); });
  });
  bindCompareHistory();

  // 保持按钮禁用态 + 恢复上次/当前对比结果
  const runBtn = page.querySelector("#run-compare");
  if (runBtn && comparing) { runBtn.disabled = true; runBtn.textContent = "生成中…"; }
  if (state.currentCompare && state.currentCompare.input === state.compareInput) {
    ["a", "b"].forEach((side) => {
      const result = state.currentCompare[side];
      if (result) fillCompareCard(side, result);
      else if (comparing) setCompareCardLoading(side);
    });
  }
}

function renderCompareCard(side, title, selectedId) {
  const isDefaultSide = side === "a";
  const options = isDefaultSide
    ? state.defaultPrompts.map((p) => `<option value="${p.id}" ${p.id === selectedId ? "selected" : ""}>${escapeHtml(p.name)}</option>`).join("")
    : state.prompts.map((p) => `<option value="${p.id}" ${p.id === selectedId ? "selected" : ""}>${escapeHtml(p.name)}${p.id === state.activePromptId ? "（激活）" : ""}</option>`).join("");
  const tagColor = isDefaultSide ? "a" : "b";
  return `
    <div class="compare-card">
      <header><span class="tag ${tagColor}">${side.toUpperCase()}</span><span class="title">${escapeHtml(title)}</span><select class="compare-prompt-select" data-side="${side}">${options}</select></header>
      <div class="answer placeholder" id="answer-${side}">点击「运行对比」生成</div>
      <div class="card-foot"><span>来源 <strong id="src-${side}">-</strong></span><span id="meta-${side}">待运行</span></div>
    </div>`;
}

function renderCompareHistory() {
  return state.compareHistory.map((r) => `
    <button class="history-item" data-history-id="${r.id}">
      <span class="history-time">${new Date(r.createdAt).toLocaleString("zh-CN")}</span>
      <span class="history-input">${escapeHtml(r.input)}</span>
      <span class="history-models">A/B 对比</span>
    </button>`).join("");
}

function bindCompareHistory() {
  document.querySelectorAll("[data-history-id]").forEach((b) => b.addEventListener("click", () => loadCompareRecord(b.dataset.historyId)));
}

function loadCompareRecord(id) {
  const r = state.compareHistory.find((x) => String(x.id) === String(id));
  if (!r) return;
  state.compareInput = r.input;
  saveState();
  renderCompare();
  fillCompareCard("a", r.a);
  fillCompareCard("b", r.b);
}

function setCompareCardLoading(side) {
  const el = document.querySelector(`#answer-${side}`);
  if (el) { el.className = "answer placeholder"; el.textContent = "正在生成…"; }
  const meta = document.querySelector(`#meta-${side}`);
  if (meta) meta.textContent = "运行中";
}

function fillCompareCard(side, block) {
  const answerEl = document.querySelector(`#answer-${side}`);
  const metaEl = document.querySelector(`#meta-${side}`);
  const srcEl = document.querySelector(`#src-${side}`);
  if (!answerEl) return;
  if (!block) { answerEl.className = "answer placeholder"; answerEl.textContent = "无记录"; metaEl.textContent = "—"; srcEl.textContent = "-"; return; }
  answerEl.className = "answer";
  answerEl.innerHTML = escapeHtml(block.answer) + renderSources(block.sources);
  metaEl.textContent = block.ok === false ? "失败" : `完成 · ${block.elapsed}ms`;
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
  if (btn) { btn.disabled = true; btn.textContent = "生成中…"; }
  state.currentCompare = { input, a: null, b: null };
  saveState();
  ["a", "b"].forEach(setCompareCardLoading);

  const jobs = [
    ["a", promptA.content, state.compareSelections.a],
    ["b", promptB.content, state.compareSelections.b],
  ].map(async ([side, promptContent, promptId]) => {
    const started = performance.now();
    const result = { side };
    try {
      const body = { question: input, system_prompt_override: promptContent };
      if (!state.defaultPrompts.find((p) => p.id === promptId)) {
        body.prompt_id = promptId; // 不是默认库里的才用 prompt_id
      }
      const data = await api("POST", "/api/query", body);
      result.answer = data.answer || "（无返回内容）";
      result.sources = data.sources || [];
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
  state.compareHistory.unshift({ id: Date.now(), input, a: map.a, b: map.b, createdAt: new Date().toISOString() });
  state.compareHistory = state.compareHistory.slice(0, 30);
  saveState();
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
  await loadPromptConfig();
  render();
  checkConnection();
})();
