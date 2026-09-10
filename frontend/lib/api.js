// ============================================================
// 网络层：token 读写 + 统一请求封装 + 错误归一化
// 与视图层解耦：会话失效通过 configureApi 注入的回调处理（api 层不反向依赖渲染）
// ============================================================

const TOKEN_KEY = "rag_token";

let _onSessionExpired = () => {};

export function configureApi({ onSessionExpired } = {}) {
  if (typeof onSessionExpired === "function") _onSessionExpired = onSessionExpired;
}

export function getToken() {
  try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
}

export function setToken(token) {
  try { localStorage.setItem(TOKEN_KEY, String(token)); } catch { /* 隐私模式等场景忽略 */ }
}

export function clearToken() {
  try { localStorage.removeItem(TOKEN_KEY); } catch { /* noop */ }
}

// 归一化错误：status / detail（服务端原文）/ code（业务错误码）/ isAuthError
export class ApiError extends Error {
  constructor(message, { status = 0, detail = "", code = null, auth = false } = {}) {
    super(message || "请求失败");
    this.name = "ApiError";
    this.status = status;
    this.detail = detail || message || "";
    this.code = code || null;
    this.isAuthError = !!auth;
  }
}

// 读取错误响应体：保留服务端 detail（可能是字符串或 FastAPI 校验数组）与业务 code
// 仅供本模块内部使用，不作为对外 API 导出
async function readError(res) {
  let detail = "";
  let code = null;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
    else if (Array.isArray(body?.detail)) {
      detail = body.detail.map((d) => d?.msg || "").filter(Boolean).join("；");
    }
    if (typeof body?.code === "string") code = body.code;
  } catch { /* 非 JSON 响应：保持空 detail，由调用方兜底文案 */ }
  return { status: res.status, detail, code };
}

/**
 * 统一请求封装。
 *
 * @param {string} path
 * @param {RequestInit} opts
 * @param {{auth?: "session"|"none"}} options
 *   auth = "session"（默认）：带 token 的业务请求；401 视为登录过期 → 触发会话失效回调
 *   auth = "none"：登录/注册/校验 token 等认证接口；401 原文抛给调用方（"用户名或密码错误"）
 */
export async function apiFetch(path, opts = {}, { auth = "session" } = {}) {
  const headers = { ...(opts.headers || {}) };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let res;
  try {
    res = await fetch(path, { ...opts, headers });
  } catch (e) {
    if (e?.name === "AbortError") throw e;
    throw new ApiError("网络异常，请检查网络后重试", { status: 0, detail: e?.message || "" });
  }
  if (res.ok) return res;

  const info = await readError(res);
  if (res.status === 401 && auth === "session") {
    _onSessionExpired();
    throw new ApiError("登录已过期，请重新登录", { ...info, auth: true });
  }
  const fallback =
    res.status === 401 ? "用户名或密码错误"
      : res.status === 403 ? "无权限执行该操作"
        : `请求失败（${res.status}）`;
  throw new ApiError(info.detail || fallback, info);
}

// JSON 便捷层：非 2xx 抛 ApiError；204 或无 body 返回 null
export async function api(method, path, body, options) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await apiFetch(path, opts, options);
  if (res.status === 204) return null;
  try { return await res.json(); } catch { return null; }
}

// 健康检查（公开接口，不带业务错误语义）
export async function checkHealth() {
  try {
    const res = await fetch("/api/health", { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}
