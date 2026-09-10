// ============================================================
// 纯文本 / 协议工具：不触碰 DOM 与网络，可单独测试
// ============================================================

export function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[c]));
}

const CODE_TOKEN = (i) => `\u0000code:${i}\u0000`;

// AI 回答 → 可信 HTML：先整体转义，再只注入本函数产生的标签。
// 任何模型原文中的标签都会以 &lt; 形式落地，不存在 XSS 直塞空间。
export function renderAnswer(raw) {
  if (raw == null) return "";
  const codes = [];
  let text = escapeHtml(raw)
    // 行首标题标记降级为普通文本（气泡已有字号层级，不再叠加 h1~h6）
    .replace(/(^|\n)#{1,6}[ \t]+/g, "$1")
    // 无序列表标记 → 视觉项目符号
    .replace(/(^|\n)([ \t]*)[-*+][ \t]+/g, "$1$2• ")
    // 行内代码先摘出来保护，避免其中的 * _ 被后续强调规则吃掉
    .replace(/`([^`\n]+)`/g, (_m, inner) => {
      codes.push(inner);
      return CODE_TOKEN(codes.length - 1);
    })
    .replace(/(\*\*|__)(?=\S)([\s\S]*?\S)\1/g, "<strong>$2</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/(^|[^_\w])_([^_\n]+)_(?![_\w])/g, "$1<em>$2</em>");
  return text.replace(/\u0000code:(\d+)\u0000/g, (_m, i) => `<code>${codes[Number(i)]}</code>`);
}

// AI 回答 → 流式期间的纯文本：去掉 Markdown 符号（**加粗**、__下划线__、*斜体*、`代码`、行首标题号）。
// 仅在逐 token 更新时使用，收尾/重绘一律走 renderAnswer（富文本）。
export function toPlainText(value) {
  if (!value) return "";
  return String(value)
    .replace(/\*\*([^*\n]+)\*\*/g, "$1")
    .replace(/__([^_\n]+)__/g, "$1")
    .replace(/(^|\n)#{1,6}[ \t]*/g, "$1")
    .replace(/\*([^*\n]+)\*/g, "$1")
    .replace(/`([^`\n]+)`/g, "$1");
}

// ---------------- SSE 帧解析 ----------------
// 后端（modules/gateway.py）按 `event: X\ndata: {json}\n\n` 发送。这里按规范补齐
// CRLF、`data:` 后无空格、多行 data、注释行与 EOF 残留帧，避免中间层改写换行时丢事件。
function parseFrame(raw) {
  let event = "message";
  const dataLines = [];
  for (const line of raw.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) event = line.slice(6).replace(/^[ \t]/, "").trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^[ \t]/, ""));
  }
  if (!dataLines.length) return null;
  let data;
  try { data = JSON.parse(dataLines.join("\n")); } catch { return null; }
  return { event, data };
}

export function createSseParser() {
  let buf = "";
  return {
    push(chunk) {
      buf += chunk;
      const frames = [];
      let match;
      while ((match = /\r?\n\r?\n/.exec(buf)) !== null) {
        const raw = buf.slice(0, match.index);
        buf = buf.slice(match.index + match[0].length);
        const frame = parseFrame(raw);
        if (frame) frames.push(frame);
      }
      return frames;
    },
    // 流结束时的残留（服务端漏掉最后一个空行也要把最后一帧交出去）
    end() {
      const rest = buf;
      buf = "";
      const frame = rest.trim() ? parseFrame(rest) : null;
      return frame ? [frame] : [];
    },
  };
}

// ---------------- 认证错误落位 ----------------
// 服务端登录/注册错误统一在 detail 里，按文案落回对应字段，避免都堆在密码下方
export function authErrorField(msg) {
  if (!msg) return "password";
  if (/密码|password/i.test(msg)) return "password";
  if (/用户名|username|账号/i.test(msg)) return "username";
  return null;
}
