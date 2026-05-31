import {
  CSSProperties,
  ChangeEvent,
  DragEvent,
  FormEvent,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import katex from "katex";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import {
  AgentProcessEvent,
  ChatMessage,
  ToolEvent,
  WorkspaceSnapshot,
  WorkspaceVariable,
  sendToAgent,
  uploadMatrixFile
} from "./api";

const DEMO_MATH = `后端尚未接通，这里先展示工作台渲染能力。

我们可以把矩阵 $A$ 的条件数写成：

$$
\\kappa(A)=\\|A\\|\\cdot\\|A^{-1}\\|
$$

当 $\\kappa(A)$ 很大时，线性方程组 $Ax=b$ 对扰动会更敏感。`;

const WELCOME_TEXT = String.raw`# NLA Agent

这是一个数值线性代数工作台。你可以直接提问，也可以附加矩阵、向量或文本文件；Agent 会读取数据、调用计算工具，并把关键结果同步到右侧 Workspace。

可以从一个问题开始，例如：“分析矩阵 A 的条件数，并求解 Ax=b。”

示例：

$$
r_k = b - A x_k, \quad \|r_k\|_2 < \varepsilon
$$
`;

type FileKind = "matrix" | "text" | "binary";

interface ProjectFile {
  id: string;
  path: string;
  fileId?: string;
  uploadUri?: string;
  serverPath?: string;
  uploadError?: string;
  name: string;
  kind: FileKind;
  size: string;
  byteSize?: number;
  mimeType?: string;
  updatedAt: string;
  summary: string;
  preview: string;
  previewTruncated?: boolean;
  source: "project" | "local";
}

type BrowserFileWithPath = File & {
  webkitRelativePath?: string;
};

type StructuredContentBlock =
  | { type: "paragraph" | "text" | "markdown"; text: string }
  | { type: "math"; latex: string; display?: boolean }
  | { type: "code"; code: string; language?: string }
  | { type: "heading"; text: string; level?: number }
  | { type: "list"; items: string[]; ordered?: boolean };

interface UIMessage extends ChatMessage {
  reasoning?: string;
  final?: string;
  toolEvents?: ToolEvent[];
  processEvents?: AgentProcessEvent[];
  relatedVariables?: string[];
  attachments?: MessageAttachment[];
}

type MessageAttachment = Pick<ProjectFile, "id" | "name" | "kind" | "size">;

const EMPTY_WORKSPACE: WorkspaceSnapshot = { status: "ok", count: 0, variables: [] };
const CHAT_STORAGE_KEY = "nla-agent:chat-sessions:v1";
const LAYOUT_STORAGE_KEY = "nla-agent:panel-layout:v1";
const WORKBENCH_GAP = 12;
const PANEL_LIMITS = {
  leftMin: 240,
  leftMax: 560,
  centerMin: 520,
  rightMin: 340,
  rightMax: 760
};
const WORKSPACE_LIST_LIMITS = {
  min: 140,
  max: 620
};

interface ChatSession {
  id: string;
  title: string;
  messages: UIMessage[];
  workspace: WorkspaceSnapshot;
  files: ProjectFile[];
  updatedAt: number;
}

interface InitialAppState {
  sessions: ChatSession[];
  activeSessionId: string;
}

interface PanelLayout {
  left: number;
  right: number;
  workspaceList: number;
}

const kindLabels: Record<FileKind, string> = {
  matrix: "Matrix",
  text: "Text",
  binary: "Binary"
};

const MATRIX_FILE_RE = /\.(csv|tsv|mtx|mm|mat|dat|npy|npz|csc|coo|csr|rb|rua|rsa)(\.gz)?$/i;
const TEXT_FILE_RE = /\.(txt|md|markdown|json|py|js|ts|tsx|jsx|log|csv|tsv|csc|mtx|mm)$/i;
const MATRIX_MARKET_RE = /\.(mtx|mm)(\.gz)?$/i;
const CSC_TEXT_MATRIX_RE = /\.(csc|txt|csv|tsv)$/i;
const TEXT_PREVIEW_BYTE_LIMIT = 12_000;
const TEXT_PREVIEW_LINE_LIMIT = 160;
const BINARY_PREVIEW_TEXT = "该文件已加入 Current Folder，但浏览器无法直接预览二进制内容。";
const MATRIX_PREVIEW_TEXT = "矩阵文件已上传或等待上传；为避免污染 Agent 上下文，前端不读取、不预览矩阵正文。";
const KATEX_OPTIONS = { throwOnError: false, strict: "ignore" } as const;
const STRUCTURED_CONTENT_RE = /<nla-content-blocks>([\s\S]*?)<\/nla-content-blocks>/i;
const DISPLAY_MATH_HINT_RE =
  /(\\begin\{|\\end\{|\\\\|\\frac|\\sqrt|\\left|\\right|\\sum|\\prod|\\int|\\quad|\\cdot|\\times|\\[A-Za-z]+|[&_^=])/;
const MISSING_DISPLAY_MATH_START_RE =
  /^(?:[=+\-*/&]|\\[A-Za-z]+|\[|\(|\{|\d|[A-Za-z]\s*(?:[=_{^]|\\(?:in|to|mapsto|approx|sim)\b))/;

interface MathDelimiter {
  start: number;
  length: 1 | 2;
}

function isEscaped(text: string, index: number): boolean {
  let slashCount = 0;
  for (let cursor = index - 1; cursor >= 0 && text[cursor] === "\\"; cursor -= 1) {
    slashCount += 1;
  }
  return slashCount % 2 === 1;
}

function findMathDelimiters(text: string): MathDelimiter[] {
  const delimiters: MathDelimiter[] = [];

  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== "$" || isEscaped(text, index)) continue;

    if (text[index + 1] === "$") {
      delimiters.push({ start: index, length: 2 });
      index += 1;
      continue;
    }

    delimiters.push({ start: index, length: 1 });
  }

  return delimiters;
}

function findNextUnescapedDisplayDelimiter(text: string, start: number): number {
  for (let index = start; index < text.length - 1; index += 1) {
    if (text[index] === "$" && text[index + 1] === "$" && !isEscaped(text, index)) {
      return index;
    }
  }
  return -1;
}

function hasUnescapedSingleDollar(text: string, start: number, end: number): boolean {
  for (let index = start; index < end; index += 1) {
    if (text[index] !== "$" || isEscaped(text, index)) continue;
    if (text[index + 1] === "$") {
      index += 1;
      continue;
    }
    return true;
  }
  return false;
}

function hasUnescapedDisplayDelimiter(text: string, start: number, end: number): boolean {
  for (let index = start; index < end - 1; index += 1) {
    if (text[index] === "$" && text[index + 1] === "$" && !isEscaped(text, index)) {
      return true;
    }
  }
  return false;
}

function looksLikeDisplayMath(body: string): boolean {
  const stripped = body.trim();
  if (!stripped) return false;
  return stripped.includes("\n") || DISPLAY_MATH_HINT_RE.test(stripped) || stripped.length >= 24;
}

function findCandidateMathBlockStarts(text: string, end: number): number[] {
  const starts = new Set<number>([0]);
  let blankLineMatch: RegExpExecArray | null;
  const blankLineRe = /\n[ \t]*\n/g;

  while ((blankLineMatch = blankLineRe.exec(text.slice(0, end))) !== null) {
    starts.add(blankLineMatch.index + blankLineMatch[0].length);
  }

  starts.add(text.lastIndexOf("\n", end - 1) + 1);
  return [...starts].sort((a, b) => b - a);
}

function findMissingDisplayOpenerStart(text: string, closeIndex: number): number | null {
  for (const start of findCandidateMathBlockStarts(text, closeIndex)) {
    const candidate = text.slice(start, closeIndex);
    const trimmed = candidate.trim();
    const textBeforeBlock = text.slice(0, start).trimEnd();
    const previousLine = textBeforeBlock.slice(textBeforeBlock.lastIndexOf("\n") + 1).trim();

    if (
      !trimmed ||
      previousLine === "$$" ||
      !MISSING_DISPLAY_MATH_START_RE.test(trimmed) ||
      hasUnescapedSingleDollar(text, start, closeIndex) ||
      hasUnescapedDisplayDelimiter(text, start, closeIndex) ||
      !looksLikeDisplayMath(trimmed)
    ) {
      continue;
    }

    return start;
  }

  return null;
}

function repairMissingDisplayOpeners(text: string): string {
  const insertPositions = new Set<number>();

  for (let index = 0; index < text.length - 1; index += 1) {
    if (text[index] !== "$" || text[index + 1] !== "$" || isEscaped(text, index)) continue;

    const openerStart = findMissingDisplayOpenerStart(text, index);
    if (openerStart !== null) {
      insertPositions.add(openerStart);
    }
    index += 1;
  }

  if (insertPositions.size === 0) return text;

  let normalized = "";
  for (let index = 0; index < text.length; index += 1) {
    if (insertPositions.has(index)) normalized += "$$\n";
    normalized += text[index];
  }
  return normalized;
}

function isLinePrefixWhitespace(text: string, index: number): boolean {
  for (let cursor = index - 1; cursor >= 0 && text[cursor] !== "\n"; cursor -= 1) {
    if (text[cursor] !== " " && text[cursor] !== "\t") return false;
  }
  return true;
}

function repairParagraphStartDisplayOpeners(text: string): string {
  let normalized = "";

  for (let index = 0; index < text.length; index += 1) {
    if (!isLinePrefixWhitespace(text, index) || text[index] !== "$" || isEscaped(text, index)) {
      normalized += text[index];
      continue;
    }

    if (text[index + 1] === "$") {
      normalized += "$$";
      index += 1;
      continue;
    }

    const closeIndex = findNextUnescapedDisplayDelimiter(text, index + 1);
    if (
      closeIndex !== -1 &&
      !hasUnescapedSingleDollar(text, index + 1, closeIndex) &&
      looksLikeDisplayMath(text.slice(index + 1, closeIndex))
    ) {
      normalized += "$$";
    } else {
      normalized += "$";
    }
  }

  return normalized;
}

function escapeUnpairedMathDelimiters(text: string): string {
  const delimiters = findMathDelimiters(text);
  const paired = new Set<number>();
  const displayRanges: Array<{ start: number; end: number }> = [];
  let openDisplayToken: number | null = null;

  delimiters.forEach((delimiter, tokenIndex) => {
    if (delimiter.length !== 2) return;

    if (openDisplayToken === null) {
      openDisplayToken = tokenIndex;
      return;
    }

    paired.add(openDisplayToken);
    paired.add(tokenIndex);
    displayRanges.push({
      start: delimiters[openDisplayToken].start,
      end: delimiter.start + delimiter.length
    });
    openDisplayToken = null;
  });

  let openInlineToken: number | null = null;

  delimiters.forEach((delimiter, tokenIndex) => {
    if (
      delimiter.length !== 1 ||
      displayRanges.some((range) => delimiter.start > range.start && delimiter.start < range.end)
    ) {
      return;
    }

    if (openInlineToken === null) {
      openInlineToken = tokenIndex;
      return;
    }

    paired.add(openInlineToken);
    paired.add(tokenIndex);
    openInlineToken = null;
  });

  const escapedPositions = new Set<number>();
  delimiters.forEach((delimiter, tokenIndex) => {
    if (paired.has(tokenIndex)) return;

    for (let offset = 0; offset < delimiter.length; offset += 1) {
      escapedPositions.add(delimiter.start + offset);
    }
  });

  if (escapedPositions.size === 0) return text;

  let normalized = "";
  for (let index = 0; index < text.length; index += 1) {
    if (escapedPositions.has(index)) normalized += "\\";
    normalized += text[index];
  }
  return normalized;
}

function findLegacyMathClose(text: string, start: number, close: "\\)" | "\\]"): number {
  for (let index = start; index < text.length - 1; index += 1) {
    if (text[index] === "\\" && text[index + 1] === close[1] && !isEscaped(text, index)) {
      return index;
    }
  }
  return -1;
}

function convertLegacyTexDelimiters(text: string): string {
  let normalized = "";

  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== "\\" || isEscaped(text, index)) {
      normalized += text[index];
      continue;
    }

    const next = text[index + 1];
    if (next !== "(" && next !== "[") {
      normalized += text[index];
      continue;
    }

    const isDisplay = next === "[";
    const closeIndex = findLegacyMathClose(text, index + 2, isDisplay ? "\\]" : "\\)");
    if (closeIndex === -1) {
      normalized += text[index];
      continue;
    }

    const delimiter = isDisplay ? "$$" : "$";
    normalized += `${delimiter}${text.slice(index + 2, closeIndex)}${delimiter}`;
    index = closeIndex + 1;
  }

  return normalized;
}

function transformOutsideInlineCode(text: string): string {
  let normalized = "";
  let plainText = "";

  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== "`") {
      plainText += text[index];
      continue;
    }

    const tickStart = index;
    while (index + 1 < text.length && text[index + 1] === "`") {
      index += 1;
    }
    const tickCount = index - tickStart + 1;
    const tickFence = "`".repeat(tickCount);
    const closingIndex = text.indexOf(tickFence, index + 1);

    if (closingIndex === -1) {
      plainText += tickFence;
      continue;
    }

    normalized += escapeUnpairedMathDelimiters(
      repairMissingDisplayOpeners(repairParagraphStartDisplayOpeners(convertLegacyTexDelimiters(plainText)))
    );
    plainText = "";
    normalized += text.slice(tickStart, closingIndex + tickCount);
    index = closingIndex + tickCount - 1;
  }

  return (
    normalized +
    escapeUnpairedMathDelimiters(
      repairMissingDisplayOpeners(repairParagraphStartDisplayOpeners(convertLegacyTexDelimiters(plainText)))
    )
  );
}

function normalizeMarkdownMath(text: string): string {
  const lines = text.split(/(\n)/);
  let normalized = "";
  let plainText = "";
  let codeBlock = "";
  let fence: { marker: "`" | "~"; length: number } | null = null;

  const flushPlainText = () => {
    if (!plainText) return;
    normalized += transformOutsideInlineCode(plainText);
    plainText = "";
  };

  for (let index = 0; index < lines.length; index += 2) {
    const line = lines[index] ?? "";
    const newline = lines[index + 1] ?? "";
    const fenceMatch = line.match(/^\s*(`{3,}|~{3,})/);

    if (fence) {
      codeBlock += line + newline;
      if (
        fenceMatch &&
        fenceMatch[1][0] === fence.marker &&
        fenceMatch[1].length >= fence.length
      ) {
        normalized += codeBlock;
        codeBlock = "";
        fence = null;
      }
      continue;
    }

    if (fenceMatch) {
      flushPlainText();
      fence = { marker: fenceMatch[1][0] as "`" | "~", length: fenceMatch[1].length };
      codeBlock = line + newline;
      continue;
    }

    plainText += line + newline;
  }

  flushPlainText();
  return normalized + codeBlock;
}

function textFromRecord(record: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return undefined;
}

function normalizeStructuredContentBlock(raw: unknown): StructuredContentBlock | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const block = raw as Record<string, unknown>;
  const rawType = typeof block.type === "string" ? block.type.toLowerCase() : "paragraph";
  const text = textFromRecord(block, ["text", "content", "markdown"]);

  if (rawType === "math" || rawType === "latex") {
    const latex = textFromRecord(block, ["latex", "text", "content"]);
    if (!latex) return undefined;
    return { type: "math", latex, display: block.display !== false && block.inline !== true };
  }

  if (rawType === "code") {
    const code = textFromRecord(block, ["code", "content", "text"]);
    if (!code) return undefined;
    const language = typeof block.language === "string" ? block.language : undefined;
    return { type: "code", code, language };
  }

  if (rawType === "heading") {
    if (!text) return undefined;
    const level = typeof block.level === "number" ? block.level : undefined;
    return { type: "heading", text, level };
  }

  if (rawType === "list" || rawType === "bullet_list" || rawType === "numbered_list") {
    if (!Array.isArray(block.items)) return undefined;
    const items = block.items
      .map((item) => (typeof item === "string" ? item : textFromRecord(item as Record<string, unknown>, ["text", "content"])))
      .filter((item): item is string => Boolean(item?.trim()));
    if (!items.length) return undefined;
    return { type: "list", items, ordered: rawType === "numbered_list" || block.ordered === true };
  }

  if (!text) return undefined;
  return { type: rawType === "markdown" ? "markdown" : "paragraph", text };
}

function contentBlocksFromPayload(payload: unknown): StructuredContentBlock[] | undefined {
  const candidate =
    Array.isArray(payload)
      ? payload
      : payload && typeof payload === "object"
      ? (payload as Record<string, unknown>).blocks ??
        (payload as Record<string, unknown>).content_blocks ??
        (payload as Record<string, unknown>).structured_final
      : undefined;

  if (Array.isArray(candidate)) {
    const blocks = candidate
      .map((item) => normalizeStructuredContentBlock(item))
      .filter((item): item is StructuredContentBlock => Boolean(item));
    return blocks.length ? blocks : undefined;
  }

  if (candidate && typeof candidate === "object") {
    return contentBlocksFromPayload(candidate);
  }

  return undefined;
}

function parseStructuredContent(text: string): { blocks?: StructuredContentBlock[]; fallbackText: string } {
  const markerMatch = text.match(STRUCTURED_CONTENT_RE);
  const fallbackText = markerMatch ? text.replace(STRUCTURED_CONTENT_RE, "").trim() : text;
  const candidates = markerMatch ? [markerMatch[1]] : [text.trim()];

  for (const candidate of candidates) {
    if (!candidate || !/^[\[{]/.test(candidate.trim())) continue;
    try {
      const blocks = contentBlocksFromPayload(JSON.parse(candidate));
      if (blocks) return { blocks, fallbackText };
    } catch {
      // 结构化块解析失败时交回 MarkdownView 兜底。
    }
  }

  return { fallbackText };
}

function createWelcomeMessage(): UIMessage {
  return {
    role: "assistant",
    content: WELCOME_TEXT,
    final: WELCOME_TEXT,
    reasoning: "",
    toolEvents: [],
    processEvents: [],
    relatedVariables: []
  };
}

function isWelcomeMessageContent(content?: string): boolean {
  return Boolean(content?.includes("# NLA Agent") && content.includes("Workspace"));
}

function createSession(): ChatSession {
  const now = Date.now();
  return {
    id: `session-${now}-${Math.random().toString(36).slice(2, 8)}`,
    title: "新对话",
    messages: [createWelcomeMessage()],
    workspace: EMPTY_WORKSPACE,
    files: [],
    updatedAt: now
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function normalizePanelLayout(raw: unknown): PanelLayout {
  if (!isRecord(raw)) return { left: 320, right: 460, workspaceList: 320 };
  return {
    left: typeof raw.left === "number" ? clamp(raw.left, PANEL_LIMITS.leftMin, PANEL_LIMITS.leftMax) : 320,
    right: typeof raw.right === "number" ? clamp(raw.right, PANEL_LIMITS.rightMin, PANEL_LIMITS.rightMax) : 460,
    workspaceList:
      typeof raw.workspaceList === "number" ? clamp(raw.workspaceList, WORKSPACE_LIST_LIMITS.min, WORKSPACE_LIST_LIMITS.max) : 320
  };
}

function loadPanelLayout(): PanelLayout {
  if (typeof window === "undefined") return { left: 320, right: 460, workspaceList: 320 };
  try {
    return normalizePanelLayout(JSON.parse(window.localStorage.getItem(LAYOUT_STORAGE_KEY) ?? "{}"));
  } catch {
    return { left: 320, right: 460, workspaceList: 320 };
  }
}

function normalizeStoredWorkspace(raw: unknown): WorkspaceSnapshot {
  if (!isRecord(raw) || !Array.isArray(raw.variables)) return EMPTY_WORKSPACE;
  return {
    status: typeof raw.status === "string" ? raw.status : "ok",
    count: typeof raw.count === "number" ? raw.count : raw.variables.length,
    variables: raw.variables.filter((item): item is WorkspaceVariable => isRecord(item) && typeof item.name === "string")
  };
}

function normalizeStoredMessage(raw: unknown): UIMessage | undefined {
  if (!isRecord(raw)) return undefined;
  const role = raw.role === "user" || raw.role === "assistant" ? raw.role : undefined;
  if (!role || typeof raw.content !== "string") return undefined;
  return {
    role,
    content: raw.content,
    reasoning: typeof raw.reasoning === "string" ? raw.reasoning : "",
    final: typeof raw.final === "string" ? raw.final : undefined,
    toolEvents: Array.isArray(raw.toolEvents) ? (raw.toolEvents as ToolEvent[]) : [],
    processEvents: Array.isArray(raw.processEvents) ? (raw.processEvents as AgentProcessEvent[]) : [],
    relatedVariables: Array.isArray(raw.relatedVariables)
      ? raw.relatedVariables.filter((item): item is string => typeof item === "string")
      : [],
    attachments: Array.isArray(raw.attachments) ? (raw.attachments as MessageAttachment[]) : []
  };
}

function normalizeStoredSession(raw: unknown): ChatSession | undefined {
  if (!isRecord(raw) || typeof raw.id !== "string") return undefined;
  const storedMessages = Array.isArray(raw.messages)
    ? raw.messages.map((message) => normalizeStoredMessage(message)).filter((message): message is UIMessage => Boolean(message))
    : [];
  const messages = storedMessages.map((message, index) =>
    index === 0 &&
    message.role === "assistant" &&
    (isWelcomeMessageContent(message.content) || isWelcomeMessageContent(message.final))
      ? createWelcomeMessage()
      : message
  );
  return {
    id: raw.id,
    title: typeof raw.title === "string" && raw.title.trim() ? raw.title : titleFromMessages(messages),
    messages: messages.length ? messages : [createWelcomeMessage()],
    workspace: normalizeStoredWorkspace(raw.workspace),
    files: Array.isArray(raw.files) ? (raw.files as ProjectFile[]) : [],
    updatedAt: typeof raw.updatedAt === "number" ? raw.updatedAt : Date.now()
  };
}

function loadInitialAppState(): InitialAppState {
  if (typeof window === "undefined") {
    const session = createSession();
    return { sessions: [session], activeSessionId: session.id };
  }
  try {
    const raw = window.localStorage.getItem(CHAT_STORAGE_KEY);
    const parsed = raw ? (JSON.parse(raw) as unknown) : undefined;
    if (isRecord(parsed) && Array.isArray(parsed.sessions)) {
      const sessions = parsed.sessions
        .map((session) => normalizeStoredSession(session))
        .filter((session): session is ChatSession => Boolean(session));
      const activeSessionId =
        typeof parsed.activeSessionId === "string" && sessions.some((session) => session.id === parsed.activeSessionId)
          ? parsed.activeSessionId
          : sessions[0]?.id;
      if (sessions.length && activeSessionId) return { sessions, activeSessionId };
    }
  } catch {
    // 历史数据损坏时回退到新会话，避免阻塞页面启动。
  }
  const session = createSession();
  return { sessions: [session], activeSessionId: session.id };
}

function persistAppState(state: InitialAppState) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // 存储空间不足时保持内存态可用。
  }
}

function titleFromMessages(messages: UIMessage[]): string {
  const firstUserMessage = messages.find((message) => message.role === "user")?.content.trim();
  if (!firstUserMessage) return "新对话";
  return firstUserMessage.length > 22 ? `${firstUserMessage.slice(0, 22)}...` : firstUserMessage;
}

function sessionSubtitle(session: ChatSession): string {
  const userTurns = session.messages.filter((message) => message.role === "user").length;
  const variableCount = session.workspace.variables.length;
  if (userTurns === 0) return "尚未开始";
  return `${userTurns} 轮 · ${variableCount} 个变量`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function detectFileKind(file: File): FileKind {
  const lowerName = file.name.toLowerCase();
  if (MATRIX_FILE_RE.test(lowerName)) return "matrix";
  if (lowerName.endsWith(".gz") && /(matrix|mat|csc|csr|mtx|sparse|矩阵|稀疏)/i.test(file.name)) {
    return "matrix";
  }
  if (lowerName.endsWith(".txt") && /(matrix|mat|矩阵|向量|vector|sparse)/i.test(file.name)) {
    return "matrix";
  }
  if (file.type.startsWith("text/") || TEXT_FILE_RE.test(lowerName)) return "text";
  return "binary";
}

function summarizeLocalFile(file: File, kind: FileKind): string {
  if (kind === "matrix") return "本地矩阵/向量数据。发送给 Agent 前会上传到后端，并基于 file_id 载入 Workspace。";
  if (kind === "text") return "本地文本文件。发送给 Agent 时会附带可读取内容。";
  return `本地二进制文件（${file.type || "未知类型"}）。`;
}

function displayPath(file: ProjectFile): string {
  return file.path.replace(/^(Current Folder\/[^/]+\/)\d+-\d+-/, "$1");
}

function canReadAsText(file: File, kind: FileKind): boolean {
  return kind !== "binary" && !file.name.toLowerCase().endsWith(".gz");
}

function matrixLoaderName(fileName: string): "load_matrix_mtx_gz" | "load_matrix_csc_file" {
  return MATRIX_MARKET_RE.test(fileName.toLowerCase()) ? "load_matrix_mtx_gz" : "load_matrix_csc_file";
}

async function readTextPreview(file: File): Promise<{ preview: string; truncated: boolean }> {
  const preview = await file.slice(0, TEXT_PREVIEW_BYTE_LIMIT).text();
  return {
    preview,
    truncated: file.size > TEXT_PREVIEW_BYTE_LIMIT || preview.split("\n").length > TEXT_PREVIEW_LINE_LIMIT
  };
}

function buildMatrixReadHint(file: ProjectFile): string {
  const path = file.uploadUri ?? displayPath(file);
  const lowerName = file.name.toLowerCase();
  if (file.fileId && file.uploadUri) {
    const toolName = matrixLoaderName(file.name);
    return `该矩阵已上传到后端。请调用 ${toolName}("${file.uploadUri}") 基于 file_id 读取，并使用工具自动保存到 Workspace 变量 A。`;
  }
  if (file.uploadError) {
    return `矩阵上传失败，后端暂时无法按 file_id 读取。错误：${file.uploadError}`;
  }
  if (!file.uploadUri) {
    return "这是浏览器本地矩阵文件，尚未获得后端 file_id；请先上传成功后再读取。";
  }
  if (MATRIX_MARKET_RE.test(lowerName)) {
    return `建议调用 load_matrix_mtx_gz("${path}") 读取，并使用工具自动保存到 Workspace 变量 A。`;
  }
  if (CSC_TEXT_MATRIX_RE.test(lowerName)) {
    return file.uploadUri
      ? `建议调用 load_matrix_csc_file("${path}") 读取，并使用工具自动保存到 Workspace 变量 A。`
      : "这是浏览器本地附件，没有后端可访问路径；请先上传成功后再读取，不要把矩阵正文直接放进对话。";
  }
  return `这是矩阵类文件。若已有 uploadUri，优先用矩阵读取工具按 URI 载入 Workspace，不要在对话中直接展开完整矩阵。URI：${path}`;
}

const markdownComponents: Components = {
  code({ className, children, ...props }) {
    const code = String(children).replace(/\n$/, "");
    const isBlock = code.includes("\n") || Boolean(className);

    if (!isBlock) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }

    return (
      <div className="code-shell">
        <button type="button" onClick={() => void navigator.clipboard?.writeText(code)}>
          复制代码
        </button>
        <pre>
          <code className={className}>{children}</code>
        </pre>
      </div>
    );
  }
};

function MarkdownView({ children }: { children: string }) {
  const normalizedChildren = useMemo(() => normalizeMarkdownMath(children), [children]);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[[rehypeKatex, KATEX_OPTIONS]]}
      components={markdownComponents}
    >
      {normalizedChildren}
    </ReactMarkdown>
  );
}

function MathBlock({ latex, display = true }: { latex: string; display?: boolean }) {
  const rendered = useMemo(
    () =>
      katex.renderToString(latex, {
        displayMode: display,
        throwOnError: false,
        strict: "ignore"
      }),
    [display, latex]
  );

  return (
    <div className={display ? "structured-math display" : "structured-math inline"}>
      <span dangerouslySetInnerHTML={{ __html: rendered }} />
    </div>
  );
}

function CodeBlock({ code, language }: { code: string; language?: string }) {
  return (
    <div className="code-shell">
      <button type="button" onClick={() => void navigator.clipboard?.writeText(code)}>
        复制代码
      </button>
      <pre>
        <code className={language ? `language-${language}` : undefined}>{code}</code>
      </pre>
    </div>
  );
}

function StructuredContentView({ blocks }: { blocks: StructuredContentBlock[] }) {
  return (
    <div className="structured-content">
      {blocks.map((block, index) => {
        if (block.type === "math") {
          return <MathBlock key={index} latex={block.latex} display={block.display} />;
        }
        if (block.type === "code") {
          return <CodeBlock key={index} code={block.code} language={block.language} />;
        }
        if (block.type === "heading") {
          const level = Math.min(Math.max(block.level ?? 2, 1), 4);
          const Tag = `h${level}` as keyof JSX.IntrinsicElements;
          return <Tag key={index}>{block.text}</Tag>;
        }
        if (block.type === "list") {
          const ListTag = block.ordered ? "ol" : "ul";
          return (
            <ListTag key={index}>
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>
                  <MarkdownView>{item}</MarkdownView>
                </li>
              ))}
            </ListTag>
          );
        }
        return <MarkdownView key={index}>{block.text}</MarkdownView>;
      })}
    </div>
  );
}

function AgentContentView({ children }: { children: string }) {
  const parsed = useMemo(() => parseStructuredContent(children), [children]);
  if (parsed.blocks) return <StructuredContentView blocks={parsed.blocks} />;
  return <MarkdownView>{parsed.fallbackText}</MarkdownView>;
}

function AgentProcess({
  reasoning,
  toolEvents = [],
  processEvents = [],
  active
}: {
  reasoning?: string;
  toolEvents?: ToolEvent[];
  processEvents?: AgentProcessEvent[];
  active: boolean;
}) {
  const hasReasoning = !!reasoning?.trim();
  const orderedEvents =
    processEvents.length > 0
      ? processEvents
      : [
          ...(hasReasoning ? [{ type: "thinking" as const, text: reasoning ?? "" }] : []),
          ...toolEvents
        ];
  if (!active && orderedEvents.length === 0) return null;

  return (
    <section className="agent-process" aria-label="Agent 执行过程">
      <div className="process-title">
        <span className={active ? "pulse-dot" : "done-dot"} />
        <strong>{active ? "执行过程" : "执行过程已完成"}</strong>
      </div>
      <div className="process-list">
        {orderedEvents.map((event, index) =>
          event.type === "thinking" ? (
            <div key={`thinking-${index}`} className="process-item thinking">
              <span>思考/进度</span>
              <div className="process-text">
                <MarkdownView>{event.text}</MarkdownView>
              </div>
            </div>
          ) : (
            <div key={`${event.type}-${event.name}-${event.id ?? index}`} className={`process-item ${event.type}`}>
              <span>{event.type === "call" ? `调用工具：${event.name}` : `工具返回：${event.name}`}</span>
              {event.summary && <small>{event.summary}</small>}
            </div>
          )
        )}
        {active && orderedEvents.length === 0 && (
          <div className="process-item thinking">
            <span>正在思考</span>
            <small>Agent 正在分析问题，等待后续事件...</small>
          </div>
        )}
      </div>
    </section>
  );
}

export default function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const workbenchRef = useRef<HTMLDivElement>(null);
  const workspacePanelRef = useRef<HTMLElement>(null);
  const initialAppState = useMemo(() => loadInitialAppState(), []);
  const initialSession = useMemo(
    () => initialAppState.sessions.find((session) => session.id === initialAppState.activeSessionId) ?? initialAppState.sessions[0],
    [initialAppState]
  );
  const [chatSessions, setChatSessions] = useState<ChatSession[]>(initialAppState.sessions);
  const [activeSessionId, setActiveSessionId] = useState(initialAppState.activeSessionId);
  const [messages, setMessages] = useState<UIMessage[]>(initialSession.messages);
  const [workspace, setWorkspace] = useState<WorkspaceSnapshot>(initialSession.workspace);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showReasoning, setShowReasoning] = useState(false);
  const [projectFiles, setProjectFiles] = useState<ProjectFile[]>(initialSession.files);
  const [attachedFiles, setAttachedFiles] = useState<ProjectFile[]>([]);
  const [folderLabel, setFolderLabel] = useState(initialSession.files.length ? "历史附件索引" : "尚未选择文件夹");
  const [selectedVariableName, setSelectedVariableName] = useState(initialSession.workspace.variables[0]?.name ?? "");
  const [panelLayout, setPanelLayout] = useState<PanelLayout>(() => loadPanelLayout());

  useEffect(() => {
    setChatSessions((current) =>
      current.map((session) =>
        session.id === activeSessionId
          ? {
              ...session,
              title: titleFromMessages(messages),
              messages,
              workspace,
              files: projectFiles,
              updatedAt: Date.now()
            }
          : session
      )
    );
  }, [activeSessionId, messages, projectFiles, workspace]);

  useEffect(() => {
    persistAppState({ sessions: chatSessions, activeSessionId });
  }, [activeSessionId, chatSessions]);

  useEffect(() => {
    window.localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(panelLayout));
  }, [panelLayout]);

  const canSend = useMemo(
    () => (input.trim().length > 0 || attachedFiles.length > 0) && !loading,
    [attachedFiles.length, input, loading]
  );
  const sortedSessions = useMemo(
    () => [...chatSessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [chatSessions]
  );
  const activeSession = chatSessions.find((session) => session.id === activeSessionId) ?? initialSession;
  const workspaceVariables = workspace.variables;
  const selectedVariable =
    workspaceVariables.find((variable) => variable.name === selectedVariableName) ??
    workspaceVariables[0] ??
    null;

  function shapeText(variable: WorkspaceVariable): string {
    if (variable.size) return variable.size;
    if (!Array.isArray(variable.shape) || !variable.shape.length) return "1x1";
    return variable.shape.length === 1 ? `${variable.shape[0]}x1` : variable.shape.join("x");
  }

  function objectKind(variable: WorkspaceVariable): string {
    const rawKind = (variable.kind || variable.className || variable.type || "object").toLowerCase();
    if (rawKind.includes("sparse") || rawKind.includes("matrix")) return "matrix";
    if (rawKind.includes("vector")) return "vector";
    if (rawKind.includes("factor")) return "factorization";
    if (rawKind.includes("diagnostic")) return "diagnostic";
    if (rawKind.includes("result") || rawKind.includes("struct")) return "result";
    if (rawKind.includes("scalar") || variable.shape.length === 0) return "scalar";
    return rawKind.replace(/\s+/g, "_");
  }

  function objectStatus(variable: WorkspaceVariable): string {
    const status = variable.status?.trim();
    if (status) return status.toLowerCase();
    if (variable.alias_of || variable.parent_refs?.length) return "derived";
    return "fresh";
  }

  function objectSource(variable: WorkspaceVariable): string {
    return variable.origin || variable.source || variable.created_by_tool || "Agent";
  }

  function storageText(variable: WorkspaceVariable): string {
    return variable.storage_type ?? (variable.isSparse ? "sparse" : "dense");
  }

  function objectTags(variable: WorkspaceVariable): string[] {
    return [
      objectKind(variable),
      variable.role,
      storageText(variable),
      variable.created_by_tool ? `tool:${variable.created_by_tool}` : ""
    ].filter((tag): tag is string => Boolean(tag?.trim()));
  }

  function densityText(variable: WorkspaceVariable): string | null {
    if (typeof variable.density !== "number") return null;
    return `density ${variable.density.toExponential(2)}`;
  }

  function fitPanelLayout(next: PanelLayout, fixedSide?: "left" | "right"): PanelLayout {
    const containerWidth = workbenchRef.current?.clientWidth ?? 0;
    const maxSideSpace = containerWidth ? containerWidth - WORKBENCH_GAP * 2 - PANEL_LIMITS.centerMin : Number.POSITIVE_INFINITY;
    let left = clamp(next.left, PANEL_LIMITS.leftMin, PANEL_LIMITS.leftMax);
    let right = clamp(next.right, PANEL_LIMITS.rightMin, PANEL_LIMITS.rightMax);

    if (left + right > maxSideSpace) {
      if (fixedSide === "left") {
        right = clamp(maxSideSpace - left, PANEL_LIMITS.rightMin, PANEL_LIMITS.rightMax);
      } else if (fixedSide === "right") {
        left = clamp(maxSideSpace - right, PANEL_LIMITS.leftMin, PANEL_LIMITS.leftMax);
      } else {
        const overflow = left + right - maxSideSpace;
        left = clamp(left - overflow / 2, PANEL_LIMITS.leftMin, PANEL_LIMITS.leftMax);
        right = clamp(maxSideSpace - left, PANEL_LIMITS.rightMin, PANEL_LIMITS.rightMax);
      }
    }

    return { ...next, left, right };
  }

  function startPanelResize(side: "left" | "right", event: ReactPointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    const startX = event.clientX;
    const startLayout = panelLayout;
    const fixedSide = side === "left" ? "right" : "left";

    function onPointerMove(moveEvent: PointerEvent) {
      const delta = moveEvent.clientX - startX;
      setPanelLayout(
        fitPanelLayout(
          {
            ...startLayout,
            left: side === "left" ? startLayout.left + delta : startLayout.left,
            right: side === "right" ? startLayout.right - delta : startLayout.right
          },
          fixedSide
        )
      );
    }

    function onPointerUp() {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      document.body.classList.remove("is-resizing-panels");
    }

    document.body.classList.add("is-resizing-panels");
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp, { once: true });
  }

  function fitWorkspaceListHeight(height: number): number {
    const panelHeight = workspacePanelRef.current?.clientHeight ?? 0;
    const maxHeight = panelHeight
      ? Math.min(WORKSPACE_LIST_LIMITS.max, Math.max(WORKSPACE_LIST_LIMITS.min, panelHeight - 180))
      : WORKSPACE_LIST_LIMITS.max;
    return clamp(height, WORKSPACE_LIST_LIMITS.min, maxHeight);
  }

  function startWorkspaceResize(event: ReactPointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = panelLayout.workspaceList;

    function onPointerMove(moveEvent: PointerEvent) {
      setPanelLayout((current) => ({
        ...current,
        workspaceList: fitWorkspaceListHeight(startHeight + moveEvent.clientY - startY)
      }));
    }

    function onPointerUp() {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      document.body.classList.remove("is-resizing-workspace");
    }

    document.body.classList.add("is-resizing-workspace");
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp, { once: true });
  }

  function syncWorkspace(snapshot?: WorkspaceSnapshot) {
    if (!snapshot) return [] as string[];
    setWorkspace(snapshot);
    const names = snapshot.variables.map((variable) => variable.name);
    setSelectedVariableName((current) => {
      if (current && names.includes(current)) return current;
      return names[0] ?? "";
    });
    return names;
  }

  function focusVariable(name: string) {
    setSelectedVariableName(name);
  }

  function selectSession(session: ChatSession) {
    if (loading || session.id === activeSessionId) return;
    setActiveSessionId(session.id);
    setMessages(session.messages);
    setWorkspace(session.workspace);
    setProjectFiles(session.files);
    setAttachedFiles([]);
    setInput("");
    setError(null);
    setSelectedVariableName(session.workspace.variables[0]?.name ?? "");
    setFolderLabel(session.files.length ? "历史附件索引" : "尚未选择文件夹");
  }

  function startNewSession() {
    if (loading) return;
    const nextSession = createSession();
    setChatSessions((current) => [nextSession, ...current]);
    setActiveSessionId(nextSession.id);
    setMessages(nextSession.messages);
    setWorkspace(nextSession.workspace);
    setProjectFiles(nextSession.files);
    setAttachedFiles([]);
    setInput("");
    setError(null);
    setSelectedVariableName("");
    setFolderLabel("尚未选择文件夹");
  }

  function buildAgentFileContext(filesForTurn: ProjectFile[]): string {
    const currentFolderCandidates =
      filesForTurn.length > 0
        ? projectFiles.filter((file) => filesForTurn.every((attached) => attached.id !== file.id)).slice(0, 8)
        : projectFiles.slice(0, 24);
    const currentFolder = currentFolderCandidates
      .map((file) => {
        const pathLine = file.fileId
          ? `file_id: ${file.fileId}\n  uploadUri: ${file.uploadUri}`
          : `browserPath: ${displayPath(file)}\n  pathAccess: 尚未上传成功，后端不能读取该浏览器本地文件`;
        return `- name: ${file.name}\n  kind: ${kindLabels[file.kind]}\n  path: ${displayPath(file)}\n  ${pathLine}\n  note: ${file.summary}`;
      })
      .join("\n");

    const attachments = filesForTurn
      .map((file) => {
        const readablePath = file.uploadUri ?? displayPath(file);
        if (file.kind === "matrix") {
          const uploadLines = file.fileId
            ? `\n  file_id: ${file.fileId}\n  uploadUri: ${file.uploadUri}`
            : file.uploadError
            ? `\n  uploadError: ${file.uploadError}`
            : "";
          return `- name: ${file.name}\n  kind: Matrix\n  size: ${file.size}\n  path: ${readablePath}${uploadLines}\n  contentPolicy: matrix_body_omitted\n  handling: ${buildMatrixReadHint(file)}`;
        }
        if (file.kind === "text" && file.preview) {
          const truncatedLine = file.previewTruncated
            ? `\n  contentTruncated: true\n  truncationNote: 仅包含文件前 ${TEXT_PREVIEW_BYTE_LIMIT} bytes / 最多 ${TEXT_PREVIEW_LINE_LIMIT} 行，若需要完整文件请让用户拆分或改用后端上传读取。`
            : "\n  contentTruncated: false";
          return `- name: ${file.name}\n  kind: Text\n  size: ${file.size}\n  path: ${readablePath}${truncatedLine}\n  content:\n${file.preview
            .split("\n")
            .slice(0, TEXT_PREVIEW_LINE_LIMIT)
            .map((line) => `    ${line}`)
            .join("\n")}`;
        }
        return `- name: ${file.name}\n  kind: ${kindLabels[file.kind]}\n  size: ${file.size}\n  path: ${readablePath}\n  note: ${file.summary}`;
      })
      .join("\n");

    return `<frontend-files>
<frontend-session id="${activeSessionId}" />
Current Folder 文件索引如下。若本轮已有聊天框附件，这里只给少量额外索引；优先处理本轮附件。若用户提到矩阵文件，必须优先使用 file_id/uploadUri；browserPath 只是前端相对路径，普通 Web 页面不能提供电脑真实绝对路径。不要按 UI 类型分组推断任务。
${currentFolder || "（当前文件夹为空）"}

本轮聊天框附件如下。请自动识别文件类型：普通文本可直接阅读 content；矩阵文件的正文已被前端省略，必须优先用 uploadUri/file_id 调用矩阵读取工具载入 Workspace，并复用工具自动保存的变量 A/ans。不要要求用户重新粘贴矩阵正文。
${attachments || "（本轮没有附件）"}
</frontend-files>

<frontend-rendering-protocol>
前端支持可选结构化最终答案。若回答包含较多数学公式，可在最终回答中追加：
<nla-content-blocks>[{"type":"paragraph","text":"..."},{"type":"math","latex":"A=QR","display":true},{"type":"code","language":"python","code":"..."}]</nla-content-blocks>
其中 type=math 的 latex 字段不要再包裹 $ 或 $$；前端会专门用 KaTeX 渲染。若无法稳定输出严格 JSON，则继续输出普通 Markdown，并且行间公式必须写成独立的 $$...$$ 块。
</frontend-rendering-protocol>`;
  }

  function removeAttachedFile(id: string) {
    setAttachedFiles((current) => current.filter((file) => file.id !== id));
  }

  async function addLocalFiles(
    fileList: FileList | null,
    options: { attachToComposer?: boolean; replaceCurrentFolder?: boolean } = {}
  ) {
    const { attachToComposer = true, replaceCurrentFolder = false } = options;
    const files = Array.from(fileList ?? []) as BrowserFileWithPath[];
    if (!files.length) return;

    const selectedFolderName = files[0].webkitRelativePath?.split("/")[0] || "本地文件";
    const nextFiles = await Promise.all(
      files.map(async (file, index): Promise<ProjectFile> => {
        const kind = detectFileKind(file);
        const relativePath = file.webkitRelativePath || file.name;
        let fileId: string | undefined;
        let uploadUri: string | undefined;
        let serverPath: string | undefined;
        let uploadError: string | undefined;
        if (kind === "matrix") {
          try {
            const uploaded = await uploadMatrixFile(file);
            fileId = uploaded.file_id;
            uploadUri = uploaded.uri;
          } catch (err) {
            uploadError = err instanceof Error ? err.message : "上传失败";
          }
        }
        const textPreview =
          kind === "text" && canReadAsText(file, kind)
            ? await readTextPreview(file)
            : { preview: kind === "matrix" ? MATRIX_PREVIEW_TEXT : BINARY_PREVIEW_TEXT, truncated: false };
        const id = `local-${Date.now()}-${index}-${file.name}`;

        return {
          id,
          path: `Current Folder/${relativePath}`,
          fileId,
          uploadUri,
          serverPath,
          uploadError,
          name: file.name,
          kind,
          size: formatFileSize(file.size),
          byteSize: file.size,
          mimeType: file.type || undefined,
          updatedAt: fileId ? "刚刚上传" : "刚刚加入",
          summary:
            kind === "matrix" && fileId
              ? `已上传到后端，可通过 file_id=${fileId} 读取。`
              : kind === "matrix" && uploadError
              ? `矩阵上传失败：${uploadError}`
              : summarizeLocalFile(file, kind),
          preview: textPreview.preview,
          previewTruncated: textPreview.truncated,
          source: "local"
        };
      })
    );

    setProjectFiles((current) => (replaceCurrentFolder ? nextFiles : [...nextFiles, ...current]));
    if (replaceCurrentFolder) {
      setFolderLabel(selectedFolderName);
    }
    if (attachToComposer) {
      setAttachedFiles((current) => [...current, ...nextFiles]);
    }
  }

  function onFileInputChange(event: ChangeEvent<HTMLInputElement>) {
    void addLocalFiles(event.target.files, { attachToComposer: true });
    event.target.value = "";
  }

  function onFolderInputChange(event: ChangeEvent<HTMLInputElement>) {
    void addLocalFiles(event.target.files, { attachToComposer: false, replaceCurrentFolder: true });
    event.target.value = "";
  }

  function onDropFiles(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    void addLocalFiles(event.dataTransfer.files, { attachToComposer: true });
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSend) return;

    const filesForTurn = attachedFiles;
    const userContent = input.trim() || "请读取并分析我附上的文件。";
    const userMessage: UIMessage = {
      role: "user",
      content: userContent,
      attachments: filesForTurn.map((file) => ({
        id: file.id,
        name: file.name,
        kind: file.kind,
        size: file.size
      }))
    };
    const nextMessages = [...messages, userMessage];
    const assistantPlaceholder: UIMessage = {
      role: "assistant",
      content: "",
      final: "",
      reasoning: "",
      toolEvents: [],
      processEvents: [],
      relatedVariables: []
    };

    setMessages([...nextMessages, assistantPlaceholder]);
    setInput("");
    setAttachedFiles([]);
    setLoading(true);
    setError(null);

    try {
      const requestSessionId = activeSessionId;
      const reply = await sendToAgent(nextMessages, {
        clientSessionId: requestSessionId,
        contextPrefix: buildAgentFileContext(filesForTurn),
        onUpdate: ({ reasoning, final, toolEvents, processEvents, workspace: nextWorkspace }) => {
          const relatedVariables = syncWorkspace(nextWorkspace);
          setMessages((prev) => {
            if (prev.length === 0) return prev;
            const clone = [...prev];
            const last = clone[clone.length - 1];
            if (last.role !== "assistant") return prev;
            clone[clone.length - 1] = {
              ...last,
              reasoning,
              final,
              toolEvents,
              processEvents,
              content: final,
              relatedVariables: relatedVariables.length ? relatedVariables : last.relatedVariables
            };
            return clone;
          });
        }
      });
      const replyRelatedVariables = syncWorkspace(reply.workspace);
      setMessages((prev) => {
        if (prev.length === 0) return prev;
        const clone = [...prev];
        const last = clone[clone.length - 1];
        if (last.role !== "assistant") return prev;
        clone[clone.length - 1] = {
          ...last,
          reasoning: reply.reasoning,
          final: reply.final,
          toolEvents: reply.toolEvents,
          processEvents: reply.processEvents,
          content: reply.final,
          relatedVariables: replyRelatedVariables.length ? replyRelatedVariables : last.relatedVariables
        };
        return clone;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "请求失败");
      setMessages((prev) => {
        if (prev.length === 0) return prev;
        const clone = [...prev];
        clone[clone.length - 1] = {
          role: "assistant",
          content: "后端尚未接通，这里先展示 KaTeX 渲染是否正常。\n\n" + DEMO_MATH,
          final: "后端尚未接通，这里先展示 KaTeX 渲染是否正常。\n\n" + DEMO_MATH,
          reasoning: "",
          toolEvents: [],
          processEvents: [],
          relatedVariables: []
        };
        return clone;
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <p className="eyebrow">Numerical Linear Algebra Agent</p>
          <h1>交互式数值代数工作台</h1>
        </div>
        <div className="topbar-actions">
          <span className="session-chip" title={activeSession.id}>
            当前会话：{activeSession.title}
          </span>
          <span className="connection-dot" />
          <span>ADK: {loading ? "生成中" : "就绪"}</span>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showReasoning}
              onChange={(event) => setShowReasoning(event.target.checked)}
            />
            显示 reasoning
          </label>
        </div>
      </header>

      <div
        ref={workbenchRef}
        className="workbench"
        style={
          {
            "--left-panel-width": `${panelLayout.left}px`,
            "--right-panel-width": `${panelLayout.right}px`,
            "--workspace-list-height": `${panelLayout.workspaceList}px`
          } as CSSProperties
        }
      >
        <aside className="panel history-panel" aria-label="历史会话">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Chats</p>
              <h2>历史聊天</h2>
            </div>
            <span className="count">{chatSessions.length}</span>
          </div>

          <button className="new-chat-button" type="button" onClick={startNewSession} disabled={loading}>
            新建数组代数任务
          </button>

          <div className="history-list" aria-label="会话列表">
            {sortedSessions.map((session) => (
              <button
                key={session.id}
                type="button"
                className={`history-item ${activeSessionId === session.id ? "selected" : ""}`}
                onClick={() => selectSession(session)}
                disabled={loading && activeSessionId !== session.id}
              >
                <span>
                  <strong>{session.title}</strong>
                  <small>{sessionSubtitle(session)}</small>
                </span>
                <span className="history-time">
                  {new Date(session.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              </button>
            ))}
          </div>

          <section className="side-note">
            <h3>附件入口</h3>
            <p>文件附件已保留在输入框下方。拖拽矩阵或点击“附加文件”，Agent 会优先通过 file_id 载入 Workspace。</p>
            {projectFiles.length > 0 && (
              <div className="file-memory">
                <span>当前会话附件索引</span>
                <strong>{projectFiles.length}</strong>
                <small>{folderLabel}</small>
              </div>
            )}
          </section>
        </aside>

        <button
          className="panel-resizer left-resizer"
          type="button"
          aria-label="调整历史聊天和主内容区宽度"
          onPointerDown={(event) => startPanelResize("left", event)}
        />

        <section className="panel content-panel" aria-label="主内容区">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Agent Output</p>
              <h2>对话与技术文档</h2>
            </div>
            <span className="status-pill">Markdown + LaTeX</span>
          </div>

          <div className="chat-panel">
            {messages.map((msg, idx) => (
              <article key={`${msg.role}-${idx}`} className={`bubble ${msg.role}`}>
                <div className="role">{msg.role === "user" ? "你" : "Agent"}</div>
                {msg.role === "assistant" ? (
                  <>
                    {showReasoning && !!msg.reasoning && (
                      <details className="reasoning-box" open>
                        <summary>Reasoning</summary>
                        <MarkdownView>{msg.reasoning}</MarkdownView>
                      </details>
                    )}
                    <AgentProcess
                      reasoning={msg.reasoning}
                      toolEvents={msg.toolEvents}
                      processEvents={msg.processEvents}
                      active={loading && idx === messages.length - 1}
                    />
                    <div className="final-box">
                      <AgentContentView>{msg.final || msg.content || " "}</AgentContentView>
                    </div>
                    {!!msg.relatedVariables?.length && (
                      <div className="related-row" aria-label="关联变量">
                        {msg.relatedVariables.map((name) => (
                          <button key={name} type="button" onClick={() => focusVariable(name)}>
                            @{name}
                          </button>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    {!!msg.attachments?.length && (
                      <div className="message-attachments" aria-label="本轮附件">
                        {msg.attachments.map((file) => (
                          <span key={file.id}>
                            {file.name}
                            <small>{file.size}</small>
                          </span>
                        ))}
                      </div>
                    )}
                    <MarkdownView>{msg.content}</MarkdownView>
                  </>
                )}
              </article>
            ))}
            {loading && <div className="status">Agent 正在流式生成...</div>}
            {error && <div className="status error">{error}</div>}
          </div>

          <form className="composer" onSubmit={onSubmit} onDrop={onDropFiles} onDragOver={(event) => event.preventDefault()}>
            <input
              ref={fileInputRef}
              className="hidden-file-input"
              type="file"
              multiple
              onChange={onFileInputChange}
              accept=".txt,.csv,.tsv,.mtx,.mtx.gz,.mm,.mm.gz,.mat,.dat,.csc,.csc.gz,.csr,.csr.gz,.coo,.coo.gz,.rb,.rua,.rsa,.gz,.md,.markdown,.json,.py,.log"
            />
            <input
              ref={folderInputRef}
              className="hidden-file-input"
              type="file"
              multiple
              onChange={onFolderInputChange}
              {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
            />
            {attachedFiles.length > 0 && (
              <div className="attachment-tray" aria-label="待发送附件">
                {attachedFiles.map((file) => (
                  <span key={file.id} className="attachment-chip">
                    <strong>{file.name}</strong>
                    <small>
                      {kindLabels[file.kind]} · {file.size}
                    </small>
                    <button type="button" onClick={() => removeAttachedFile(file.id)} aria-label={`移除 ${file.name}`}>
                      ×
                    </button>
                  </span>
                ))}
              </div>
            )}
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="输入问题，或直接附上文件让 Agent 读取"
              rows={4}
            />
            <div className="composer-actions">
              <button className="attach-button" type="button" onClick={() => fileInputRef.current?.click()}>
                附加文件
              </button>
              <button className="attach-button" type="button" onClick={() => folderInputRef.current?.click()}>
                附加文件夹
              </button>
              <span>也可拖拽文件到输入区，矩阵文件会优先载入 Workspace</span>
              <button type="submit" disabled={!canSend}>
                发送
              </button>
            </div>
          </form>
        </section>

        <button
          className="panel-resizer right-resizer"
          type="button"
          aria-label="调整主内容区和 Workspace 宽度"
          onPointerDown={(event) => startPanelResize("right", event)}
        />

        <aside ref={workspacePanelRef} className="panel workspace-panel" aria-label="Workspace 区">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Workspace</p>
              <h2>数学对象</h2>
            </div>
            <span className="count">{workspace.count || workspaceVariables.length}</span>
          </div>

          <div className="workspace-hint">
            当前会话的状态真相：Agent、矩阵加载器和计算工具写入的矩阵、向量、分解结果与标量。
          </div>

          {workspaceVariables.length > 0 ? (
            <div className="workspace-object-list" aria-label="数学对象列表">
              {workspaceVariables.map((variable) => (
                <button
                  key={variable.name}
                  type="button"
                  className={`object-card ${selectedVariableName === variable.name ? "selected" : ""}`}
                  onClick={() => focusVariable(variable.name)}
                  title={variable.summary ?? variable.preview}
                >
                  <span className="object-card-top">
                    <span className={`object-type ${objectKind(variable)}`}>{objectKind(variable)}</span>
                    <span className={`object-status ${objectStatus(variable)}`}>{objectStatus(variable)}</span>
                  </span>
                  <span className="object-name-row">
                    <strong>{variable.display_name || variable.name}</strong>
                    <code>@{variable.ref ?? variable.name}</code>
                  </span>
                  <span className="object-meta-line">
                    <span>{shapeText(variable)}</span>
                    <span>{storageText(variable)}</span>
                    <span>v{variable.version ?? 1}</span>
                  </span>
                  <span className="object-origin">
                    来源：{objectSource(variable)}
                    {variable.updatedAt ? ` · ${variable.updatedAt}` : ""}
                  </span>
                  <span className="tag-row compact">
                    {objectTags(variable).map((tag) => (
                      <span key={tag}>{tag}</span>
                    ))}
                    {typeof variable.nnz === "number" && <span>nnz {variable.nnz}</span>}
                    {densityText(variable) && <span>{densityText(variable)}</span>}
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-workspace">
              <strong>Workspace 为空</strong>
              <span>可以对 Agent 说：“记住 A=[[4,1],[1,3]] 和 b=[1,2]”。</span>
            </div>
          )}

          {workspaceVariables.length > 0 && selectedVariable && (
            <button
              className="workspace-section-resizer"
              type="button"
              aria-label="调整变量列表和变量详情高度"
              onPointerDown={startWorkspaceResize}
            />
          )}

          {selectedVariable && (
            <section className="variable-detail">
              <div className="detail-header">
                <div>
                  <p>当前变量</p>
                  <h3>{selectedVariable.name}</h3>
                </div>
                <span className={`health ${objectStatus(selectedVariable)}`}>{objectStatus(selectedVariable)}</span>
              </div>

              <dl className="meta-grid">
                <div>
                  <dt>Size</dt>
                  <dd>{shapeText(selectedVariable)}</dd>
                </div>
                <div>
                  <dt>Class</dt>
                  <dd>{selectedVariable.kind ?? selectedVariable.className ?? selectedVariable.type}</dd>
                </div>
                <div>
                  <dt>dtype</dt>
                  <dd>{selectedVariable.dtype ?? "unknown"}</dd>
                </div>
                <div>
                  <dt>Storage</dt>
                  <dd>{selectedVariable.storage_type ?? (selectedVariable.isSparse ? "sparse" : "dense")}</dd>
                </div>
                <div>
                  <dt>Version</dt>
                  <dd>{selectedVariable.version ?? 1}</dd>
                </div>
                {selectedVariable.role && (
                  <div>
                    <dt>Role</dt>
                    <dd>{selectedVariable.role}</dd>
                  </div>
                )}
                {typeof selectedVariable.nnz === "number" && (
                  <div>
                    <dt>nnz</dt>
                    <dd>{selectedVariable.nnz}</dd>
                  </div>
                )}
                {typeof selectedVariable.density === "number" && (
                  <div>
                    <dt>Density</dt>
                    <dd>{selectedVariable.density.toExponential(3)}</dd>
                  </div>
                )}
                {typeof selectedVariable.norm === "number" && (
                  <div>
                    <dt>Norm</dt>
                    <dd>{selectedVariable.norm.toExponential(3)}</dd>
                  </div>
                )}
                {typeof selectedVariable.cond_est === "number" && (
                  <div>
                    <dt>Cond. est.</dt>
                    <dd>{selectedVariable.cond_est.toExponential(3)}</dd>
                  </div>
                )}
                <div>
                  <dt>来源</dt>
                  <dd>{selectedVariable.origin ?? selectedVariable.source ?? "Agent"}</dd>
                </div>
              </dl>

              {selectedVariable.summary && <p className="notes">{selectedVariable.summary}</p>}
              {selectedVariable.notes && <p className="notes">{selectedVariable.notes}</p>}
              {selectedVariable.alias_of && <p className="notes">Alias of {selectedVariable.alias_of}</p>}
              <pre>{selectedVariable.preview}</pre>
              <small className="updated-at">最近更新：{selectedVariable.updatedAt || "本轮对话"}</small>
            </section>
          )}
        </aside>
      </div>
    </main>
  );
}
