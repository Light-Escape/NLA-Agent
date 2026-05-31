export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface WorkspaceVariable {
  name: string;
  ref?: string;
  display_name?: string;
  role?: string;
  origin?: string;
  alias_of?: string;
  parent_refs?: string[];
  created_by_tool?: string;
  type: string;
  kind?: string;
  className?: string;
  shape: number[];
  size?: string;
  dtype?: string;
  storage_type?: string;
  isSparse?: boolean;
  isComplex?: boolean;
  nnz?: number;
  density?: number;
  norm?: number;
  cond_est?: number;
  status?: string;
  version?: number;
  fingerprint?: string;
  summary?: string;
  preview_policy?: string;
  source?: string;
  updatedAt?: string;
  notes?: string;
  preview: string;
}

export interface WorkspaceSnapshot {
  status: string;
  count: number;
  variables: WorkspaceVariable[];
}

interface AdkPart {
  text?: string | null;
  thought?: boolean | null;
  functionCall?: {
    id?: string | null;
    name?: string | null;
    args?: unknown;
  } | null;
  function_call?: {
    id?: string | null;
    name?: string | null;
    args?: unknown;
  } | null;
  functionResponse?: {
    id?: string | null;
    name?: string | null;
    response?: unknown;
  } | null;
  function_response?: {
    id?: string | null;
    name?: string | null;
    response?: unknown;
  } | null;
}

interface AdkEvent {
  author?: string;
  content?: {
    role?: string | null;
    parts?: AdkPart[] | null;
  } | null;
}

export interface AgentTurn {
  reasoning: string;
  final: string;
  toolEvents: ToolEvent[];
  processEvents: AgentProcessEvent[];
  workspace?: WorkspaceSnapshot;
}

export interface StreamUpdate extends AgentTurn {
  event: AdkEvent;
}

export interface ToolEvent {
  id?: string;
  type: "call" | "response";
  name: string;
  summary: string;
}

export type AgentProcessEvent =
  | {
      type: "thinking";
      text: string;
    }
  | ({
      type: "call" | "response";
    } & ToolEvent);

interface SendToAgentOptions {
  onUpdate?: (update: StreamUpdate) => void;
  contextPrefix?: string;
  clientSessionId?: string;
}

export interface UploadedMatrixFile {
  status: "ok";
  file_id: string;
  uri: string;
  original_name: string;
  size: number;
  content_type?: string;
}

interface CreateSessionResponse {
  id: string;
}

const BASE_URL =
  (import.meta.env.VITE_AGENT_API_BASE as string | undefined)?.trim() || "http://127.0.0.1:8000";
const UPLOAD_BASE_URL =
  (import.meta.env.VITE_UPLOAD_API_BASE as string | undefined)?.trim() || "http://127.0.0.1:8001";
const APP_NAME = (import.meta.env.VITE_ADK_APP_NAME as string | undefined)?.trim() || "NLA_Master";
const USER_ID =
  (import.meta.env.VITE_ADK_USER_ID as string | undefined)?.trim() || "frontend_user";
const ENABLE_MOCK = ["1", "true", "yes", "on"].includes(
  ((import.meta.env.VITE_AGENT_USE_MOCK as string | undefined) || "").trim().toLowerCase()
);

const SESSION_CACHE_STORAGE_KEY = "nla-agent:adk-session-map:v1";

function loadSessionIdCache(): Map<string, string> {
  if (typeof window === "undefined") return new Map();
  try {
    const raw = window.localStorage.getItem(SESSION_CACHE_STORAGE_KEY);
    if (!raw) return new Map();
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") return new Map();
    return new Map(
      Object.entries(parsed as Record<string, unknown>).filter(
        (entry): entry is [string, string] => typeof entry[1] === "string" && entry[1].trim().length > 0
      )
    );
  } catch {
    return new Map();
  }
}

function persistSessionIdCache(cache: Map<string, string>) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(SESSION_CACHE_STORAGE_KEY, JSON.stringify(Object.fromEntries(cache)));
  } catch {
    // localStorage 不可用时只保留本次页面生命周期内的映射。
  }
}

const sessionIdCache = loadSessionIdCache();

function buildUrl(path: string): string {
  return `${BASE_URL.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

function buildUploadUrl(path: string): string {
  return `${UPLOAD_BASE_URL.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

async function safeFetch(url: string, init: RequestInit, action: string): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch {
    throw new Error(
      `${action}失败：无法连接后端 (${BASE_URL})。如果暂时没有后端，请在 frontend/.env.local 中设置 VITE_AGENT_USE_MOCK=true。`
    );
  }
}

export async function uploadMatrixFile(file: File): Promise<UploadedMatrixFile> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await safeFetch(
    buildUploadUrl("/nla/uploads/matrix"),
    {
      method: "POST",
      body: formData
    },
    "上传矩阵文件"
  );

  if (!response.ok) {
    let detail = "";
    try {
      const data = (await response.json()) as { detail?: string };
      detail = data.detail ? `：${data.detail}` : "";
    } catch {
      // 忽略非 JSON 错误体。
    }
    throw new Error(`上传矩阵文件失败: ${response.status}${detail}`);
  }

  const data = (await response.json()) as UploadedMatrixFile;
  if (!data?.file_id || !data?.uri) {
    throw new Error("上传矩阵文件成功但返回中缺少 file_id/uri。");
  }
  return data;
}

function parseAssistantText(events: AdkEvent[]): string {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    const parts = event.content?.parts ?? [];
    const text = parts
      .filter((p) => typeof p.text === "string" && p.text.trim().length > 0 && p.thought !== true)
      .map((p) => p.text!.trim())
      .join("\n\n")
      .trim();
    if (text) return text;
  }
  throw new Error("ADK /run 已返回事件，但未找到可显示的文本内容。");
}

function normalizeWorkspaceVariable(raw: unknown): WorkspaceVariable | undefined {
  if (typeof raw === "string") {
    return {
      name: raw,
      ref: raw,
      type: "unknown",
      className: "unknown",
      shape: [],
      size: "1x1",
      preview: "工具返回了简略变量名；最终回答同步后会展示完整摘要。"
    };
  }
  if (!raw || typeof raw !== "object") return undefined;
  const item = raw as Partial<WorkspaceVariable>;
  if (typeof item.name !== "string" || !item.name.trim()) return undefined;
  const shape = Array.isArray(item.shape)
    ? item.shape.filter((dim): dim is number => typeof dim === "number" && Number.isFinite(dim))
    : [];
  return {
    ...item,
    name: item.name,
    ref: item.ref ?? item.name,
    type: item.type ?? "unknown",
    className: item.className ?? item.kind ?? item.type ?? "unknown",
    shape,
    size: item.size,
    preview: typeof item.preview === "string" ? item.preview : "暂无预览。"
  };
}

function normalizeWorkspace(raw: unknown): WorkspaceSnapshot | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const candidate = raw as Partial<WorkspaceSnapshot>;
  if (Array.isArray(candidate.variables)) {
    const variables = candidate.variables
      .map((item) => normalizeWorkspaceVariable(item))
      .filter((item): item is WorkspaceVariable => Boolean(item));
    return {
      status: typeof candidate.status === "string" ? candidate.status : "ok",
      count: typeof candidate.count === "number" ? candidate.count : variables.length,
      variables
    };
  }
  return undefined;
}

function findWorkspaceInObject(raw: unknown): WorkspaceSnapshot | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const obj = raw as Record<string, unknown>;
  return normalizeWorkspace(obj.workspace) ?? normalizeWorkspace(raw);
}

function extractWorkspaceMarker(text: string): { text: string; workspace?: WorkspaceSnapshot } {
  const markerRe = /<nla-workspace>([\s\S]*?)<\/nla-workspace>/g;
  let workspace: WorkspaceSnapshot | undefined;
  const cleaned = text.replace(markerRe, (_match, payload: string) => {
    try {
      workspace = normalizeWorkspace(JSON.parse(payload.trim())) ?? workspace;
    } catch {
      // 标记解析失败时只隐藏标记，不影响用户阅读最终回答。
    }
    return "";
  });
  return { text: cleaned.trim(), workspace };
}

function stripWorkspaceMarkerFromTurn(turn: AgentTurn): AgentTurn {
  const extracted = extractWorkspaceMarker(turn.final);
  return {
    ...turn,
    final: extracted.text || turn.final,
    workspace: extracted.workspace ?? turn.workspace
  };
}

function summarizePayload(payload: unknown): string {
  if (payload === undefined || payload === null) return "";
  if (typeof payload === "string") {
    return payload.length > 180 ? `${payload.slice(0, 180)}...` : payload;
  }
  try {
    const text = JSON.stringify(payload);
    return text.length > 180 ? `${text.slice(0, 180)}...` : text;
  } catch {
    return String(payload);
  }
}

function splitEventText(event: AdkEvent): AgentTurn {
  const parts = event.content?.parts ?? [];
  let reasoning = "";
  let final = "";
  const toolEvents: ToolEvent[] = [];
  const processEvents: AgentProcessEvent[] = [];
  let workspace: WorkspaceSnapshot | undefined;
  for (const part of parts) {
    const functionCall = part.functionCall ?? part.function_call;
    const functionResponse = part.functionResponse ?? part.function_response;
    if (functionCall?.name) {
      const toolEvent: ToolEvent = {
        id: functionCall.id ?? undefined,
        type: "call",
        name: functionCall.name,
        summary: summarizePayload(functionCall.args)
      };
      toolEvents.push(toolEvent);
      processEvents.push(toolEvent);
    }
    if (functionResponse?.response) {
      const toolEvent: ToolEvent = {
        id: functionResponse.id ?? undefined,
        type: "response",
        name: functionResponse.name ?? "tool",
        summary: summarizePayload(functionResponse.response)
      };
      toolEvents.push(toolEvent);
      processEvents.push(toolEvent);
      workspace = findWorkspaceInObject(functionResponse.response) ?? workspace;
    }
    if (typeof part.text === "string" && part.text.length > 0) {
      if (part.thought === true) {
        reasoning += part.text;
        processEvents.push({ type: "thinking", text: part.text });
      } else {
        const extracted = extractWorkspaceMarker(part.text);
        final += extracted.text;
        workspace = extracted.workspace ?? workspace;
      }
    }
  }
  return { reasoning, final, toolEvents, processEvents, workspace };
}

function accumulateTurn(current: AgentTurn, event: AdkEvent): AgentTurn {
  const delta = splitEventText(event);
  return {
    reasoning: current.reasoning + delta.reasoning,
    final: current.final + delta.final,
    toolEvents: [...current.toolEvents, ...delta.toolEvents],
    processEvents: [...current.processEvents, ...delta.processEvents],
    workspace: delta.workspace ?? current.workspace
  };
}

function emitUpdate(
  onUpdate: SendToAgentOptions["onUpdate"],
  current: AgentTurn,
  event: AdkEvent
): AgentTurn {
  const next = accumulateTurn(current, event);
  onUpdate?.({ ...next, event });
  return next;
}

function parseSseDataBlock(block: string): string | null {
  const lines = block.split(/\r?\n/);
  const dataLines = lines
    .map((line) => line.trim())
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());
  if (dataLines.length === 0) return null;
  return dataLines.join("\n").trim();
}

async function readStreamingEvents(
  response: Response,
  onUpdate?: SendToAgentOptions["onUpdate"]
): Promise<AgentTurn> {
  if (!response.body) {
    throw new Error("流式响应中缺少 body。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let turn: AgentTurn = { reasoning: "", final: "", toolEvents: [], processEvents: [] };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const data = parseSseDataBlock(block);
      if (!data || data === "[DONE]") continue;
      try {
        const event = JSON.parse(data) as AdkEvent;
        turn = emitUpdate(onUpdate, turn, event);
      } catch {
        // 忽略非 JSON data 行
      }
    }
  }

  buffer += decoder.decode();
  const tailData = parseSseDataBlock(buffer);
  if (tailData && tailData !== "[DONE]") {
    try {
      const event = JSON.parse(tailData) as AdkEvent;
      turn = emitUpdate(onUpdate, turn, event);
    } catch {
      // 忽略非 JSON 尾块
    }
  }

  return stripWorkspaceMarkerFromTurn(turn);
}

async function ensureSessionId(clientSessionId = "default"): Promise<string> {
  const cachedSessionId = sessionIdCache.get(clientSessionId);
  if (cachedSessionId) return cachedSessionId;

  const response = await safeFetch(
    buildUrl(`/apps/${APP_NAME}/users/${USER_ID}/sessions`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    },
    "创建会话"
  );
  if (!response.ok) {
    throw new Error(`创建会话失败: ${response.status}`);
  }

  const data = (await response.json()) as CreateSessionResponse;
  if (!data?.id) throw new Error("创建会话成功但返回中缺少 session id。");
  sessionIdCache.set(clientSessionId, data.id);
  persistSessionIdCache(sessionIdCache);
  return data.id;
}

export async function sendToAgent(
  messages: ChatMessage[],
  options: SendToAgentOptions = {}
): Promise<AgentTurn> {
  const latestUserText = [...messages].reverse().find((m) => m.role === "user")?.content || "";
  const agentUserText = options.contextPrefix
    ? `${latestUserText}\n\n${options.contextPrefix}`
    : latestUserText;
  if (ENABLE_MOCK) {
    const final = `当前是前端 Mock 模式（未接后端）。\n\n你刚才的问题是：${latestUserText || "（空消息）"}\n\n示例公式：\n\n$$\nAx=b\n$$`;
    return {
      reasoning: "（Mock 模式）这里是示例 reasoning。",
      final,
      toolEvents: [
        { type: "call", name: "mock_tool", summary: "{\"demo\":true}" },
        { type: "response", name: "mock_tool", summary: "{\"status\":\"ok\"}" }
      ],
      processEvents: [
        { type: "thinking", text: "（Mock 模式）这里是示例 reasoning。" },
        { type: "call", name: "mock_tool", summary: "{\"demo\":true}" },
        { type: "response", name: "mock_tool", summary: "{\"status\":\"ok\"}" }
      ],
      workspace: {
        status: "ok",
        count: 2,
        variables: [
          {
            name: "A",
            type: "list",
            className: "matrix",
            shape: [2, 2],
            size: "2x2",
            dtype: "double",
            source: "Mock",
            updatedAt: "demo",
            preview: "[[4, 1], [1, 3]]"
          },
          {
            name: "b",
            type: "list",
            className: "vector",
            shape: [2],
            size: "2x1",
            dtype: "double",
            source: "Mock",
            updatedAt: "demo",
            preview: "[1, 2]"
          }
        ]
      }
    };
  }

  const sessionId = await ensureSessionId(options.clientSessionId);
  const payload = {
    app_name: APP_NAME,
    user_id: USER_ID,
    session_id: sessionId,
    new_message: {
      role: "user",
      parts: [{ text: agentUserText }]
    },
    streaming: false
  };

  const requestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  };

  let response = await safeFetch(buildUrl("/run_sse"), requestInit, "调用 ADK /run_sse");
  if (response.status === 404 || response.status === 405) {
    response = await safeFetch(buildUrl("/run"), requestInit, "调用 ADK /run");
  }
  if (!response.ok) {
    throw new Error(`调用 ADK 失败: ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("text/event-stream")) {
    const streamed = await readStreamingEvents(response, options.onUpdate);
    if (!streamed.final.trim()) {
      throw new Error("流式响应结束，但未收到可显示的最终答案。");
    }
    return stripWorkspaceMarkerFromTurn(streamed);
  }

  const events = (await response.json()) as AdkEvent[];
  if (!Array.isArray(events)) {
    throw new Error("ADK 返回格式异常：预期 Event[]。");
  }
  let turn: AgentTurn = { reasoning: "", final: "", toolEvents: [], processEvents: [] };
  for (const event of events) {
    turn = emitUpdate(options.onUpdate, turn, event);
  }
  if (!turn.final.trim()) {
    turn.final = parseAssistantText(events);
  }
  return stripWorkspaceMarkerFromTurn(turn);
}
