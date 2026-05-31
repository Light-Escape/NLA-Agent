# NLA Frontend (Vite + React + KaTeX)

最小可运行前端模板，专门用于渲染 Agent 输出中的 Markdown 与数学公式。

## 1) 安装与启动

在 `NLA_Master/frontend` 目录执行：

```bash
npm install
cd NLA_Master/frontend
npm run dev
```

默认地址：`http://localhost:5173`

## 2) 公式渲染支持

- 行内公式：`$...$`
- 块级公式：`$$...$$`
- 已集成 `remark-math + rehype-katex + katex.css`

## 3) 对接 ADK 后端 API

当前前端默认请求 ADK Web 接口：

- `POST /apps/{appName}/users/{userId}/sessions`
- `POST /run`
- `POST /nla/uploads/matrix`（项目自定义上传接口，用于把浏览器本地矩阵保存到后端并返回 `file_id/uploadUri`）

### 可配置环境变量（可选）

新建 `.env.local`：

```bash
VITE_AGENT_API_BASE=/adk
VITE_UPLOAD_API_BASE=http://127.0.0.1:8001
VITE_ADK_APP_NAME=NLA_Master
VITE_ADK_USER_ID=frontend_user
VITE_AGENT_USE_MOCK=false
```

矩阵上传服务可单独启动：

```bash
uvicorn NLA_Master.upload_api:app --port 8001
```

上传成功后，前端会在附件上下文中传入 `file_id` 和 `nla-upload://<file_id>`，Agent 后续会基于 `file_id` 调用矩阵读取工具；后端保存路径不会传给模型。

### 无后端时的本地开发（避免 `Failed to fetch`）

如果你暂时还没接后端，可以在 `.env.local` 里开启 mock：

```bash
VITE_AGENT_USE_MOCK=true
```

开启后，前端会直接返回示例文本，不再发起网络请求。

### 跨域说明（前端 5173 -> 后端 8000）

开发环境已在 `vite.config.ts` 配置代理：

- 前端请求 `/adk/...`
- Vite 转发到 `http://127.0.0.1:8000/...`

这样可以绕过浏览器跨域限制，避免由于后端未配置 CORS 导致的 `Failed to fetch`。

## 4) 你可能需要改的地方

- 如果后端端口不是 `8000`，修改 `VITE_AGENT_API_BASE`。
- 如果 ADK app 名称或用户标识不同，修改 `VITE_ADK_APP_NAME`、`VITE_ADK_USER_ID`。
