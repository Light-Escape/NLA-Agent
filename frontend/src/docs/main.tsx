import React from "react";
import ReactDOM from "react-dom/client";
import "katex/dist/katex.min.css";
import "./styles.css";

type Status = "已完成" | "进行中" | "待完善" | "规划中";

interface NavItem {
  id: string;
  label: string;
}

interface ModuleItem {
  name: string;
  path: string;
  status: Status;
  description: string;
}

interface ProgressItem {
  area: string;
  status: Status;
  detail: string;
}

interface TestGroup {
  name: string;
  scope: string;
  status: string;
}

const navItems: NavItem[] = [
  { id: "overview", label: "项目概况" },
  { id: "structure", label: "项目结构" },
  { id: "backend", label: "后端与 Agent" },
  { id: "frontend", label: "前端工作台" },
  { id: "progress", label: "阶段进展" },
  { id: "tests", label: "测试与评估" },
  { id: "usage", label: "运行方式" },
  { id: "next", label: "下一步计划" }
];

const modules: ModuleItem[] = [
  {
    name: "Agent 编排",
    path: "NLA_Master/agent.py",
    status: "已完成",
    description: "Google ADK root_agent 入口，注册数值计算、Workspace、记忆、文件读取等工具。"
  },
  {
    name: "数值后端",
    path: "NLA_Master/linalg_backend.py",
    status: "已完成",
    description: "封装 LAPACK/BLAS 能力，包括线性方程、最小二乘、GEMM 和通用 driver 调用。"
  },
  {
    name: "稀疏计算",
    path: "NLA_Master/sparse_backend.py",
    status: "已完成",
    description: "基于 SciPy sparse 支持稀疏直接法、CG/GMRES 和部分特征值计算。"
  },
  {
    name: "矩阵解析",
    path: "NLA_Master/parsers.py",
    status: "已完成",
    description: "支持 CSC 文本和 Matrix Market .mtx.gz，加载结果进入 Workspace 句柄。"
  },
  {
    name: "Workspace",
    path: "NLA_Master/workspace.py",
    status: "进行中",
    description: "保存矩阵、向量、分解结果和诊断量，提供摘要、统计、结构和切片接口。"
  },
  {
    name: "策略工具",
    path: "NLA_Master/policy.py",
    status: "进行中",
    description: "负责任务路由、前提检查、矩阵性质判断和算法推荐。"
  },
  {
    name: "长期记忆",
    path: "NLA_Master/memory/",
    status: "已完成",
    description: "保存问题模式、解法模式、失败经验和待审核记忆条目。"
  },
  {
    name: "前端工作台",
    path: "NLA_Master/frontend/",
    status: "进行中",
    description: "Vite + React 三栏工作台，支持对话、文件上传、Markdown/LaTeX 和 Workspace 展示。"
  }
];

const progressItems: ProgressItem[] = [
  {
    area: "4/20 汇报基线",
    status: "已完成",
    detail: "已形成以 Google ADK root_agent 为入口、NumPy/SciPy/LAPACK/BLAS 为计算后端、Workspace 保存中间对象的 MATLAB 替代型 Agent 雏形。"
  },
  {
    area: "数值计算工具链",
    status: "已完成",
    detail: "线性方程、最小二乘、矩阵乘法、特征值、SVD、伪逆、矩阵函数、稀疏求解和迭代算法均已纳入能力测试范围。"
  },
  {
    area: "文件与大矩阵处理",
    status: "进行中",
    detail: "CSC 文本、Matrix Market 对称格式、上传 URI 与 Workspace 句柄流程已可用于评测；后续继续强化大对象受控读取、审计与内存保护。"
  },
  {
    area: "前端工作台与说明网站",
    status: "进行中",
    detail: "已具备 Vite + React 三栏交互工作台和独立 /docs/ 静态汇报页；后续重点是组件拆分、会话隔离、对象卡片和诊断面板。"
  },
  {
    area: "评测闭环",
    status: "待完善",
    detail: "40 题人工 benchmark 已全部记录 3/3，下一阶段需要把高价值用例固化为自动化回归和更真实的数据集评测。"
  }
];

const testGroups: TestGroup[] = [
  { name: "第一组：基础 MATLAB 替代", scope: "SPD 方程、一般方阵、奇异系统、Hilbert 病态、BLAS GEMM", status: "5/5，全部 3/3" },
  { name: "第二组：最小二乘与秩亏", scope: "普通拟合、秩亏最小二乘、欠定最小范数", status: "3/3，全部 3/3" },
  { name: "第三组：特征值能力", scope: "对称、非对称复特征值、不可对角化、部分特征值选择", status: "4/4，全部 3/3" },
  { name: "第四组：稀疏矩阵与文件读取", scope: "CSC 解析、稀疏求解、稀疏特征值", status: "3/3，全部 3/3" },
  { name: "第五组：交互、策略与鲁棒性", scope: "教练模式、缺失信息、维度错误、拒绝显式求逆、工作区变量", status: "5/5，全部 3/3" },
  { name: "第六组：高级线性系统与稳定性", scope: "多右端项、Cholesky 回退、对称不定、病态最小二乘、迭代改进", status: "5/5，全部 3/3" },
  { name: "第七组：SVD、秩与伪逆", scope: "有效秩、截断 SVD、Moore-Penrose 伪逆、列主元 QR、岭回归", status: "5/5，全部 3/3" },
  { name: "第八组：高级特征值与矩阵函数", scope: "广义特征值、Hermitian 复矩阵、Schur、矩阵指数、奇异值区别", status: "5/5，全部 3/3" },
  { name: "第九组：大规模稀疏与迭代算法", scope: "CG/GMRES、预条件器、Matrix Market symmetric、大稀疏矩阵方案", status: "5/5，全部 3/3" }
];

const projectTree = `NLA_Master/
  agent.py                  # ADK root_agent 组装与工具注册
  policy.py                 # 任务路由、前提检查、算法选择
  linalg_backend.py         # LAPACK/BLAS 后端封装
  sparse_backend.py         # 稀疏矩阵求解与迭代算法
  parsers.py                # CSC / Matrix Market 读取
  workspace.py              # 会话级 Workspace 与句柄协议
  upload_api.py             # 矩阵上传 API
  upload_store.py           # 上传文件存储
  executors.py              # Python snippet 执行器
  retrieval.py              # NumPy/SciPy 文档检索
  memory/                   # 长期记忆模块
  tests/                    # 后端单元测试
  frontend/                 # React 前端工作台与说明网站
  NLA_AGENT_TEST_CASES.md   # 40 题能力评估文档`;

function App() {
  return (
    <div className="doc-layout">
      <aside className="sidebar">
        <div className="brand">
          <span>NLA Agent</span>
          <strong>项目说明文档</strong>
        </div>
        <nav aria-label="文档目录">
          {navItems.map((item) => (
            <a href={`#${item.id}`} key={item.id}>
              {item.label}
            </a>
          ))}
        </nav>
      </aside>

      <main className="content">
        <header className="doc-header" id="overview">
          <p className="kicker">Numerical Linear Algebra Agent</p>
          <h1>数值代数 Agent 项目说明</h1>
          <p>
            本项目是一个面向中文数值线性代数任务的 Agent 系统。它以 Google ADK 为编排层，结合
            NumPy/SciPy、LAPACK/BLAS、稀疏矩阵工具、Workspace 状态管理和 React 前端工作台，用于辅助完成线性方程组、最小二乘、特征值、SVD、矩阵文件读取、大规模稀疏问题和数值稳定性分析等任务。
          </p>
          <dl className="summary-grid">
            <div>
              <dt>当前定位</dt>
              <dd>科研/课程场景下的数值线性代数辅助 Agent</dd>
            </div>
            <div>
              <dt>主要成果</dt>
              <dd>后端工具链成型，前端工作台和说明网站已有可展示版本</dd>
            </div>
            <div>
              <dt>评估材料</dt>
              <dd>40 题能力测试全部记录 3/3，总分 120/120</dd>
            </div>
            <div>
              <dt>说明网站</dt>
              <dd>纯静态文档页，便于阶段汇报和持续维护</dd>
            </div>
          </dl>
        </header>

        <DocSection id="structure" title="项目结构">
          <p>项目主体位于 <Code>NLA_Master</Code>，后端 Agent、数值计算工具、记忆模块、测试文档和前端工程都在该目录下。</p>
          <pre className="tree"><code>{projectTree}</code></pre>
        </DocSection>

        <DocSection id="backend" title="后端与 Agent 架构">
          <p>
            后端采用单 <Code>root_agent</Code> 架构。LLM 负责理解用户意图和组织解释，实际数值计算由 Python 工具完成。Workspace
            保存真实矩阵对象，避免把大矩阵直接放入模型上下文。
          </p>
          <div className="module-table" role="table" aria-label="后端模块状态">
            <div className="module-row heading" role="row">
              <span>模块</span>
              <span>路径</span>
              <span>状态</span>
              <span>说明</span>
            </div>
            {modules.map((module) => (
              <div className="module-row" role="row" key={module.path}>
                <strong>{module.name}</strong>
                <Code>{module.path}</Code>
                <StatusBadge status={module.status} />
                <span>{module.description}</span>
              </div>
            ))}
          </div>
        </DocSection>

        <DocSection id="frontend" title="前端工作台现状">
          <p>
            前端位于 <Code>NLA_Master/frontend</Code>，使用 Vite + React + TypeScript。当前主要页面是三栏工作台：左侧历史/文件，中间对话，右侧
            Workspace。新增的说明网站作为独立 <Code>/docs/</Code> 静态入口存在，不影响工作台主界面。
          </p>
          <ul className="plain-list">
            <li>已具备 Markdown、GFM、LaTeX/KaTeX 渲染能力。</li>
            <li>已接入 ADK Web API 和矩阵上传 API 的前端调用层。</li>
            <li>当前 <Code>App.tsx</Code> 体量较大，后续重构应优先拆分组件与状态模型。</li>
            <li>工作台目标是让会话、文件、Workspace 数学对象和诊断信息保持一致。</li>
          </ul>
        </DocSection>

        <DocSection id="progress" title="阶段进展">
          <p>
            4/20 阶段汇报确认了项目目标和基础架构：用 Agent + Python 数值工具链替代部分 MATLAB 工作流，重点覆盖矩阵读入、Workspace 状态保存和数值计算解释。本轮更新基于
            <Code>NLA_AGENT_TEST_CASES.md</Code>，把评测范围扩展到 40 题并完成全部评分记录。
          </p>
          <div className="progress-list">
            {progressItems.map((item) => (
              <article key={item.area}>
                <div>
                  <h3>{item.area}</h3>
                  <StatusBadge status={item.status} />
                </div>
                <p>{item.detail}</p>
              </article>
            ))}
          </div>
        </DocSection>

        <DocSection id="tests" title="测试与评估">
          <p>
            项目评估主要由两部分组成：后端单元测试和 <Code>NLA_AGENT_TEST_CASES.md</Code> 中的 40 题人工能力测试。40
            题覆盖 MATLAB 替代、稀疏矩阵、交互策略、SVD、矩阵函数和大规模迭代算法等方向。当前记录显示每题均达到“结果正确、算法选择合理、能主动检查条件并解释数值风险”的 3/3 标准。
          </p>
          <div className="score-card">
            <strong>40 / 40</strong>
            <span>全部测试题已有 3/3 评分记录，总分 120/120</span>
            <p>评测重点已经从“补齐题目”转向“沉淀自动化回归”：优先固化算法路由、残差验证、病态提示、稀疏迭代收敛信息和大矩阵内存保护。</p>
          </div>
          <div className="test-list">
            {testGroups.map((group) => (
              <article key={group.name}>
                <h3>{group.name}</h3>
                <p>{group.scope}</p>
                <span>{group.status}</span>
              </article>
            ))}
          </div>
        </DocSection>

        <DocSection id="usage" title="运行方式">
          <p>说明网站本身是静态页面；完整 Agent 运行需要分别启动 ADK 服务、上传 API 和前端开发服务。</p>
          <pre><code>{`# 后端 Agent CLI
adk run NLA_Master

# ADK Web 服务
adk web --port 8000

# 矩阵上传 API
uvicorn NLA_Master.upload_api:app --port 8001

# 前端工作台与说明网站
cd NLA_Master/frontend
npm run dev
# 说明网站: http://localhost:5173/docs/`}</code></pre>
        </DocSection>

        <DocSection id="next" title="下一步计划">
          <ol className="next-list">
            <li>将 40 题人工 benchmark 中的关键用例迁移为自动化回归测试，并保留人工评测记录用于阶段汇报。</li>
            <li>拆分前端工作台组件，建立 Workspace 对象卡片、诊断面板和会话隔离模型。</li>
            <li>强化大矩阵句柄协议，完善摘要、统计、结构、切片和审计接口。</li>
            <li>将 supervisor 从单测模块接入运行时，减少 LLM 工具编排的不确定性。</li>
            <li>增加完整 ADK 对话链的 E2E 回归测试，降低后续迭代风险。</li>
            <li>在说明网站中继续补充截图、架构图、阶段汇报记录和实验案例。</li>
          </ol>
        </DocSection>
      </main>
    </div>
  );
}

function DocSection({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section className="doc-section" id={id}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return <code className="inline-code">{children}</code>;
}

function StatusBadge({ status }: { status: Status }) {
  return <span className={`status ${statusClass(status)}`}>{status}</span>;
}

function statusClass(status: Status) {
  if (status === "已完成") return "done";
  if (status === "进行中") return "active";
  if (status === "待完善") return "todo";
  return "planned";
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
