# NLA_Master - 数值线性代数 Agent

基于 Google ADK 的数值线性代数助手，使用中文提示词，可求解线性方程组、特征值/特征向量、矩阵运算等。

## 环境要求

- Python 3.10+
- 已安装：`google-adk`、`litellm`、`numpy`、`scipy`
- 本项目测试与 Agent 加载依赖 Conda 环境 `agent`，不要用 `base` 环境直接跑测试。

## 配置

1. 复制环境变量示例并填入 DeepSeek API Key：
   ```bash
   copy .env.example .env
   ```
2. 在 `.env` 中设置 `DEEPSEEK_API_KEY="你的密钥"`（在 [DeepSeek 开放平台](https://platform.deepseek.com) 获取）。**请使用你自己申请的密钥**，不要使用示例里的占位密钥。
3. 本 Agent 通过 LiteLLM 使用 DeepSeek 的 `deepseek-chat` 模型，需已安装 `litellm`、`python-dotenv`（见下方依赖）。  
   **Windows 用户**：若出现 `UnicodeDecodeError`，可在当前会话执行 `$env:PYTHONUTF8 = "1"` 或在系统环境变量中设置 `PYTHONUTF8=1`。

### 若出现 `AuthenticationError: DeepseekException - Authentication Fails (governor)`

- 表示 DeepSeek 返回了 401，即 **API Key 无效或未正确传入**。请检查：
  1. 已把 `.env.example` 复制为 `.env`，并在 **`.env`** 里填写你在 [DeepSeek 开放平台](https://platform.deepseek.com) 申请的 **有效** API Key（不要用示例中的占位 key）。
  2. 将 `.env` 放在 **NLA_Master 目录内**（与 `agent.py` 同级），程序会优先从该位置加载。
  3. 若 Key 曾泄露或过期，请在平台重新生成并更新 `.env`。

## 运行

在**上一级目录**（即包含 `NLA_Master` 的目录）下执行：

```bash
# 命令行对话
adk run NLA_Master

# 或启动 Web 界面（默认 http://localhost:8000）
adk web --port 8000
```

在 Web 界面左上角选择 **NLA_Master** 后即可用中文提问，例如：
- “解线性方程组：2x+y=5, x-y=1”
- “求矩阵 [[1,2],[2,1]] 的特征值和特征向量”
- “计算矩阵 [[1,0],[0,1]] 和 [[2,3],[4,5]] 的乘积”

## 测试

测试必须在 **`agent` Conda 虚拟环境** 中运行，否则会因为 `google.adk` 等依赖缺失导致 `NLA_Master.agent` 无法完整加载。

在**上一级目录**（即包含 `NLA_Master` 的目录）下执行：

```powershell
conda activate agent
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$PWD\NLA_Master;$PWD"
python -m unittest discover -s NLA_Master/tests -t .
```

如果不想激活环境，也可以显式使用该环境的解释器：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$PWD\NLA_Master;$PWD"
& "D:\Anaconda\envs\agent\python.exe" -m unittest discover -s NLA_Master/tests -t .
```

说明：
- `-t .` 会按包级路径发现测试，确保 `NLA_Master.__init__` 和 `NLA_Master.agent` 能被完整加载。
- `PYTHONPATH` 同时包含仓库根目录和 `NLA_Master`，用于兼容历史测试中的包内裸导入。

### 交互式“数值代数教练”模式

本项目现支持教练式求解流程，默认遵循：
1. 先确认你想要**精确解**还是**近似解**；
2. 先提示并检查任务所需的必要条件；
3. 逐步确认矩阵关键性质（方阵、秩、对称性、条件数等）；
4. 再给出算法方案与（按需）数值验证。

你可以这样触发：
- “我想一步步理解，不要直接给最终答案。”
- “先判断是否能精确解，不行再给近似方案。”
- “请先检查必要条件，再带我确认矩阵性质。”

### 读取用户上传的矩阵列压缩格式（CSC）文件

Agent 支持从**列压缩格式（CSC）**文件中读取矩阵，便于处理稀疏矩阵或从外部导出的数据。

**文件格式**（纯文本，UTF-8，共 4 行）：

| 行号 | 含义 | 示例 |
|------|------|------|
| 1 | `m n`：行数、列数（空格分隔） | `2 3` |
| 2 | `indptr`：列指针，共 n+1 个整数 | `0 1 2 3` |
| 3 | `indices`：非零元的行下标（从 0 开始） | `0 1 0` |
| 4 | `data`：非零元的值 | `1.0 3.0 2.0` |

上述示例对应 2×3 矩阵 `[[1, 0, 2], [0, 3, 0]]`。

**使用方式**：
运行方式: uvicorn NLA_Master.upload_api:app --port 8001

- Web 前端选择矩阵文件后，会先上传到后端并返回 `file_id` 与 `nla-upload://<file_id>`；后续 Agent 只用上传 URI 调用读取工具，例如：“请读取刚上传的矩阵并求其特征值”。后端绝对路径不会进入 Agent 上下文。
- 也可以在对话中说清后端可访问的文件路径，或把小型文本内容粘贴给 Agent，例如：“请读取矩阵文件 `matrix.csc` 并求其特征值”。
- 工具：`load_matrix_csc_file(文件路径)` 或 `load_matrix_csc_content(文件内容字符串)`；解析得到的矩阵可直接用于解方程、求特征值、乘法等后续计算。

### 读取 .mtx.gz 文件（Matrix Market 格式）

Agent 支持从 **.mtx.gz**（Matrix Market 稀疏矩阵的 gzip 压缩）文件中读取矩阵。

- **格式**：标准 Matrix Market coordinate 格式，首行 `%%MatrixMarket matrix coordinate [real|integer|pattern|complex] [general|symmetric|skew-symmetric]`，注释行以 `%` 开头，随后一行 `M N L`（行数、列数、非零元个数），再为 L 行 `i j [value]`（1-based 下标）。支持 real/integer/pattern，复数取实部；symmetric/skew-symmetric 会按对称性补全。
- **使用**：通过前端上传后提供 `file_id` / `nla-upload://<file_id>`，或提供后端可访问的 .mtx.gz 文件路径，例如：“请用 `load_matrix_mtx_gz` 读取 `matrix.mtx.gz` 并求特征值”。解析得到的 `A_rows` 可直接用于解方程、特征值、乘法、行列式等工具。  
  **Windows 注意**：如果你是在对话里直接给出绝对路径，建议用正斜杠写法（如 `C:/Users/xxx/matrix.mtx.gz`），或把反斜杠写成 `\\`，避免上层参数解析出现 `Unterminated string...` 之类错误。

### 矩阵大小与上下文长度限制

为避免单次请求超出模型上下文，**通过文件加载的矩阵**会自动保存为 Workspace 矩阵句柄（默认 `A`），工具返回只包含 shape、dtype、nnz、density 等摘要，不会把完整 `A_rows/A_csc` 展开给模型。后续计算应使用 `A_ref="A"` 等引用；若出现 `ContextWindowExceededError`，请减少直接粘贴到对话中的矩阵规模。

## 项目结构

```
NLA_Master/
  agent.py         # Root Agent 组装与工具注册
  parsers.py       # 矩阵读取与格式转换（CSC / Matrix Market）
  retrieval.py     # NumPy/SciPy 文档检索与摘要
  executors.py     # 本地 Python 执行（含静态安全检查）
  linalg_backend.py # LAPACK/BLAS 后端封装（gesv/posv/gemm/lstsq-driver）
  policy.py        # 任务路由、矩阵性质分析、算法选择
  memory/          # 长期记忆模块（向量检索、去重、合并、版本）
  memory_cli.py    # 记忆入库/检索示例命令
  memory_config.json
  sample_memories.json
  tests/
  __init__.py
  .env.example
  README.md
  matrix_csc_example.txt   # 可选：CSC 格式示例文件
```

## 长期记忆模块（数值代数问题经验库）

本项目已内置一个可直接调用的长期记忆组件，核心能力：

- 结构化记忆条目：问题模式、矩阵性质、解法模式、适用前提、失败模式等；
- 向量检索：新问题先召回历史相似问题；
- 去重与合并：相似条目自动去重/合并并维护版本；
- 待审核区：支持用户确认后再转正式记忆；
- 模式切换：支持只读检索模式与自动写入模式。

### 记忆模块快速使用

```bash
# 安装依赖
pip install -r requirements.txt

# 导入示例记忆
python memory_cli.py --seed sample_memories.json

# 查询相似记忆
python memory_cli.py --query "病态最小二乘推荐什么方法" --top-k 5
```

### 单元测试

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## LAPACK/BLAS 工具（新增）

当前 Agent 已提供可直接调用的后端工具（由 `scipy.linalg.lapack/blas` 的 f2py 包装层驱动）：

- `get_linalg_backend_info`：查看 NumPy/SciPy 与 LAPACK/BLAS 可用性；
- `call_lapack(func_name, arrays, kwargs, dtype, output_names)`：MATLAB 风格通用 LAPACK driver 调用入口；
- `call_blas(func_name, arrays, kwargs, dtype, output_names)`：MATLAB 风格通用 BLAS routine 调用入口；
- `solve_linear_lapack(A_rows, b, assume)`：线性方程组，`assume` 支持 `auto/pos/gen`；
- `least_squares_lapack(A_rows, b, driver)`：最小二乘，`driver` 支持 `gelsd/gelss/gelsy`；
- `gemm_blas(A_rows, B_rows, alpha, beta, C_rows, trans_a, trans_b)`：BLAS GEMM 矩阵乘法。

通用入口允许按 LAPACK/BLAS 函数名调用 SciPy 已暴露的例程，例如 `func_name="gejsv"` 或 `func_name="dgejsv"`，后端会根据 `dtype` 和输入数组解析到 `dgejsv/sgejsv/...`。在 Windows 上不要直接查找 `lapack.dll`、`.so` 或 `dgejsv_` 符号；正确路径是通过 SciPy 包装层调用。常见 Jacobi SVD 示例：

```python
call_lapack(
    func_name="gejsv",
    arrays={"a": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]},
    kwargs={"jobu": 1, "jobv": 1},
    output_names=["s", "u", "v", "workout", "iworkout", "info"],
)
```

这些任务会优先走后端工具，`run_python_snippet` 作为通用兜底执行方式。

### 绘图执行说明（run_python_snippet）

- 当代码中使用 `matplotlib` 绘图时，执行器会强制使用无界面后端（`Agg`），避免 `plt.show()` 阻塞导致超时；
- 会自动尝试设置中文字体（如 `Microsoft YaHei`、`SimHei` 等）并设置 `axes.unicode_minus=False`，降低中文乱码概率；
- `plt.show()` 会自动保存为 PNG 到运行目录下的 `generated_images/`；
- 工具返回结果中会包含 `image_files` 字段，列出可直接下载/查看的图片绝对路径。
