# NLA 长期记忆模块

该模块为数值代数 Agent 提供长期记忆能力：把历史问答抽象成结构化知识条目，存入向量数据库并支持检索、去重、合并、版本管理与简单评估。

## 1. 方案说明

- 向量库方案：`ChromaDB`（本地持久化，零外部服务依赖，便于在单机工程直接集成）。
- 嵌入方案：默认使用 `HashEmbeddingFunction`（离线可运行、测试稳定）；可后续替换为更强模型。
- 记忆策略：
  - 入库前归一化（中英混合、数学符号文本化）
  - 相似去重（高阈值直接去重）
  - 同类合并（中阈值触发合并并升版本）
  - 低质量过滤
  - 支持待审核区（pending）与人工确认后转正

## 2. 目录结构

```text
memory/
  __init__.py
  config.py
  schema.py
  normalizer.py
  embedder.py
  extractor.py
  store.py
  agent_integration.py
  README.md
```

## 3. 核心接口

- `add_memory(item)`
- `search_memory(query, top_k=5, filters=None)`
- `update_memory(id, item)`
- `delete_memory(id)`
- `summarize_memory(query_results)`
- `extract_memory_from_dialogue(user_question, assistant_answer)`

这些接口已在 `NLAMemoryStore` 和 `agent.py` 工具函数中提供。

## 4. 配置

编辑 `memory_config.json`：

- `auto_write_mode`: 是否自动写入正式库
- `readonly_mode`: 只读检索模式（禁写）
- `dedup_similarity_threshold`: 去重阈值
- `merge_similarity_threshold`: 合并阈值
- `min_quality_score`: 最低质量分

## 5. 启动与示例

在 `NLA_Master` 下执行：

```bash
python memory_cli.py --seed sample_memories.json
python memory_cli.py --query "稀疏SPD系统怎么解" --top-k 3
```

## 6. Agent 集成行为

- 回答前：自动检索历史记忆并注入用户问题上下文
- 回答后：在 `auto_write_mode=true` 时自动抽取上一轮问答并入库（仅写入用户问题的原子子问题，不写入答案文本）
- 只读模式：`readonly_mode=true` 时仅检索，不会写入/更新/删除
