# SolarAI-X架构说明

## 1. 服务关系

SolarAI-X由三个独立进程组成：

```text
Gradio UI（5173）
        ↓ HTTP / SSE
FastAPI（8001）
        ↓ MCP SSE
MCP Server（12346）
        ↓
RAG、仿真、预测、绘图工具
```

拆成三个进程的原因是：

- 页面、业务API和工具服务职责分离；
- MCP Server可以被其他MCP Client复用；
- 模型工具失败时不必让前端直接处理模型细节；
- 后续可以单独部署和扩缩容。

## 2. Agent工具调用流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant UI as Gradio
    participant API as FastAPI
    participant LLM as DeepSeek
    participant MCP as MCP Server
    participant Tool as 领域工具

    User->>UI: 输入自然语言任务
    UI->>API: POST /api/chat/send
    API->>MCP: list_tools()
    MCP-->>API: 工具名称、描述、Schema
    API->>LLM: messages + tools
    LLM-->>API: tool_calls(name, arguments)
    API->>MCP: call_tool(name, arguments)
    MCP->>Tool: 执行RAG/仿真/预测
    Tool-->>MCP: 文字、数值、文件或图像
    MCP-->>API: 工具执行结果
    API->>LLM: tool result
    LLM-->>API: 最终自然语言回答
    API-->>UI: SSE流式事件
    UI-->>User: 回答、推理和图表
```

## 3. RAG流程

```text
太阳能领域文档
→ TextEmbedding.process_directory
→ BGE向量
→ embeddings.pkl + file_info.pkl
→ search_embedded_text
→ Top-N相似片段及来源
→ DeepSeek生成回答
```

当前向量保存在本地文件中，适合教学和演示。数据规模扩大后可以替换为FAISS、Qdrant、Milvus或pgvector。

## 4. 模型降级设计

SolarAI-X包含多组彼此独立的模型。如果在模块导入时强制要求所有模型存在，那么缺少一个外部目录就会导致整个API无法启动。

维护版改为：

```text
服务启动
→ 尝试加载各模型组
→ 记录每组模型的available/error状态
→ 其他功能继续启动
→ 调用缺失模型时返回清晰的503错误
```

`GET /api/health`用于查看当前实际能力，而不是只返回一个无法解释的`ok`。

## 5. 数据生命周期

| 数据 | 当前存储 | 生命周期 |
|---|---|---|
| 对话消息 | Python内存字典 | FastAPI重启后消失 |
| Embedding | 本地PKL文件 | 持久保存 |
| AutoGluon模型 | 外部模型目录 | 持久保存 |
| 老化模型 | 项目内PTH文件 | 持久保存 |
| 生成图表 | `outputs/` | 手动清理 |

后续可以使用SQLite或PostgreSQL持久化对话，并为图表增加定期清理策略。
