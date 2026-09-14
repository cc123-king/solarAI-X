# SolarAI-X五分钟演示脚本

## 准备

1. 运行`run_all.bat`；
2. 打开<http://127.0.0.1:8001/api/health>；
3. 打开<http://127.0.0.1:5173>；
4. 确认`.env`已经配置自己的DeepSeek Key。

## 第一步：展示工程可观测性

先展示健康接口，解释：

- `llm_configured`表示DeepSeek是否配置；
- `mcp_connected`表示FastAPI是否连接MCP；
- `models`表示三组外部预测模型是否就绪；
- 某个模型缺失不会拖垮整个系统。

## 第二步：展示RAG

问题示例：

```text
请从知识库中查找影响钙钛矿太阳能电池稳定性的因素，并说明资料来源。
```

讲解重点：模型先调用`search_embedded_text`，检索结果返回后再组织最终答案。

## 第三步：展示Tool Calling和MCP

问题示例：

```text
使用默认参数模拟一组硅太阳能电池，先告诉我准备使用哪些参数，确认后生成JV曲线。
```

讲解重点：工具列表来自MCP Server，DeepSeek选择`simulate_solar_cell`并生成JSON参数。

## 第四步：展示领域预测页面

依次打开：

- 钙钛矿电池老化预测；
- 钙钛矿带隙预测。

强调预测结果用于研究辅助和演示，不能替代真实实验。

## 第五步：展示代码

推荐只打开三处：

1. `api/main.py`中的MCP Client和流式工具循环；
2. `api/mcpserver.py`中的`@mcp.tool()`工具；
3. `api/embed.py`中的向量检索。

不要在现场逐行讲1500行文件，而是围绕一次请求的数据流解释。
