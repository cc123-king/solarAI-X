# 固定12任务真实验收

qwen-plus与真实本机MCP：8道正向、4道负向均通过，新增23次聊天、0次向量请求。JSON保留结果与实际工具事件；offline_rescore记录从持久事件重新评分的结果。单次固定任务结果不代表通用Agent准确率或并发性能。

原始结果中的plot_results路径为运行时生成路径；对应图已复制为[真实曲线产物](../../docs/examples/live-agent-curve.png)。公开文献模型的30条论文分组留出预测、训练规则与系数见models/bandgap，不能与Agent任务通过率混用。

[完整结果与限制](../../docs/REAL_AGENT_ACCEPTANCE_V5.md)
