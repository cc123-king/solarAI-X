# 本地模型验证与未完成项

模型由用户批准下载到独立资源目录，不随仓库提交。运行时只加载显式指定的本地模型，保持零新增付费模型调用。实验环境复用原有 CPU PyTorch，另行安装 Transformers 4.57.1、PEFT 0.17.1、Accelerate 1.10.1 与 SentenceTransformers；这是本地隔离环境复用，不是从空机器安装验收。

## 真实语义模型

`evaluation/semantic-frozen-v1.json` 是执行前冻结的 12 条中英文诊断，与模型训练种子隔离；只做一次词法/语义对照，没有按结果修改题目或调路由。`scripts/evaluate_semantic_memory.py` 使用真实本地 encoder，并通过实际 MCP STDIO 发现九个工具的描述和 Schema。`reports/semantic-frozen-v1-results.json` 保存每题结果、模型版本、文件哈希与源码指纹。

| 方法 | 记忆 Hit@1 | 路由保留目标工具 | 平均保留工具数 |
|---|---:|---:|---:|
| 词法 | 7/12 | 9/12 | 7.00 |
| all-MiniLM-L6-v2 | 7/12 | 8/12 | 4.58 |

依赖工具固定保留会提高路由召回，因此这个指标不能等同于模型选工具成功率。英文小模型在混合语言样本上没有测出优势；语义能力有真实运行证据，质量提升仍未达到目标。后续应选合适的中英文模型与新的未见问题，不能反复把这 12 条当独立验收。

## 实际训练、重载和工具调用

`reports/local-lora-training-v1.json` 记录真实 CPU 训练：Qwen2.5-0.5B-Instruct 的精确下载版本、基座权重 SHA256、LoRA 参数、适配器 SHA256、12 条训练种子、4 条训练验证、3 步训练和约 515 秒训练运行时间（含训练器的轮末评估，另有一次最终评估）。验证 loss 约 0.00985；它是助手 token 上的 teacher-forced loss，不是任务成功率。种子规模不足以得出泛化收益。

训练后重载适配器，用真实回环 HTTP 服务连接实际 MCP STDIO，在执行前冻结的四条新探针上各调用基座与 LoRA 一次。`reports/local-lora-mcp-probe-v1.json` 保留了每次模型参数、usage、耗时和原始报告 SHA256。两个模型各有 4/4 条符合基本 Schema 且包含预期参数的调用，基座 3/4、LoRA 4/4 次 MCP 未报告执行错误。四个工具被限定在探针目录中；这里没有测九工具自由路由或完整多轮研究任务，也没有证明普遍提高准确率。

首次测量暴露两个绘图契约问题：空的可选行标签导致绘表错误，`save_file=false` 时生成图片但丢弃返回值。现已修复为使用默认标签，并在不落盘时返回真正的 MCP ImageContent。修复前的探针结果保留；修复后以真实 MCP 回归核对 PNG 字节和执行链路，不改写首次模型成绩，也不将回放当新冻结验收。自动生成的 Row/Column 标签使用字体可直接显示的英文。

复现时自行准备已批准的本地资源路径，不使用密钥或云 API：

```powershell
python scripts/train_lora.py --evaluation-manifest finetuning/evaluation-group-ids.json --model <local-base-model> --base-model-id Qwen/Qwen2.5-0.5B-Instruct --output <new-adapter-directory> --max-length 4096 --max-steps 3 --threads 4 --allow-cpu --run
python scripts/serve_cpu_lora.py --base-model <local-base-model> --solar-adapter <adapter-directory> --port 19018 --threads 4
python scripts/verify_local_lora_mcp.py --training-manifest <adapter-directory>/training_manifest.json --output <new-report.json>
```

## 可靠性和监控

错误指标 `solar_failures_total` 按固定阶段与类别统计 timeout、非法 JSON、参数对象、Schema、未授权工具、MCP 错误、连接问题、模型流错误和最终回答校验失败。未知工具名、用户文本和异常消息不成为标签。`deploy/agent-alerts.yml` 与 `agent-alert-tests.yml` 由真实 promtool 验证告警触发与恢复；独立 CI 的 OTLP/HTTP collector 接收同 trace 的 task/model/tool spans，并把回执保存在 CI artifact。这是公开隔离实验，不是企业生产监控验收。

## 仍需要外部资源的能力

- 三个原 AutoGluon 预测器目录仍缺失，旧接口继续明确不可用。已另行以作者 CC BY 4.0 的公开 TOPCon 数据训练数值森林，新增完整18字段单条/批量预测；278条同源留出测试 Eff MAE 0.255个百分点、Voc MAE 0.00437，实际MCP单条/批量与结果引用检查通过。来源、适用范围与未验证部分见 [公开TOPCon恢复](PUBLIC_TOPCON_RECOVERY.md)。此模型不等同于原13字段模型，不能用于钙钛矿。
- 更大视觉模型和 MinerU 引擎的下载与验证由用户决定延后。
- 未验证 vLLM、QLoRA、DPO 或分布式训练；它们不能由本次 CPU LoRA 实验替代。
- 旧文件能证实的首轮历史任务成绩为 32/40；尚未找到 37/40 的可复核证据，不将它写为本轮成果。
- 科学模型报告的总体论文留出 MAE 0.02703 eV 对应 30 条测试，MAPbIBr 子集为 17 条、MAE 0.01790 eV；不能混写样本范围。其他小样本家族保持禁用。
