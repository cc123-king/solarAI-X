# 本地模型与 Agent 功能集成

本轮先接通功能入口，按用户要求将全量测试、真实模型对照和负载验收后置。代码集成状态不代表效果已经达标，旧报告只适用于其中记录的源码和协议。

## 已有本地模型的一次启动

需要已有基础模型、完成训练的 LoRA 目录，以及包含依赖的 Python 环境。启动器不会训练、下载、修改 `.env` 或启动容器；四项服务都只绑定回环地址。没有模型资产时仍可用 README 的传统方式启动 MCP/API/UI，查看能力状态和使用可用的本地工具。

基础 API/MCP/UI 环境安装 `requirements.txt`；模型环境还需安装 `finetuning/requirements.txt` 与基础 API 依赖，可使用同一 Python。语义记忆另需 `requirements-semantic-memory.txt`。现有环境可以通过 `--python` 和 `--model-python` 分开指定。

在项目根目录运行以下 PowerShell 命令，先查看启动计划：

```powershell
python scripts/run_local_stack.py `
  --base-model D:/AgentValidationAssets/models/qwen2.5-0.5b-instruct `
  --solar-adapter D:/AgentValidationAssets/runs/solar-lora-20261102-v1 `
  --encoder D:/AgentValidationAssets/models/multilingual-e5-small `
  --encoder-profile e5 `
  --dry-run
```

确认依赖齐备后去掉 `--dry-run` 即为实际启动命令。需要不同环境时加 `--python <API环境的python.exe> --model-python <模型环境的python.exe>`。路径示例对应本任务已有资产，其他机器应替换为自己的目录。

默认模型端口 18018、MCP 18346、API 18001、UI 15173。启动前检查端口和依赖；任一被占用即退出，不停止占用者。模型须提供 `agent-base` 与 `solar-tools` 两个别名；API 就绪要求 MCP 已连接。独立数据库、四项进程的 stdout/stderr 日志和启动锁保存在 Git 忽略的 `.local/resume-stack/`。API 本身的原生日志仍在 `api/api.log`。使用 `--state-dir` 选择其他独立状态目录；同一目录不能同时启动两套服务。异常退出后的残留锁需人工核实 PID 后处理。

打开 `http://127.0.0.1:15173/`，在模型选择框切换本地基座或 LoRA，发起真实聊天与 MCP 工具调用。Ctrl+C 或服务异常时启动器清理自己创建的进程树。它将 Hugging Face 设为离线，关闭本次子进程的外部 OTLP 导出和 Gradio analytics；要接 Collector 可另按常规部署配置。科学预测、带隙和 RAG 的资产状态仍以 `/api/health` 为准；模型服务启动不意味着这些资产存在。

## 长期记忆的完整使用流程

智能对话右栏的“长期记忆”支持刷新列表、读取具体标识、编辑内容/来源/TTL、保存、删除和召回预览。内容须由用户确认；系统不把模型草稿自动保存成科研事实。

新标识版本为 0；更新已有标识先读取当前版本。服务端通过 CAS 核对版本，冲突返回 409。过期或已删除标识也能读取当前版本再重新保存；查询不返回其失效正文。删除清空当前内容并增加版本，后续召回排除本条；数据库历史备份和旧聊天内容需另按运维策略处理。TTL 为从本次保存起的 1–365 天。

“预览召回”只调用记忆检索，不调用聊天模型，展示当前查询会注入的偏好/案例、召回方法与字符预算。预览与正式聊天都调用同一 `ResearchMemory.recall`，实际发送前还会受整轮上下文预算约束。`--encoder` 配合 `--encoder-profile e5` 启用本地 E5；未提供 encoder 时记忆为词法模式。工具路由可用 `--tool-routing all|lexical|semantic` 显式覆盖；语义模式缺少 encoder 会拒绝启动。

接口对应：`GET/PUT /api/memory`、`GET /api/memory/entry?kind=case&key=...`、`DELETE /api/memory/{kind}/{key}`、`GET /api/memory/recall?query=...`。目前是本机单用户工作台，namespace 用于隔离实验，不提供企业身份认证。

## 简历功能的入口与资产边界

| 简历能力 | 使用入口 | 当前边界 |
|---|---|---|
| MCP、白名单、Schema、超时取消、执行轨迹 | 智能对话、`/api/tasks`、`MCP_TOOL_ALLOWLIST`、`MCP_TOOL_TIMEOUT_SECONDS` | 当前目录有 10 个工具（包含后来新增的公开 TOPCon）；部分科学工具因资产缺失不可用 |
| 上下文裁剪、结果压缩、引用回填、图表数值核对 | 正式聊天工作流、`api/context_budget.py`、`api/execution_evidence.py` | 字符预算；依赖真实执行账本，不用新测试成绩替代质量结论 |
| Transformers/PEFT LoRA、response mask、adapter 重载 | `scripts/train_lora.py`、`scripts/serve_cpu_lora.py`、统一启动器、模型选择框 | 本轮不新增训练或对照；CPU 服务不承诺生产吞吐 |
| 公开论文带隙基线与分组留出 | 带隙页面、`scripts/train_bandgap_baseline.py`、历史公开基线报告 | 历史 MAE 属于对应数据和切分；未重新训练，不适用于未开放材料家族 |
| Buffer、长期向量记忆、语义工具路由 | 对话历史、长期记忆面板、encoder/route 配置 | 本地显式记忆；语义策略需要外部模型目录 |
| 日志、指标、trace、告警规则 | `/api/metrics`、`deploy/agent-alerts.yml`、`SOLAR_OTLP_ENDPOINT` | 本轮补齐公开 TOPCon 工具的观测目录；企业监控和新版本运行表现待验收 |
| 原硅电池/窄带隙/宽带隙钙钛矿预测 | 各业务页面及健康接口 | 三组原 AutoGluon 资产缺失；明确显示不可用。公开 TOPCon 仅作为独立工具提供 |

简历中涉及的具体准确率、留出 MAE 和延迟仍要逐项引用原报告。本轮的功能集成不产生新的业务成绩。
