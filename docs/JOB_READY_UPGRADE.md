# SolarAI-X 求职项目升级

已实现异步模型流、同步科学计算的线程隔离、SQLite 会话与任务、取消执行及真实模型评测。
本轮进一步修复工具错误状态、FastMCP 响应封装，以及生成图表与下载目录不一致的问题。

## 本地运行和任务记录

使用现有安装脚本和三个服务启动脚本。API 启动时将 queued/running 历史任务标记为 interrupted，不会自动重放工具。
默认数据库为 data/state.sqlite3，可用 SOLAR_STATE_DB 指定。当前面向本机单用户、单 API worker。
聊天接口返回 X-Task-ID 响应头和 task 类型 SSE 事件，并保留 content/context/image/error/done。

- GET /api/tasks 列出任务。
- GET /api/tasks/{id} 查看状态、工具事件、参数、结构化结果和回答。
- POST /api/tasks/{id}/cancel 取消后续编排；底层线程计算不保证立即终止。
- GET /api/chat/history 在重启后恢复会话。

工具失败支持标准 MCP isError，以及旧工具的 text.error、text.status=error/failed/unavailable；同时识别 FastMCP 的 result 封装。
明确的业务错误不会被当成可重试网络错误。检索文档中普通的 error 字样不会触发失败。
成功事件保留限长结构化结果，排除图像/向量字节；失败事件保存错误类型。

## 图表路径

曲线、表格与仿真生成文件使用项目根目录下的 plot_results 或 simulation_results，并返回带正斜杠的相对路径。
OUTPUT_DIR 仅接受这两个目录名；其他旧配置会回到对应工具的默认结果目录，保证 /api/files 能访问。
下载仍检查目录、路径穿越和目录重定向，不扩大到整个 outputs 或项目目录。

## 真实模型评测

先启动 MCP 服务，再在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_agent_live.py --live --max-cases 2
```

命令使用本机模型配置，消耗模型额度；默认最多 1 个任务，可选择 2 个。每轮最多 512 输出 token，工具轮次最多 3。
固定任务要求最终返回 JSON，逐个核对 Br 比例、带隙、单位、实际工具结果及全部曲线坐标；曲线任务还检查返回路径对应有效 PNG。
只在回答中出现 1.64，或否定后又给出错误值，都不能通过。首次内容耗时与总耗时分别记录。
这是两个确定性任务的功能验证，不代表通用准确率、吞吐能力或材料模型科学精度。error_type 表示评测未完成。
离线替身负责异常、权限、重试、取消等回归；真实评测与其分开。

## 容器与 CI

准备 .env 后执行 docker compose up --build；UI 为 5173、API 为 8001。
基础镜像不安装 AutoGluon/FlagEmbedding；模型文件见 models/README.md，缺少能力时工具报告不可用。
CI 仅运行离线测试，不带密钥或调用真实模型。Docker 镜像及 Linux CI 实际运行须单独验证。
SQLite 与图表使用命名卷保存；同机运行另一个占用 8001 的项目时调整主机端口。

## v3 图片完整性与评测失败

PNG 必须有完整 IEND 结尾，通过结构/校验和检查，再重新打开完成像素解码后才会判定有效。
截断文件、错误 CRC、压缩数据损坏等情况返回失败及安全的错误类型，不会因为读取过尺寸而误判通过。
单条任务异常会保留为失败记录并继续下一条受限任务；模型配置或 MCP 连接等全局失败仍在报告顶层保留 error_type。
本轮用新评测器复评已有两条真实模型/MCP 执行记录与图表，无新增模型请求；离线测试覆盖损坏图片和异常继续执行。
