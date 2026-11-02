# SolarAI-X 能力补齐与复现

基于 GitHub 主分支合入原本仅在本机的上下文预算与 CPU LoRA 工具，再补跨会话记忆、工具路由和运行观测。历史模型任务成绩不作为本轮新源码的准确率。

| 能力 | 实现和证据 | 边界 |
|---|---|---|
| 上下文 | 固定当前问题、工具轨迹和服务端记忆，整轮裁剪旧对话，完整数值从原执行账本引用 | 预算单位为 JSON 字符；未解析 JSON 和非对象参数提前拒绝 |
| 分层记忆 | SQLite 显式偏好/确认案例，版本 CAS、TTL、删除、命名空间及独立预算；测试核对实际模型输入 | 默认词法召回。删除清空当前记录，非磁盘取证级擦除；项目仍是本机单用户，命名空间不替代认证 |
| 工具路由 | 只从注册 MCP 工具选择，保留检索/绘图依赖并记录方法与数量 | 默认 `SOLAR_TOOL_ROUTING=all`；`lexical` 可直接演示。`semantic` 要求本地 encoder，不以关键词冒充语义；无新的成功率提升结论 |
| 可观测性 | `/api/metrics` 提供任务、模型、工具、usage 和延迟直方图；OpenTelemetry task/model/tool 同一 trace | 默认不外传正文、记忆、参数和异常原文。只有 provider 上报 usage 才计 token；缺失不代表零成本。真实 SDK 内存 exporter 独立测试 |
| LoRA | 助手响应损失掩码、分组清单校验、本地 PEFT、CPU API、输出/忙时限制及 Schema 验证 | 合入代码与作者构造的 12/4 条种子，不上传权重；未重新训练，不将历史四条模板样例当泛化收益，vLLM 未验证 |

## 演示与配置

启动原 MCP/API/UI；继续只绑定回环地址。CORS 默认只允许本机 UI 的 5173 端口。

`PUT /api/memory` 示例：

```json
{"kind":"preference","key":"units","value":"Use eV for bandgap values.","source":"explicit user request","expected_version":0,"ttl_days":30}
```

通过 `GET /api/memory` 查看。新会话会注入偏好；`DELETE /api/memory/preference/units?expected_version=1` 后下一次输入不再出现。系统不自动把未审核模型回答保存成长期事实。

可选语义模式安装 `requirements-semantic-memory.txt`，设置 `SOLAR_ENCODER_PATH` 为已有 SentenceTransformer 目录、`SOLAR_TOOL_ROUTING=semantic`。只加载本地文件，缺失配置报错，不自动下载/调用云模型。当前语义单测使用确定性 encoder，只证明接线，未验证真实 embedding 效果。

Prometheus 抓取 `/api/metrics`，用 `solar_duration_seconds_bucket` 计算 P95，比较 model calls 与 usage observations 看 token 覆盖率。告警样例位于 `deploy/agent-alerts.yml`；企业监控尚未接入。

可选安装 `requirements-observability.txt` 并设置 `SOLAR_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces` 才启用 exporter。兼容性不足的模型服务可设置 `LLM_STREAM_USAGE=false`。

## 训练与检查

```powershell
python scripts/train_lora.py --evaluation-manifest path/to/evaluation-group-ids.json
python scripts/serve_cpu_lora.py --help
python -m pytest
python scripts/verify_execution_contracts.py --output path/to/new-output-directory
```

manifest 格式为 `{"fixture_groups":["development-id","sealed-group-id"]}`，只含分组标识。训练器不读取封存题；只检查声明的隔离，调用者负责清单完整。没有 `--run` 不训练，训练也只读取明确指定的本机模型并拒绝覆盖输出，超长样本直接报错。

可选 `python -m pytest integration_tests/test_opentelemetry_spans.py` 使用真实 SDK 的内存 exporter，无网络外传。本轮新增付费云调用为 0，未改动原数据库/模型服务；历史科学 MAE 仍遵守原公开数据分组和留出范围。
