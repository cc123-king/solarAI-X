# 模型目录说明

SolarAI-X可以在部分模型未配置时降级启动，并通过`GET /api/health`报告功能可用状态。

完整的AutoGluon模型目录结构如下：

```text
models/
├── solar/
│   ├── Vm/
│   ├── Im/
│   ├── Voc/
│   ├── Jsc/
│   ├── FF/
│   └── Eff/
├── perovskite_narrow/
│   └── 同样包含六个参数子目录
└── perovskite_wide/
    └── 同样包含六个参数子目录
```

如果模型保存在其他位置，在`.env`中设置绝对路径：

```dotenv
MODEL_DIR=D:\models\solar
PSKNA_DIR=D:\models\perovskite_narrow
PSKWD_DIR=D:\models\perovskite_wide
```

仓库已经保留老化预测所需的`api/quxian_high500.pth`和`api/quxian_shiyan.xlsx`。大型AutoGluon模型建议通过模型制品仓库、Release或对象存储单独分发。

新增公开 TOPCon 数值森林替代模型，由 `scripts/train_topcon_public.py` 在外部资源目录生成 `forest.npz` 和 `model-card.json`；以 `TOPCON_PUBLIC_MODEL_DIR` 指定并重启 API/MCP。模型使用原始18字段，不兼容旧13字段，不能替代钙钛矿预测器。数据许可、下载与质量边界见 [公开TOPCon说明](../docs/PUBLIC_TOPCON_RECOVERY.md)。权重不随 Git 分发。
