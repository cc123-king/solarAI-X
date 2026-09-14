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
