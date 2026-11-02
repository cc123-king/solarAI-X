# TOPCon 公开数据预测替代方案

原始三个 AutoGluon 模型目录遗失时，语言模型不能补造科学预测。这里新增可复现的 TOPCon 数据代理模型与 `predict_topcon_public` MCP 工具，解决硅电池领域单条、批量计算与后续绘图的一部分需求。旧 13 字段工具仍需要原权重；新模型使用完整 18 字段，绝不静默改名、换单位或填入缺失参数。钙钛矿窄/宽带隙模型继续明确不可用。

## 数据来源和下载

Chenhui Zhang，*AutoGluon-regression-model-for-TOPCon*，Mendeley Data v2，DOI [10.17632/s2vnyfgc4s.2](https://doi.org/10.17632/s2vnyfgc4s.2)，**CC BY 4.0**。关联论文 *Machine Learning Guided Device-Level Design for High-Efficiency Tunnel Oxide Passivating Contact Solar Cells*，DOI [10.1002/smll.202506958](https://doi.org/10.1002/smll.202506958)。

作者压缩包约 6.06 GB，内部模型压缩包约 5.48 GB，CSV 仅 368,188 bytes。下载器通过标准 HTTP Range 仅取得约 136 KB 压缩 CSV；服务器若不支持范围请求，立即停止，避免自动下载整个包。逐文件 CRC32 与固定 CSV SHA256 都必须匹配：

`14479b042a29137d107aa7198eea4d82ab2804177c2e583074bf674cb54287d2`

本项目只重新训练自己的数值森林，没有下载/执行作者的 pickle/AutoGluon 文件，没有复制其训练代码。源数据和训练权重不入 Git。受 CC BY 4.0 许可的数据用于另行训练 ExtraTrees，此修改不代表作者认可本项目。

```powershell
python scripts/download_topcon_public.py --output <data-directory>
python scripts/train_topcon_public.py --csv <data-directory>/dataset/regression.csv --output <new-model-directory>
$env:TOPCON_PUBLIC_MODEL_DIR = '<new-model-directory>'
python scripts/verify_topcon_public.py --model <new-model-directory> --csv <data-directory>/dataset/regression.csv --output <new-report.json>
```

训练依赖已包含在基础 requirements。训练器拒绝覆盖已有实验目录；运行前固定的规格在 `evaluation/topcon-public-v1-spec.json`，选型只看开发集，随后仅一次测试。部署制品是 `forest.npz` 与 `model-card.json`；NPZ 只含数值数组，`allow_pickle=False`，校验大小、SHA256、数组结构、无环树索引、来源和质量门槛。API 与 MCP 进程都需要设置环境变量并重启；`/api/health` 的 `public_topcon` 单独报告实际可用状态。

## 已测结果

CSV 共有 1,893 条记录，18 个输入、6 个目标，无缺失数值。按完整输入的哈希分组划分训练 1,337 条、开发 278 条、测试 278 条；重复输入不会跨分组。开发集在预先声明的 ExtraTrees/RandomForest 两个候选间选择，最终以训练加开发集拟合 ExtraTrees 128 棵树。测试记录从未进入拟合。

| 278 条公开数据测试 | MAE | R² | 训练中位数常数基线 MAE |
|---|---:|---:|---:|
| Vm | 0.006366 | 0.9582 | 0.029497 |
| Im | 0.403313 | 0.8730 | 1.146009 |
| Voc | 0.004372 | 0.9753 | 0.031825 |
| Jsc | 0.264265 | 0.9346 | 1.095637 |
| FF | 0.147580 | 0.9195 | 0.682890 |
| Eff | 0.255294 | 0.9686 | 1.580941 |

数值遵循源 CSV 单位；FF/Eff 的百分数尺度上 MAE 对应百分点。输入范围与训练邻域检查接纳测试集 275/278 条（98.92%），其余明确拒绝。表格是完整测试集的离线模型指标，部署不返回被拒样本的估计；接纳子集指标同时保存在 `reports/topcon-public-v1.json`。全部六个目标满足预先固定的门槛，不修改题目或按测试成绩调参。

通过实际 MCP STDIO 验证了注册、单条结果与直接推理一致、双条批量、整批拒绝越界、拒绝旧字段、后续图表结果引用、阻断未绑定图表以及真实表格图片返回。此软件链路没有调用 LLM，不能当作自由自然语言任务准确率。数值 NPZ 推理另与真实 scikit-learn 训练器对所有测试记录核对一致。

## 输入与使用边界

每条记录必须提供：`T_Si, T_SiO2, T_PolySi, Me, Mh, Emitter_Depth, Inner_Diffusion_Depth, Rs, Na_Top, Nd_Rear, Dit_Si_SiOx, Dit_SiOx_Poly, Dit_Emitter, Cap_Area, Nt_top, Nt_rear, Cap_Area_Defect, Taun_PolySi`。工具接受 `inputs` 列表，每次 1–32 条；扫描参数时显式构造完整记录。未知字段、缺失字段、非有限数、超出训练极值或联合邻域均整批拒绝，不返回部分预测。

输入单位使用原数据数值约定；公开 README 未逐一解释全部字段的物理单位，因此接口不猜测换算，也不能把旧工具的 1.4 nm 直接填入范围约 0.0008–0.00196 的 `T_SiO2`。使用者须依据源论文/仿真定义准备记录；未确认单位的新器件不能作为已验证输入。

这是**单一公开研究数据集上的留出测试**。源数据效率范围约 20.00–28.88%，存在明显选择范围；不能推断低效率器件。未知作者划分和相邻仿真样本的独立程度；按输入哈希隔离只解决完全相同特征泄漏。未验证新论文、新批次实验、生产器件或生产容量；未预测完整 JV 曲线、未声称恢复原 AutoGluon，不能用于窄/宽带隙钙钛矿。原公开论文成绩不作为本模型成绩。
