> **This repository:** run from the repository root with `bash extensions/balanced/run_extensions.sh ...` (environment `dn_repro`, `BMI5101_V3_ROOT` set automatically). Shared features are in `extensions/data/processed/` (local only, git-ignored). Paths such as `/mnt/d/...` below refer to the original author's machine.

# V3扩展：balanced 简化计算版

2026-10-14：原严格版已完成第1次重复的前6个外层折。原代码和缓存保留，这个子目录使用独立方法与缓存，不混入旧预测。

## 直接切换

先在原来的训练终端按 **Ctrl+C**。已完成折不会丢失；正在拟合的未完成折没有保存完整预测时，需重新计算。

从原扩展目录启动简化版：

```bash
cd /mnt/d/NUS_Course/BMI5101/dn_mortality_repro_V3_extensions
bash balanced/run_extensions.sh full --experiments static_rf,static_xgb,static_stacking
```

后续仍可分项运行：

```bash
bash balanced/run_extensions.sh full --experiments temporal_lr
bash balanced/run_extensions.sh full --experiments burden_lr,embedding_lr
bash balanced/run_extensions.sh full --experiments combined_stacking
```

`full` 仍为种子42–46的5×10外层验证。若只检查接线和耗时，可用 `pilot`；pilot不是正式结论。本次仅用合成数据冒烟测试，没有启动新版本真实患者训练。

## 减少了什么

| 内容 | 原严格版 | balanced full |
|---|---|---|
| 外层评估 | 5×10折 | 5×10折保留 |
| stacking交叉拟合 | 5折 | 3折 |
| 基模型调参 | 每个stacking子折再做5折GridSearchCV | 固定参数，无GridSearchCV |
| LASSO选择 | 10折×100λ | 5折×30λ |
| 预处理缓存 | LR与树的表示分别拟合 | 公共表示缓存共享，LR单独做轻量缩放 |

固定参数方案在这版实验开始前写入配置，不依据正在运行的6折结果选“最佳参数”：

| 模型 | 固定设置 |
|---|---|
| LR | L2，C=1 |
| RF | 200棵树、max_depth=10、min_samples_leaf=10、max_features=sqrt |
| XGBoost | 100棵树、max_depth=2、learning_rate=0.1、subsample=0.8、colsample_bytree=0.6 |
| 元模型 | L2 LR，C=1 |

这些参数是一套明确的计算适中配方，并不是声明其已经最优。静态基线和所有扩展都用同一配方。RF棵树、λ网格和超参数策略变化可能改变性能与入选变量，不能把新旧两版折结果拼接起来。

## 保留的有效性约束

- 同一个V3 CLIN队列、结局、开发集划分以及外层折。
- 不读取固定留出集标签或评分。
- 缺失率筛选、插补、缩放、LASSO、TF-IDF、SVD在相应训练子集上拟合。
- stacking元模型使用折外预测；各子折重新训练Pipeline，没有提前在整个开发集筛特征。
- LASSO λ仍使用1-SE原则，内部各折也重新预处理。
- 同外层折配对比较，并报告AUROC、AUPRC、Brier、校准与DCA。

只有嵌套的基模型参数搜索被取消；外层独立评估仍然有效。整个项目已经探索过多个方案，因此新结果仍是开发集探索性证据，不是独立确认性结论。

## 数据与结果位置

- 共享只读特征：`../data/processed/development_features.parquet`；不需要重新prepare。
- 新缓存：`balanced/data/cache/`，与原严格版缓存分开。
- 新结果：`balanced/results/full/<签名>/`。
- 新Notebook：`balanced/notebooks/01–05`，完成实验后自动更新结论。
- 原严格版继续使用上级目录；其前6折和旧代码不会被删掉或覆盖。

从原目录执行 `bash run_extensions.sh full ...` 仍会跑旧严格版；简化版必须使用带 `balanced/` 的入口。

减少拟合次数不等于保证相同AUC，也不能仅凭合成测试承诺真实耗时。先观察第一个真实外层折，再估算整轮时间。


## RF / XGBoost 新特征消融（2026-10-15）

在现有入口和 Pipeline 上增加实验名，不改 config.yaml 的参数与默认实验列表。
从 `dn_mortality_repro_V3_extensions` 目录运行：

```bash
# 一次完成四个请求的消融；自动补齐同模型静态/诊断负担对照。
bash balanced/run_extensions.sh full --experiments temporal_rf,temporal_xgb,embedding_rf,embedding_xgb --reuse-from ebf3f368bb8beb5c
```

也可以分两次执行相同命令，分别把 `--experiments` 设为
`temporal_rf,temporal_xgb` 和 `embedding_rf,embedding_xgb`。
在命令末尾加 `--check-only` 只校验，不训练、不写入结果。
脚本仍通过 WSL 的 `conda run -n ml` 执行。

| 实验 | 特征 | 主对照 |
|---|---|---|
| temporal_rf / temporal_xgb | 静态 + 0–24h 时间特征 | 同模型 static |
| burden_rf / burden_xgb | 静态 + 4 项既往诊断负担 | 同模型 static |
| embedding_rf / embedding_xgb | 静态 + 诊断负担 + ICD TF-IDF/SVD | 同模型 burden |

诊断负担组会自动补齐，用来区分诊断数量与 embedding 的贡献；embedding 还另报相对静态模型的总增益。
每个模型的参数、种子42–46、5×10折、20%变量缺失率规则、插补与 LASSO 1-SE 均保持不变。
TF-IDF（min_df=10）和 SVD（最多32维）在每个训练折内拟合。验证数据只 transform。
树模型仍使用与 LR 一致的 LASSO 特征筛选，以便比较；这不等于搜索所有树模型专用筛选方案。

修改代码后会产生新的结果签名。`--reuse-from` 只接受修改前已审查的版本：
核对完整数据、配置、软件版本、训练函数和每折索引后，将原来已完成的预测复制到新缓存。
原结果目录不会覆盖；复用来源会记录在新结果的 `cache_reuse.json`。
相同输入的预处理 joblib 缓存也可以复用；旧 stacking 预测只参与已有结果汇总，不会启动 stacking 重训。
如果参数、训练算法、数据或环境不匹配，程序报错，不自动放宽校验。

运行结束后，结果写到新的 `results/full/<签名>/`，并自动更新 Notebook 03、04、06 的结果与结论。
Notebook 06 保存方法解释和本次运行开关；默认不启动训练。
注意：更新后的 Holm p 值以当前比较家族为准，旧 Notebook 手工记录是当时的历史结果。
