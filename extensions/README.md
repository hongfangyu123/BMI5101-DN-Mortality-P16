> **This repository:** the extensions live in `extensions/` inside the V3 project; the run scripts set `BMI5101_V3_ROOT` to the repository root and use the `dn_repro` environment. Only feature preparation (`check`, `prepare`) is kept here; the incomplete strict experiment branch is not included. The experiments reported in the paper and in `results/report_en.md` Part IV are in [`balanced/`](balanced/README.md). See the root `README.md` for current commands.

# V3：stacking、时间轨迹与合并症 embedding

该目录是 V3 的独立扩展，读取 `../dn_mortality_repro_V3/dn_mortality_repro` 的开发集与子集 DuckDB。原始 V3 文件、冻结模型、留出结果和旧缓存不改动。

## 已实现与尚未执行

- 三类扩展、共同静态基线、联合消融、逐折缓存、AUROC/AUPRC/Brier/校准/DCA及配对比较。
- 中文 Notebook 01–05，并在每次真实开发集实验后自动写入结果和实验结论。
- 只执行轻量预检与120行合成数据的冒烟测试；没有运行真实患者特征提取或正式交叉验证。当前没有任何提升结论。

## 在 WSL 运行

```bash
conda activate ml
cd /mnt/d/NUS_Course/BMI5101/dn_mortality_repro_V3_extensions

# 检查输入和结构，不训练。
bash run_extensions.sh check

# 第一次运行：只提取开发集轨迹和既往诊断，可能需要几分钟，不做CV。
bash run_extensions.sh prepare

# 可选：先用1×3外层折检查真实数据流程和耗时。
bash run_extensions.sh pilot --experiments static_stacking

# 正式实验可以一个一个运行，已完成折自动复用缓存。
bash run_extensions.sh full --experiments static_rf,static_xgb,static_stacking
bash run_extensions.sh full --experiments temporal_lr
bash run_extensions.sh full --experiments burden_lr,embedding_lr
bash run_extensions.sh full --experiments combined_stacking
```

不加 `--experiments` 会运行配置内的全部八组实验。为便于排查，建议按上述顺序逐项运行。不同profile使用独立缓存；pilot结果不会混入full。中断后使用相同命令继续；改代码、数据或配置会产生新签名，避免错误复用旧缓存。同一目录不要同时启动两个训练进程。

`config.yaml` 的 `jobs: 2` 控制内部调参并行；内层模型线程固定为1，stacking内部顺序执行，避免递归并行占满内存。可加 `--jobs 1`。正式严格 stacking 仍可能很慢，不能按旧固定参数版本的耗时推算。

Notebook 内核选择 **Python (ml - WSL)**；运行开关默认False。最方便的方式是在终端运行，再打开对应 Notebook 查看自动写入的结论。

## 方法与对照

- 复用V3 CLIN样本、1年死亡标签、开发/留出划分；外层5×10折和种子42–46，顺序与V3一致。
- 不读取固定留出集标签，也不评估其性能。V3 的 holdout 已看过，不能用来反复选扩展。
- 缺失筛选继承V3的20%变量规则及8个静态WBC differential豁免；患者不因变量缺失被删除。轨迹变量不豁免。
- 静态特征维持V3 −6至24h口径；轨迹仅0至24h，同次hadm_id/stay_id，避免跨住院拼接。缺hadm_id的化验不用于轨迹。
- LASSO复用V3的glmnet-style路径与λ.1se原则；本实现连LASSO内部折也重做预处理。数值无变化/全缺失变量自动剔除；无入选变量用常数列，模型退化为基率预测。
- 模型调参为固定小网格：LR4组、RF8组、XGBoost8组。比V3穷举小，但每个内层调参折重新训练完整Pipeline。新静态基线与新扩展配对比较，不把V3旧表当作可直接相减的结果。
- stacking基模型本身是GridSearchCV(Pipeline)，其OOF子折内重新调参和表示学习；元模型用OOF概率，外层验证患者完全独立。
- 既往文档仅来自 `previous dischtime < index admittime`。TF-IDF词表、IDF、SVD及缩放均训练折拟合。embedding最多32维，不是临床文本或因果合并症表示。
- 三组ICD消融用于区分既往诊断数量与embedding增益；联合模型只作探索性比较。

## 输出

`results/full/<签名>/` 或 `results/pilot/<签名>/`：

- `summary.csv`：折均值和每次重复合并OOF均值。
- `paired_comparisons.csv`：扩展−参考的Nadeau–Bengio校正95% CI、p值；按当前可用对比每个指标做Holm校正。
- `fold_metrics.csv`、`pooled_oof_metrics.csv`、`pooled_dca.csv`。
- `fit_metadata.json`：各折入选变量、embedding维数及参数。
- `run_identity.json`：数据、代码、配置身份及运行模式。
- `report.md`：结果与解释边界；同步写回对应Notebook。

`data/` 只保存在本机且被Git忽略：开发集特征、ICD文档、折外概率与Pipeline缓存。聚合结果没有患者ID。不要公开患者级特征/文档/模型缓存。

AUROC/AUPRC差值>0更好，Brier差值<0更好。pilot不能作正式结论，smoke完全是合成数据。正式实验也属于开发集探索性证据，不应承诺一定超过论文0.780或已获得独立验证。

可用 `bash run_extensions.sh smoke` 再检查代码连接；其结果只写入 `results/smoke/`，不会写成真实Notebook结论。


## 2026-10-14：计算适中版本

原严格版完整保留。简化版见 [balanced/README.md](balanced/README.md)。

先在原训练终端按Ctrl+C，再执行：

```bash
bash balanced/run_extensions.sh full --experiments static_rf,static_xgb,static_stacking
```

新方法保留5×10外层评估、训练折预处理与stacking折外预测；取消基模型重复网格搜索，LASSO改5折/30λ、stacking改3折。两版使用独立缓存，前6折旧预测没有拼接进新结果。无需重新prepare。
