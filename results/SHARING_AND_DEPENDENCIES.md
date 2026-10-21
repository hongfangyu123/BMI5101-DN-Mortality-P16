> **Note (this repository):** sections 3–4 below describe the original two-folder layout used when the extensions were shared. In this repository the extensions live in `extensions/` inside the V3 project and find it automatically; see the root `README.md` for the current commands.

# 给同学的文件与原 V3 依赖说明

整理日期：2026-10-21。分析内容截至 2026-10-15，最终扩展签名为 `27464ba98305edbc`。

## 1. 先看哪份报告

- [完整中文报告](report_zh.md)：从原 V3 最新中文报告原样复制，包含原论文复现、CLIN、比较与局限、扩展实验四部分。
- [完整英文报告](report_en.md)：保留原英文报告的前三部分，并补齐最新摘要、第 17–18 节更新、第四部分第 19–25 节及附录 C；不是只翻译扩展摘要。
- [报告目录说明与校验记录](README.md)：说明配套文件和版本来源。

`reports/` 已带上报告引用的图、聚合结果表、决策和进度记录。单独复制整个 `reports/` 就能阅读报告及其配套结果，不需要原 V3、数据库或 Python。`decisions.md` 和 `progress.md` 是原中文记录；英文报告中引用的数据表列名和实验名沿用原始名称。

## 2. 为什么运行扩展还需要原 V3

扩展是接在原 V3 的 CLIN 数据流程后面的实验模块，没有重复实现原队列构建与全部公共函数。

| 依赖 | 原 V3 项目内的路径 | 用途 |
|---|---|---|
| 变量定义 | `src/variables.py` | 导入 `FEATURES` 和 `MODE_VARS`，确定 55 个静态候选及众数插补变量；扩展另加 `wbc_diff_missing` |
| LASSO 公共函数 | `src/lasso.py` | 直接导入 `_deviance`、`_path`、`lambda_max`，复用惩罚路径与二项偏差计算；balanced 的折内流程由扩展自身实现 |
| 评估公共函数 | `src/metrics.py` | 直接导入校准、DCA、Nadeau–Bengio 校正函数 |
| CLIN 队列 | `data/processed/cohort_clin.parquet` | 开发患者的住院/ICU 标识和时间，用于同次住院及 24 h 窗口校验 |
| CLIN 静态特征 | `data/processed/features_clin.parquet` | 复用已完成的静态预测变量与开发集结局 |
| 原始开发/留出划分 | `data/processed/split_clin.parquet` | 固定沿用原划分与开发集行顺序；留出集只读取 `stay_id` 检查互斥，不读取留出标签 |
| 子集数据库 | `data/interim/mimic_subset.duckdb` | 准备新增特征及预检时，查询原始测量、既往住院和诊断 |

数据库需包含 `mimiciv_derived` 下的 `chemistry`、`complete_blood_count`、`blood_differential`、`coagulation`、`vitalsign`，以及 `mimiciv_hosp.admissions` 和 `mimiciv_hosp.diagnoses_icd`。不是任意一个只有原始表的 DuckDB 都能直接替代。

**有原始 MIMIC 数据，不等于已经有扩展所需的 V3 中间文件。** 如果上表的 parquet 和概念表缺失，需要先按原 V3 的 `REPRODUCE.md` 完成相应的队列、概念表、CLIN 共病/特征和划分步骤。若这些输入已经齐全，就不必为了运行扩展重跑原 V3 的全部模型。

## 3. 目录怎么放

默认要求两个项目是同级目录，保留 V3 解压后的内层目录：

```text
任意工作目录/
├── dn_mortality_repro_V3/
│   └── dn_mortality_repro/
│       ├── src/
│       └── data/
│           ├── processed/
│           │   ├── cohort_clin.parquet
│           │   ├── features_clin.parquet
│           │   └── split_clin.parquet
│           └── interim/mimic_subset.duckdb
└── dn_mortality_repro_V3_extensions/
    ├── reports/
    ├── data/processed/development_features.parquet
    └── balanced/
```

严格版 `config.yaml` 的 `source_project` 是 `../dn_mortality_repro_V3/dn_mortality_repro`；balanced 配置是 `../../dn_mortality_repro_V3/dn_mortality_repro`。它们分别相对各自代码根目录解析，最终指向同一个 V3 项目，和终端最初所在目录无关。

如果同学的原 V3 放在别处，在同一个终端设置下面的环境变量即可，两版都会使用它，无需改配置文件。路径指向直接包含 `src/` 和 `data/` 的那一层：

```bash
export BMI5101_V3_ROOT="/实际路径/dn_mortality_repro"
```

## 4. 从同学已有的 V3 输入运行 balanced

以下命令使用 Bash/WSL，从 `dn_mortality_repro_V3_extensions` 目录运行。它们是给同学的执行说明；本次整理报告没有运行训练。

```bash
cd /实际路径/dn_mortality_repro_V3_extensions

# 只检查原 V3 输入与概念表，不训练。
bash run_extensions.sh check

# 首次运行或原 V3 输入改变后，准备新增特征；这里不会训练模型。
bash run_extensions.sh prepare

# 只检查 balanced 输入和实验列表，不训练、不写实验结果。
bash balanced/run_extensions.sh full --check-only

# 可选：先以少量折检查流程。pilot 不能作为正式结论。
bash balanced/run_extensions.sh pilot --experiments static_stacking

# 从零运行报告对应的全部 14 组；无需我这台电脑的旧缓存。
bash balanced/run_extensions.sh full --experiments static_lr,static_rf,static_xgb,static_stacking,temporal_lr,temporal_rf,temporal_xgb,burden_lr,burden_rf,burden_xgb,embedding_lr,embedding_rf,embedding_xgb,combined_stacking
```

上面的 shell 入口默认使用 `$HOME/miniconda3` 中名为 `ml` 的 Conda 环境。Miniconda 装在别处时可设置 `BMI5101_CONDA_BASE`。若同学用的是原 V3 的 `dn_repro` 或另一个已配好依赖的环境，可以激活自己的环境后直接调用 Python，不必把环境重命名为 `ml`：

```bash
conda activate dn_repro
python scripts/00_preflight.py
python scripts/01_prepare_features.py
python balanced/scripts/02_run_experiments.py --profile full --check-only

# 正式训练时在上一条命令中去掉 --check-only，并加上上面列出的 --experiments。
```

依赖包括 NumPy、pandas、SciPy、scikit-learn、XGBoost、DuckDB、joblib、PyYAML、statsmodels、nbformat，以及 parquet 读写引擎。原实验的关键版本见 [run_identity.json](extensions/balanced_27464ba98305edbc/run_identity.json)。复用 V3 源码时需注意 API 兼容性；例如当前 `src/lasso.py` 使用 `LogisticRegression(l1_ratio=...)`，不能假定任意旧版 scikit-learn 都能运行。

**balanced 共享上一级扩展项目的特征**：`data/processed/development_features.parquet`，不是 `balanced/data/processed/`。`balanced/run_extensions.sh prepare` 不负责提取特征，应使用上级入口的 `prepare`。该 parquet 来自同学本机的 V3 输入，包含静态特征、125 个轨迹候选、4 个诊断负担变量及既往 ICD 文档；之后训练折内才拟合词表、SVD、插补和筛选。

即使扩展特征已经准备好，正式训练仍会读取原 V3 的三个 parquet 来核验开发集及静态列是否改变，因此不能只保留扩展特征文件。DuckDB 则在 `prepare` 和 `check` 阶段使用，已有特征下的正式训练不直接查询它。

不加 `--experiments` 时，balanced 配置默认只有 8 组，不能据此认为会自动重跑报告中的全部 14 组。`--reuse-from ebf3f368bb8beb5c` 只适用于持有本机旧预测缓存且兼容性校验通过的情况；仅拿到聚合结果时不需要该参数。机器、软件或输入变化也可能生成不同结果签名，不应强行改成原来的目录名。

## 5. 发给同学的范围

- 只看报告：发整个 `reports/`，或从扩展项目 README 点开两份报告。
- 看代码并自行运行：发扩展代码、配置、Notebook、聚合结果和报告；同学在自己的原 V3 项目上准备上述输入。
- 本次新增的 `reports/` 不包含患者级 parquet、DuckDB、逐人预测、训练缓存、模型序列化文件或单患者 force plot。现有扩展项目的 `data/` 和 `balanced/data/` 仍是本机运行数据；它们不是阅读报告所需的文件。

English: the complete bilingual reports and their aggregate supporting assets are self-contained in `reports/`. Running the extensions additionally requires the original V3 source helpers, CLIN cohort/features/split parquet files, and the subset DuckDB during preparation/checking. Set `BMI5101_V3_ROOT` if your original V3 project is elsewhere. A fresh run does not require the author's caches; explicitly select all 14 experiments to reproduce the full extension design.
