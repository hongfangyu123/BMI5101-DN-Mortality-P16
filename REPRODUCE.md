# 复现指南

本指南说明如何在另一台电脑上从原始数据重新运行整个项目，得到与仓库中相同的聚合结果（`results/`）。所有命令都在项目根目录（解压后的 `dn_mortality_repro/`）下的终端中运行。

## 1. 前提

- **PhysioNet 认证**：需要你本人的 PhysioNet 账号、完成 CITI 培训，并签署 MIMIC-IV 的数据使用协议（DUA）。
- **下载 MIMIC-IV v3.1**（https://physionet.org/content/mimiciv/3.1/），例如：
  ```bash
  wget -r -N -c -np --user <你的 PhysioNet 用户名> --ask-password https://physionet.org/files/mimiciv/3.1/
  ```
  下载后的目录（下文称 `MIMIC_ROOT`）应直接包含 `hosp/` 和 `icu/` 两个子目录。本项目读取其中 10 个表：`hosp/` 的 patients、admissions、diagnoses_icd、services、labevents、d_labitems，`icu/` 的 icustays、chartevents、outputevents、d_items（`.csv.gz`，无需解压）。
- **数据不得与他人共享**：原始数据和由它派生的任何患者级文件（`data/` 下的 DuckDB 库、parquet、模型文件、预测值，以及单患者 SHAP force plot `fig6c–e_*`）都只能留在你自己的电脑上。本仓库和压缩包里只有代码和聚合结果。
- **硬件**：与原作者同型号的 Mac（Apple silicon，8 GB 内存）。8 GB 内存足够（DuckDB 内存上限设为 4 GB），运行时建议关闭其他占内存的程序；磁盘至少留 10 GB 空闲。

## 2. 安装

```bash
# 1) conda 环境（名称 dn_repro，Python 3.11；environment.yml 锁定核心包版本，跨平台可用）
conda env create -f environment.yml
#    与原作者 Mac 完全相同的全部依赖：conda env create -f environment-macos-full.yml
#    Linux 显式锁文件：conda create -n dn_repro --file environment-linux-64.lock.txt

# 2) mimic-code：克隆到 external/mimic-code，并切换到固定的 commit（见 external/MIMIC_CODE_COMMIT.txt）
git clone https://github.com/MIT-LCP/mimic-code.git external/mimic-code
git -C external/mimic-code checkout 303d26c623dcc9c49cc0f204468d4acc2f063797

# 3) 指向你的 MIMIC-IV v3.1 目录（包含 hosp/ 与 icu/ 的那一层）
export MIMIC_ROOT=/path/to/mimic-iv-3.1
#    想长期生效可以写入 ~/.zshrc：
#    echo 'export MIMIC_ROOT=/path/to/mimic-iv-3.1' >> ~/.zshrc
```

- `external/mimic-code/` 在 `.gitignore` 中，不在仓库和压缩包里，必须按上面的命令自己克隆；只记录 commit 号。
- 没有设置 `MIMIC_ROOT` 时，程序使用 `config/config.yaml` 中的 `paths.mimic_root`（默认 `~/Desktop/mimic-iv-3.1`）。目录不存在时程序会报错并说明如何设置。
- 不要修改 `src/` 和 `config/config.yaml` 中的其他内容：阶段 8 的冻结模型（`results/clinical/frozen_model_clin.json`）记录了 `src/models.py`、`src/lasso.py`、`src/preprocess.py`、`src/variables.py`、`src/metrics.py` 的哈希，改动后 `04h`/`04j` 会拒绝运行。

## 3. 运行

### 3.1 环境自检

```bash
conda run -n dn_repro python scripts/00_check_env.py
```

逐项输出 OK / FAIL：Python 与关键包版本是否与 `environment.yml` 一致；`MIMIC_ROOT` 是否存在、10 个原始表是否齐全（只检查文件是否存在，不读取内容）；`external/mimic-code` 的 commit 是否正确；内存与磁盘空间。"available memory now"只是提示（WARN），不会导致失败。`run_all.sh` 和 `run_stage8.sh` 开头会自动运行这一步，有 FAIL 就停止。

### 3.2 阶段 0–7：`run_all.sh`

| 脚本 | 内容 | 大致耗时（8 GB Mac 上的经验值） |
|---|---|---|
| `01_cohort.py` | 队列构建 + 子集 DuckDB 库 + mimic-code 概念表（扫描 chartevents/labevents，最耗内存的一步） | 5–20 分钟 |
| `01b_outcome_check.py` | 1 年死亡结局核查 | < 1 分钟 |
| `02_features.py` | 55 个特征 | 1–2 分钟 |
| `03_preprocess.py` | 划分、插补、Table 1 | < 1 分钟 |
| `04_model.py` | LASSO、9 个模型、调参、DCA、SHAP | 20–30 分钟 |
| `04b`、`04c` | 校准、阈值、DeLong、LR 解释 | 几分钟 |
| `04d_nested_cv.py` | "仅 LASSO 嵌套"的 CV | 约 10 分钟 |
| `04e_sensitivity_pre2020.py` | 排除 2020–2022 年的敏感性分析 | 约 10 分钟 |
| `05_report.py` | 生成报告 | 秒级 |

合计约 1–1.5 小时。

### 3.3 阶段 8：`run_stage8.sh`（必须在 `run_all.sh` 完成之后）

按顺序运行：`01c`（CLIN 队列）→ `02b`（口径 C 共病时间）→ `02b --track clinical`（CLIN 共病）→ `02_features --track clinical`（CLIN 特征）→ `03c`（CLIN Table 1）→ `04f` 五次完整嵌套 CV（口径 C 主分析约 1.5 h、口径 C comorb_time 约 1.2 h、CLIN 主分析约 2.2 h、CLIN prior_only 约 1.8 h、CLIN no_wbc_diff 约 1.5 h）→ `04g`（口径 C 留出集）→ `04h`（拟合冻结的 CLIN 模型）→ `04j`（CLIN 图表与核对）→ `05`（报告）。合计约 **9–10 小时**（原作者在 8 GB 内存的 Mac 上运行）。每步日志写在 `results/logs/<步骤名>.log`。

`04f` 每完成一折就把结果缓存到 `data/interim/nested_cache/`；中途中断后重新运行 `run_stage8.sh`，已完成的折会直接复用。

### 3.4 长任务在终端后台运行

用 `nohup` 防止关闭终端后中断，用 `caffeinate -i` 防止 Mac 休眠：

```bash
cd /path/to/dn_mortality_repro
export MIMIC_ROOT=/path/to/mimic-iv-3.1

# 阶段 0–7 和阶段 8 依次运行（约 11 小时）
nohup caffeinate -i sh -c "bash run_all.sh && bash run_stage8.sh" > results/logs/run_everything.out 2>&1 &

# 或者分开运行：
nohup caffeinate -i bash run_all.sh   > results/logs/run_all.out    2>&1 &
nohup caffeinate -i bash run_stage8.sh > results/logs/run_stage8.out 2>&1 &   # 等上一条结束后再运行
```

查看进度：

```bash
tail -f results/logs/run_everything.out          # 当前在哪一步（Ctrl-C 退出查看，不影响运行）
ls -lt results/logs | head                       # 最近更新的日志
tail -n 5 results/logs/04f_nested_full_clin.log  # 某一步的最新进度（04f 每折一行，共 5 × 10 折）
pgrep -fl "scripts/"                             # 正在运行的脚本；没有输出说明已经结束
```

## 4. 关于 04i（CLIN 留出集的确认性评估）

- CLIN 的留出集（512 人）按预设**只评估一次**（decisions.md D8.15）。`04i_holdout_clin.py` 已在原作者电脑上运行过一次，**复现时不运行**，`run_stage8.sh` 也不包含它。仓库中的 `results/clinical/holdout_clin.csv`、`holdout_dca_clin.csv`、`coefs_clin.csv`、`shap_clin.csv` 就是正式结果。
- `04i` 自带防线：输出文件已存在时拒绝运行，不会覆盖仓库中的结果。
- **复现核对**：`04j_plots_clin.py` 用你重新拟合的冻结模型（`04h`）对留出集重新预测，把 `holdout_clin.csv` 的全部 33 列和 `holdout_dca_clin.csv` 的 112 行逐项重算并与仓库中的值比较（差值须 < 1e-9），不一致时报错退出。`04j` 运行成功即说明你的冻结模型和留出集评估与正式结果一致。
- `04h` 会重写 `results/clinical/final_model_clin.json`。其中的模型文件哈希 `bundle_sha256` 取决于序列化的字节，在不同电脑上可能与仓库中的值不同；其余字段（入选变量、λ、C、EPV）应完全相同。

## 5. 如何核对复现成功

全部运行完后：

```bash
git status --short results/                        # 应只有日志有改动（若用 git 管理解压后的目录）
git diff --stat -- results/ ':!results/logs'       # 聚合结果：应无差异
```

- 如果你拿到的是压缩包而不是 git 仓库，先在解压后的目录中执行 `git init && git add -A && git commit -qm original`，再运行流程，最后用上面的命令比较。
- **预期**：同型号 Mac、同一 conda 环境下，`results/tables/`、`results/replication/`、`results/clinical/` 中的 CSV/JSON 和 `results/report.md`、`report_en.md` 应**逐字节一致**（特征计算已改为与行顺序和线程数无关，D8.13）。例外：`final_model_clin.json` 的 `bundle_sha256` 可能不同（见第 4 节）；`results/logs/` 含运行时间和本机路径，必然不同；PNG 图片在不同 matplotlib 构建下可能有元数据差异。
- 其他型号或操作系统上，浮点运算顺序不同，可能出现 1e-10 量级的差异（例如 Table 1 中个别 p 值的末位）；这类差异不影响任何显示的数字和结论。若出现更大的差异，请检查 `00_check_env.py` 的结果、mimic-code 的 commit 以及 MIMIC-IV 的版本（必须是 v3.1）。

## 6. 数据合规

- `data/`（子集 DuckDB 库、parquet、模型文件、留出集预测、嵌套 CV 缓存）和单患者 force plot（`results/figures/fig6c_*`、`fig6d_*`、`fig6e_*`）是患者级数据，**只能留在本机**：不得上传、发送（含即时通讯软件和网盘）或提交到 git。它们已在 `.gitignore` 中。
- 不要用 `head()`、`print(df)`、`cat`/`zcat` 等方式查看原始数据或 `data/` 下的文件；检查数据时只看行数、缺失率、汇总统计等聚合信息。详见 `DATA_ACCESS.md`。
- 提交或公开前运行 `python tools/check_public_release.py`：它只检查 git 跟踪和暂存的文件，发现 `data/` 路径、`.parquet`/`.duckdb`/`.pkl` 等后缀、`fig6c–e`、含患者级字段的 CSV/JSON 表头或密钥时报错。

## 7. 扩展实验（stacking、首日时间轨迹、既往诊断 embedding；探索性）

扩展代码在 `extensions/`，最终采用的是 `extensions/balanced/`（固定参数，5 × 10 外层折，只用 CLIN 开发集 2,045 人，不读取留出集标签）。结果快照在 `results/extensions/balanced_27464ba98305edbc/`，报告见 `results/report_en.md` / `report.md` 第四部分（§19–25、附录 C）。严格版（每个 stacking 子折再调参）只完成 6 折，未完成，其实验代码不在仓库中，结果也不进报告。

- **依赖**：扩展读取本项目的 `src/`（变量、LASSO、指标函数）和阶段 8 产生的 `data/processed/{cohort_clin,features_clin,split_clin}.parquet`；准备特征时还查询 `data/interim/mimic_subset.duckdb`。所以必须先完成第 3 节（至少到 CLIN 特征和划分）。
- **找到 V3 项目**：`extensions/run_extensions.sh` 和 `extensions/balanced/run_extensions.sh` 会自动把环境变量 `BMI5101_V3_ROOT` 设为仓库根目录，并使用 `dn_repro` 环境（可用 `BMI5101_CONDA_ENV`、`BMI5101_CONDA_BASE` 覆盖）。直接用 `python` 运行扩展脚本时，请先 `export BMI5101_V3_ROOT=$(pwd)`（仓库根目录）。`config.yaml` 中的 `source_project` 保留原值，不要修改：它属于结果签名的一部分，改动会使签名与已发布的 `27464ba98305edbc` 不一致。

```bash
bash extensions/run_extensions.sh check            # 检查 V3 输入和概念表，不训练
bash extensions/run_extensions.sh prepare          # 准备扩展特征 -> extensions/data/processed/（患者级，git 忽略）
bash extensions/balanced/run_extensions.sh full --check-only
bash extensions/balanced/run_extensions.sh full --experiments static_lr,static_rf,static_xgb,static_stacking,temporal_lr,temporal_rf,temporal_xgb,burden_lr,burden_rf,burden_xgb,embedding_lr,embedding_rf,embedding_xgb,combined_stacking
```

- 不加 `--experiments` 时只运行配置中的 8 组；报告中的完整设计是上面的 14 组，需要数小时。新环境不要加 `--reuse-from`（逐折缓存不在仓库中）。
- 新运行写入 `extensions/balanced/results/full/<签名>/`。机器、操作系统或软件版本不同可能得到不同签名和末位数字；报告引用的是经过哈希校验的快照 `results/extensions/balanced_27464ba98305edbc/`，不会被新运行自动替换。
- 只用合成数据的接线检查：`BMI5101_V3_ROOT=$(pwd) conda run -n dn_repro python extensions/balanced/tests/test_tree_ablations.py`。

