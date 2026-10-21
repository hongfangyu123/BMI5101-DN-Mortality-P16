# 进度记录（progress.md）

## 阶段 0：项目设置（2026-09-30）
- 目录、conda 环境 `dn_repro`（Python 3.11.16；duckdb 1.5.6、pandas 3.0.6、scikit-learn 1.9.1、xgboost 3.4.2、shap 0.51.0）、`environment.yml`、`config/config.yaml`、`.gitignore`、`CLAUDE.md`、`.claude/settings.json`（数据访问拒绝规则）。
- mimic-code 克隆到 `external/mimic-code`，commit 303d26c6（含官方 `concepts_duckdb`）。
- 论文 PDF 与两个旧 notebook（已清除输出）复制到 `docs/`。

## 阶段 1：队列构建
- 子集库：3,046 名候选患者（口径 A–D 的并集，已过 LOS/年龄），chartevents 2,962 万行，24 个 mimic-code 概念表全部成功生成（无需简化任何依赖链）。
- DN 编码：8,992 名患者 / 24,834 次住院（ICD-9 与 ICD-10 版本匹配无冲突）；其中 3,189 名有含 DN 编码且有 ICU 的住院。
- 各口径最终人数（论文 1,313）：A 837、B 1,228、**C 1,960（主分析）**、D 1,685。
- 口径 C 流程：3,189 → LOS>48h 3,046 → 年龄>18 3,046 → CBC 完整 1,960。各步保留比例与论文几乎一致（LOS 95.5% vs 95.4%；CBC 64.3% vs 64.2%）；年龄步骤两者都未排除任何人。CBC 缺失几乎完全来自白细胞分类（35.6%），CBC 本体仅缺 1%。
- 结局（口径 C，n=1,960）：1 年死亡（自 ICU 入科）494 例（25.2%）；自入院起算相同（25.2%）；院内死亡 7.1%；"dod 非空"口径 42.8%。
- 与论文 38.2% 的差距：论文很可能把任何 `dod` 非空都算作死亡（该口径在我们队列为 42.8%，且 v3.1 纳入 2020–2022 年患者、`dod` 随访随末次出院延长）。有 `dod` 的出院后死亡中 49% 发生在 ICU 后 >365 天，不属于 1 年结局。旧代码 24.9% 与本结果（25.2%）一致。
- 队列中 247 人（12.6%）来自 2020–2022 年组（v3.x 新增），这是人数高于论文的原因之一；另一个原因是 DN 患者数本身（8,992 vs 4,137）。
- 决策：D1.1–D1.7（见 decisions.md）。

## 阶段 2：特征提取
- 特征表：1,960 行 × 55 个预测变量（2 人口学 + 6 合并症 + 31 化验 + 4 生命体征 + 6 评分 + 6 组合指标）+ id/结局，保存为 `data/processed/features.parquet`。
- 缺失率：全部 ≤ 7.04%（Calcium、PTT 7.04%；PT/INR 6.79%；体温 2.19%；其余 < 1%），没有变量超过 20% 的剔除阈值。
- 与论文 Table 1 总体列相比（`feature_vs_paper.csv`）：人口学、心梗/心衰/慢性肺病/肾病/肥胖、全部电解质/肾功能/凝血/CBC/生命体征、SAPS-II、OASIS 与论文几乎相同。APS-III（45 vs 47）、LODS（4 vs 5）、GCS（15 vs 14）、CCI（7.11 vs 7.80）略低。
- 单位混杂得到证实：51146 是嗜碱性粒细胞 %；论文的白细胞分类存在跨患者的尺度混杂（D2.11）。按正确口径计算后，NLR/LMR/SII/PWR 与论文一致，PLR/PNR 不一致。
- 高血压：完整定义 84.6%，而无并发症高血压 35.4% ≈ 论文 37.3%（D2.9）。
- 决策：D2.1–D2.11。

## 阶段 3：预处理与 Table 1
- 划分：训练集 1,568 人（死亡 395，25.19%）/ 验证集 392 人（死亡 99，25.26%）。
- 没有变量缺失 > 20%，55 个变量全部保留；生成 faithful / noleak 两套插补矩阵（原始尺度 + 标准化）。
- Table 1（`results/tables/table1.csv`，观测数据）：没有连续变量通过 Shapiro-Wilk，全部报中位数 (IQR)。39 个变量组间差异 p < 0.05（论文 42 个）。与论文方向一致的有：年龄、心衰、肾病、肥胖（存活组更高）、阴离子间隙、肌酐/BUN、Hb/RBC、RDW、氯（死亡组更低）、PT/PTT/INR、MBP、呼吸频率、全部评分、NLR/LMR/SII。
- 与论文不一致：慢性肺病、高血压、glucose、WBC、PWR 在我们的数据中不显著；bicarbonate_min、potassium_max 显著而论文不显著。高血压不显著是因为采用了完整定义（D2.9）。
- 决策：D3.1–D3.6。

## 阶段 4：建模与评估（含校准）
- LASSO（二项偏差，10 折 CV）：忠实版 λ.min = 0.01065（20 个变量）、λ.1se = 0.02700（12 个）；无泄露版 λ.min = 0.01052、λ.1se = 0.02667，选出的 12 个变量相同。论文为 0.01056 / 0.02687（21 / 12 个）。λ.1se 的变量：age, aniongap_min, rdw_mean, lymphocytes_mean, chloride_max, pt_min, sapsii, apsiii, oasis, cci, nlr, sii，与论文 12 个重合 7 个（age, aniongap_min, rdw_mean, chloride_max, pt_min, apsiii, cci）；论文的 obesity、neutrophils_mean、ptt_min、lmr 出现在我们的 λ.min 集合中（λ.min 下重合 11/12，只缺 pwr）。高血压敏感性分析的选择不变。
- 验证集 AUC（无泄露 + 调参）：LR 0.803、SVM 0.802、RF 0.793、XGBoost 0.782（95% CI 宽约 ±0.05，相互重叠）。论文参数：LR 0.803、XGB 0.787、RF（字面参数）0.780（与论文 0.780 相同）、SVM 0.764、RF（R 默认）0.756。
- 过拟合：RF（R 默认，9 棵完全生长的树）训练集 AUC 0.999 / 验证集 0.756（差 0.243），与论文 RF 0.999 / 0.780 的模式完全一致；RF（字面参数）差 0.106；调参后所有模型的差 ≤ 0.057。
- 校准（验证集）：LR Brier 0.145（slope 1.21）、SVM 0.146（0.99）、XGB 0.148（0.88）、RF 调参版 0.153（slope 1.76，概率被压向患病率，阈值 0.5 时 recall 仅 0.13）；RF（R 默认）slope 0.17，严重过度自信。
- 主分析模型（按 CV AUC 选）：调参 RF（与 XGBoost 并列）。DCA 在阈值 0.07–0.59 时净获益高于 treat-all/none。SHAP 前 5：rdw_mean, cci, sapsii, apsiii, chloride_max（论文：rdw_mean, neutrophils_mean, cci, chloride_max, pt_min）；论文配置（忠实版 + 论文参数 + RF）的前 5：rdw_mean, sapsii, cci, age, apsiii。
- 忠实版与无泄露版的结果几乎相同（均值插补的泄露在缺失率 ≤ 7% 时影响可以忽略）。
- 决策：D4.1–D4.12。

## 阶段 5：复现报告
- `results/report.md` 由 `scripts/05_report.py` 从聚合表自动生成，内容包括：队列流程与四种口径、结局与死亡率差异、55 个特征与论文 Table 1 的比较、单位混杂分析、Table 1、LASSO、AUC 表与相对排序、阈值指标、校准、过拟合检查、DCA/CIC、SHAP、差异汇总、不明确之处与处理口径、局限性与后续工作。
- 合规检查：`results/tables` 中没有任何 ID 列；单患者 SHAP force plot（fig6c–e）加入 .gitignore，只保留在本机。
- 运行顺序：01 → 02 → 03 → 04 → 04b → 05（每个脚本可单独重跑）。

## 阶段 6：阈值、模型比较与推荐模型（2026-09-30，`04c_threshold_compare.py`）
- 阈值：每个模型在训练集 10 折 OOF 预测上取 Youden 最佳阈值（调参模型 0.23–0.26，接近患病率），在验证集上报告 accuracy/precision/recall/F1/specificity。推荐 LR 阈值 0.228：recall 0.77、specificity 0.67、precision 0.44、F1 0.56（0.5 时 recall 0.37）。阈值 0.5 保留作对照，报告说明了它不合适的原因（D4.13）。
- 模型比较：验证集配对 DeLong + Holm 校正，所有模型对均不显著。调参 RF vs LR：ΔAUC −0.010（−0.025 至 +0.006），p = 0.21（D4.14）。
- 推荐模型改为调参 LR（AUC 0.803，调参模型中 Brier 最低，可解释），RF 为次要结果（D4.15，属事后规则，已在报告局限性中说明）。LR 解释：无惩罚 LR 的 OR（95% CI）、VIF、SHAP。显著因素：RDW、CCI、年龄、APS-III、OASIS（危险），chloride_max、淋巴细胞（保护）；LR 与 RF 的 SHAP 前 5 有 4 个相同（D4.16）。
- 高血压：报告新增 3.3 节。仅无并发症定义（35.4%，≈ 论文）会把 964 名只有并发症编码的患者记为"无"，产生虚假保护效应（OR 0.49）；完整定义 OR 1.07（p = 0.65）。主分析维持完整定义，LASSO 选择不受影响。
- DCA 有用区间改为 D4.17 口径；新增各调参模型 DCA 对比图（Fig. 5B）和 LR OR 图（Fig. 8）。
- 代码：模型定义移到 `src/models.py`（04 与 04c 共用；04c 重建的模型与阶段 4 保存的预测一致，误差 < 1e-6）；`src/metrics.py` 新增 specificity、Youden 阈值、配对 DeLong、DCA 有用区间。

## 阶段 7：审阅意见修改（2026-09-30）
- **任务 1（核查 + 嵌套 CV，`04d_nested_cv.py`，D4.18）**：核查结论是阶段 4 的 LASSO 在整个训练集上一次完成，CV 在固定的 12 个变量上进行，因此 CV AUC 含有变量选择偏倚。嵌套版本在每折内部重做插补/标准化和 LASSO（λ.1se），超参数固定。偏倚为 +0.008 至 +0.015（RF R 默认 +0.030）：推荐 LR 0.793 → 0.781，调参 RF 0.800 → 0.785。各折选出 9–15 个变量，阶段 4 的 12 个中有 6 个 10/10 入选。偏倚很小，主要结论不变；非嵌套复算与阶段 4 一致（差 < 1e-3）。
- **任务 2（推荐模型改为无惩罚 LR，`04c`，D4.15/D4.16 修订）**：
  - 验证集：AUC 0.803（0.752–0.853），Brier 0.145（0.1445，此前因两次四舍五入写作 0.144），校准斜率 0.96（L2 LR 为 1.21）。
  - 训练集 OOF 校准斜率：0.93 vs 1.20，说明差异不只出现在验证集上。
  - OOF Youden 阈值 0.259，验证集 recall/specificity/precision/F1 = 0.74/0.74/0.49/0.59（L2 LR 为 0.228，0.77/0.67/0.44/0.56）。
  - DCA 有用区间 0.01–0.71，SHAP 前 5：rdw_mean, cci, age, apsiii, oasis。
  - 与调参 RF 的 DeLong：+0.010（−0.008 至 +0.028），p = 0.26。
  - 新增 `lr_variants.csv`，DeLong 表新增 `recommended` 组。
- **任务 3**：报告第 0 节按"预设规则选出 RF → DeLong 无差异 → 事后改为 L2 LR → 再改为无惩罚 LR"写明全过程，并注明需要外部验证。
- **任务 4**：对论文方法的推断改为谨慎表述，包括高血压定义、RF 完全生长树（补"无法排除其他原因"）、结局口径、单位混杂、插补后统计、阳性类、钙单位、中性粒细胞伪信号等；decisions.md 的 D2.9、D4.4 同步修改。事实性比较（NLR/SII 一致、PLR/PNR 偏离）保留。
- **任务 5（`04e_sensitivity_pre2020.py`，D4.19）**：
  - 队列：排除 anchor_year_group 2020–2022 的 247 人，剩余 1,713 人（死亡 25.0%），重新 80/20 划分（seed 42）。
  - LASSO λ.1se 0.0264，选 11 个变量（新增 ptt_min、plr；缺 aniongap_min、oasis、nlr）。
  - 验证集 AUC：推荐 LR 0.799（0.747–0.852），调参 RF 0.802（0.750–0.855），DeLong p = 0.77。
  - 校准斜率：LR 0.93，RF 1.65。SHAP 前 2 仍为 rdw_mean、cci。
  - 按预设标准判定结论稳健。报告新增第 9 节，原第 9–11 节顺延为 10–12，删去"后续工作"原第 6 条。
- 代码：新增 `src/preprocess.py`（03/04d/04e 共用）；`cal_intercept_slope` 移入 `src/metrics.py`（04b/04c/04e 共用）。重构后重跑 03、04b，所有结果表逐字节不变。

## 阶段 7b：收尾
- **主图改为以推荐的无惩罚 LR 为主**（D5.2）：
  - 新增 `fig3_roc_noleak_main.png` 和 `fig7_calibration_noleak_main.png`，包含无惩罚 LR 与四个调参模型。
  - `04_model.py` 中 primary 改为无惩罚 LR（`*_primary_noleak_recommended.*`：fig5 DCA/CIC、fig6a–e SHAP），调参 RF 改名为 `*_secondary_noleak_tuned.*`；旧的 `*_primary_noleak_tuned.*` 已删除。
  - `04c` 不再重复画 LR 的 SHAP 图。
  - 报告第 7、8 节的文字和图片路径已同步，新增推荐 LR 的 CIC 描述。
- **表格分组**：
  - 无惩罚 LR 在所有表格中单独列为 `recommended` 组，调参 LR 记为 `LR (L2)`；6.1 节的 AUC 表改为长表（含嵌套 CV 列），并说明 recommended 与论文参数组的 LR 是同一模型。
  - DeLong 比较分为 paper（不含 LR，6 对）、tuned（6 对）、recommended（无惩罚 LR vs 其余 8 个模型）三组，各自做 Holm 校正；recommended 组最小校正 p = 0.19，结论不变。
- **措辞**：
  - 第 10 节"RF 训练集 AUC"一行改为"与论文 RF 为完全生长树的解释一致"。
  - 第 0、9 节的"按预先写明的标准"改为"按在敏感性分析运行前写明的标准（在本次修订中制定，并非项目开始时预设）"；D4.19 同步。
- **精度**：`calibration.csv`、`lr_variants.csv` 改存 6 位小数，避免两次四舍五入造成报告与图例不一致（无惩罚 LR 的 Brier 0.1445，统一显示为 0.145）。
- **复核**：重跑 04 → 04b → 04c → 05。模型性能、阈值、LASSO、超参数表逐字节不变；calibration 只有精度变化；delong_pairwise 因重新分组而变化。

## 阶段 8：拆分复现线与临床线（2026-10-01 起）
- **8a 分析计划**：写入 decisions.md 第 8 节（D8.1–D8.9）。用户确认时提出 5 处修改，已写入：
  - 白细胞分类只设 1 个指示变量，比值变量直接插补；
  - 等价规则改为"差值 ≤ 0.01，或（CI 跨 0 且差值 ≤ 0.02）"；
  - 586/N19 先计数，≥ 1% 时做加回的敏感性分析；
  - 降为 3 次重复时，配对比较只用共同划分；
  - CLIN 流程表逐步列出排除人数。
- **8b 一年死亡结局核查**（`scripts/01b_outcome_check.py`，D8.10）：
  - 新定义为"死亡日期 ∈ [date(intime), date(intime) + 365 天]"，院内死亡用 deathtime，院外死亡用 dod。
  - 口径 A/B/C/D 中，新旧定义不一致的人数均为 0（死亡数 189/283/494/400 不变），不重跑。
  - dod 早于入科、院内死亡超过 365 天、dod 与 deathtime 日期不一致的人数均为 0；1 例 hospital_expire_flag = 1 但 deathtime 为空，按规则改用 dod。
  - 已对照 PhysioNet v3.1 文档：dod 截尾于末次出院后 1 年，入科后 365 天内的死亡可完整观察。报告 12.5 节已据此修正（只改了这一处）。
- **英文版报告**（2026-10-01）：`scripts/05_report.py` 现在同时生成 `results/report.md`（中文）和 `results/report_en.md`（英文）。
  - 英文模板在 `scripts/templates/report_en.md`，与中文版共用同一套数字和表格，表头和文字为英文。
  - 中文报告逐字节不变；英文报告中没有残留中文，没有未渲染的占位符，图片链接全部有效。
  - 阶段 8f 重组报告时，两个模板需要同步修改。
- **8c 第 0 步：小修正与确定化**（2026-10-02 至 10-03）：
  - **改动**：
    - `01_cohort.py` 的主结局改为 D8.10 新定义，保留旧定义并断言口径 A–D 最终队列逐人一致（通过）；
    - config 中 `min_age` 拆为 `min_age_replication` / `min_age_clinical`，并新增 `nested_cv` 节；
    - progress.md 中第一个"阶段 8：收尾"标题改为"阶段 7b：收尾"。
  - **第一次重跑 01 → 05**：38 个 CSV 中 36 个逐字节相同，两份报告逐字节相同；`table1.csv`、`table1_after_imputation.csv` 中 wbc_mean、pwr 的 p 值在第 7 位有效数字以后变化，按要求停下报告。
  - **诊断**：连续两次运行未修改的 02，只有 wbc_mean（最大差 7e-15，220 人）和 pwr（1.4e-14，205 人）不同。原因是 DuckDB 多线程 `avg()` 的浮点求和顺序随运行而变。
  - **确定化**：02 的连接改为单线程（`SET threads=1`）；连续两次运行 02，特征表完全相同。与旧的多线程结果相比，hemoglobin/wbc/rbc/rdw 均值和 pwr 的差 ≤ 3e-14。
  - **split 核对**：训练 1,568 / 验证 392，stay_id 集合与原 split.parquet 相同。
  - **重跑 03 → 05 后与上次提交（847a84d）比较**：
    - 38 个 CSV 中 32 个逐字节相同；
    - 4 个只有浮点末位差异：`lasso_coefs_faithful` 1.3e-15、`lasso_coefs_noleak` 6.1e-16、`shap_importance` 2.8e-16、`shap_importance_lr` 2.8e-16；
    - `table1.csv`、`table1_after_imputation.csv`：hemoglobin/wbc/rbc/rdw_mean 与 pwr 的 Mann-Whitney p 值变化，最大绝对差 0.0011（wbc_mean：0.7461 → 0.7472，显示值 0.746 → 0.747）。原因是末位差异打破或形成了并列值，改变了秩与并列校正。其余列不变。
    - `report.md`、`report_en.md` 逐字节相同，报告中显示的数字没有变化。
- **8c 第 1–2 步：完整嵌套 CV 框架**（`scripts/04f_nested_full.py`，2026-10-03）：
  - **种子**（每次重复一个，config `nested_cv.seeds = [42, 43, 44, 45, 46]`，下列各处都用该次重复的种子）：
    - 外层切分：`StratifiedKFold(10, shuffle=True, random_state=seed)`；
    - LASSO 内层 10 折：`cv_lasso(..., n_folds=10, seed=seed)`，内部为 `StratifiedKFold(random_state=seed)`，liblinear 也用该 `random_state`；
    - 调参内层 5 折：`GridSearchCV(cv=StratifiedKFold(5, shuffle=True, random_state=seed))`（D8.5 修订 2026-10-02）；
    - 模型自身的 `random_state`：`paper_models(seed)`、`tuning_grids(p, seed)`（RF、XGBoost、SVC）；LR 用 lbfgs，无随机性；
    - SVC Platt 缩放的内部 CV：调参后重新拟合时 `SVC(probability=True, random_state=seed)`；论文参数 SVM 同样由 `paper_models(seed)` 设定。
  - **防泄漏**：留出验证集不读入（parquet 行过滤，断言 n = 1,568）。缺失筛选、插补/标准化、LASSO 和每个模型的拟合前都断言拟合行与外层验证折无交集。
  - **Nadeau–Bengio 单元检查**（`tests/check_nadeau_bengio.py`，模拟数据）：
    - 公式核对通过：σ² 为全部 k·r 个配对差值的样本方差（ddof = 1），t 分布自由度 k·r − 1，校正因子 1/(k·r) + n_test/n_train；
    - 零假设模拟（200 个数据集，10 × 5 CV）：校正检验的拒绝率 0.040，未校正 t 检验 0.380。
  - **一致性自检**（`--selfcheck`：种子 42、关闭内层调参、阶段 4 超参数）：9 个模型的 CV AUC 与 `nested_cv_auc.csv`（04d）一致，最大差 4.1e-5 < 1e-3（来自该表 4 位小数的舍入）。
  - **耗时估计**（`--timing`：重复 0、第 0 折，9 个模型全部运行，`results/replication/nested_timing_C.csv`）：
    - 单折耗时：LASSO（各模型共用）35 s；调参 RF 60 s；调参 XGBoost 10 s；调参 SVM 1.9 s；调参 LR 0.1 s；5 个论文参数模型合计 < 0.4 s；
    - 外推到 5 × 10 = 50 折：LASSO 0.49 h + 调参 RF 0.83 h + 调参 XGBoost 0.14 h + 其余 < 0.05 h，合计约 **1.5 h**；
    - 远低于 48 h，不触发 D8.5 降级方案，9 个模型均跑满 5 次重复；
    - 第 0 折的结果已写入缓存，全量运行时直接复用。各折入选变量数不同，实际耗时可能有出入。
- **8c 第 3 步：口径 C 完整嵌套 CV 全量运行与模型选择**（2026-10-03，D8.11）：
  - 9 个模型 × 5 次重复 × 10 折全部完成，无 0 变量折，未降级；LASSO 每折入选 8–16 个变量（中位数 12）。
  - 外层 AUROC（折均值）：XGBoost 论文参数 0.791（M\*）、XGBoost 调参 0.790、LR L2 0.788、RF 调参 0.788、SVM 调参 0.788、LR 无惩罚 0.786、RF 论文字面 0.767、RF R 默认 0.724、SVM 论文参数 0.723（SD 均约 0.035–0.044）。
  - 按 D8.8：等价集 = XGBoost×2、LR×2、RF 调参、SVM 调参（Δ 均 ≤ 0.005）；两种 LR 都等价，按 OOF 校准斜率选 **无惩罚 LR**（0.854 vs L2 1.189）。推荐模型与阶段 6–7 一致，报告推荐不变；无惩罚 LR 的 AUROC 在等价集中最低（比 M\* 低 0.005），报告中如实写出。
  - ~~下一步：口径 A 的完整嵌套 CV（04f `--cohort A`），然后做 A/C 一致性判定（D8.7）与口径 C 留出集次要估计。~~（2026-10-03 修订后作废，见下）
- **8c 第 3 步补充：LR 校准斜率精确重算与校准不收敛处理**（2026-10-03，D8.11）：
  - 从缓存的各折预测重算每次重复合并 OOF 的校准斜率，按"每次重复 |ln 斜率| 再平均"：无惩罚 LR 0.158，L2 LR 0.170，仍选 **无惩罚 LR**，结论不变；L2 在 3/5 次重复中更接近 1，差距主要来自种子 44，校准在报告中写作"两种 LR 相当"，推荐理由以简约和可解释性为主。
  - 04f 的校准拟合加入收敛检查（不收敛或 |值| > 1e3 记为 NaN），summary 新增不收敛折数列和每次重复的斜率列。只有 RF（R 默认）有 5 折截距不收敛，其校准只报告合并 OOF 值；其余模型数值不变。
  - 从缓存重跑 04f（未重新拟合模型）：dca、pairwise、params、selection、oof_threshold 与之前逐字节相同；folds、summary 只新增列，RF（R 默认）的折级校准值改为 NaN。日志 `results/logs/04f_nested_full_C_recal.log`。
- **计划修订：取消口径 A 建模**（2026-10-03，D8.1、D8.7 修订）：
  - 口径 A 不再做完整嵌套 CV，也不做 A/C 一致性判定。该决定在口径 A 没有任何模型结果时做出。
  - 报告中口径 A、B、D 只保留队列流程与死亡率；第 2.2 节只陈述四种口径的人数与死亡率；措辞为"本研究采用口径 C"。
- **剩余工作**（修订后）：
  1. 口径 C 留出验证集的次要估计（D8.6、D8.7）：无惩罚 LR，阈值取 Youden 0.229；
  2. 8d：`diabetes_with_cc` 比例核查；口径 C 共病时间敏感性分析（推荐 LR + 调参 RF，D8.2）；
  3. 8e：CLIN 队列、特征与完整嵌套 CV，按 D8.8 选模型，留出集确认性评估（D8.1–D8.4）；
  4. 8f：报告重组，中英文模板同步。按 D8.9 的结构写，并落实本次修订：口径 A/B/D 只保留流程与死亡率、改写第 2.2 节、使用"本研究采用口径 C"的措辞。
- **8d 第 1 步：口径 C 留出集次要估计**（2026-10-03，D8.12，`scripts/04g_holdout_C.py`）：
  - 无惩罚 LR：AUROC 0.803（0.753–0.853），AUPRC 0.581（0.500–0.683，基线 0.253），Brier 0.145，校准斜率 0.96。
  - 调参 RF：AUROC 0.793（0.741–0.845），AUPRC 0.605（0.521–0.693），Brier 0.153，校准斜率 1.76。
  - 阈值取嵌套 CV OOF 的 Youden 中位数（LR 0.229，RF 0.247）：LR 的 recall 0.78、specificity 0.70；RF 的 recall 0.77、specificity 0.66。
  - EPV 32.9（395 例事件 / 12 个入选变量）。该验证集参与过事后模型选择，只作为次要估计。
- **8d 第 2 步：共病时间敏感性分析**（D8.12，`scripts/02b_comorb_time.py` + 04f `--variant comorb_time`）：
  - 子集库的诊断数据完整：与原始文件行数一致。
  - `diabetes_with_cc` 有 <10 人为 0，原因是 249.6x 编码，需更正 D8.2 的表述。
  - 重算后 843 人的 CCI 改变，21 人的 SAPS-II 改变；其中 304 人的心梗仅来自索引住院的急性编码。
  - 冒烟测试通过，外推全量约 1.2 h。**全量待运行**，运行后写入 D8.12。
  - **全量完成**（2026-10-03）：与主分析相比，LR 的 ΔAUROC 为 +0.001（−0.006, 0.008），RF 为 +0.002（−0.006, 0.010）。AUPRC、Brier 和 LR 的校准斜率基本不变（0.854 → 0.852）；RF 的校准斜率 1.48 → 1.59。CCI 仍在 50/50 折入选。结论不受共病编码时间归属的影响。
  - 肾病"失去 = 0"的原因：口径 C 索引住院中有 586/N19 的人数为 0。整个 MIMIC-IV v3.1 只有 119 行 586/N19；本数据中 AKI 编码为 584/N17。
- **剩余工作**（更新）：
  1. 8e：临床改良模型（CLIN）；
  2. 8f：报告重组。
- **8e-1：CLIN 队列与特征；特征均值改为与行顺序无关**（2026-10-03，D8.13）：
  - **子集库**：扩充到 3,189 名患者（复现线候选 ∪ CLIN 候选）。扩充后 `features.parquet` 有 5 列出现约 1e-14 的差异，原因是建库时行顺序不固定，加上 `avg()` 按行顺序累加。
  - **02 的修正**：均值改为先排序再求平均，与行顺序无关。重跑 02 → 05：
    - LASSO 变量、λ、调参超参数、全部模型指标不变，中英文报告逐字节相同。
    - 显示值只有 Table 1 中 rbc_mean 的中位数 3.44 → 3.43，该值不在报告中。
    - 04d、04f、04g 沿用旧特征。
  - **CLIN 队列**：3,189 → 2,557 人（排除转出 594 人、24 h 前死亡 38 人），1 年死亡 680 例（26.6%）。训练集 2,045 / 留出集 512，死亡率均为 26.6%。
  - **CLIN 特征**：
    - `wbc_diff_missing = 1` 的有 873 人（34.1%），部分缺失 21 人。
    - 缺失率 > 20% 的只有 8 个白细胞分类变量。
    - 有既往住院的 1,452 人（56.8%）。
    - 586/N19 影响 0 人，不做加回的敏感性分析。
    - `diabetes_with_cc` 为 99.9%。
  - **新增脚本**：`01c_cohort_clin.py`；02 和 02b 新增 `--track clinical`；一次性核对脚本移到 `tests/check_subset_extension.py`。
- **剩余工作**（更新）：
  1. 8e-2：CLIN 完整嵌套 CV（04f `--track clinical`），按 D8.8 选模型，留出集确认性评估；
  2. 8f：报告重组。
- **8e-3：CLIN 模型选择**（2026-10-03，D8.14，**只选择、尚未冻结**；未读取留出集）：
  - 04f `--track clinical` 全量：9 个模型 × 5 × 10 折完成，未降级，无 0 变量折；LASSO 每折入选 13–23 个变量（中位数 17）；只有 RF（R 默认）1 折校准截距不收敛。
  - 外层 AUROC：LR L2 0.801（M\*）、LR 无惩罚 0.799、SVM 调参 0.798、RF 调参 0.797、XGBoost 调参 0.795、XGBoost 论文 0.794、RF 论文字面 0.781、SVM 论文 0.756、RF R 默认 0.740。
  - 按 D8.8：等价集 = 两种 LR、SVM/RF/XGBoost 调参、XGBoost 论文参数（Δ ≤ 0.008）；两种 LR 的 |ln 斜率|（每次重复再平均）L2 0.102 对无惩罚 0.109 → 选 **LR（L2）**。差距很小，无惩罚 LR 在 3/5 次重复中更接近 1，报告写作"两种 LR 校准相当"。
  - L2 LR：合并 OOF 校准斜率 1.108，Youden 阈值 0.258；调参 RF：1.465，0.241。
  - `wbc_diff_missing` 入选 13/50 折；9 个变量 50/50 入选（age、aniongap_min、cci、oasis、apsiii、resp_rate_mean、rdw_mean、chloride_max、obesity）。
  - 下一步：L2 LR + 调参 RF 的两项预设敏感性分析（prior_only、no_wbc_diff），然后冻结并做留出集确认性评估。
- **8e-4 第 1 步：冻结 CLIN 模型**（2026-10-04）：`frozen_model_clin.json` + `scripts/04h_final_clin.py`。训练集上 LASSO λ.1se 0.0168，入选 22 个变量；LR（L2）C = 0.01；调参 RF max_depth 10 / max_features 2 / min_samples_leaf 20；EPV 24.7。冻结时未读取留出集。敏感性分析改到留出集评估之后运行（不影响冻结模型）。
- **8e-4 第 2 步：CLIN 留出集确认性评估**（2026-10-04，D8.15，`scripts/04i_holdout_clin.py`，只运行一次）：
  - 三道防线 + 训练集冒烟测试通过后正式运行一次。
  - LR（L2）：AUROC 0.750（0.702–0.797），AUPRC 0.533（基线 0.266），Brier 0.165，校准斜率 0.82；调参 RF：AUROC 0.752（0.704–0.799），校准斜率 1.14；配对 DeLong p = 0.83。
  - 留出集 AUROC 比嵌套 CV（0.801）低约 0.05，两模型同步。训练集表观 AUROC 0.817，不像过拟合；按 OOF 预测抽样，≤ 0.750 的概率约 1.4%，属于抽样下尾。如实报告，不更换模型。
  - SHAP 前 3 名：rdw_mean、cci、age（与口径 C 相同）。
  - 剩余：两项敏感性分析；8f 报告重组。
- **8e 敏感性分析**（2026-10-04，D8.16；LR（L2）+ 调参 RF，5 × 10 嵌套 CV，与主分析同划分）：
  - prior_only：LR ΔAUROC −0.014（−0.023, −0.005），RF −0.008（−0.016, −0.0001）。obesity 入选 50 → 4/50。索引住院的慢性病编码有一定增量，作为保守部署情景报告（AUROC 约 0.787）。
  - no_wbc_diff：LR +0.001（−0.005, 0.007），RF −0.002（−0.010, 0.005），入选变量数中位数 17 → 13。白细胞分类变量没有增量。
  - 冻结模型与主分析不变。剩余：8f 报告重组。
- **8f：报告重组**（2026-10-04，D8.17）：
  - 新增 `scripts/04j_plots_clin.py`：用冻结模型对 CLIN 留出集重新预测并画 ROC、校准、DCA（另加嵌套 CV OOF 校准图）。重算指标与 `holdout_clin.csv`、`holdout_dca_clin.csv` 逐项一致（差 0），同时把 D8.15 的事后核查写成 `holdout_gap_diagnostics_clin.csv`。
  - 新增 `scripts/03c_table1_clin.py`：CLIN Table 1（2,557 人，按结局分组）、`wbc_diff_missing` 两组死亡率（28.6% vs 25.5%，p = 0.092）、肥胖编码来源表。
  - `05_report.py` 与英文模板重写为三部分结构（口径 C / CLIN / 比较与局限），中英文同步；阶段 4–7 的详细表格移到附录 A。按 D8.1/D8.7 修订，口径 A/B/D 只列流程与死亡率；`plans/stage8_prompt.md` 中"口径 A 与 C 并列"不再执行。
  - 更正：CLIN 留出集 LR (L2) AUROC 的 CI 上限为 0.797（此前误写 0.798），配对 DeLong 差值 CI 上限为 0.016（此前误写 0.017）。
  - 审阅后补充：二分类变量的 OR 改报"有 vs 无"（obesity 0.63，wbc_diff_missing 1.33，gender_male 1.31）；6.1 补充种子 42 对照（0.781，与阶段 7 一致）；标题改为"复现与临床改良报告"。
  - 新增 `scripts/04k_shap_clin.py`（论文 Figure 4a）：冻结 LR (L2) 在 CLIN 留出集（512 人）上的 SHAP 图，不重新拟合；模型文件、fit record、frozen spec 哈希校验同 04j。输出 `results/figures/figC5_shap_bar_clin_holdout.{png,pdf}`、`figC6_shap_beeswarm_clin_holdout.{png,pdf}`（22 个变量，300 dpi）。一致性检查通过：mean |SHAP| 与 `clinical/shap_clin.csv` 6 位小数舍入后最大差 0，排名一致。
- 2026-10-22 合并扩展与整理项目（分支 `merge-extensions`，备份标签 `pre-merge-2026-10-22`）：同学的扩展代码放入 `extensions/`（balanced 为最终版；严格版只保留特征准备链，其实验代码移到本地 `archive/`），扩展结果快照 `results/extensions/balanced_27464ba98305edbc/`；`report.md` / `report_en.md` 拼接第四部分（§19–25、附录 C、开头摘要），第一至第三部分及全部数字不变；合并 `scripts/report_extensions.py` 与 `05_report.py` 的 3 行接口，04h/04j 保留本地版；新增 `paper/`、`DATA_ACCESS.md`、`tools/check_public_release.py`（只检查 git 将发布的文件），重写 README，更新 REPRODUCE（第 7 节扩展）；`CLAUDE.md`、`.claude/`、`results/logs/`、`docs/topic16.pdf` 停止跟踪（仍在本机）。检查：发布检查通过且能拦截反例；冻结模型、fit record、frozen spec、留出集结果与 `src/` 哈希与标签时一致；合成数据 smoke 测试（Nadeau–Bengio、balanced test_tree_ablations、smoke_checks）通过。
