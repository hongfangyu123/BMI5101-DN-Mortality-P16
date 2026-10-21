"""把已归档的 balanced 聚合结果合并入中文 V3 报告；不读取患者表、不训练模型。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PART = '# 第四部分：时间轨迹、诊断 embedding 与 stacking 扩展（探索性）'
END = '<!-- END BALANCED EXTENSIONS -->'
START = '<!-- BEGIN BALANCED EXTENSIONS -->'
APPENDIX = '<!-- BEGIN BALANCED EXTENSION SOURCES -->'
APPENDIX_END = '<!-- END BALANCED EXTENSION SOURCES -->'

LABELS = {
    'static_lr': '静态 LR (L2)', 'static_rf': '静态 RF', 'static_xgb': '静态 XGBoost',
    'static_stacking': '静态 stacking', 'temporal_lr': '静态＋轨迹 LR',
    'temporal_rf': '静态＋轨迹 RF', 'temporal_xgb': '静态＋轨迹 XGBoost',
    'burden_lr': '静态＋诊断负担 LR', 'burden_rf': '静态＋诊断负担 RF',
    'burden_xgb': '静态＋诊断负担 XGBoost', 'embedding_lr': '静态＋诊断负担＋embedding LR',
    'embedding_rf': '静态＋诊断负担＋embedding RF', 'embedding_xgb': '静态＋诊断负担＋embedding XGBoost',
    'combined_stacking': '静态＋轨迹＋诊断表示 stacking',
}
ORDER = ['static_lr', 'static_rf', 'static_xgb', 'static_stacking',
         'temporal_lr', 'temporal_rf', 'temporal_xgb', 'burden_lr', 'burden_rf', 'burden_xgb',
         'embedding_lr', 'embedding_rf', 'embedding_xgb', 'combined_stacking']


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def remove_block(text, begin, end):
    if begin not in text:
        return text
    left, rest = text.split(begin, 1)
    if end not in rest:
        raise ValueError(f'报告补充块缺少结束标记：{end}')
    return left.rstrip('\n') + '\n\n' + rest.split(end, 1)[1].lstrip('\n')


def integrate_extensions_report(report: str, results_dir: Path) -> str:
    """可重复调用：只替换本补充块；保留原报告的主要结果和历史段落。"""
    results_dir = Path(results_dir)
    latest = results_dir / 'extensions/latest.txt'
    if not latest.exists():
        return report
    snapshot_name = latest.read_text(encoding='utf-8').strip()
    if Path(snapshot_name).name != snapshot_name:
        raise ValueError('扩展快照名称无效')
    folder = latest.parent / snapshot_name
    provenance = json.loads((folder / 'source_provenance.json').read_text(encoding='utf-8'))
    if provenance['source_signature'] != '27464ba98305edbc':
        raise ValueError('此补充模板仅对应已核对的27464ba98305edbc完整实验，请先更新解释再使用新快照')
    for name, digest in provenance['files_sha256'].items():
        if Path(name).name != name or hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'扩展聚合文件已变化：{name}；请重新核对快照')
    summary = pd.read_csv(folder / 'summary.csv').set_index('experiment')
    paired = pd.read_csv(folder / 'paired_comparisons.csv')
    stability = pd.read_csv(folder / 'repeat_stability.csv')
    evidence = json.loads((folder / 'extension_evidence.json').read_text(encoding='utf-8'))
    identity = json.loads((folder / 'run_identity.json').read_text(encoding='utf-8'))
    if not set(ORDER) <= set(summary.index) or not summary.loc[ORDER, 'n_outer_folds'].eq(50).all():
        raise ValueError('扩展结果尚未包含完整14组×50折，不能生成最终补充报告')
    if identity['holdout_labels_loaded'] or identity['synthetic']:
        raise ValueError('补充报告只接受未读取留出标签的真实开发集实验')
    options = identity['options']
    if options['seeds'] != [42, 43, 44, 45, 46] or options['outer_folds'] != 10:
        raise ValueError('当前报告模板要求完整5×10折；评估设置改变后需更新说明')
    h = evidence['history']
    if identity['n_development'] != h['n_development']:
        raise ValueError('历史诊断聚合人数与运行身份不一致')
    linkbase = f'extensions/{snapshot_name}'
    def link(name):
        return f'[{name}]({linkbase}/{name})'
    def comparison(exp, ref, metric='AUROC'):
        rows = paired[(paired.experiment == exp) & (paired.reference == ref) & (paired.metric == metric)]
        if len(rows) != 1:
            raise ValueError(f'缺少唯一配对结果：{exp}/{ref}/{metric}')
        return rows.iloc[0]
    stack = summary.loc['static_stacking']; lr = summary.loc['static_lr']; combined = summary.loc['combined_stacking']
    stackdiff = comparison('static_stacking', 'static_lr')
    jointdiff = comparison('combined_stacking', 'static_stacking')
    nfamily = evidence['metric_comparison_family']
    if not paired.groupby('metric').size().eq(nfamily).all():
        raise ValueError('多重比较家族与聚合表不一致')
    main_table = table(['模型', '外层折数', 'AUROC：均值 ± 折间SD', 'AUPRC均值', 'Brier均值↓'], [
        [LABELS[e], 50, f'{summary.loc[e,"AUROC_fold_mean"]:.4f} ± {summary.loc[e,"AUROC_fold_sd"]:.4f}',
         f'{summary.loc[e,"AUPRC_fold_mean"]:.4f}', f'{summary.loc[e,"Brier_fold_mean"]:.4f}'] for e in ORDER])
    auroc_pairs = paired[paired.metric == 'AUROC']
    pair_table = table(['扩展', '参考', 'ΔAUROC', '校正95%差值CI', '原始p', 'Holm p'], [
        [f'`{r.experiment}`', f'`{r.reference}`', f'{r.mean_diff:+.5f}', f'[{r.lo:+.5f}, {r.hi:+.5f}]',
         f'{r.p_value:.4f}', f'{r.p_holm_current_family:.4f}'] for r in auroc_pairs.itertuples()])
    calibration_table = table(['模型', '完整OOF AUROC', '完整OOF AUPRC', '完整OOF Brier↓', '校准截距', '校准斜率'], [
        [LABELS[e], f'{summary.loc[e,"AUROC_pooled_mean"]:.4f}', f'{summary.loc[e,"AUPRC_pooled_mean"]:.4f}',
         f'{summary.loc[e,"Brier_pooled_mean"]:.4f}', f'{summary.loc[e,"cal_intercept_pooled_mean"]:.4f}',
         f'{summary.loc[e,"cal_slope_pooled_mean"]:.4f}'] for e in ORDER])
    st = stability[(stability.experiment == 'static_stacking') & (stability.reference == 'static_lr')].set_index('seed')
    co = stability[(stability.experiment == 'combined_stacking') & (stability.reference == 'static_stacking')].set_index('seed')
    repeat_table = table(['重复seed', '静态stacking − LR：OOF ΔAUROC', '联合 − 静态stacking：OOF ΔAUROC'], [
        [seed, f'{st.loc[seed,"AUROC_difference"]:+.5f}', f'{co.loc[seed,"AUROC_difference"]:+.5f}'] for seed in options['seeds']])
    dca_table = table(['扩展 / 参考', '净获益较高阈值数', '较低', '相同', '净获益差范围'], [
        [f'`{r["experiment"]}` / `{r["reference"]}`', r['higher'], r['lower'], r['equal'],
         f'[{r["minimum_difference"]:+.5f}, {r["maximum_difference"]:+.5f}]'] for r in evidence['dca']])
    temporal = evidence['selection']['temporal_lr']
    selection_table = table(['时间特征', '被LASSO选中的外层折数'], [[f'`{name}`', f'{count}/50'] for name, count in temporal['top']])
    burden_table = table(['学习器', '静态AUROC', '＋诊断负担AUROC', '＋负担＋embedding AUROC'], [
        [name, f'{summary.loc[f"static_{key}","AUROC_fold_mean"]:.4f}',
         f'{summary.loc[f"burden_{key}","AUROC_fold_mean"]:.4f}', f'{summary.loc[f"embedding_{key}","AUROC_fold_mean"]:.4f}']
        for key, name in [('lr','LR'),('rf','RF'),('xgb','XGBoost')]])
    part = f'''{START}
{PART}

**补充日期：2026-10-15。**本部分是在已查看原 V3 分析后进行的开发集探索，不属于 D8.8 写定时的原始模型选择计划。现已完成时间轨迹、既往诊断表示、静态 stacking 和联合 stacking 共14组实验，每组5次重复×10折。原 V3 的冻结模型与留出集结果保留；本部分不据此改写原模型的确认性评估。

## 19. 扩展设计、可比性与评估口径

### 19.1 队列和数据边界

- 直接沿用 CLIN 队列及其开发集：全队列2,557人，开发集 **{identity['n_development']:,}人、死亡{identity['events_development']}人**；原512人留出集不参与本次扩展选择、训练或评分。预测时点仍为 ICU 入科后24 h，结局沿用第1节的入科起1年全因死亡。
- 静态候选共{evidence['n_static_candidates']}列（55个原预测变量＋wbc_diff_missing）。不重新筛选队列，不因为缺少历史诊断或动态测量删患者。死亡为阳性类；开发集死亡比例为 {identity['events_development']/identity['n_development']:.2%}。
- 新时间特征仅用入科后0–24 h；静态特征保持 V3 原窗口（化验 −6–24 h，生命体征0–24 h）。历史诊断只用索引住院之前已经结束的住院。现有静态共病仍继承 CLIN 的既往∪索引慢性编码假设，其时间可得性局限未被新增 embedding 消除。

### 19.2 balanced 固定参数配方

为减少 stacking 内重复网格搜索的计算，本部分采用独立记录的 balanced 配方；各静态对照与扩展严格共享这套配方：

| 组件 | 设置 |
|---|---|
| 外层评估 | 分层10折，seed 42–46，共50折；同折配对比较 |
| 数值变量处理 | 训练折内按20%缺失率筛选、均值/众数插补；原白细胞分类8项保留既定豁免；零方差变量删除 |
| LASSO | 训练集内部5折、30个λ，二项偏差及1-SE选择；LASSO内部折也重新拟合缺失筛选、插补和标准化 |
| LR | L2，C=1，max_iter=10000；分类前在相应训练集拟合 StandardScaler |
| RF | 200棵树，max_depth=10，min_samples_leaf=10，max_features=sqrt |
| XGBoost | 100棵树，max_depth=2，learning_rate=0.1，subsample=0.8，colsample_bytree=0.6，CPU hist |
| stacking | LR/RF/XGBoost基模型；内部3折交叉拟合；元模型为L2 LR（C=1）；passthrough=False |
| 诊断表示 | TF-IDF min_df=10，TruncatedSVD最多32维；均在相应训练集拟合 |

**这里保留折内变量选择，但取消了基模型的内部超参数网格搜索。**因此不能把本部分写成原 V3 的“完整嵌套调参模型”，也不能直接把 stacking 的 {stack.AUROC_fold_mean:.3f} 与第11节 LR 的约0.801相减来估计增益。stacking 的直接对照是本配方下静态 LR 的 {lr.AUROC_fold_mean:.4f}。严格版的未完成折和 pilot / 合成测试没有混入正式结果。

### 19.3 指标与推断

下文主要表格是**50个验证折上分别计算指标后的均值**。每次重复也将10折 OOF 预测拼回完整开发集，计算一次整体指标，再对5次重复取平均，作为另一个汇总口径（第24节）。这不是先平均5次预测概率再计算指标；两种汇总不能混用。

配对差值用 Nadeau–Bengio 方法修正重复交叉验证的训练样本重叠；置信区间是**模型间差值的95%区间**，不是某个模型自身 AUROC 的区间。最终按每个指标对当前 **{nfamily}组比较**作 Holm 校正。早期结果中较小的 Holm p 值属于当时更小的比较家族，本报告统一使用完整家族。折间SD仅描述波动，50折不等于50个独立验证队列；差异未显著不等于证明等效。

## 20. 时间轨迹：构建、入选与消融

### 20.1 时间特征如何构建

在原静态候选基础上，对25项化验/生命体征各增加5种统计，共 **{evidence['n_temporal_candidates']}个候选时间特征**。化验限定同一subject_id与hadm_id，生命体征限定同一subject_id与stay_id；charttime ∈ [intime, intime+24 h]。同一时刻重复测量先取平均，不把重复记录当作独立时间点。

| 后缀 | 定义 | 测量不足时 |
|---|---|---|
| delta | 最后值减最初值 | 少于2个不同时刻则缺失 |
| slope_per_hour | 对实际测量小时数拟合线性斜率 | 少于2个不同时刻则缺失 |
| std | 不同测量时刻数值的样本标准差 | 少于2个不同时刻则缺失 |
| late_minus_early | 12–24 h均值减0–<12 h均值 | 任一半段没有测量则缺失 |
| count | 有效的不同测量时刻数 | 无测量记0 |

变量缺失率在训练折计算；时间变量没有白细胞分类豁免，缺失过多可被剔除，患者保留。count 反映检测强度，也可能受到就诊过程影响，不能全部解释为生理变化。

### 20.2 新特征实际参与训练

每个外层最终拟合都选中了时间变量：每折中位数 **{temporal['selected_temporal_median']:g}项**，范围 **{temporal['selected_temporal_min']}–{temporal['selected_temporal_max']}项**；静态＋时间数值变量合计入选中位数 {temporal['selected_numeric_median']:g}项。LR、RF、XGBoost 使用同一套 LASSO 选择，故以下频次相同；这不是三种分类器各自的特征重要性：

{selection_table}

50折累积入选中，测量次数类共{temporal['count_feature_instances']}次，其他动态类共{temporal['other_temporal_instances']}次，不能把本次结果解释成“时间特征全被删掉了”。入选频次也不是独立验证的生物学或因果证据。

### 20.3 性能发现

相对同模型静态对照，时间特征使平均 AUROC 在 LR、RF、XGBoost 中分别变化 **{comparison('temporal_lr','static_lr').mean_diff:+.4f}、{comparison('temporal_rf','static_rf').mean_diff:+.4f}、{comparison('temporal_xgb','static_xgb').mean_diff:+.4f}**。三者 AUPRC 均略低、Brier 均略高，配对差值区间均包含0（完整结果见第23节）。因此，当前0–24 h统计量、20%缺失阈值和 LASSO 配方没有显示额外收益；不能推广为所有动态表示均无效。可能解释包括：静态均值/极值和评分已包含部分信息、化验重复测量有限、部分轨迹缺失过多、LASSO可能丢弃只通过非线性或交互发挥作用的变量；这些是待检验解释。

## 21. 诊断 embedding：来源、表示和模型依赖

### 21.1 历史诊断如何变成向量

1. 仅取同一患者以前住院，要求 **历史dischtime < 索引admittime**；不加入本次住院或未来住院的诊断。
2. ICD码去小数点、转大写，区分ICD-9与ICD-10。每个码同时生成完整码和前三字符分组 token，例如ICD-10 E11.9 → `v10_E119`＋`c10_E11`。前三字符只是简单编码分组，不是完整医学疾病层级。
3. 每位患者将所有历史token去重组成一份文档；同一诊断反复出现不增加文档词频。
4. 在训练折建立 TF-IDF 词表，保留至少10位训练患者中出现的token；验证集未知码不扩展词表。因为token已去重，sublinear_tf不额外表达重复诊断次数；频次由下述burden补充。
5. 训练折拟合 TruncatedSVD，把稀疏TF-IDF压缩至最多32维，验证集只transform。三种embedding模型的50个最终外层拟合均得到32维；TF-IDF/SVD本身不使用死亡标签，不保证其主要结构最有利于死亡预测。
6. 与LASSO选中的静态数值变量拼接后交给分类器；四个burden指标和32维embedding是在LASSO之后直接追加，不再经过LASSO筛选。LR对拼接结果做折内标准化，RF/XGBoost不增加StandardScaler。SVD维度不能直接命名为某种疾病，各折坐标也可能不同。

这是**结构化ICD诊断的压缩表示**。目前没有引入临床笔记或医学语言模型，不能写成已完成clinical notes融合。新增embedding严格使用既往诊断，不会修复原静态共病仍使用索引慢性编码的时间假设。

### 21.2 历史覆盖和诊断负担对照

开发集{h['n_development']:,}人中，{h['with_prior_admission']:,}人有已结束的既往住院，**{h['with_prior_diagnosis_document']:,}人（{h['document_coverage']:.1%}）有可用历史诊断文档**；其余{h['without_prior_diagnosis_document']:,}人保留空文档、诊断向量为0，不代表他们没有合并症。有历史文档者不同确切诊断码数量的中位数为{h['distinct_icd_median_among_document']:g}。历史仅覆盖本数据库记录，不等于完整院外病史。

burden含既往住院次数、不同确切诊断码数、诊断事件数（同次住院同码先去重）、有无历史诊断；前三项log1p转换。三组比较分开回答问题：`static → burden`检验历史记录数量，`burden → embedding`检验额外诊断组合表示，`static → embedding`检验整个新增诊断模块的总增益。

{burden_table}

### 21.3 性能与解释

- **LR**：burden与静态几乎相同；再加embedding的ΔAUROC为{comparison('embedding_lr','burden_lr').mean_diff:+.4f}，未显示增益。
- **RF**：embedding相对burden增加{comparison('embedding_rf','burden_rf').mean_diff:+.4f}，相对静态增加{comparison('embedding_rf','static_rf').mean_diff:+.4f}；5次完整OOF的AUROC均略高于静态RF，但差值区间仍跨0。其Brier反而由{summary.loc['static_rf','Brier_fold_mean']:.4f}升至{summary.loc['embedding_rf','Brier_fold_mean']:.4f}，不能仅按AUROC称为整体改善。
- **XGBoost**：embedding相对burden仅增加{comparison('embedding_xgb','burden_xgb').mean_diff:+.4f}，相对静态变化{comparison('embedding_xgb','static_xgb').mean_diff:+.4f}；同样没有明确增益证据。

结果方向依赖学习器，补充RF/XGBoost避免了只依据LR作泛化判断。历史覆盖不足、稀有token过滤、无监督压缩目标及维度增加可能限制收益，但本轮没有直接验证哪一个是原因。

## 22. Stacking：结构、增益与重复稳定性

### 22.1 如何避免元模型训练泄漏

每个外层训练集内部再分3折；每个基模型只用其中训练子集拟合自己的缺失筛选、插补、LASSO及（如适用）TF-IDF/SVD，然后预测内部验证子集，生成LR/RF/XGBoost三个折外死亡概率列。元模型L2 LR（C=1）只用这些内部OOF列训练，不使用基模型在自身训练行上的拟合概率。之后基模型用完整外层训练集重新拟合，预测外层验证集，再交由元模型组合。passthrough=False，元模型不额外接收原始特征。

静态stacking只使用静态输入；联合stacking的三个基模型都使用静态＋轨迹＋burden＋embedding。联合结果不是把单独LR的预测直接附到静态stacking上，也没有跳过内部折的预处理。补充的独立RF/XGBoost消融无需再做这层stacking交叉拟合。

### 22.2 静态stacking相对单模型

静态stacking的平均AUROC为 **{stack.AUROC_fold_mean:.4f}**，高于同配方LR的{lr.AUROC_fold_mean:.4f}，Δ={stackdiff.mean_diff:+.4f}（约{stackdiff.mean_diff*100:.2f}个百分点）。AUPRC增加{comparison('static_stacking','static_lr','AUPRC').mean_diff:+.4f}，Brier减少{abs(comparison('static_stacking','static_lr','Brier').mean_diff):.4f}。5次重复的完整OOF AUROC和AUPRC均更高，50折中{int(st.iloc[0].AUROC_fold_wins_total)}折AUROC更高，说明当前重复划分中方向一致。

但AUROC差值校正95% CI为 **[{stackdiff.lo:+.4f}, {stackdiff.hi:+.4f}]**，原始p={stackdiff.p_value:.4f}，最终Holm p={stackdiff.p_holm_current_family:.4f}；AUPRC、Brier差值区间也包含0。应写“小幅数值提升、尚未统计学确认”，不能写“显著优于LR”。五次重复使用同一批患者，不等于五个外部验证。预测互补是可能的机制，本轮未通过相关性/权重消融直接证明原因。

### 22.3 联合扩展是否再提高stacking

联合stacking平均AUROC **{combined.AUROC_fold_mean:.4f}**，相对静态仅 **{jointdiff.mean_diff:+.6f}**（约{jointdiff.mean_diff*100:.3f}个百分点），校正95% CI [{jointdiff.lo:+.4f}, {jointdiff.hi:+.4f}]，p={jointdiff.p_value:.4f}。AUPRC变化{comparison('combined_stacking','static_stacking','AUPRC').mean_diff:+.6f}，Brier变化{comparison('combined_stacking','static_stacking','Brier').mean_diff:+.6f}。完整OOF AUROC在5次中3次上升、2次下降，50折中{int(co.iloc[0].AUROC_fold_wins_total)}折上升，不能据此证明额外提升或等效。

{repeat_table}

**本轮可把静态stacking作为候选，保留LR作为简约对照；联合扩展没有显示值得替换静态stacking的性能收益。**这是探索性候选判断，不替代第11–12节已冻结的原V3推荐模型。

## 23. 完整性能表与同折配对推断

### 23.1 14组模型的50折结果

{main_table}

来源：{link('summary.csv')}、{link('fold_metrics.csv')}。SD是折间标准差，不能作为95% CI使用。表中的小数差异在第23.2节按同折差值检验，不能凭四舍五入后的排序宣布胜负。

### 23.2 最终15组AUROC比较

Δ为扩展减参考；正值对AUROC有利。下面包含单模型、轨迹、burden、embedding增量/总增益及联合模型比较；每个指标的Holm家族包含15组，当前全部Holm p为1。

{pair_table}

全部AUROC、AUPRC、Brier三类共45项配对结果及校正标准误见 {link('paired_comparisons.csv')}。这些是观察到原V3结果之后的探索性比较，未达到显著不能证明临床等效。

## 24. 校准、DCA与计算代价

### 24.1 完整OOF汇总

以下每个指标均先在每次重复的完整开发集OOF上计算，再取5次平均；校准理想截距为0、斜率为1：

{calibration_table}

来源：{link('pooled_oof_metrics.csv')}。静态stacking的OOF斜率{stack.cal_slope_pooled_mean:.3f}，较LR的{lr.cal_slope_pooled_mean:.3f}更接近1，但Brier差值未显著。RF加embedding的斜率由{summary.loc['static_rf','cal_slope_pooled_mean']:.3f}变为{summary.loc['embedding_rf','cal_slope_pooled_mean']:.3f}，离1更远，且Brier数值变差，说明AUROC略升不能代替完整评估。本轮未做扩展模型独立留出集再校准或外部校准验证。

### 24.2 DCA：没有统一净获益优势

在每次重复完整OOF上计算0.05–0.60、步长0.01的56个阈值，再按阈值取5次平均，比较模型净获益：

{dca_table}

来源：{link('pooled_dca.csv')}、{link('dca_comparison.csv')}。静态stacking并非所有阈值均优于LR，联合扩展也没有统一优势。这里未计算DCA差值区间，也没有预先确定临床决策阈值，不能用“赢了多少阈值点”作显著性检验或宣称临床效用已改善；不同阈值点彼此相关。

### 24.3 计算与缓存

静态stacking各折记录耗时合计约{stack.seconds_total/60:.1f}分钟，联合stacking约{combined.seconds_total/3600:.2f}小时；轨迹LR约{summary.loc['temporal_lr','seconds_total']/60:.1f}分钟。诊断/轨迹后续RF/XGBoost能共享先前拟合的表示缓存，不能把其少量记录耗时解释为从零构建全部特征的固有代价。这些用时还受并行运行与缓存命中影响，不是严格受控的速度基准。

新增树模型实验前，对源签名`ebf3f368bb8beb5c`的数据、配置、环境、训练函数与每折索引作兼容性校验后，复用了400个旧折预测，并新增6组×50折。最终签名为 **`{provenance['source_signature']}`**；700个折记录来自同一开发集与划分，未混入未完成的严格版结果。四次重复的中途快照仅作为历史记录，最终表统一采用完整五次结果。

## 25. 综合结论与扩展局限

1. **Stacking**：相对本配方的静态LR有约0.0054 AUROC的数值增益，重复划分方向一致，但CI跨0、最终Holm p=1；可作为候选，保留LR对照。
2. **时间轨迹**：LR、RF、XGBoost均未显示额外收益；时间变量确实入选，当前结论限定于0–24 h统计量及既定缺失/选择规则，不覆盖所有时间模型。
3. **诊断embedding**：RF有微小数值增益，LR/XGBoost没有明确改善；RF的Brier和校准斜率未同步改善。无监督压缩、病史不完整和稀有码过滤的影响尚未隔离。
4. **联合扩展**：比静态stacking的AUROC仅高约0.00009，其他指标基本持平，当前没有增加模型复杂度的性能依据。
5. **泛化边界**：这是已多轮探索的单中心开发集证据，固定参数不代表最优；相同LASSO便于控制变量，但可能不利于只通过交互起作用的树模型特征。未评估治疗效应，也不能将入选变量称为可干预因果风险因素。
6. **下一步**：在课程工作量内优先围绕选定候选完成解释与临床阈值论证。如继续寻求泛化证据，需新的独立评估或明确隔离的时间验证；已看过的512人留出集不能重新称为全新未触碰测试集。原CLIN留出AUROC约0.750的结果仍须报告，不能被这里约0.805的开发集数字替代。
7. **完成与未完成**：已完成首日轨迹、结构化诊断表示、独立RF/XGBoost消融、stacking、校准估计及DCA点估计；尚未加入临床文本、扩展模型SHAP交互或反事实解释，也没有通过额外模型发现经过验证的可行动因果因素。原V3的SHAP结果仍属于原模型，不能直接归给新stacking。

与论文约0.780的数值比较只能说明本研究完成了不同设计下的扩展探索；队列、结局解释及评估设置不同，**不能据此声称已确认超越论文**。
{END}
'''
    report = remove_block(report, START, END)
    report = remove_block(report, APPENDIX, APPENDIX_END)
    # 更新摘要，保持原 V3 的三部分主要结果不被替换。
    old_intro = '报告分三部分：第一部分为论文复现（口径 C），第二部分为临床改良模型（CLIN），第三部分为与论文的比较、局限性和后续工作。'
    new_intro = '报告分四部分：第一部分为论文复现（口径 C），第二部分为临床改良模型（CLIN），第三部分为与论文的比较、局限性和后续工作，第四部分为时间轨迹、诊断 embedding 与 stacking 的开发集探索性扩展（2026-10-15补充）。扩展聚合快照及来源见附录 C；本报告不含患者标识或逐人预测。'
    report = report.replace(old_intro, new_intro)
    summary_start = '<!-- BEGIN EXTENSION SUMMARY -->'; summary_end = '<!-- END EXTENSION SUMMARY -->'
    report = remove_block(report, summary_start, summary_end)
    brief = (f'{summary_start}\n\n- **新增扩展（第四部分，2026-10-15）**：同一CLIN开发集2,045人、544例死亡，固定参数balanced方案完成14组×50折。静态stacking AUROC {stack.AUROC_fold_mean:.4f} vs 同方案LR {lr.AUROC_fold_mean:.4f}，Δ={stackdiff.mean_diff:+.4f}，差值95% CI [{stackdiff.lo:+.4f}, {stackdiff.hi:+.4f}]，原始p={stackdiff.p_value:.4f}、最终Holm p={stackdiff.p_holm_current_family:.4f}；方向一致但提升未得到统计学确认。时间特征在LR/RF/XGBoost上均无明确增益；诊断embedding仅RF有微小数值改善，未伴随Brier改善；联合stacking {combined.AUROC_fold_mean:.4f}与静态几乎相同。本轮不读取留出集标签，不能替代原CLIN留出AUROC约0.750。\n\n{summary_end}\n\n')
    marker = '- **最重要的预测因子**'
    if marker not in report:
        raise ValueError('找不到结论速览插入位置')
    before, after = report.split(marker, 1)
    report = before.rstrip('\n') + '\n\n' + brief + marker + after
    report = report.replace('6. **首日特征**：只用 ICU 首日数据，不含治疗信息（如血管活性药物、机械通气）和时间动态；部分评分与原始变量有重叠和共线性。',
                            '6. **首日特征**：原V3静态分析只用首日汇总值，不含治疗信息（如血管活性药物、机械通气）；第四部分已补充0–24 h时间轨迹消融，但未发现明确增益，也未扩展至更长观察窗。部分评分与原始变量仍有重叠和共线性。')
    report = report.replace('4. **特征**：加入 HbA1c、治疗变量和纵向特征；',
                            '4. **特征**：首日轨迹、诊断表示及stacking已完成探索性消融（第四部分）；后续可按课程工作量评估 HbA1c、治疗变量或更有针对性的动态表示；')
    if '\n# 附录\n' not in report:
        raise ValueError('找不到附录插入位置')
    report = report.replace('\n# 附录\n', '\n' + part.strip() + '\n\n# 附录\n', 1)
    source_text = f'''{APPENDIX}
## 附录 C：扩展结果来源与复现

- 聚合快照：`{linkbase}/`；源实验：`dn_mortality_repro_V3_extensions/balanced/results/full/{provenance['source_signature']}`。
- 主性能、同折配对及完整OOF：{link('summary.csv')}、{link('fold_metrics.csv')}、{link('paired_comparisons.csv')}、{link('pooled_oof_metrics.csv')}。
- 补充证据：{link('extension_evidence.json')}、{link('temporal_selection_frequency.csv')}、{link('repeat_stability.csv')}、{link('pooled_dca.csv')}、{link('dca_comparison.csv')}。
- 运行身份、缓存复用及SHA-256来源：{link('run_identity.json')}、{link('cache_reuse.json')}、{link('source_provenance.json')}。这些文件均为聚合结果，不含subject_id、hadm_id、stay_id、原始ICD文档或逐人预测概率。
- 代码与中文记录：相邻扩展项目的 `balanced/dn_extensions/{{features,modeling,evaluation}}.py`、`balanced/notebooks/02–06`。报告补充生成器 `scripts/report_extensions.py` 只读此归档快照，由 `scripts/05_report.py` 调用；重新生成中文报告会保留本扩展部分。

补充树模型的原执行命令（从 `dn_mortality_repro_V3_extensions` 目录运行）：

```bash
bash balanced/run_extensions.sh full --experiments temporal_rf,temporal_xgb,embedding_rf,embedding_xgb --reuse-from ebf3f368bb8beb5c
```

全部原有静态/时间/诊断/stacking实验可通过相同入口按实验名分项运行。完整参数在快照的run_identity中固定，当前归档报告不要求重新训练；`python scripts/05_report.py`只重新汇总报告。
{APPENDIX_END}
'''
    return report.rstrip() + '\n\n' + source_text.strip() + '\n'
