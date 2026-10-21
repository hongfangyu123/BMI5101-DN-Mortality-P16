"""生成带中文方法说明与运行开关的 Notebook，不执行其中的训练代码。"""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks'
SETUP = '''from pathlib import Path
import sys
import pandas as pd

# 从扩展目录或 notebooks 子目录定位项目，避免硬编码 Windows 盘符。
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "dn_extensions" / "settings.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("请从 dn_mortality_repro_V3_extensions 或其 notebooks 子目录启动。")
sys.path.insert(0, str(ROOT))
print("扩展目录：", ROOT)
'''


def result_cell():
    c = nbformat.v4.new_markdown_cell('## 实验状态与结论（自动更新）\n\n简化版尚未运行真实患者实验。原严格版前6折已保存，未混入此版。')
    c.metadata['extension_results'] = True
    return c


def save(name, title, method, experiments=None):
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python (ml - WSL)', 'language': 'python', 'name': 'ml-wsl'}
    nb.metadata['language_info'] = {'name': 'python'}
    nb.cells = [nbformat.v4.new_markdown_cell(f'# {title}\n\n{method}'), nbformat.v4.new_code_cell(SETUP)]
    if experiments is None:
        nb.cells.append(nbformat.v4.new_code_cell('''# 默认只展示说明，不扫描数据库。准备特征时改为 True。
RUN_FEATURE_PREPARATION = False
if RUN_FEATURE_PREPARATION:
    from dn_extensions.features import build_features
    from dn_extensions.reporting import update_preparation_notebook
    manifest = build_features()
    update_preparation_notebook(manifest)
else:
    print("balanced共享上级已准备的特征，无需重新prepare；缺少特征时到上级运行prepare。")
'''))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(f'''# pilot 是1×3外层小规模试跑；正式实验将 PROFILE 改为 "full"。
# 仅把 RUN_EXPERIMENT 改成 True 才会启动；full可能需要较长时间。
PROFILE = "pilot"
RUN_EXPERIMENT = False
EXPERIMENTS = {experiments!r}
if RUN_EXPERIMENT:
    from dn_extensions.evaluation import run_evaluation
    from dn_extensions.settings import profile_options
    output, summary, comparisons = run_evaluation(profile_options(PROFILE), EXPERIMENTS)
    display(summary)
    display(comparisons)
else:
    print("尚未启动本Notebook中的模型训练。")
'''))
        nb.cells.append(nbformat.v4.new_code_cell('''# 优先查看正式 full 结果；没有结果时打印状态，不重新训练。
RESULT_PROFILE = "full"
latest = ROOT / "results" / RESULT_PROFILE / "latest.txt"
if latest.exists():
    folder = latest.parent / latest.read_text().strip()
    summary = pd.read_csv(folder / "summary.csv")
    display(summary[summary.experiment.isin(EXPERIMENTS)])
    print("详细记录：", folder / "report.md")
else:
    print("该模式尚无实验结果。pilot 结果需把 RESULT_PROFILE 改为 pilot 才显示。")
'''))
    nb.cells.append(result_cell())
    OUT.mkdir(parents=True, exist_ok=True)
    nbformat.validate(nb)
    nbformat.write(nb, OUT / name)


def main():
    save('01_prepare_features.ipynb', 'V3：扩展特征准备', '''复用 V3 CLIN 的队列、结局和固定划分，只处理开发集，不读取留出集标签。

时间轨迹：入 ICU 后0–24小时，化验同次 hadm_id、生命体征同次 stay_id；取首尾差值、每小时斜率、SD、后12小时减前12小时均值、测量次数。相同时刻先平均。少于两个时刻时斜率和差值缺失，缺失不删除患者。

ICD 文档：只纳入索引住院开始前已结束的住院；同时保留 ICD9/10 版本、确切码与三级类目。没有既往诊断的患者仍保留，文档为空。此步只准备文档，没有提前拟合 TF-IDF/SVD。

V3 静态列仍沿用原来的−6至24小时窗口，因此轨迹增加的是动态信息而非重定义原始基线。描述性缺失率只用于了解数据，后续缺失率筛选在训练折内进行。

主分析继承 V3 的20%变量缺失规则及8个白细胞分类变量豁免。轨迹变量没有豁免，测量次数是检查强度指标，不应作因果风险因素解读。''')
    save('02_stacking.ipynb', 'V3：LR/RF/XGBoost stacking', '''在共同外层折中比较静态 LR、RF、XGBoost 和 stacking。

第一层是三个固定参数的 Pipeline；没有在stacking子折里重复网格调参。第二层 LR 只用第一层在 stacking 内部验证折上的预测概率训练。每个基模型在该内部训练子集重新插补、标准化和LASSO选择，不在全体开发集上先生成特征，也不使用训练集拟合概率冒充折外预测。

外层验证患者从未参与上述过程。LR/RF/XGBoost 每个实验使用相同外层划分；第二层 C=1 预先固定，passthrough=False；stacking内部3折；LASSO内部5折/30λ。所有基模型参数见config.yaml，未依据当前6折结果重新挑选。

对比的新静态基线与扩展共享完整 Pipeline。它遵循 V3 的 λ.1se 原则，但 LASSO内部折也重新预处理，并采用固定基模型参数，不能要求与 V3 旧表逐位一致。''',
         ['static_lr', 'static_rf', 'static_xgb', 'static_stacking'])
    save('03_temporal_trajectories.ipynb', 'V3：时间轨迹消融', '''比较 static_lr 与 temporal_lr，保持模型类型、外层折、固定参数和插补规则相同。

先在每个训练折计算变量缺失率，再训练插补、缩放和 LASSO。只为 V3 静态白细胞分类保留既定豁免，时间斜率缺失过多仍被剔除。不会为了凑完整轨迹删除患者。

差值表示变化幅度；斜率表示每小时变化速度；SD表示波动；前后半天差值表示早晚阶段变化；count 表示测量强度。静态均值/min/max与轨迹可能相关，LASSO λ.1se 为可对照的主分析，不保证时间特征一定提高性能。''', ['static_lr', 'temporal_lr'])
    save('04_comorbidity_embedding.ipynb', 'V3：既往合并症 ICD embedding', '''三组消融：static_lr → burden_lr → embedding_lr。

burden 包括既往住院次数、不同确切码数、诊断事件数和有无既往诊断；前三个用log1p压缩。embedding_lr 在 burden_lr 上增加 TF-IDF + TruncatedSVD（最多32维），以区分“诊断更丰富”与“潜在表示”带来的增益。

词表、IDF、SVD和缩放只在训练子集中拟合，验证集只 transform。没历史、空词表、仅一个词均有保留患者的回退逻辑。未知验证码不会增加词表。

这属于结构化诊断表示，不是 clinical notes；当前没有临床文本数据。静态 CLIN 共病仍继承 V3 的既往∪索引慢性编码假设，新增 embedding 完全使用既往已结束住院。后续部署仍需说明原静态编码在24小时时是否可得。''',
         ['static_lr', 'burden_lr', 'embedding_lr'])
    save('05_combined_extensions.ipynb', 'V3：三类扩展联合模型', '''比较 static_stacking 与 combined_stacking，分别使用静态特征和静态+轨迹+既往ICD表示。

联合模型每个基模型的训练 Pipeline 都重新拟合自身的轨迹筛选、插补、LASSO、TF-IDF、SVD及缩放，stacking使用3折交叉拟合，基模型不重复调参。不能先在完整开发集生成 embedding 再堆叠。

这是探索性的联合消融。必须先看独立时间/embedding实验，再判断联合模型是否改善AUROC、AUPRC、Brier、校准和DCA。当前不使用已看过的固定留出集选择最优扩展，也没有外部验证结论。''',
         ['static_stacking', 'combined_stacking'])


if __name__ == '__main__':
    main()
