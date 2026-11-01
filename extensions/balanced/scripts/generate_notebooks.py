"""Generate notebooks with method descriptions (originally in Chinese, translated to English) and run switches; the training code in them is not executed."""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks'
SETUP = '''from pathlib import Path
import sys
import pandas as pd

# Locate the project from the extension directory or the notebooks subdirectory, to avoid hard-coding a Windows drive letter.
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "dn_extensions" / "settings.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("start from dn_mortality_repro_V3_extensions or its notebooks subdirectory.")
sys.path.insert(0, str(ROOT))
print("extension directory:", ROOT)
'''


def result_cell():
    c = nbformat.v4.new_markdown_cell('## Experiment status and conclusions (updated automatically)\n\nThe simplified version has not run any real patient experiment yet. The first 6 folds of the original strict version are saved and not mixed into this version.')
    c.metadata['extension_results'] = True
    return c


def save(name, title, method, experiments=None):
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python (ml - WSL)', 'language': 'python', 'name': 'ml-wsl'}
    nb.metadata['language_info'] = {'name': 'python'}
    nb.cells = [nbformat.v4.new_markdown_cell(f'# {title}\n\n{method}'), nbformat.v4.new_code_cell(SETUP)]
    if experiments is None:
        nb.cells.append(nbformat.v4.new_code_cell('''# By default only show the description and do not scan the database. Set to True to prepare features.
RUN_FEATURE_PREPARATION = False
if RUN_FEATURE_PREPARATION:
    from dn_extensions.features import build_features
    from dn_extensions.reporting import update_preparation_notebook
    manifest = build_features()
    update_preparation_notebook(manifest)
else:
    print("balanced shares the features prepared by the parent directory, no need to rerun prepare; if features are missing, run prepare in the parent directory.")
'''))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(f'''# pilot is a small 1×3 outer-fold trial run; for the formal experiment set PROFILE to "full".
# Starts only when RUN_EXPERIMENT is set to True; full may take a long time.
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
    print("model training in this notebook has not been started.")
'''))
        nb.cells.append(nbformat.v4.new_code_cell('''# Show the formal full results first; without results print the status, no retraining.
RESULT_PROFILE = "full"
latest = ROOT / "results" / RESULT_PROFILE / "latest.txt"
if latest.exists():
    folder = latest.parent / latest.read_text().strip()
    summary = pd.read_csv(folder / "summary.csv")
    display(summary[summary.experiment.isin(EXPERIMENTS)])
    print("detailed record:", folder / "report.md")
else:
    print("no experiment results for this mode yet. Set RESULT_PROFILE to pilot to show pilot results.")
'''))
    nb.cells.append(result_cell())
    OUT.mkdir(parents=True, exist_ok=True)
    nbformat.validate(nb)
    nbformat.write(nb, OUT / name)


def main():
    save('01_prepare_features.ipynb', 'V3: extension feature preparation', '''Reuses the V3 CLIN cohort, outcome and fixed split; processes only the development set and does not read the held-out labels.

Temporal trajectories: 0–24 hours after ICU admission, labs from the same hadm_id and vital signs from the same stay_id; last-minus-first difference, slope per hour, SD, mean of the last 12 hours minus the first 12 hours, and number of measurements. Measurements at the same time point are averaged first. With fewer than two time points the slope and difference are missing; missingness does not remove the patient.

ICD documents: only hospitalizations that ended before the index hospitalization began are included; the ICD9/10 version, the exact code and the three-character category are all kept. Patients with no prior diagnoses are kept with an empty document. This step only prepares the documents; TF-IDF/SVD is not fitted in advance.

The V3 static columns keep the original −6 to 24 hour window, so the trajectories add dynamic information rather than redefining the original baseline. Descriptive missing rates are only for understanding the data; the later missing-rate screening is done within the training folds.

The main analysis inherits the V3 20% variable missingness rule and the exemption of the 8 white-cell differential variables. Trajectory variables are not exempt; the number of measurements is an indicator of testing intensity and should not be interpreted as a causal risk factor.''')
    save('02_stacking.ipynb', 'V3: LR/RF/XGBoost stacking', '''Compares static LR, RF, XGBoost and stacking on common outer folds.

The first level is three fixed-parameter Pipelines; there is no repeated grid tuning in the stacking sub-folds. The second-level LR is trained only on the first-level predicted probabilities on the stacking inner validation folds. Each base model redoes imputation, standardisation and LASSO selection on that inner training subset; features are not generated on the whole development set first, and training-set fitted probabilities are not passed off as out-of-fold predictions.

The outer validation patients never take part in the above. Each LR/RF/XGBoost experiment uses the same outer split; the second level has C=1 fixed in advance, passthrough=False; stacking inner 3 folds; LASSO inner 5 folds/30 λ. All base-model parameters are in config.yaml and were not re-chosen on the basis of the current 6 folds.

The new static baseline used for comparison shares the complete Pipeline with the extensions. It follows the V3 λ.1se rule, but also redoes preprocessing inside the LASSO folds and uses fixed base-model parameters, so it cannot be required to match the old V3 tables digit for digit.''',
         ['static_lr', 'static_rf', 'static_xgb', 'static_stacking'])
    save('03_temporal_trajectories.ipynb', 'V3: temporal-trajectory ablation', '''Compares static_lr with temporal_lr, keeping the model type, outer folds, fixed parameters and imputation rules the same.

Variable missing rates are computed first in each training fold, then imputation, scaling and LASSO are trained. The established exemption is kept only for the V3 static white-cell differential; temporal slopes with too much missingness are still dropped. No patient is removed to obtain complete trajectories.

The difference measures the size of change; the slope the rate of change per hour; SD the fluctuation; the late-minus-early half-day difference the change between early and late phases; count the measurement intensity. Static mean/min/max may be correlated with the trajectories; LASSO λ.1se is the comparable main analysis, and temporal features are not guaranteed to improve performance.''', ['static_lr', 'temporal_lr'])
    save('04_comorbidity_embedding.ipynb', 'V3: prior-comorbidity ICD embedding', '''Three ablation groups: static_lr → burden_lr → embedding_lr.

burden includes the number of prior hospitalizations, the number of distinct exact codes, the number of diagnosis events and whether there is any prior diagnosis; the first three are compressed with log1p. embedding_lr adds TF-IDF + TruncatedSVD (at most 32 dimensions) on top of burden_lr, to separate the gain from "richer diagnoses" from that of the "latent representation".

The vocabulary, IDF, SVD and scaling are fitted only on the training subset; the validation set is only transformed. No history, an empty vocabulary and a single-term vocabulary all have fallback logic that keeps the patient. Unknown codes in validation do not extend the vocabulary.

This is a structured diagnosis representation, not clinical notes; there are currently no clinical text data. The static CLIN comorbidities still inherit the V3 assumption of prior ∪ index chronic codes, while the new embedding uses only prior hospitalizations that have ended. Deployment would still need to state whether the original static codes are available at 24 hours.''',
         ['static_lr', 'burden_lr', 'embedding_lr'])
    save('05_combined_extensions.ipynb', 'V3: combined model of the three extension types', '''Compares static_stacking with combined_stacking, using static features and static + trajectory + prior-ICD representations respectively.

In the combined model, the training Pipeline of every base model refits its own trajectory screening, imputation, LASSO, TF-IDF, SVD and scaling; stacking uses 3-fold cross-fitting and the base models are not re-tuned. The embedding must not be generated on the complete development set first and then stacked.

This is an exploratory combined ablation. The separate temporal/embedding experiments must be looked at first, before judging whether the combined model improves AUROC, AUPRC, Brier, calibration and DCA. The fixed held-out set that has already been looked at is not used to choose the best extension, and there is no external validation conclusion.''',
         ['static_stacking', 'combined_stacking'])


if __name__ == '__main__':
    main()
