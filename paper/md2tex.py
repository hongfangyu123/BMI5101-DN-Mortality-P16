import re, subprocess, sys
import os, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
# source Markdown: next to this script if present, otherwise one level up
SRC = os.path.join(HERE, "paper_v2_en.md")
if not os.path.exists(SRC): SRC = os.path.join(HERE, "..", "paper_v2_en.md")
OUT = os.path.join(HERE, "body.tex")
TMP = tempfile.mkdtemp()
s = open(SRC).read()

KEYS = {1:"popbusui2017",2:"feldman2019",3:"huang2025",4:"johnson2023",5:"tibshirani1996",6:"lundberg2017",7:"collins2015",
        8:"varma2006",9:"cawley2010",10:"quan2005",11:"nadeau2003",12:"delong1988",13:"vancalster2019",14:"vickers2006",
        15:"bazick2011",16:"tani2012",17:"vanhouwelingen2007",18:"friedman2010",19:"yuan2012",20:"breiman2001",21:"chen2016",
        22:"cortes1995",23:"platt1999",24:"sun2014",25:"steyerberg2010",26:"youden1950"}

# keep only the body between "## 1. Introduction" and "## References"; abstract is handled in main.tex
body = s[s.index("## 1. Introduction"):s.index("## References")]
supp = s[s.index("## Supplementary material"):]

body = re.sub(r"\s*\*\*\[verify\]\*\*", "", body)
body = re.sub(r"\s*\[verify\]", "", body)
body = body.replace("Huang et al. [3]", r"\textcite{huang2025}").replace("Sun and Xu [24]", r"\textcite{sun2014}")
body = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", lambda m: r"\parencite{" + ",".join(KEYS[int(x)] for x in m.group(1).split(",")) + "}", body)

# headings: drop manual numbers, LaTeX numbers them
body = re.sub(r"^(#{2,3}) \d+(?:\.\d+)?\.?\s+", r"\1 ", body, flags=re.M)
body = body.replace("## Supplementary material", "## Supplementary material {.unnumbered}")
body = re.sub(r"^---\s*$", "", body, flags=re.M)

# table captions: "**Table N.** caption" before a table -> pandoc "Table: caption" after it
def move_caption(text):
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        m = re.match(r"\*\*Table \d+\.\*\*\s*(.*)", lines[i])
        if m:
            cap = m.group(1); i += 1
            while i < len(lines) and not lines[i].startswith("|"):
                if lines[i].startswith("<!-- widths"): out.append(lines[i])
                i += 1
            while i < len(lines) and lines[i].startswith("|"): out.append(lines[i]); i += 1
            out += ["", "Table: " + cap, ""]
        else:
            out.append(lines[i]); i += 1
    return "\n".join(out)
body = move_caption(body)

def set_widths(text):
    lines = text.split("\n"); i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|(\s*:?-+:?\s*\|)+\s*$", lines[i+1]):
            j = i
            while j < len(lines) and lines[j].startswith("|"): j += 1
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for k, l in enumerate(lines[i:j]) if k != 1]
            ncol = len(rows[0]); w = []
            mw = re.match(r"<!-- widths: ([\d,]+) -->", lines[i-1]) if i > 0 else None
            if mw:
                w = [float(x) for x in mw.group(1).split(",")]
            for c in ([] if mw else range(ncol)):
                body_len = max(len(re.sub(r"\\\w+\{[^}]*\}", "", r[c])) for r in rows[1:] if c < len(r))
                head_len = max(len(x) for x in rows[0][c].split())
                w.append(max(body_len, head_len + 2, 10) ** 0.8)
            tot = sum(w); lines[i+1] = "|" + "|".join("-" * max(3, round(120 * x / tot)) for x in w) + "|"
            i = j
        else:
            i += 1
    return "\n".join(lines)
body = set_widths(body)

FIG = {
 1: r"""\begin{figure}[tbp]
\centering
\includegraphics[width=\textwidth]{figures/fig1_cohort_flow.pdf}
\caption{Cohort flow. Both cohorts start from the same index admissions. The reference cohort applies the benchmark's inclusion criteria; the landmark cohort includes all adults alive and in the ICU 24\,h after admission.}
\label{fig:flow}
\end{figure}""",
 2: r"""\begin{figure}[tbp]
\centering
\includegraphics[width=\textwidth]{figures/fig2_nested_cv.pdf}
\caption{Repeated nested cross-validation. Every data-dependent step, including variable selection and tuning, is fitted inside each outer training fold and evaluated on the untouched outer validation fold.}
\label{fig:nestedcv}
\end{figure}""",
 3: r"""\begin{figure}[tbp]
\centering
\includegraphics[height=5.2cm]{figures/figC1_roc_clin_holdout.png}\hfill
\includegraphics[height=5.2cm]{figures/figC2_calibration_clin_holdout.png}\\[4pt]
\includegraphics[height=5.6cm]{figures/figC3_dca_clin_holdout.png}
\caption{Frozen landmark models on the held-out set (512 patients, 136 deaths): (a) ROC curves, (b) calibration, (c) decision curves, with dotted vertical lines marking each model's frozen Youden threshold.}
\label{fig:holdout}
\end{figure}""",
 4: r"""\begin{figure}[tbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/figC6_shap_beeswarm_clin_holdout.pdf}
\caption{SHAP summary of the frozen landmark LR (L2) on the held-out set (512 patients), log-odds scale, predictors ordered by mean $|\text{SHAP}|$. Colour shows the predictor value. For patients without a white-cell differential, the differential-derived variables take their imputed training means and appear as a cluster near zero. The corresponding plots for the reference cohort are in Supplementary S2.}
\label{fig:shap}
\end{figure}""",
}
def fig_repl(m):
    n = int(m.group(1)); return "\n" + FIG[n] + "\n"
body = re.sub(r"^\*\*Figure (\d)\.\*\*.*$", fig_repl, body, flags=re.M)

supp = re.sub(r"^## Supplementary material", "# Supplementary material", supp, flags=re.M)
supp = re.sub(r"^---\s*$", "", supp, flags=re.M)
open(os.path.join(TMP, "supp.md"),"w").write(supp)
subprocess.run(["pandoc","-f","markdown+tex_math_dollars+raw_tex","-t","latex",os.path.join(TMP, "supp.md"),"-o",os.path.join(HERE, "supplementary.tex")],check=True)
open(os.path.join(TMP, "body.md"), "w").write(body)
subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars+raw_tex+pipe_tables", "-t", "latex", "--columns=80", "--shift-heading-level-by=-1",
                os.path.join(TMP, "body.md"), "-o", OUT], check=True)
print("ok")
