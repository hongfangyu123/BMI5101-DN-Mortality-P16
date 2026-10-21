# LaTeX version of the paper

- `main.tex`: preamble (pdfLaTeX, Times, APA 7 via biblatex-apa + biber), title, abstract, keywords.
- `abstract.tex`, `body.tex`, `supplementary.tex`: `body.tex` and `supplementary.tex` are generated from `paper_v2_en.md` (in this folder) by `python md2tex.py` (needs pandoc); `abstract.tex` is edited by hand. `numbers_v2.md` is the ledger tracing every number to its source. Edit the Markdown and regenerate, or edit these files directly once the Markdown is frozen.
- `references.bib`: all 26 references with DOIs where they exist.
- `figures/`: Figure 1–2 (drawn from report numbers), Figure 3 panels (copied from results/figures). Figure 4a is figC6 from the local run of scripts/04k_shap_clin.py.

Overleaf: upload `dn_paper_latex.zip`, set the compiler to pdfLaTeX; biber runs automatically.
Local: `latexmk -pdf main.tex`.
