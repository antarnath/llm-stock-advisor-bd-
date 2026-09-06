# Report Maintenance Protocol — `report/main.tex`

This document is the **canonical instruction set for every agent
(Claude / Cursor / human contributor) that works on this repo.** Its
purpose is to keep `report/main.tex` continuously aligned with the
artifacts produced by the rest of the project (CSVs, plots, summary
reports, log files, model checkpoints).

If you change anything under `results/`, `data/`, `models/`, `logs/` or
`scripts/`, **you must also update `report/main.tex` (and any tables /
figures it references) in the same change set.** A user-visible
`REPORT_MAINTENANCE.md` checklist is provided at the bottom of this
file.

---

## 1. What the report sources from

The report is **evidence-grounded** — every quantitative claim is
backed by a file under the repo:

| Report section | Source artifact(s) |
|---|---|
| §3 Methodology | `src/training/*.py`, `docs/PROGRESS.md`, `scripts/run_pipeline.py` |
| §4.2 Dataset description | `data/raw/stocks/`, `data/raw/indices/`, `results/sentiment/news_scored.csv` |
| §4.4 Experimental setup | `requirements.txt`, `src/utils/config.py` |
| §4.5 Hyper-parameters | `src/training/baseline_trainer.py`, `src/training/deep_learning_trainer.py`, `src/training/multimodal_trainer.py` |
| §4.7 Phase 3 results | `results/baseline/baseline_results_v2.csv`, `results/baseline/summary_report.txt` |
| §4.7 Phase 4 results | `results/deep_learning/deep_learning_results.csv`, `results/deep_learning/summary_report.txt` |
| §4.7 Phase 6 results | `results/sentiment/correlation_summary.json`, `results/sentiment/summary_report.txt`, `results/sentiment/correlation_per_stock.csv` |
| §4.7 Phase 7 results | `results/multimodal/multimodal_results.csv`, `results/multimodal/summary_report.txt` |
| §4.7 Phase 3 vs Phase 4 vs Phase 7 ablation | `results/deep_learning/phase3_vs_phase4_comparison.csv` |
| §4.7 Plots | `report/figures/*.png` (mirrored from `results/*/plots/*.png`) |
| §5 Conclusion / Future Work | `docs/PROGRESS.md`, `docs/phases/phase_*xxx_*.md` |

The **single source of truth** is the on-disk artifact. If the report
disagrees with the artifact, **the artifact wins** and the report must
be fixed.

---

## 2. What an agent MUST do when artifacts change

### 2.1 When a phase script regenerates a CSV

1. Re-run the canonical summary command:
   ```bash
   python scripts/phase_summary.py --phase <phase_id>
   ```
   (If this script does not exist yet, the convention is to read the
   CSV directly with `pandas` and copy the aggregate metrics into the
   report.)
2. Open `report/main.tex` and update the matching `\begin{table}` /
   inline numbers in §4.7.
3. If a stock is added / removed, also update §4.2 (dataset
   description) and any `n=30` claims throughout the document.

### 2.2 When a plot changes

1. Re-copy the plot to `report/figures/` with a stable, descriptive
   name:
   ```bash
   cp results/<phase>/plots/<file>.png \
      report/figures/<phase>_<short_description>.png
   ```
   The **file name must not change silently** because `\includegraphics`
   references break. If you rename, update every `\includegraphics{...}`
   call in `report/main.tex`.
2. Update the figure caption if the content has shifted in meaning
   (not just style).

### 2.3 When a new phase is added (e.g. Phase 8)

1. Add a new subsection to §4.7 following the existing pattern.
2. Add the new phase's table in the §4.7 aggregate-comparison table.
3. Add the new phase's plots under `report/figures/`.
4. Add any new top-of-funnel numbers to the Abstract and the §1.3
   objectives list.
5. Add an entry to `docs/PROGRESS.md` describing the new phase; the
   report's §5 Future Work should mirror that entry.

### 2.4 When a metric definition changes

If you change, say, Dir\_Acc to use a 5-day rolling window:

1. Update §4.6 *Model Performance Evaluation Metrics*.
2. Update every numeric table in §4.7.
3. Update §5.1 *Conclusion* if the headline interpretation changes.
4. Add a `CHANGELOG.md` entry under `report/` describing the change.

### 2.5 When code / library versions change

Update §4.4 *Experimental Setup* (Hardware and Software Environment).

---

## 3. Style rules for `report/main.tex`

These rules exist to keep the report consistent across agents:

1. **British English spellings** (`colour`, `behaviour`,
   `modelled`) are used throughout. Existing sections follow this;
   preserve it.
2. **Numbers.** Always report averages across 30 stocks to 6 decimals
   for RMSE/MAE and 2 decimals for percentages, unless the table
   explicitly says otherwise.
3. **Tables.** Use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`)
   — never vertical rules.
4. **Figures.** Always include a caption with `\label{fig:...}` and a
   reference in the text body. Never leave a dangling figure.
5. **Citations.** Add a `\bibitem{key}` entry to the
   `\begin{thebibliography}` block when you cite a new source.
6. **Acronyms.** Define an acronym with `\acrodef{...}[...]{...}` if
   you introduce a new one.
7. **Sectioning depth.** The blueprint defines up to three levels
   (`section`, `subsection`, `subsubsection`). Do not introduce
   `paragraph` headings.

---

## 4. Build & verify

The report compiles with `pdflatex` (twice) or `latexmk`:

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # second pass for the TOC
```

If `pdflatex` is not installed locally, use a Docker image:

```bash
docker run --rm -v "$(pwd)/..":/src -w /src/report \
    texlive/texlive:latest pdflatex -interaction=nonstopmode main.tex
```

### CI suggestion

Add a `make report` target at the project root that runs the two
`pdflatex` passes and fails the build if any `LaTeX Warning: Reference
... undefined` appears. The reference graph that *must* resolve:

- `\ref{tab:dataset}`, `\ref{tab:phase3}`, `\ref{tab:phase4}`,
  `\ref{tab:phase6}`, `\ref{tab:phase7-fusion}`, `\ref{tab:headline}`,
  `\ref{tab:env}`, `\ref{tab:rf}`, `\ref{tab:lr}`, `\ref{tab:lstm}`,
  `\ref{tab:mm}`.
- `\ref{fig:baseline-dist}`, `\ref{fig:baseline-comp}`,
  `\ref{fig:baseline-best}`, `\ref{fig:lstm-summary}`,
  `\ref{fig:lstm-curves}`, `\ref{fig:sent-labels}`,
  `\ref{fig:sent-by-sector}`, `\ref{fig:sent-by-event}`,
  `\ref{fig:sent-confusion}`, `\ref{fig:sent-corr}`,
  `\ref{fig:phase7-ablation}`, `\ref{fig:sent-contribution}`,
  `\ref{fig:mm-summary}`.
- Cross-chapter references: `\ref{ch:lit}`, `\ref{ch:method}`,
  `\ref{ch:results}`, `\ref{ch:conclusion}`.
- Cross-section references: `\ref{sec:metrics}`, `\ref{sec:v2-leak-free}`,
  `\ref{sec:phase7-results}`, `\ref{sec:phase7-inference}`.

---

## 5. Per-change checklist (paste into your PR description)

```text
[report/main.tex maintenance]

- [ ] Affected section(s): §
- [ ] Re-generated table(s): §
- [ ] Re-copied plot(s): §
- [ ] Cross-reference IDs updated (§\ref{...})
- [ ] `pdflatex main.tex` passes twice with no warnings
- [ ] Cross-checked headline numbers against source CSV(s)
- [ ] `report/CHANGELOG.md` updated
```

---

## 6. Triggering the maintenance automatically

The simplest way to keep agents honest across sessions is to add a
**pre-commit hook** that runs the build:

```yaml
# .pre-commit-config.yaml (excerpt)
repos:
  - repo: local
    hooks:
      - id: report-build
        name: Build report/main.tex
        entry: bash scripts/build_report.sh
        language: system
        pass_filenames: false
```

where `scripts/build_report.sh` is:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../report"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
```

If the build breaks, the agent's commit is rejected and the
maintenance protocol kicks in.

A second (optional) hook inspects the diff and refuses the commit if
a CSV under `results/` changes without an accompanying edit to
`report/main.tex`:

```bash
#!/usr/bin/env bash
# scripts/check_report_aligned.sh
set -euo pipefail
git diff --cached --name-only | grep -q '^results/' || exit 0
git diff --cached --name-only | grep -q '^report/main.tex' \
    || { echo "results/ changed but report/main.tex not updated"; exit 1; }
```

---

## 7. File inventory

| File | Purpose | Updated by |
|---|---|---|
| `report/main.tex` | The thesis itself. | Every agent that touches `results/`, `data/`, `models/`, `scripts/`, `docs/`. |
| `report/figures/` | Plot copies referenced by the report. | Re-copied when source plots change. |
| `report/tables/` | Optional helper tables (CSV→`.tex` converters). | Reserved for future. |
| `report/CHANGELOG.md` | Section-level log of what changed. | Every report update. |
| `report/REPORT_MAINTENANCE.md` | This file. | Rarely — only when the protocol itself evolves. |

---

## 8. Final word

The thesis is a **living document**. The report does not get
re-generated at the end of the project; it gets re-generated
**continuously** as the underlying experiments evolve. If you find
yourself changing a CSV, a plot, a script or a result, ask yourself:
*"does the report still say the truth?"* If not, fix the report in
the same change set.