# AGENTS.md — Project rules for AI assistants (Claude, Cursor, …)

This file is read automatically by AI coding assistants whenever they
join this project. **Read it first; obey its rules for the entire
session.** The goal is to keep the thesis (`report/main.tex`) in lock-
step with the underlying code, data and results.

---

## 1. Always read these files first

```
report/REPORT_MAINTENANCE.md      ← authoritative maintenance protocol
docs/PROGRESS.md                  ← current phase + status
overview.md                       ← high-level project overview
report/CHANGELOG.md               ← last few report changes
```

Do not skip this. If you are about to change anything under
`results/`, `data/`, `models/`, `scripts/`, or `docs/`, you are also
about to change the report. Plan for both.

---

## 2. Hard rules

1. **Never edit `report/main.tex` in isolation.** A change to the
   report that is not accompanied by the underlying experiment is a
   fabrication and must not happen.
2. **Always cite a source artifact.** Every number in the report must
   trace to a CSV, JSON, PNG or log file under the repo. If it does
   not, either add the source or remove the claim.
3. **Never use `today_close` (or any current-day indicator) as a
   feature when the target is next-day return.** This is the v1
   leakage bug; reproducing it invalidates the entire thesis.
4. **Always preserve the file-naming convention for figures.** Plots
   in `report/figures/` are referenced from `main.tex` by their
   prefixed name (`baseline_*`, `lstm_*`, `mm_*`, `sent_*`). Renaming
   breaks the build. If you must rename, do it in one commit
   together with the LaTeX update.
5. **Run `python scripts/report/phase_summary.py` before committing**
   to verify your numbers match the report.
6. **Use the pre-commit hook** at `scripts/report/check_report_aligned.sh`.
   It will refuse a commit that touches `results/` without also
   touching `report/main.tex`.

---

## 3. Common tasks — recipes

### "I just re-ran Phase X and got new metrics"

1. Re-run:
   ```bash
   python scripts/report/phase_summary.py --phase X
   ```
2. Compare the printed values to the corresponding table in
   `report/main.tex` §4.7.
3. If anything changed, edit the table in place; do not invent new
   metrics.
4. Re-copy the plots:
   ```bash
   bash scripts/report/sync_figures.sh
   ```
5. Rebuild the PDF (if pdflatex is available):
   ```bash
   bash scripts/report/build_report.sh
   ```
6. Update `report/CHANGELOG.md` with a one-line entry describing the
   change.

### "I'm adding a new phase"

See §2.3 of `report/REPORT_MAINTENANCE.md`. The minimum is:
- New §4.7 subsection.
- New entry in the §4.7 aggregate-comparison table.
- New plots under `report/figures/`.
- Update §1.3 objectives and §5 Future Work.

### "I want to know what's left to do"

```bash
cat docs/PROGRESS.md        # current phase + status
ls docs/phases/             # 16 phase specs + extensions
```

---

## 4. Forbidden behaviours

- Adding a metric to the report without regenerating the experiment.
- Editing the bibliography without verifying the source actually exists.
- Changing the headline RMSE without changing every other place that
  number is cited (table, abstract, conclusion, executive summary).
- Committing `report/main.tex` that fails `pdflatex`.
- Copy-pasting prose from another thesis without attribution.

---

## 5. Contact

Owner: Antar Chandra Nath. Open an issue on the GitHub tracker if any
of the rules above are unclear or impossible to follow.