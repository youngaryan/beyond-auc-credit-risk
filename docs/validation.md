# Completed validation

Validated on 27 October 2026 using Python 3.12.10 on Windows. Exact package versions
are in requirements-lock.txt and results/run_manifest.json.

- Editable installation with development dependencies succeeded using the documented command.
- Full experiment and presentation pipeline completed, including the locked holdout and 2,000 bootstrap replicates.
- Two complete model runs produced identical bytes for all seven compared numerical artifacts: metrics, CV results, CV predictions, holdout predictions, bootstrap intervals, tuning results and logistic coefficients.
- Final test suite: 28 passed, 0 failed, 0 errors, 0 skipped.
- Ruff checks passed after formatting.
- Artifact verification passed: 16 PDF pages, 14 numbered figures, PNG and SVG pairs, and 32 hashed report/figure outputs.
- Every PDF page was rendered with Poppler and inspected. The final prevalence-table header was adjusted and re-inspected; figures, equations, captions and page footers are legible without overflow.
- All local links in the README and research report resolve.
- The final run's Python source hashes match the delivered source files.

The CI workflow is supplied for GitHub, but no remote CI execution or publication
was performed. These checks establish local software and artifact consistency,
not independent model validation or portfolio suitability.
