# Public data provenance

Source: **South German Credit**, UCI Machine Learning Repository (2019),
[DOI: 10.24432/C5X89F](https://doi.org/10.24432/C5X89F).
[Official record](https://archive.ics.uci.edu/dataset/522/south+german+credit).
The original archive and its three members are retained under `raw/` so a clone
can reproduce this study offline. `data.py` downloads the official archive if
absent and checks the SHA-256 of each member. A changed source fails loudly.

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit also belongs to Ulrike Grömping, *South German Credit Data: Correcting a
Widely Used Data Set*, Report 4/2019, Beuth University of Applied Sciences Berlin.
[Original technical report](https://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf).

The released file has 21 columns: **20 predictors plus the target**, despite
the current landing-page metadata listing 21 features. Source `kredit=0` means
bad; `kredit=1` means good. We explicitly create `bad_credit=(kredit==0)` and
rename predictor columns using the supplied codebook. No observations are removed.
There are 300 bad and 700 good observations, drawn from 1973-1975 with bad credits
deliberately oversampled. Their 30% share is a sample-design property, not a
population default-rate estimate. The outcome is contract compliance, not a
specified modern regulatory default definition with a prediction horizon.

Only duration (months), age (years), and amount (transformed historical amount)
are continuous. The amount's monotonic transformation is unknown, so we do not
interpret its unit as a usable currency exposure. Other integer codes represent
categories or discretised quantities. Sex cannot be reliably recovered from
`personal_status_sex`; in particular code 2 combines different groups.

The original-file route is an explicitly permitted alternative to `ucimlrepo`:
it preserves the codebook alongside the data and avoids ambiguity in metadata.
Python's ZIP reader supports this archive's BZip2 compression; some Windows ZIP
utilities do not. No employer or private borrower data is used.
