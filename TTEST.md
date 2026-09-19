# Classical t-tests

Entry point: **Exploratory Data Analysis → Comparison → t-Test**.

Select quantitative numerical variables using Comparison's existing checkboxes. Select a categorical variable for stacked independent samples. Independent separate columns and paired samples need two distinct numerical variables. Study design is explicit: two columns do not imply pairing.

## Controls and reporting

- One-Sample t-Test: numerical response and finite hypothesized mean.
- Two Independent Samples t-Test: stacked response/group with two explicitly selected categories (including selections from more than two categories), or separate numerical sample columns. **Assume equal variances** defaults off (Welch). On selects the pooled Student test; no preliminary variance test switches methods.
- Paired Samples t-Test: two numerical columns, one matched pair per row.
- Two-sided, greater and less alternatives apply to **Sample 1 - Sample 2**, **Variable 1 - Variable 2**, or **sample mean - hypothesized mean**. Reversing roles reverses the directional hypothesis.
- Confidence level defaults to 95%, configurable from 80% to 99.9%. Significance alpha tracks 1 - confidence. Greater gives a lower confidence bound with +infinity above; less gives -infinity below and an upper bound.

Tabs: Test Configuration (study design, hypotheses and row accounting), Descriptive Statistics, Diagnostics, t-Test Results, Interpretation. Descriptive and diagnostic information remains visible when a test cannot be estimated. CSV exports provide descriptive statistics and full-precision test results with configuration, subtraction direction, bounds and effect-size definition.

## Calculations

SciPy's `ttest_1samp`, `ttest_ind`, `ttest_rel` provide t, df, p and alternative-aware `confidence_interval` outputs. For one-sample tests SciPy returns an interval for the population mean; subtract the hypothesized mean from both bounds to obtain the requested difference interval. Sample SD uses ddof=1.

For independent groups, let n1/n2 be sample sizes and v1/v2 sample variances:

- Welch SE = sqrt(v1/n1 + v2/n2).
- Welch df = (v1/n1 + v2/n2)^2 / ((v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1)).
- Pooled variance = ((n1-1)*v1 + (n2-1)*v2)/(n1+n2-2).
- Pooled SE = sqrt(pooled variance * (1/n1 + 1/n2)), df = n1+n2-2.

Standardized effects:

- One sample: Cohen's d = (mean - hypothesized mean)/sample SD.
- Paired: Cohen's dz = mean paired difference/SD of paired differences.
- Independent: Cohen's d = (mean1 - mean2)/sqrt(pooled variance), for both methods. Under Welch this pooled SD is a descriptive standardizer only; inference still uses unequal variances. This convention depends on relative sample sizes and must be interpreted with care under unequal variances. The formula is displayed alongside results and documented in code. No automatic small/medium/large labels, Hedges correction, or effect-size confidence intervals are supplied.

## Cleaning, semantic types, and validation

Only selected Numerical → Discrete/Continuous variables are candidates. Categorical/Binary overrides are respected even if storage was integer. Boolean, complex, datetime and nonnumeric responses are excluded. There is no Identifier subtype in the app; a conservative identifier-name screen (ID/identifier tokens or an uppercase ID suffix) prevents automatic inclusion. A deliberate Numerical override allows such a column. Identifier detection is a heuristic, not a guarantee of measurement suitability.

One sample drops missing/nonfinite responses. Stacked samples drop missing groups and missing/nonfinite responses, and separately count valid rows belonging to unselected groups. Separate independent columns drop missing/nonfinite values separately; unequal sample sizes are retained. Paired tests use complete rows only and describe both original variables on these same pairs. Outlier row positions refer to original input positions (1-based), so duplicate dataframe indexes are safe.

Require at least two usable observations in each independent sample or two complete pairs. One-sample values and paired differences require positive variance. Independent tests require positive finite combined variation; one constant group is estimable if the other varies, with a caution. Both constant groups are blocked. Invalid columns, duplicate roles, absent groups, nonfinite null means, invalid confidence levels, numerical overflow or nonfinite results produce user-friendly messages. No raw exception text is exposed for unexpected errors.

## Assumptions and diagnostics

No Shapiro-Wilk or other formal normality test is run. Diagnostics include sample size, skewness, the existing histogram/box plot, a normal Q-Q plot and the existing 1.5 x IQR outlier method. No flagged observations are automatically removed. Small N and substantial skewness are cautions rather than gates. A Q-Q plot needs at least three observations and nonconstant values.

One-sample diagnostics concern the response; independent diagnostics concern each selected group; paired diagnostics concern differences only. Inference about means can be reasonably robust to moderate non-normality with adequate samples, but small samples, severe skewness and extreme values need judgment. Independence cannot be checked automatically. Independent designs require independence within and between samples; paired designs require independence between pairs. Users receive an explicit study-design note, not an automatic independence test.

Results emphasize the signed estimate, uncertainty, numerical effect size and strength of evidence. Insufficient evidence is not proof of equal means; the test does not establish causation. No automatic multiple-comparison correction or nonparametric alternatives are included.

## Architecture and verification

- `utils/ttest_analysis.py`: semantic candidates, validation, cleaning/accounting, summaries, diagnostics and classical tests.
- `utils/ttest_reporting.py`: result tables, directional interpretation and precision-preserving export.
- `utils/ttest_ui.py`: controls, five result tabs and charts, reusing chart styling, formatting and outlier helpers.
- `utils/comparison_ui.py`: selection-driven navigation.
- `tests/test_ttest.py`: deterministic numerical checks against SciPy and independent statsmodels confidence intervals, plus explicit SE/df/effect formulas; all alternatives, Welch/pooled, unbalanced sizes/variances, missingness, constants, groups, role reversal, types/overrides, outliers and export.
- `tests/test_ttest_ui.py`: all modes, chart rendering, narrowing selections, stale roles/groups, nonnumeric semantic exclusion and constant-data messages.

Run `python -m unittest discover -s tests -p "test_*.py"` using the project dependencies. No new dependency is required; supported SciPy >=1.11 already provides confidence intervals.

## Library references

- [SciPy one-sample t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)
- [SciPy independent t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
- [SciPy paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)

Verification completed: all 110 application tests passed, including the new numerical and UI cases. A full-app Streamlit AppTest run exercised one-sample, independent stacked and separate-column, paired tests, charts, and the existing ANOVA flow with database tracking mocked.
