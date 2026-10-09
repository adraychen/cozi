# Guided One-way ANOVA

EDA → Comparison → One-way ANOVA uses one quantitative numerical response and one categorical factor. Welch is the only method. Neither diagnostic p-values nor AI can switch methods or bypass numerical validation. Factorial ANOVA is unchanged.

## Implementation files

| Files | Change |
| --- | --- |
| `utils/anova_analysis.py` | Welch-only validation, SS decomposition, variance-weighted effect size, Games–Howell engine and group mean intervals |
| `utils/anova_independence.py` | New evidence-based identifier, cluster and chronological screening |
| `utils/anova_charts.py` | New shared-result group mean, pairwise forest and residual plots |
| `utils/anova_ui.py` | Guided results, diagnostics, interpretation, conditional design clarification, CSV and report integration |
| `utils/ai_interpreter.py` | Compact-summary AI explanation with no authority over eligibility |
| `utils/comparison_ui.py`, `utils/dataset_workspace.py` | Quantitative-response eligibility and safe handling of the new AI button state |
| `tests/test_anova.py`, `tests/test_comparison_ui.py` | Statistical reference and UI verification; automatic Welch workflow expectations |
| `README.md`, `docs/ui-controls.md`, `ANOVA.md` | Updated methods, controls, formulas, privacy and limitations |

## Statistical calculations

Welch F and its adjusted denominator degrees of freedom use `statsmodels.stats.oneway.anova_oneway(..., use_var='unequal', welch_correction=True)`. Every included group needs at least two observations and positive finite variance. Missing/nonfinite responses and missing factors are excluded with counts; singleton/constant groups are not silently discarded. Groups without any usable response are reported explicitly.

The unequal-variance Cohen f² plug-in estimate uses `effectsize_oneway(..., use_var='unequal', ddof_between=0)`: weights wᵢ = nᵢ/sᵢ², weighted mean μw = Σwᵢȳᵢ/Σwᵢ, f² = Σwᵢ(ȳᵢ−μw)²/N. It is Welch-compatible, not bias corrected, not conventional eta-squared or a percentage of explained variation, and not derived from the Welch-corrected F. Interpretation depends on the measurement scale and study context.

The **Variation Decomposition** table is descriptive: SS between = Σnᵢ(ȳᵢ−ȳ)², SS within = ΣΣ(yᵢⱼ−ȳᵢ)², and SS total is independently calculated as Σ(y−ȳ)². The identity SS between + SS within ≈ SS total is verified. DF are k−1, N−k, N−1; total MS is shown as an em dash. No classical F is calculated from this table.

For k ≥ 3, **Games–Howell** runs for all pairs without an overall-significance gate. Difference = mean A − mean B; SE = √(sA²/nA + sB²/nB); df = (sA²/nA+sB²/nB)² / [(sA²/nA)²/(nA−1)+(sB²/nB)²/(nB−1)]. q = √2|difference|/SE. Adjusted p-values use the studentized-range survival function with k groups and pair-specific df. Intervals are difference ± q(1−α, k, df)SE/√2. Significance uses adjusted p < α. Family confidence is 1−α (95% by default); all pairs remain in the family when filtering the plot.

The pairwise table, CSV and forest plot share exactly these calculations. Positive differences favor A; negative differences favor B. Color and marker shape distinguish significant comparisons. Plot pages contain at most 20 comparisons. Computational work is limited to 30 groups/435 pairs; above that, Welch can still run and an explicit Games–Howell unavailable message is shown. No subsets are silently substituted. Pairwise computational failure yields no fabricated p-values or intervals.

Group mean intervals are separate, pointwise 95% Student-t intervals: mean ± t(0.975,n−1)SD/√n. They are not the simultaneous Games–Howell intervals. Group boxplots are retained, with residual plots and Q–Q plots in Diagnostics. Shared Plotly rendering retains responsive sizing and image-export controls.

## Diagnostics and study design

Group summaries include sizes, variances, SD, skewness, excess kurtosis, missing counts and 1.5 × IQR flags. Size imbalance above 4:1, groups below 20 observations, |skewness| ≥ 2 and |excess kurtosis| ≥ 7 are screening cautions, not eligibility thresholds or formal normality decisions. Median-centered Levene describes variance differences. No Shapiro–Wilk or automatic Kruskal–Wallis switch is used.

Independence assessment has three states: **No dependence indicators detected**, **Potential dependence detected**, and **Insufficient information**. Named entity/ID and cluster/batch columns are inspected using actual analyzed-row coverage and repetition. Repeated category labels, duplicated response values or repeated labels alone do not prove dependence. A unique candidate identifier provides structural information without establishing independent entities. Files without usable design information proceed under the independence assumption, explicitly qualified.

A single unambiguous chronological column permits within-group serial screening only for ≥12 complete observations with distinct, approximately equally spaced times. Data is sorted by that chronology; arbitrary file order is never tested. Ljung–Box p-values are Bonferroni-adjusted across tested groups. Adjusted p < 0.05 plus |lag-1 correlation| ≥ 0.4 flags potential serial dependence. Durbin–Watson is descriptive. Ties, irregular spacing, small groups and multiple plausible time columns produce explanations instead of forced tests. Nonsignificant results never verify independence.

Meaningful repeated/clustered or serial indicators withhold ordinary inference until the user clarifies that observations represent independent units. The detected evidence and qualification remain visible even after clarification. Repeated-measures, mixed-effects or time-aware models are recommended where relevant; they are not implemented automatically. Clarification resets when the dataset or variable/data context changes. Other numeric failures cannot be overridden.

## Guidance, privacy and reports

Deterministic overall, pairwise, variation and effect-size explanations work without AI. Optional AI interpretation uses the existing Groq infrastructure with bounded group/result/diagnostic and column-metadata summaries; no individual records, IDs or timestamps are sent. Instructions prohibit fabricated values, eligibility overrides, assumption certification or causal claims. Errors leave statistical outputs available; changed result contexts invalidate cached explanations. AI explanations remain advisory.

Report capture preserves Welch numeric results, descriptive variation, group/diagnostic/pairwise tables, plots, qualifications and any current AI interpretation. Pairwise CSV contains full precision, group names, SE and df in addition to the displayed comparison columns.

## References and verification

- [statsmodels Welch ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html)
- [statsmodels unequal-variance effect size](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.effectsize_oneway.html)
- [SciPy Games–Howell reference via `tukey_hsd(equal_var=False)`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html)
- [Ljung–Box diagnostic](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html)

`tests/test_anova.py` checks Welch against SciPy and independent df/effect formulas, Games–Howell adjusted p-values and intervals against SciPy, balanced/unbalanced/heteroskedastic groups, SS decomposition, signs and zero-crossings, cleaning, invalid/constant/extreme values, alpha handling, plot/table agreement, independence states/serial order, AI failure/privacy, pagination, dataset changes and report snapshots. Existing Comparison and Factorial ANOVA tests are included in the full suite.
