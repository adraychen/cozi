# Numerical formatting audit

Audit performed against current `app.py` runtime and every Python module in `utils`, before implementation. Approved formatting is opt-in at known statistical-result sites; no blanket numeric-column guessing. All unlisted rows below require review and retain their current display. Source inventory at the end records every numeric/display site found, including archived variants.

## Editing precision

Edit `NUMERICAL_PRECISION.toml` in the project root, save, then rerun/refresh Cozy. It is an executable configuration document, with commented instructions. `[decimal_places]` controls each approved fixed-decimal type (including fractional DF and percentages); `[significant_digits]` controls coefficients and ratios. `[display] precision = 4` overrides suitable fixed-decimal types; restore `"Standard"` to use individual settings. `[display] p_value_mode = "Detailed"` enables detailed p-values. Counts remain integers; p-values stay significance-aware. These are global application display settings, not transformations. There is no new settings menu.

Examples:

```toml
[display]
precision = "Standard"
p_value_mode = "Detailed"

[decimal_places]
mean = 4
sd = 3

[significant_digits]
coefficient = 4
```

Do not add unlisted types to the configuration. Configuration errors are visible; unsupported settings do not silently change the application. Saved reports retain the display rows captured when added. Raw result frames and explicit CSV exports keep numerical values. An older generic display-only regression formatter already coerces unlisted diagnostic columns to strings; that existing behavior is retained, and raw statistical engine outputs are unchanged.

## Covered Result Types

| Module | Analysis / Feature | Result Type | Approved Format | Current Format | Action |
|---|---|---|---|---|---|
| Inspect | exploration.py; dataset_views.py; descriptive_statistics_display.py | Mean, median, min/max, quartiles/percentiles; SD; SE; mean CI | 2 dp; SD/SE 3 dp; CI 2 dp | Mostly 2 dp including SD/SE; strings; CI 2 dp | Central semantic scalar formatters; preserve raw previews |
| Inspect | quality.py; variable_overview_display.py; detector.py | Observation/missing/unique/outlier counts and percentages; mean, min/max | Integer; percentages 1 dp; mean/min/max 2 dp | Counts native; percentages 1–2 dp; mean rounded early | Format calculated summaries only; remove early summary rounding |
| Inspect / Analyze | exploration.py; temporal_ui.py; distribution_fit_ui.py | Frequency counts, percentages, used/excluded observations | Integer; percentages 1 dp | Native integers; 2 dp percentages | Explicit result schemas; preserve temporal labels |
| Transform | transform_ui.py; table_transforms.py; table_editing.py; currency_conversion.py | Preview row counts; transformation accounting counts | Integer | Native counts | Preserve all working/derived values and conversions; display counts only |
| Comparison | anova_ui.py; anova_charts.py; anova_analysis.py | Welch F; p; numerator/fractional denominator DF; Cohen f²; SS; variation percentages; group means/SD/SE/CI/N | F/f²/SS/SD/SE 3 dp; p special; DF integer or 2 dp; mean/CI 2 dp; % 1 dp | Mostly 6 significant digits; p 4 significant digits; group summaries native; plot 4 significant digits | Explicit table, chart, finding semantics; unrounded inference |
| Comparison | factorial_anova_ui.py; factorial_followup.py | Type II/HC3 F, t, p, DF, SS, MS; EMM, mean CI, SE and N | Test/effect/SS/MS 3 dp; DF integer or 2 dp; EMM/CI 2 dp; SE 3 dp | Native numeric tables; narratives 3–4 significant digits | Central formatters; preserve interaction decisions |
| Comparison | ttest_ui.py; ttest_reporting.py | Means, SD/SE, mean CI, t, DF, p, Cohen d (including pooled SD variant) | Means/CI 2 dp; SD/SE/t/d 3 dp; DF auto; p special | Generic 5 significant digits; generic regression table formatter 4 significant digits | Semantic results and narratives; raw CSV unchanged |
| Comparison | anova_effects_ui.py | Fixed coefficients, SE, z, asymptotic p; descriptive group N/mean/SD/SE/CI and percentages | Coefficient 4 significant digits; SE/z 3 dp; p special; descriptive rules | Native numbers; NumberColumn 4 significant digits for p | Fixed coefficient schema only; random variance components remain review |
| Relationship | correlation_ui.py; charts.py; dataset_views.py | Pearson/Spearman r, p, usable-pair N; correlation heatmap | r 3 dp; p special; N integer | r metric 4 dp; heatmap 2 dp; hover 3 dp; p 4 significant digits | Central matrix formatting; preserve actual correlation values in plots |
| Relationship | contingency_ui.py; contingency_table.py | Observed counts, DF, p and percentages | Counts/DF integer; p special; percentages 1 dp | Chi-square module p 4 significant digits; cell counts 0 dp | Leave χ²/expected frequency/residuals unchanged pending review |
| Analyze groups / periods | group_statistics_ui.py; period_comparison_ui.py | Selected mean/median/min/max/SD/variance/count aggregations; expanding mean, rolling mean; percentage changes/shares | By explicitly selected aggregation; % 1 dp | Blanket 2 dp including counts; percent 2 dp | Resolve from selected statistic, never dataset column name; sums/deltas remain review |
| Regression | regression_ui.py; regression_reporting.py; regression_statistics.py; regression_plots.py; regression_effect_plots.py | Coefficients, SE, t/F, p, DF, R²/adjusted R²; centering means and EMM | Coefficient 4 significant digits; SE/t/F/R² 3 dp; means/mean CI 2 dp | Generic numeric 4 significant digits; equation 5; findings mixed | Explicit coefficient/inference/performance schemas and narratives |
| Logistic regression | logistic_ui.py; logistic_analysis.py | Coefficients, SE, z, p, odds ratios, counts, DF and percentages | Coefficient 4 significant digits; SE/z 3 dp; OR 3 significant digits; p special; count/DF integer; % 1 dp | Generic numeric 4 significant digits; equation 5; findings OR 4 significant digits | Explicit regression/OR schema; keep OR CI review |
| Predictive analytics | dataset_views.py; modelling.py; modelling_config.py; charts.py | Regression R², coefficients, odds ratios; observation counts; percentages | R² 3 dp; coefficients 4 significant digits; OR 3 significant digits; counts integer; % 1 dp | Metric R² 3 dp, fit interpretation 2 dp; coefficients/OR rounded to 4 dp | Preserve full coefficient/OR outputs; centralized display; error/classification metrics review |
| Interpretations | ai_interpreter.py; regression_reporting.py; ttest_reporting.py; anova_ui.py; logistic_ui.py | Approved numbers in deterministic / AI narrative | Same semantic rules and current precision settings | Independent templates; unconstrained LLM numeric prose | Pass approved display evidence; do not rewrite arbitrary generated numbers by guessing meaning |
| Report | report_capture.py; report_state.py; report_ui.py; report_export.py; report_dataset.py; report_ai.py | New approved result-table values and narrative/chart snapshots | Same captured display as analysis; raw numeric snapshot alongside | Raw capture loses Styler formatting; exports generic 8 significant digits | Freeze display rows with raw rows for NEW tables; legacy snapshots untouched |
| Unavailable features | temporal_target_guidance.py | Hazard ratio | 3 significant digits when implemented | No fitted survival/hazard-ratio output exists | Utility supports hazard ratios; do not invent outputs |

## Unlisted Result Types — Requires Review

| Module | Analysis / Feature | Result Type | Example Value | Current Format | Proposed Format | Reason |
|---|---|---|---|---|---|---|
| Inspect | exploration.py; dataset_views.py | Skewness; kurtosis / excess kurtosis | 0.312 | 2 dp / native | 3 dp | Shape statistics are not listed — **Requires review** |
| Inspect | exploration.py | IQR fences and IQR width | 12.3456 | 2 dp | 2 dp | Not percentiles or CI bounds; preserve existing — **Requires review** |
| Inspect / distributions | distribution_fit_ui.py; distribution_fit.py | SSE, AIC, KS statistic, bootstrap simulation uncertainty, fitted distribution parameters | AIC=123.4; KS=.082 | SSE/KS 3 dp; AIC 0 dp; raw candidate parameters | AIC 2 dp; KS 3 dp; params 4 significant digits | Distribution fit and parameter classes not approved — **Requires review** |
| Comparison / t-test | ttest_analysis.py; ttest_reporting.py | Mean differences; Cohen dz; Shapiro statistic; outlier measurement values | difference=4.75184; dz=.625 | 5 significant digits / generic 4 significant digits / raw | Differences 2 dp; dz 3 dp; Shapiro 3 dp; outlier data unchanged | Difference is not an absolute mean; paired effect variant not explicitly approved — **Requires review** |
| Comparison / factorial | factorial_followup.py; factorial_anova_ui.py | Partial η² (distinct from ordinary η²), pairwise differences, interaction/simple-effect contrast estimates; diagnostic condition numbers; covariance/rank measures | 4.75184 | Raw / 4 significant digits | Partial η² 3 dp; differences 2 dp; condition numbers 4 significant digits | Partial η² is ambiguous under the approved η² rule; contrasts are not automatically regression coefficients or means — **Requires review** |
| Random / mixed models | anova_effects_ui.py; anova_effects.py | Variance components, component SD, total modeled variance, ICC, near-boundary tolerances, likelihood/Hessian diagnostics | variance=7.92515; ICC=.3803 | Raw tables; variance narratives 5 significant digits; ICC 4 dp | Component variances/SD 3 dp; ICC 3 dp | Explicitly called out for review in request; ordinary variance rule does not approve model components — **Requires review** |
| Diagnostics | anova_ui.py; anova_independence.py; factorial_diagnostics.py; ttest_ui.py; regression_validation_ui.py; logistic_diagnostics.py | Levene/Brown–Forsythe, Jarque–Bera, Ljung–Box, Durbin–Watson, Shapiro statistics; size/variance ratios; serial screen metrics | DW=1.932; JB=8.34 | 3–4 significant digits / native | Test statistics 3 dp, ratios 3 significant digits | Not F/t/z unless explicitly that statistic; their p-values ARE covered — **Requires review** |
| Regression / logistic | regression_statistics.py; regression_ui.py; logistic_ui.py | Coefficient confidence intervals; OR confidence intervals; standardized coefficients; covariance matrices | OR CI=[.12345,2.3456] | 4 significant digits; prediction probability CI 4 dp | Coefficient CI 4 significant digits; ratio CI 3 significant digits | Request explicitly lists coefficient CI as review despite general CI rule — **Requires review** |
| Regression diagnostics | regression_statistics.py; regression_ui.py; logistic_diagnostics.py; regression_reporting.py | Residual SE, RMSE, MAE, MSE, PRESS, leverage, Cook distance, studentized residuals, DFBETAs, VIF, tolerance | RMSE=3.238; VIF=2.145 | 2–4 significant digits; residual-SE 4 significant digits | Errors in response units 3 dp; influence 3 significant digits | Not ordinary descriptive SD/SE; explicit review examples — **Requires review** |
| Logistic | logistic_ui.py; logistic_analysis.py | Log likelihood, deviance, AIC/BIC, LR χ²/Wald χ², McFadden pseudo-R², Brier score | AIC=147.234; Brier=.1834 | 4–5 significant digits | Likelihood/AIC 2 dp; test/Brier 3 dp | Pseudo-R² is not ordinary R²; χ² not F/t/z — **Requires review** |
| Prediction | logistic_ui.py; regression_ui.py; predictive_controls.py; modelling.py | Prediction probabilities, calibration proportions/intervals, prediction intervals, individual fitted/predicted values | probability=.72145 | Probability 4 dp; OLS predictions/intervals 5 significant digits; raw | Probabilities 3 dp; predictions/intervals scale-aware 2 dp | Explicit review examples; not ordinary group mean CI — **Requires review** |
| Predictive analytics | modelling.py; dataset_views.py; charts.py; modelling_config.py | Accuracy, F1 (weighted/macro), precision, recall, sensitivity/specificity, ROC-AUC, FPR/TPR, confusion rates, CV SD and gaps | F1=.8267 | 2–3 dp; CV summaries ±3 dp | Metrics 3 dp | Classification performance not correlation/effect size — **Requires review** |
| Predictive analytics | modelling.py; dataset_views.py; charts.py | Feature importance, SHAP contributions, tuning scores/parameters, learning-curve variability | importance=.21456 | 4 dp / native | Importance 3 significant digits; SHAP response-scale | Not coefficients; preserve model settings and hyperparameter meaning — **Requires review** |
| Groups / periods / trends | group_statistics.py; group_statistics_ui.py; period_comparison.py; relationship_ui.py; visual_analysis.py | Sums, cumulative sums, absolute changes, percentage-point differences, generic values, slope per time unit | change=7000; sum=12345.67 | 2 dp / 4 significant digits / raw | Response-scale format; slopes 4 significant digits | Unit-specific sums/differences are not means or regression coefficients by assumption — **Requires review** |
| Relationship / contingency | contingency_ui.py; contingency_table.py | χ², expected counts, standardized residuals; association diagnostics | χ²=12.345; expected=3.25 | χ² 4 dp; expected 2 dp; residual 4 dp | χ² 3 dp; expected 2 dp; residual 3 dp | Expected counts are fractional, not observed N; χ² not F — **Requires review** |
| Temporal / raw / configuration | temporal_ui.py; temporal.py; date_components.py; currency_conversion.py; transform_ui.py; manual_input.py; loader.py | Dates, durations, years/month codes, IDs, currency rates, user measurements, thresholds, α/confidence inputs, model hyperparameters | year=2026; rate=1.234567 | Source precision / temporal-specific representation | Preserve current meaning and precision | Not statistical results; never apply scientific notation or numerical display rounding — **Requires review** |
| Legacy variants | appV15.py; appV16.py; utils/*V*.py; utils/qualityV1.py | Archived versions of descriptive, chart and model displays | Various | Legacy mixed formats | Retain historical variants | Not imported by current app.py runtime; no retrospective rewrite — **Requires review** |

### Architecture / approval notes

- User chose an editable precision-control document instead of UI controls: NUMERICAL_PRECISION.toml. It is re-read on render; invalid configuration fails visibly. No new settings panel is introduced.
- Native Streamlit table downloads use underlying numeric data, while Styler is display-only. CSV/Excel callers must continue using numeric frames. Legacy `format_table` returns strings for display only; callers must not export it as raw evidence.
- New report captures will store approved display rows separately from raw rows. Existing saved snapshots have no new display metadata and will retain their old rendering.
- Fitted values / probabilities / component variances / coefficient intervals remain under review. No global regex will reformat arbitrary numbers in free-form AI text; approved evidence is supplied explicitly.

### Source-site inventory (pre-change)

Each site below was located by scanning function definitions, display calls, format specs, rounding and numeric result producers. Grouped audit rows above classify their meaning. Raw-value previews and archived files are included to make exclusions reviewable.

#### `utils/anova_analysis.py`

```text
67: summary = pd.DataFrame(rows, columns=['Group', 'Count', 'Mean', 'Median', 'SD', 'Variance',
124: ], columns=['Source', 'DF', 'Sum of Squares (SS)', 'Mean Square (MS)'])
```

#### `utils/anova_charts.py`

```text
20: hovertemplate='%{x}<br>Mean: %{y:.4g}<br>N: %{customdata[0]}<br>95% CI: [%{customdata[1]:.4g}, %{customdata[2]:.4g}]<extra></extra>',
47: hovertemplate=('%{y}<br>Mean A − mean B: %{x:.4g}<br>CI: [%{customdata[0]:.4g}, '
48: '%{customdata[1]:.4g}]<br>Adjusted p: %{customdata[2]:.4g}<extra>%{fullData.name}</extra>'),
```

#### `utils/anova_effects_ui.py`

```text
67: presentation.dataframe(pd.DataFrame(counts), hide_index=True)
72: presentation.dataframe(evidence['evidence'], hide_index=True)
104: presentation.dataframe(prepared['groups'], hide_index=True)
106: presentation.dataframe(prepared['relationships'], hide_index=True)
111: presentation.dataframe(result['variance'], hide_index=True)
112: presentation.write(f"Total modeled variance: {result['total_variance']:.5g}")
114: presentation.write(f"ICC: {result['icc']:.4f} — correlation between two observations sharing the same random group.")
120: presentation.dataframe(result['fixed'], hide_index=True, column_config={'p-value (asymptotic)': st.column_config.NumberColumn(format='%.4g')})
121: presentation.caption(f'{1-alpha:.1%} asymptotic normal Wald intervals and two-sided z tests, when available; these are large-sample coefficient tests, not Type II F tests or small-sample corrected inference. Reference levels use first observed order. Estimates compare the selected fixed groups conditional on random variation. Small group counts can make inference unreliable even above the five-group suppression threshold.')
123: presentation.dataframe(reference, hide_index=True)
155: presentation.table(pd.DataFrame([
160: ], columns=['Model detail', 'Value']).astype(str))
163: presentation.dataframe(result['variance'], hide_index=True)
164: presentation.write(f"Total modeled variance: {result['total_variance']:.5g}")
166: presentation.write(f"ICC: {result['icc']:.4f} — correlation between two observations sharing the same random group.")
171: presentation.dataframe(descriptive['summary'], hide_index=True)
183: presentation.dataframe(prepared['groups'], hide_index=True)
186: presentation.dataframe(prepared['relationships'], hide_index=True)
200: presentation.dataframe(evidence['evidence'], hide_index=True)
207: presentation.write(f"{row['Component']}: estimated variance {row['Variance']:.5g}.")
209: presentation.write(f"ICC = {result['icc']:.4f}: the estimated share of modeled variance associated with shared group membership.")
```

#### `utils/anova_independence.py`

```text
119: 'evidence': pd.DataFrame(evidence, columns=['Column', 'Indicator', 'Detail']),
```

#### `utils/anova_ui.py`

```text
101: presentation.dataframe(independence['evidence'], hide_index=True)
125: presentation.table(pd.DataFrame([
131: ], columns=['Model detail', 'Value']))
138: presentation.dataframe(pd.DataFrame([result]).rename(columns={
140: }).style.format({
141: 'F': '{:.6g}', 'p-value': lambda p: 'p < 0.001' if p < .001 else f'{p:.4g}', 'df numerator': '{:.6g}',
142: 'df denominator': '{:.6g}', "Effect Size (Cohen's f²)": '{:.6g}', 'α': '{:.3g}',
145: presentation.caption(f"Exact p-value: {result['p-value']:.17g}. Statistical significance does not establish practical importance.")
150: display = variation.drop(columns=['Mean Square (MS)']).copy()
154: presentation.dataframe(display.style.format({'Variation (%)': '{:.2f}%'}, na_rep='—'), hide_index=True)
163: presentation.dataframe(prepared['summary'], hide_index=True)
183: presentation.dataframe(table[columns].style.format({
184: 'Mean Difference': '{:.6g}', 'Adjusted p-value': '{:.4g}',
185: pairwise['lower']: '{:.6g}', pairwise['upper']: '{:.6g}',
192: presentation.dataframe(pd.DataFrame([
198: ], columns=['Diagnostic', 'Value']), hide_index=True)
203: presentation.write(f"Median-centered Levene (Brown–Forsythe variance screen): statistic = {prepared['levene_stat']:.4g}; p = {prepared['levene_p']:.4g}.")
208: presentation.dataframe(prepared['distribution'], hide_index=True)
229: presentation.dataframe(independence['evidence'], hide_index=True)
231: presentation.dataframe(independence['serial'], hide_index=True)
254: 'Finding': f"{row['Group A']} has a {'higher' if row['Mean Difference'] > 0 else 'lower'} mean than {row['Group B']} by {abs(row['Mean Difference']):.4g} response units.",
256: presentation.dataframe(pd.DataFrame(findings), hide_index=True, height=420)
258: presentation.write(f'Variance-weighted effect size f² = {result[EFFECT_SIZE]:.4g}. Statistical significance does not establish practical importance.')
```

#### `utils/charts.py`

```text
65: hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
75: hovertemplate="%{x}<extra></extra>",
99: hovertemplate="%{x}: %{y}<extra></extra>",
132: hovertemplate="%{y}<extra></extra>",
157: hovertemplate="%{x}: %{y}<extra></extra>",
188: hovertemplate=f"{col}: %{{y}}<extra></extra>",
208: z      = corr_df.values.round(2)
210: text   = [[f"{v:.2f}" for v in row] for row in z]
229: hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
264: hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<extra></extra>",
295: hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<extra></extra>",
308: hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<br>Density: %{{z}}<extra></extra>",
357: hovertemplate=f"{col_a}: %{{y}}<br>{col_b}: %{{x}}<br>Count: %{{z}}<extra></extra>",
395: hovertemplate=f"{col_b}={val_b} / {col_a}: %{{x}}<br>Count: %{{y}}<extra></extra>",
449: hovertemplate=f"{cat_col}: %{{x}}<br>{num_col}: %{{y}}<extra></extra>",
473: hovertemplate="%{y}: %{x:.4f}<extra></extra>",
515: hovertemplate="Size: %{x:,}<br>R²: %{y:.3f}<extra></extra>",
531: hovertemplate="Size: %{x:,}<br>R²: %{y:.3f}<extra></extra>",
569: text = [[f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in z]
585: hovertemplate=f"{cat_a}: %{{y}}<br>{cat_b}: %{{x}}<br>Avg {num_col}: %{{z:.2f}}<extra></extra>",
617: hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
637: name=f"ROC  (AUC = {auc_score:.3f})",
639: hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
672: hovertemplate="%{y}: %{x:.4f}<extra></extra>",
```

#### `utils/chartsV8.py`

```text
65: hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
75: hovertemplate="%{x}<extra></extra>",
99: hovertemplate="%{x}: %{y}<extra></extra>",
132: hovertemplate="%{y}<extra></extra>",
157: hovertemplate="%{x}: %{y}<extra></extra>",
188: hovertemplate=f"{col}: %{{y}}<extra></extra>",
208: z      = corr_df.values.round(2)
210: text   = [[f"{v:.2f}" for v in row] for row in z]
229: hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
264: hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<extra></extra>",
295: hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<extra></extra>",
308: hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<br>Density: %{{z}}<extra></extra>",
357: hovertemplate=f"{col_a}: %{{y}}<br>{col_b}: %{{x}}<br>Count: %{{z}}<extra></extra>",
395: hovertemplate=f"{col_b}={val_b} / {col_a}: %{{x}}<br>Count: %{{y}}<extra></extra>",
449: hovertemplate=f"{cat_col}: %{{x}}<br>{num_col}: %{{y}}<extra></extra>",
473: hovertemplate="%{y}: %{x:.4f}<extra></extra>",
514: hovertemplate="Size: %{x:,}<br>R²: %{y:.3f}<extra></extra>",
530: hovertemplate="Size: %{x:,}<br>R²: %{y:.3f}<extra></extra>",
568: text = [[f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in z]
584: hovertemplate=f"{cat_a}: %{{y}}<br>{cat_b}: %{{x}}<br>Avg {num_col}: %{{z:.2f}}<extra></extra>",
616: hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
636: name=f"ROC  (AUC = {auc_score:.3f})",
638: hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
671: hovertemplate="%{y}: %{x:.4f}<extra></extra>",
```

#### `utils/contingency_table.py`

```text
31: f'<div>{result["residual"].loc[r,c]:.4f}</div></td>')
```

#### `utils/contingency_ui.py`

```text
41: st.download_button("Download observed and expected counts (CSV)", pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig"),
49: presentation.metric(a, "Chi-square", f"{result['chi2']:.4f}")
50: presentation.metric(b, "Degrees of freedom", result["dof"])
51: presentation.metric(c, "p-value", f"{result['p']:.4g}")
```

#### `utils/correlation_ui.py`

```text
37: left.metric(f'{method} correlation', f'{coefficient:.4f}')
38: right.metric('p-value (two-sided)', f'{p:.4g}')
44: fig.update_traces(customdata=counts.to_numpy(), hovertemplate='%{y} / %{x}<br>Correlation: %{z:.3f}<br>Usable pairs: %{customdata}<extra></extra>')
46: presentation.dataframe(matrix, use_container_width=True)
51: presentation.dataframe(counts, use_container_width=True)
```

#### `utils/dataset_views.py`

```text
44: return frame.to_csv(index=False).encode('utf-8-sig')
58: presentation.dataframe(component_preview(preview_df, st.session_state.get('variable_metadata', {})), use_container_width=True, height=300)
109: mn  = f"{series.min():.2f}"  if has_data else "—"
110: mx  = f"{series.max():.2f}"  if has_data else "—"
111: avg = f"{series.mean():.2f}" if has_data else "—"
127: ("Missing %",      f"{missing_pct:.2f}%"),
129: ("Unique %",       f"{round(unique / total * 100, 2):.2f}%"),
134: presentation.dataframe(pd.DataFrame([(label, str(value)) for label, value in statistics], columns=["Statistic", "Value"]), hide_index=True)
164: ("Missing %",          f"{missing_pct:.2f}%"),
166: ("Unique %",           f"{round(unique / total * 100, 2):.2f}%"),
169: ("Leading/Trailing Whitespace", f"{leading_trailing_whitespace_pct(raw[col_name]):.2f}%"),
171: presentation.dataframe(pd.DataFrame([(label, str(value)) for label, value in statistics], columns=["Statistic", "Value"]), hide_index=True)
203: ("Missing %",              f"{missing_pct:.2f}%"),
205: ("Unique %",               f"{unique_pct:.2f}%"),
211: presentation.dataframe(pd.DataFrame([(label, str(value)) for label, value in statistics], columns=["Statistic", "Value"]), hide_index=True)
300: gap:.35rem; border-bottom:1px solid #8886; overflow:visible;
303: border:1px solid #8885; border-radius:.6rem .6rem 0 0;
304: padding:.65rem 1.1rem; height:auto; margin-bottom:-1px;
446: presentation.dataframe(
449: .format({"Percentage": "{:.2f}%"}),
508: presentation.dataframe(stats_df, use_container_width=True, hide_index=True)
656: presentation.caption(f"Correlation coefficient (r): **{corr_val:.2f}**")
780: presentation.dataframe(config_df, use_container_width=True, hide_index=True)
791: presentation.dataframe(config_df, use_container_width=True, hide_index=True)
802: presentation.dataframe(config_df, use_container_width=True, hide_index=True)
820: presentation.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)
881: f"Test R²: {wm['test_r2']:.3f}  /  Train R²: {wm['train_r2']:.3f}  /  Gap: {gap} ({severity})\n"
882: f"CV Mean R²: {wm['cv_mean']:.3f} ± {wm['cv_std']:.3f}\n"
883: f"Test RMSE: {wm['test_rmse']:.3f}  /  Test MAE: {wm['test_mae']:.3f}"
889: f"Test F1: {wm['test_f1']:.3f}  /  Train F1: {wm['train_f1']:.3f}  /  Gap: {gap} ({severity})\n"
890: f"Test Accuracy: {wm['test_accuracy']:.3f}  /  ROC-AUC: {wm['roc_auc']:.3f}\n"
891: f"Precision: {wm['test_precision']:.3f}  /  Recall: {wm['test_recall']:.3f}\n"
892: f"CV Mean F1: {wm['cv_mean']:.3f} ± {wm['cv_std']:.3f}"
899: return "\n".join(f"{i+1}. {clean(n)}: {v:.4f}" for i, (n, v) in enumerate(top.items()))
907: return "\n".join(f"{i+1}. {clean(n)}: {v:.4f} (mean across classes)" for i, (n, v) in enumerate(all_c.items()))
924: return "\n".join(f"{i+1}. {n}: {v:.4f}" for i, (n, v) in enumerate(pairs))
955: presentation.dataframe(
956: cdf.style.apply(_style_cmp, axis=1).format({"Lasso": "{:.3f}", "Random Forest": "{:.3f}"}),
963: presentation.metric(mc1, "Test R²",    f"{wm['test_r2']:.3f}")
964: presentation.metric(mc2, "Test RMSE",  f"{wm['test_rmse']:.3f}")
965: presentation.metric(mc3, "Test MAE",   f"{wm['test_mae']:.3f}")
966: presentation.metric(mc4, "CV Mean R²", f"{wm['cv_mean']:.3f} ± {wm['cv_std']:.3f}")
1026: presentation.dataframe(
1028: .format({"Logistic Regression": "{:.3f}", "Random Forest": "{:.3f}"}),
1035: presentation.metric(mc1, "Test F1",       f"{wm['test_f1']:.3f}")
1036: presentation.metric(mc2, "Test Accuracy", f"{wm['test_accuracy']:.3f}")
1037: presentation.metric(mc3, "Precision",     f"{wm['test_precision']:.3f}")
1038: presentation.metric(mc4, "Recall",        f"{wm['test_recall']:.3f}")
1039: presentation.metric(mc5, "ROC-AUC",       f"{wm['roc_auc']:.3f}")
1073: presentation.dataframe(
1074: results["odds_ratio_df"].style.format({"Coefficient": "{:.4f}", "Odds Ratio": "{:.4f}"}),
1107: presentation.metric(mc1, "Accuracy",  f"{results['accuracy']:.3f}")
1108: presentation.metric(mc2, "F1 Score",  f"{results['f1']:.3f}")
1109: presentation.metric(mc3, "Precision", f"{results['precision']:.3f}")
1110: presentation.metric(mc4, "Recall",    f"{results['recall']:.3f}")
1111: presentation.metric(mc5, "ROC-AUC",   f"{results['roc_auc']:.3f}")
1158: presentation.dataframe(
1160: .format({_lr_col: "{:.3f}", _rf_col: "{:.3f}"}),
1167: presentation.metric(mc1, "Test F1 (weighted)",        f"{wm['test_f1']:.3f}")
1168: presentation.metric(mc2, "Test Accuracy",             f"{wm['test_accuracy']:.3f}")
1169: presentation.metric(mc3, "Precision (weighted)",      f"{wm['test_precision']:.3f}")
1170: presentation.metric(mc4, "Recall (weighted)",         f"{wm['test_recall']:.3f}")
1171: presentation.metric(mc5, "Weighted ROC-AUC",          f"{wm['roc_auc']:.3f}")
1200: presentation.dataframe(
1202: .style.format({"Coefficient": "{:.4f}", "Odds Ratio": "{:.4f}"}),
1242: presentation.metric(mc1, "Accuracy",             f"{results['accuracy']:.3f}")
1243: presentation.metric(mc2, "F1 (weighted)",         f"{results['f1']:.3f}")
1244: presentation.metric(mc3, "Precision (weighted)", f"{results['precision']:.3f}")
1245: presentation.metric(mc4, "Recall (weighted)",    f"{results['recall']:.3f}")
1246: presentation.metric(mc5, "ROC-AUC (OvR)",        f"{results['roc_auc']:.3f}")
```

#### `utils/dataset_workspace.py`

```text
136: font-size:1rem; color:var(--text-color); opacity:.65;
140: background:var(--secondary-background-color, #f0f2f6); border-radius:.5rem; padding:.5rem .75rem;
145: width:120px; height:40px; min-height:40px; padding:.25rem .75rem;
146: background:var(--background-color, #fff); border:1px solid #8884; border-radius:.5rem;
149: .st-key-manual_entry_panel p {color:var(--text-color); opacity:.65; margin:0;}
```

#### `utils/descriptive_statistics_display.py`

```text
65: columns=['Statistic', 'Value'])
66: presentation.dataframe(frame, hide_index=True, use_container_width=True)
```

#### `utils/distribution_fit_ui.py`

```text
37: st.text(f"SSE:          {best['SSE']:.3f}\nAIC:          {best['AIC']:.0f}\nKS statistic: {best['KS statistic']:.3f}")
43: presentation.dataframe(pd.DataFrame(result['candidates']), hide_index=True, use_container_width=True)
```

#### `utils/duplicate_overview.py`

```text
26: presentation.dataframe(pd.DataFrame({'Column to remove': list(duplicates), 'Column to keep': list(duplicates.values())}),
39: total.metric('Number of rows', f'{len(frame):,}')
40: duplicated.metric('Duplicated rows', f'{count:,}')
68: presentation.dataframe(styled, hide_index=True, width='stretch')
```

#### `utils/exploration.py`

```text
20: confidence_interval = f"[{mean - margin:.2f}, {mean + margin:.2f}]" if np.isfinite(margin) else "N/A"
23: "Mean":             f"{clean.mean():.2f}",
24: "Median":           f"{clean.median():.2f}",
25: "Std Dev":          f"{clean.std():.2f}",
26: "Standard Error (Mean)": f"{standard_error:.2f}" if np.isfinite(standard_error) else "N/A",
27: "Coefficient of Variation": f"{cv:.2f}%" if np.isfinite(cv) else "N/A",
29: "Skewness":         f"{clean.skew():.2f}",
30: "Kurtosis":         f"{clean.kurtosis():.2f}",
31: "Minimum":          f"{clean.min():.2f}",
32: "Quartile 1 (25%)": f"{q1:.2f}",
33: "Quartile 2 (50%)": f"{q2:.2f}",
34: "Quartile 3 (75%)": f"{q3:.2f}",
35: "Maximum":          f"{clean.max():.2f}",
59: "pct":         f"{pct:.2f}%",
60: "lower_bound": f"{lower:.2f}",
61: "upper_bound": f"{upper:.2f}",
75: top_pct = f"{clean.value_counts().max() / total * 100:.2f}%" if not clean.empty else "—"
80: "Missing %":      f"{missing / total * 100:.2f}%",
91: pct = (counts / total * 100).round(2)
115: "Percentage": rare.round(2).values,
132: "Missing %": f"{missing / total * 100:.2f}%",
144: "Missing %":         f"{missing / total * 100:.2f}%",
```

#### `utils/explorationV1.py`

```text
20: "Missing %":      f"{series.isna().mean() * 100:.2f}%",
21: "Mean":           f"{clean.mean():.2f}",
22: "Median":         f"{clean.median():.2f}",
23: "Std Dev":        f"{clean.std():.2f}",
24: "Skewness":       f"{clean.skew():.2f}",
25: "Kurtosis":       f"{clean.kurtosis():.2f}",
26: "Minimum":        f"{clean.min():.2f}",
27: "Q1 (25%)":       f"{q1:.2f}",
28: "Q2 (50%)":       f"{q2:.2f}",
29: "Q3 (75%)":       f"{q3:.2f}",
30: "Maximum":        f"{clean.max():.2f}",
54: "pct":         f"{pct:.2f}%",
55: "lower_bound": f"{lower:.2f}",
56: "upper_bound": f"{upper:.2f}",
70: top_pct = f"{clean.value_counts().max() / total * 100:.2f}%" if not clean.empty else "—"
75: "Missing %":      f"{missing / total * 100:.2f}%",
86: pct = (counts / total * 100).round(2)
110: "Percentage": rare.round(2).values,
127: "Missing %": f"{missing / total * 100:.2f}%",
139: "Missing %":         f"{missing / total * 100:.2f}%",
```

#### `utils/explorationV2.py`

```text
17: "Mean":               f"{clean.mean():.2f}",
18: "Median":             f"{clean.median():.2f}",
19: "Std Dev":            f"{clean.std():.2f}",
20: "Skewness":           f"{clean.skew():.2f}",
21: "Kurtosis":           f"{clean.kurtosis():.2f}",
22: "Minimum":            f"{clean.min():.2f}",
23: "Quartile 1 (25%)":   f"{q1:.2f}",
24: "Quartile 2 (50%)":   f"{q2:.2f}",
25: "Quartile 3 (75%)":   f"{q3:.2f}",
26: "Maximum":            f"{clean.max():.2f}",
50: "pct":         f"{pct:.2f}%",
51: "lower_bound": f"{lower:.2f}",
52: "upper_bound": f"{upper:.2f}",
66: top_pct = f"{clean.value_counts().max() / total * 100:.2f}%" if not clean.empty else "—"
71: "Missing %":      f"{missing / total * 100:.2f}%",
82: pct = (counts / total * 100).round(2)
106: "Percentage": rare.round(2).values,
123: "Missing %": f"{missing / total * 100:.2f}%",
135: "Missing %":         f"{missing / total * 100:.2f}%",
```

#### `utils/factorial_anova_ui.py`

```text
32: presentation.caption(f"Showing {'all selected terms' if include_all else 'all significant terms'} using Type II ANOVA at α = {result['alpha']:.3g}. Each main effect and interaction has its own plot.")
43: presentation.caption(f"p = {p:.4g} · {'Significant' if p < result['alpha'] else 'Insufficient evidence at this α'}")
71: presentation.dataframe(summary, hide_index=True)
79: presentation.dataframe(observed, hide_index=True)
110: presentation.dataframe(estimated_marginal_means(prepared, result, [factor]), hide_index=True)
112: presentation.dataframe(comparisons, hide_index=True, width='stretch')
136: presentation.dataframe(item['means'], hide_index=True)
137: presentation.dataframe(item['comparisons'], hide_index=True, width='stretch')
153: presentation.write(f"{display_name(factor)}: p = {row['p-value']:.4g}; partial η² = {row['Partial η²']:.3g}.")
166: presentation.write(f"The highest adjusted estimate is {display_name(high['Level'])} ({high[MEAN]:.4g}); the lowest is {display_name(low['Level'])} ({low[MEAN]:.4g}). This ranking alone does not establish a pairwise difference; review confidence intervals and adjusted comparisons.")
177: presentation.write(f"{display_name(name)}: p = {row['p-value']:.4g}; partial η² = {row['Partial η²']:.3g}.")
191: presentation.dataframe(significant.drop(columns=['p-value', 'SE']), hide_index=True)
195: presentation.write(f"No selected interaction shows sufficient evidence at α = {result['alpha']:.3g}. You may retain the interactions based on the analysis plan or explicitly fit a reduced main-effects model for simpler interpretation. No terms are removed automatically; main effects must remain whenever their interaction remains.")
237: presentation.caption(f'Type II Sum of Squares · α = {alpha:.3g}. Type II tests respect marginality: a main-effect test does not adjust for interactions containing that factor. Review interactions before interpreting main effects. The method never switches automatically.')
266: presentation.dataframe(table, hide_index=True, width='stretch')
271: presentation.dataframe(robust_type2(prepared, result), hide_index=True, width='stretch')
277: presentation.table(diagnostics['table'].iloc[:-1], hide_index=True)
288: presentation.write(f'Residual Jarque–Bera diagnostic: statistic = {jb:.4g}; p = {normal_p:.4g}; skewness = {skew:.3g}; kurtosis = {kurtosis:.3g} (normal reference: 3).')
302: presentation.write(f'Median-centered Levene across observed factor combinations: statistic = {statistic:.4g}; p = {p:.4g}.')
```

#### `utils/factorial_followup.py`

```text
32: columns=[prepared['aliases'][f] for f in factors])
89: table = pd.DataFrame(labels, columns=['Level' if len(target) == 1 else f'Factor {i+1} level'
```

#### `utils/group_statistics.py`

```text
62: return result.drop(columns=['_period']), planned['keys']
```

#### `utils/group_statistics_ui.py`

```text
131: presentation.dataframe(display.style.format(precision=2, na_rep="N/A"), use_container_width=True)
132: st.download_button("Download statistics (CSV)", display.to_csv(index=False).encode("utf-8-sig"),
```

#### `utils/logistic_diagnostics.py`

```text
40: return pd.DataFrame(rows,columns=['Predictor','Assessment','Evidence','Recommendation','p-value','Holm-adjusted p-value'])
123: r['issues']=pd.DataFrame(issues,columns=['Severity','Issue','Evidence','Recommended action'])
```

#### `utils/logistic_ui.py`

```text
57: presentation.caption(f'Complete-case analysis: {len(df):,} total rows; {int(complete.sum()):,} included; {excluded:,} excluded for missing/nonfinite values ({100*excluded/len(df) if len(df) else 0:.1f}%). The sample stays fixed during elimination.')
69: presentation.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
88: summary=[('Response / Dependent Variable',str(spec['target'])),('Modeled event',repr(r['event'])),('Non-event',repr(r['non_event'])),('Predictors / Independent Variables',', '.join(map(str,spec['predictors']))),('Original rows',str(r['original_n'])),('Used rows',str(r['n'])),('Excluded missing/nonfinite rows',str(r['excluded'])),('Percentage excluded',f"{100*r['excluded']/r['original_n']:.1f}%"),('Events',str(r['events'])),('Non-events',str(r['non_events'])),('Estimated parameters (including intercept)',str(r['k'])),('Confidence level',f'{confidence:g}%'),('Model-building method','Backward elimination — AIC' if r['elimination'] else 'No elimination')]
89: presentation.dataframe(pd.DataFrame(summary,columns=['Model detail','Value']))
91: eta=f'{model.params[0]:.5g}'+''.join(f' {b:+.5g} × {name}' for name,b in zip(r['names'][1:],model.params[1:]))
97: presentation.dataframe(pd.DataFrame([(name,f'{v:.5g}' if isinstance(v,(float,np.floating)) else str(v)) for name,v in stats],columns=['Measure','Value']))
99: presentation.markdown('**Selected terms**');presentation.dataframe(selected_terms_table(r))
101: presentation.dataframe(pd.DataFrame([{'Predictor':c,'Reference category':repr(spec['levels'][c][0]),'Observed levels':len(spec['levels'][c])} for c in spec['categorical']]))
104: else: presentation.dataframe(format_table(r['history']),hide_index=True,use_container_width=True)
108: presentation.dataframe(format_table(r['coefficients']),hide_index=True,use_container_width=True)
110: presentation.markdown('**Joint term tests**');presentation.dataframe(format_table(r['term_tests']),hide_index=True,use_container_width=True)
112: st.download_button('Download logistic coefficients (CSV)',r['coefficients'].to_csv(index=False),'logistic_coefficients.csv',key='rel_log_coeff_download')
115: presentation.dataframe(pd.DataFrame({'Measure':['Events','Non-events','Parameters including intercept','Smaller outcome count per parameter'],'Value':[r['events'],r['non_events'],r['k'],min(r['events'],r['non_events'])/r['k']]}))
119: else: presentation.dataframe(format_table(r['linearity']),hide_index=True,use_container_width=True)
122: presentation.markdown('**Category-by-outcome counts**');presentation.dataframe(r['sparse'],hide_index=True,use_container_width=True)
131: else: presentation.dataframe(format_table(r['vif']),hide_index=True,use_container_width=True)
135: presentation.dataframe(format_table(inf),hide_index=True,use_container_width=True)
137: st.download_button('Download observation diagnostics (CSV)',inf.to_csv(index=False),'logistic_observations.csv',key='rel_log_inf_download')
140: presentation.dataframe(format_table(r['dfbetas']),hide_index=True,use_container_width=True)
167: if not settings.empty: presentation.dataframe(settings,hide_index=True,use_container_width=True)
178: presentation.dataframe(format_table(cal),hide_index=True,use_container_width=True)
179: presentation.caption(f"In-sample, quantile-grouped calibration with {100*r['confidence']:g}% Wilson intervals for observed proportions. Ties can reduce the number of groups. Brier score: {r['brier']:.4g}. Grouping and limited sample size affect appearance; this is not independent validation. An in-sample calibration intercept/slope is omitted because it can be ideal by construction.")
213: presentation.write(f'In-sample ROC-AUC: {roc_auc_score(y,r["model"].fittedvalues):.3f}')
221: pred.columns=['Event probability',f'{confidence:g}% CI lower',f'{confidence:g}% CI upper']
224: presentation.dataframe(pred,hide_index=True,use_container_width=True)
226: st.download_button('Download fitted probabilities (CSV)',pred.to_csv(index=False),'logistic_probabilities.csv',key='rel_log_prob_download')
238: presentation.metric(st, f"Probability of {r['spec']['target']} = {r['event']!r}",f"{predicted['mean']:.4f}")
239: presentation.write(f"{confidence:g}% confidence interval: {predicted['mean_ci_lower']:.4f} to {predicted['mean_ci_upper']:.4f}")
251: if not r['issues'].empty: presentation.dataframe(r['issues'],hide_index=True,use_container_width=True)
258: lines.append(f"The overall likelihood-ratio test (χ²={r['lr']:.4g}, df={r['lr_df']}, p={p_text(r['lr_p'])}) "+('provides evidence of better fit than the intercept-only model.' if r['lr_p']<.05 else 'does not provide strong evidence of better fit than the intercept-only model.'))
276: lines.append(f"After adjusting for the other model variables, {row['Term']} has an estimated odds ratio of {row['Odds Ratio']:.4g} {comparison} for {event} ({100*r['confidence']:g}% CI {row['OR CI Lower']:.4g}–{row['OR CI Upper']:.4g}; p={p_text(row['p-value'])}). This is an association in odds, not a probability ratio or causal effect.")
277: lines.append(f"McFadden Pseudo-R²={r['pseudo_r2']:.4g}; this is not the percentage of variance explained. The in-sample Brier score is {r['brier']:.4g}; assess it alongside the calibration plot.")
```

#### `utils/manual_input.py`

```text
89: edited = st.data_editor(pd.DataFrame('', index=range(500), columns=[f'Column {i+1}' for i in range(25)]),
97: presentation.dataframe(preview.head(5), hide_index=True, use_container_width=True)
```

#### `utils/modelling.py`

```text
805: "Coefficient": top20.values.round(4),
806: "Odds Ratio":  np.exp(top20.values).round(4),
1008: "Coefficient": top20.values.round(4),
1009: "Odds Ratio":  np.exp(top20.values).round(4),
```

#### `utils/modelling_config.py`

```text
63: reasons.append(f"{n_unique / n_rows * 100:.0f}% of values are unique")
156: return f"✅ **Strong fit** (R² = {r2:.2f}) — the model explains {r2*100:.0f}% of the variation in the target column."
158: return f"ℹ️ **Moderate fit** (R² = {r2:.2f}) — the model captures most patterns but leaves room for improvement."
160: return f"⚠️ **Weak fit** (R² = {r2:.2f}) — consider reviewing your predictor selection or data quality."
176: return f"✅ **Strong F1 Score** ({f1:.2f}) — the model balances precision and recall well."
178: return f"ℹ️ **Moderate F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
179: return f"⚠️ **Low F1 Score** ({f1:.2f}) — consider reviewing predictor selection or addressing class imbalance."
208: return f"✅ **Strong weighted F1 Score** ({f1:.2f}) — the model classifies all categories well."
210: return f"ℹ️ **Moderate weighted F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
211: return f"⚠️ **Low weighted F1 Score** ({f1:.2f}) — some categories may be difficult to distinguish. Review class distribution and predictor selection."
```

#### `utils/modellingV3.py`

```text
71: reasons.append(f"{n_unique / n_rows * 100:.0f}% of values are unique")
159: return f"✅ **Strong fit** (R² = {r2:.2f}) — the model explains {r2*100:.0f}% of the variation in the target column."
161: return f"ℹ️ **Moderate fit** (R² = {r2:.2f}) — the model captures most patterns but leaves room for improvement."
163: return f"⚠️ **Weak fit** (R² = {r2:.2f}) — consider reviewing your predictor selection or data quality."
297: return f"✅ **Strong F1 Score** ({f1:.2f}) — the model balances precision and recall well."
299: return f"ℹ️ **Moderate F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
300: return f"⚠️ **Low F1 Score** ({f1:.2f}) — consider reviewing predictor selection or addressing class imbalance."
```

#### `utils/modellingV4.py`

```text
83: reasons.append(f"{n_unique / n_rows * 100:.0f}% of values are unique")
187: return f"✅ **Strong fit** (R² = {r2:.2f}) — the model explains {r2*100:.0f}% of the variation in the target column."
189: return f"ℹ️ **Moderate fit** (R² = {r2:.2f}) — the model captures most patterns but leaves room for improvement."
191: return f"⚠️ **Weak fit** (R² = {r2:.2f}) — consider reviewing your predictor selection or data quality."
325: return f"✅ **Strong F1 Score** ({f1:.2f}) — the model balances precision and recall well."
327: return f"ℹ️ **Moderate F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
328: return f"⚠️ **Low F1 Score** ({f1:.2f}) — consider reviewing predictor selection or addressing class imbalance."
495: return f"✅ **Strong weighted F1 Score** ({f1:.2f}) — the model classifies all categories well."
497: return f"ℹ️ **Moderate weighted F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
498: return f"⚠️ **Low weighted F1 Score** ({f1:.2f}) — some categories may be difficult to distinguish. Review class distribution and predictor selection."
```

#### `utils/modellingV5.py`

```text
83: reasons.append(f"{n_unique / n_rows * 100:.0f}% of values are unique")
186: return f"✅ **Strong fit** (R² = {r2:.2f}) — the model explains {r2*100:.0f}% of the variation in the target column."
188: return f"ℹ️ **Moderate fit** (R² = {r2:.2f}) — the model captures most patterns but leaves room for improvement."
190: return f"⚠️ **Weak fit** (R² = {r2:.2f}) — consider reviewing your predictor selection or data quality."
324: return f"✅ **Strong F1 Score** ({f1:.2f}) — the model balances precision and recall well."
326: return f"ℹ️ **Moderate F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
327: return f"⚠️ **Low F1 Score** ({f1:.2f}) — consider reviewing predictor selection or addressing class imbalance."
494: return f"✅ **Strong weighted F1 Score** ({f1:.2f}) — the model classifies all categories well."
496: return f"ℹ️ **Moderate weighted F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
497: return f"⚠️ **Low weighted F1 Score** ({f1:.2f}) — some categories may be difficult to distinguish. Review class distribution and predictor selection."
```

#### `utils/modellingV6.py`

```text
83: reasons.append(f"{n_unique / n_rows * 100:.0f}% of values are unique")
187: return f"✅ **Strong fit** (R² = {r2:.2f}) — the model explains {r2*100:.0f}% of the variation in the target column."
189: return f"ℹ️ **Moderate fit** (R² = {r2:.2f}) — the model captures most patterns but leaves room for improvement."
191: return f"⚠️ **Weak fit** (R² = {r2:.2f}) — consider reviewing your predictor selection or data quality."
325: return f"✅ **Strong F1 Score** ({f1:.2f}) — the model balances precision and recall well."
327: return f"ℹ️ **Moderate F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
328: return f"⚠️ **Low F1 Score** ({f1:.2f}) — consider reviewing predictor selection or addressing class imbalance."
495: return f"✅ **Strong weighted F1 Score** ({f1:.2f}) — the model classifies all categories well."
497: return f"ℹ️ **Moderate weighted F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
498: return f"⚠️ **Low weighted F1 Score** ({f1:.2f}) — some categories may be difficult to distinguish. Review class distribution and predictor selection."
982: "Coefficient": top20.values.round(4),
983: "Odds Ratio":  np.exp(top20.values).round(4),
```

#### `utils/modellingV7.py`

```text
83: reasons.append(f"{n_unique / n_rows * 100:.0f}% of values are unique")
188: return f"✅ **Strong fit** (R² = {r2:.2f}) — the model explains {r2*100:.0f}% of the variation in the target column."
190: return f"ℹ️ **Moderate fit** (R² = {r2:.2f}) — the model captures most patterns but leaves room for improvement."
192: return f"⚠️ **Weak fit** (R² = {r2:.2f}) — consider reviewing your predictor selection or data quality."
326: return f"✅ **Strong F1 Score** ({f1:.2f}) — the model balances precision and recall well."
328: return f"ℹ️ **Moderate F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
329: return f"⚠️ **Low F1 Score** ({f1:.2f}) — consider reviewing predictor selection or addressing class imbalance."
496: return f"✅ **Strong weighted F1 Score** ({f1:.2f}) — the model classifies all categories well."
498: return f"ℹ️ **Moderate weighted F1 Score** ({f1:.2f}) — reasonable performance with room for improvement."
499: return f"⚠️ **Low weighted F1 Score** ({f1:.2f}) — some categories may be difficult to distinguish. Review class distribution and predictor selection."
983: "Coefficient": top20.values.round(4),
984: "Odds Ratio":  np.exp(top20.values).round(4),
1194: "Coefficient": top20.values.round(4),
1195: "Odds Ratio":  np.exp(top20.values).round(4),
```

#### `utils/period_comparison_ui.py`

```text
24: hovertemplate='%{x}<br>Value: %{y:,.2f}<br>Inserted zero: %{customdata[2]}<extra>%{fullData.name}</extra>'))
28: hovertemplate='%{x}<br>Change: %{y:+.2f}%<br>Baseline: %{customdata[0]}<br>Absolute change: %{customdata[1]}<br>%{customdata[2]}<extra>%{fullData.name}</extra>'))
35: hovertemplate='%{x}<br>Change %: N/A<br>%{customdata}<extra>%{fullData.name}</extra>'))
75: presentation.dataframe(pd.DataFrame([{'Frequency': name, 'Status': status, 'Explanation': reason}
105: presentation.dataframe(planned['gaps'].head(100), hide_index=True)
172: presentation.dataframe(display.style.format({'Value': '{:,.2f}', 'Previous': '{:,.2f}', 'Change': '{:+,.2f}', 'Change %': '{:+.2f}%'}, na_rep='N/A'), hide_index=True)
173: st.download_button('Download statistics (CSV)', display.to_csv(index=False, na_rep='N/A').encode('utf-8-sig'),
```

#### `utils/presentation.py`

```text
59: return target.metric(label, value, *args, **kwargs)
85: return st.dataframe(data, **options)
```

#### `utils/regression_effect_plots.py`

```text
28: settings = [{'Predictor':str(c), 'Held at':repr(value) if c in spec['categorical'] else f'{value:.6g}',
39: slices = [(f'{moderator}: P{q} = {value:.4g}', float(value))
54: return curves, pd.DataFrame(settings, columns=['Predictor','Held at','Setting'])
83: customdata=custom,hovertemplate='%{x}<br>Adjusted mean: %{y:.4g}<br>CI: %{customdata[0]:.4g} to %{customdata[1]:.4g}<extra>%{fullData.name}</extra>'))
109: presentation.dataframe(settings,hide_index=True,use_container_width=True)
```

#### `utils/regression_estimability.py`

```text
31: rows.append({'Term': names[i], 'Issue': f'Largest absolute generated value: {magnitudes[i]:.4g}',
```

#### `utils/regression_reporting.py`

```text
7: return 'N/A' if not np.isfinite(p) else '< 0.001' if p < .001 else f'{p:.3f}'
16: display[c] = display[c].map(lambda v:'N/A' if np.isnan(v) else f'{v:.4g}')
22: return result['spec']['target'] + ' = ' + f'{values[0]:.5g}' + ''.join(
28: lines.append(f'The model explains {100*result["r2"]:.1f}% of the observed variation in {y} (Adjusted R² = {result["adjusted_r2"]:.3f}).')
31: lines.append(f'The overall regression {"shows evidence of an association" if p < .05 else "does not show strong evidence of an association"} (F = {result["f_stat"]:.4g}, p {"" if p < .001 else "= "}{p_text(p)}).')
49: lines.append(f'{c} has an estimated {"positive" if coef["Coefficient"] >= 0 else "negative"} association with {y}: {coef["Coefficient"]:.4g} units per one-unit increase, holding other predictors constant (95% CI {coef["95% lower"]:.4g} to {coef["95% upper"]:.4g}; p {"" if p < .001 else "= "}{p_text(p)}; {"statistically significant" if p < .05 else "not statistically significant"} at α = 0.05).')
67: lines.append(f'Durbin–Watson = {result["dw"]:.3f}; values toward 0 or 4 can suggest positive or negative first-order autocorrelation. No fixed pass/fail cutoff is applied.')
89: means='; '.join(f'{c}: {spec["means"][c]:.6g}' for c in numeric) if spec['centered'] else 'Not applied'
95: factors.append(f'({c} − {spec["means"][c]:.6g})')
104: return pd.DataFrame(rows,columns=['Term','Type','Final status','Centering - Mean Subtracted','Centering - Transformation'])
```

#### `utils/regression_ui.py`

```text
54: presentation.dataframe(pd.DataFrame(exc.details), hide_index=True, width='stretch')
73: presentation.dataframe(pd.DataFrame(summary,columns=['Model detail','Value']))
78: presentation.metric(a, 'Adjusted R²',f"{r['adjusted_r2']:.3f}",help='R² adjusted for fitted model size.')
79: presentation.metric(b, 'R²',f"{r['r2']:.3f}")
80: presentation.metric(c, 'Residual standard error',f"{r['residual_se']:.4g}")
81: stats = [('RMSE',f"{r['rmse']:.4g}"),('MAE',f"{r['mae']:.4g}"),
83: ('Overall F-statistic',f"{r['f_stat']:.4g}" if np.isfinite(r['f_stat']) else 'N/A'),('Overall F-test p-value',p_text(r['f_p']))]
84: presentation.dataframe(pd.DataFrame(stats,columns=['Performance measure','Value']))
87: presentation.dataframe(selected_terms_table(r))
92: presentation.dataframe(pd.DataFrame(r['coding']))
98: presentation.dataframe(format_table(r['history']),hide_index=True,use_container_width=True)
103: presentation.dataframe(format_table(r['coefficients']),hide_index=True,use_container_width=True)
106: presentation.dataframe(format_table(r['term_tests'].drop(columns='block')),hide_index=True,use_container_width=True)
108: st.download_button('Download regression coefficients (CSV)',r['coefficients'].to_csv(index=False),file_name='regression_coefficients.csv',key='reg_coeff_download')
119: display['VIF'] = display['VIF'].map(lambda v:f'{v:.2f}')
120: display['Tolerance'] = display['Tolerance'].map(lambda v:f'{v:.3f}')
121: presentation.dataframe(display,hide_index=True,use_container_width=True)
134: presentation.dataframe(format_table(ranked),hide_index=True,use_container_width=True)
136: presentation.dataframe(format_table(inf.sort_values('Leverage',ascending=False).head(10)),hide_index=True,use_container_width=True)
143: st.download_button('Download all observation diagnostics (CSV)',inf.to_csv(index=False),file_name='regression_influence.csv',key='reg_inf_download')
188: presentation.metric(st, f"Predicted {spec['target']}",f"{result['mean']:.5g}")
192: presentation.write(f"95% confidence interval for mean response: {result['mean_ci_lower']:.5g} to {result['mean_ci_upper']:.5g}")
193: presentation.write(f"95% prediction interval for an individual future observation: {result['obs_ci_lower']:.5g} to {result['obs_ci_upper']:.5g}")
```

#### `utils/regression_validation_ui.py`

```text
34: presentation.dataframe(format_table(diagnostic), hide_index=True, use_container_width=True)
51: st.metric('Durbin–Watson statistic', f"{result['dw']:.3f}")
65: presentation.dataframe(influence.loc[flagged], hide_index=True, use_container_width=True)
```

#### `utils/relationship_ui.py`

```text
61: st.write(f'Linear trend: {slope:.4g} units per elapsed {unit}; R² = {trend["r2"]:.3f}.')
62: presentation.dataframe(chart,hide_index=True,use_container_width=True)
```

#### `utils/report_ui.py`

```text
101: presentation.dataframe(pd.DataFrame(value['rows'], columns=value['columns']), hide_index=True)
```

#### `utils/temporal_ui.py`

```text
21: presentation.dataframe(table, hide_index=True, width='stretch')
38: summary = pd.DataFrame(rows, columns=['Statistic', 'Value'])
40: presentation.dataframe(summary, hide_index=True, use_container_width=True)
42: presentation.table(summary)
```

#### `utils/transform_ui.py`

```text
61: output.metric('Output rows', f"{report['Output rows']:,}")
62: matched.metric('Matched left rows', f"{report['Matched left rows']:,}")
71: presentation.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
81: presentation.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
100: presentation.dataframe(component_preview(frame.head(100), metadata), width='stretch')
379: presentation.dataframe(exc.invalid, hide_index=True, width='stretch')
385: presentation.dataframe(exc.issues.head(20), hide_index=True)
394: presentation.dataframe(component_preview(saved['frame'].head(100), saved['metadata']), use_container_width=True)
399: presentation.dataframe(issues.head(20), hide_index=True)
```

#### `utils/ttest_analysis.py`

```text
131: 'accounting': pd.DataFrame(accounting, columns=['Data detail', 'Count']),
```

#### `utils/ttest_reporting.py`

```text
12: return f'{value:.5g}'
25: return pd.DataFrame(rows, columns=['Statistic', 'Value'])
33: p = ('p < 0.001' if r['p'] < .001 else f'p = {r["p"]:.4g}')
38: f'{evidence} at alpha = {alpha:.3g}. '
45: return pd.DataFrame([r]).to_csv(index=False).encode('utf-8-sig')
```

#### `utils/ttest_ui.py`

```text
88: presentation.caption(f'Significance level alpha = {1-confidence:.3g}. Choose a directional hypothesis before reviewing the results. One-sided intervals have one infinite bound.')
109: ('Alternative hypothesis', alt_label), ('Confidence level', f'{confidence:.1%}'),
110: ('Significance level', f'{1-confidence:.3g}')]
115: presentation.table(pd.DataFrame(config, columns=['Test detail', 'Value']))
116: presentation.table(prepared['accounting'])
120: presentation.dataframe(format_table(prepared['descriptive']), hide_index=True, use_container_width=True)
123: st.download_button('Download descriptive statistics (CSV)', prepared['descriptive'].to_csv(index=False).encode('utf-8-sig'),
129: presentation.dataframe(format_table(prepared['diagnostics']), hide_index=True, use_container_width=True)
134: presentation.dataframe(prepared['outliers'], hide_index=True, use_container_width=True)
154: presentation.table(result_table(result))
```

#### `utils/variable_metadata_ui.py`

```text
29: presentation.table(pd.DataFrame({'Category number':range(1,len(proposed)+1),'Observed value':[repr(v) for v in proposed]}))
```

#### `utils/variable_overview_display.py`

```text
54: rows.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+f'<td{style}>{missing:.2f}</td></tr>')
85: presentation.dataframe(
87: columns=['Detail', 'Value']), hide_index=True,
108: styled = overview.style.format({'Missing %': '{:.2f}'}).map(missing_color, subset=['Missing %'])
109: presentation.dataframe(styled, hide_index=True, column_config={
```

#### `utils/visual_analysis.py`

```text
298: presentation.dataframe(table.rename(columns=labels),hide_index=True,use_container_width=True)
```


## Integration notes

- Formatting schemas are passed explicitly at authored result sites. Shared `format_table` has no semantic name guessing: callers must pass the appropriate schema. DFBETA matrices with a predictor named Mean or F retain the previous diagnostic formatting.
- Random component variance, ICC, likelihoods, prediction probabilities/intervals, coefficient/odds-ratio intervals, error and classification metrics retain existing behavior. Approved p-values within these tables are formatted separately.
- Existing early rounding in descriptive missing/frequency percentages and predictive coefficient/OR/R² display tables was removed so detailed display has access to original calculated values. No fitted model, raw dataset, transformation or statistical test was changed. Unlisted performance metrics retain their existing rounded display-table behavior.
- Plotly hover fields append display text alongside original numeric customdata. Plotted means, differences, bounds, coefficients and heatmap values remain unrounded; chart axes remain scale-aware. Correlation heatmap colors now use full correlation values instead of pre-rounded values.
- AI explanations receive approved display evidence and must copy numerical literals verbatim. Unsupported numerical restatements cause the existing deterministic fallback, rather than silently rewriting or re-rounding generated prose. This is a formatting guard, not a claim that all AI prose is statistically verified.
- Hazard-ratio output, fitted survival analysis, reliability and forecasting are not implemented in this app. The utility supports approved hazard-ratio formatting for future callers; no such analyses were added.

### Modules connected to the formatter

- Inspect and descriptive/quality results: `dataset_views`, `exploration`, `quality`, `loader`, `variable_overview_display`, `distribution_fit_ui`, `temporal_ui`.
- Comparison and group/period outputs: `anova_ui`, `anova_charts`, `anova_effects_ui`, `factorial_anova_ui`, `ttest_ui`, `ttest_reporting`, `group_statistics_ui`, `period_comparison_ui`, `period_comparison`.
- Relationship and models: `correlation_ui`, `contingency_ui`, `relationship_ui`, `regression_ui`, `regression_reporting`, `regression_validation_ui`, `regression_plots`, `regression_effect_plots`, `logistic_ui`, `modelling`, `modelling_config`, `charts`.
- Shared presentation, AI and reporting: `presentation`, `ai_interpreter`, `report_capture`, `report_dataset`, `report_ui`, `report_export`; `app.py` validates the configuration document.

Statistical engines and transformation algorithms are unchanged. Some display-table preparation helpers stopped early rounding as described above. Existing hard-coded model diagnostics and unlisted statistics retain their prior behavior, with one verified presentation correction: the logistic LR finding now uses the selected confidence level’s α instead of a hard-coded 0.05. The LR statistic and p-value calculations are unchanged; the narrative uses the unrounded p-value.

## Verification outcome

- Full regression run: 380 tests, 378 passed. Two previously documented unrelated failures remain: `test_fill_missing.FillMissingUITests.test_group_navigation_preview_apply_and_create` (obsolete dropdown placeholder expectation) and `test_temporal_target_guidance.GuidanceUITests.test_workspace_preserves_controls_without_persisting_action` (missing target checkbox).
- Latest focused verification after final presentation refinements: 19 numerical-formatting and t-test UI tests passed. Includes document reload, standard/detailed/alpha-aware p-values, special values, tiny/large numbers, scalar/table/chart consistency, unlisted diagnostic-name collisions, numeric CSV/Excel exports, immutable legacy/new reports, live AI invalidation and the selected-alpha logistic narrative check.
- Statistical-reference tests for ANOVA, factorial/mixed effects, t-tests and regression ran in the full suite without a numerical-engine regression.
- Raw previews, transformations and dataset CSV values are untouched. Formatting is explicit at result sites and report snapshots retain their original captured display.
- No manual browser verification was performed for this change; Streamlit integration tests inspect the actual serialized display-value payload.
