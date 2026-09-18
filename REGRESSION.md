# Linear Regression in EDA → Relationship

Select a numeric response and one or more numeric/categorical predictors. Choose **Regression** and a Numerical response. This is an exploratory, descriptive OLS analysis; all fit measures describe the observations used for estimation.

## Model specification

- Continuous variables remain in original units for ordinary additive models. Categorical factors use one observed reference level and readable indicator labels, with the reference displayed separately.
- Select two-way interactions explicitly: numeric × numeric, numeric × categorical, or categorical × categorical. The model includes all relevant indicator products for a selected factor interaction.
- Target, predictors, interactions and polynomial powers use checkboxes. Exactly one numeric target is selected; checking a different target switches it and excludes it from predictors. Predictors can be numeric or categorical.
- Select quadratic/cubic powers separately (A², A³, etc.), or use Select all / Clear all. Selecting a cubic checks and locks its square until the cubic is unchecked. All selected predictors supply main effects; interactions preserve both main effects. Removing a predictor prunes its dependent selections.
- When interactions or powers are selected, all selected continuous predictors are centered using complete-case training means. These values are displayed; Y is not standardized. Prediction applies the same centers and category levels.
- Complete cases are determined once for the original specification, including all selected variables; the sample stays fixed during elimination. Exclusions include missing and infinite values. Original row positions and index labels are retained for observation diagnostics.

## Backward Elimination — Hierarchy Preserved

The default is **No elimination**. The optional elimination procedure fits the user-specified model and removes the eligible whole term with the highest partial F-test p-value above the configurable threshold (default 0.05). A categorical factor or interaction is a block, not a series of independently removed indicators. Lower powers and main effects remain protected while a dependent power or interaction remains. The intercept is never removed; an intercept-only result is possible.

The history shows the removal p-value before removal and adjusted R² after refitting. The joint-term table identifies hierarchy-protected terms. Perfect fits have no meaningful residual-based selection and do not trigger removal. Inference after selection is exploratory: standard p-values and intervals are conditional on the selected model and omit selection uncertainty.

## Results

1. **Model Overview:** a two-column Model detail / Value summary table for response, specified predictors, row counts and model-building method, followed by the full fitted equation. The Selected terms table includes term type, final status and term-specific centering means/transformations. Includes model performance: R², adjusted R², residual standard error, RMSE, MAE, degrees of freedom and overall F-test. Elimination history remains expandable.
2. **Predictor Analysis:** coefficients, standard errors, t statistics, p-values, 95% intervals, and joint term tests. Standardized coefficients are supplied only for ordinary numeric main effects without retained interactions/powers. Other term types show N/A. CSV export is available.
3. **Diagnostics:** residual/fitted plot with a smooth trend, histogram, normal Q–Q plot, Omnibus/Jarque–Bera, and the Koenker variant of Breusch–Pagan. Durbin–Watson is interpreted only after meaningful original observation order is confirmed. Analysts can flag visible patterns for inclusion in Findings; no visual pattern is asserted solely from an unreviewed plot.
4. **Multicollinearity & Influence:** VIF/tolerance for readable encoded columns; primary correlation matrix for original selected numeric predictors only; internally studentized residuals, leverage and Cook’s distance with original row/index identifiers, plots, rankings, and export of all observations. No rows are automatically deleted.
5. **Visual Analysis:** one-predictor line or polynomial curve over its observed range, with both 95% mean-response confidence and individual-response prediction bands. All models provide actual/fitted plots with a perfect-agreement line and coefficient intervals. Partial residual plots are provided for continuous main effects (including their retained powers) without interactions.
6. **Prediction:** original variables only; known-category dropdowns and numeric entries constrained to observed marginal ranges. The exact fitting transformation generates predictions and both intervals. Unusual multivariable combinations can still be unsupported even within marginal ranges.
7. **Regression Findings:** deterministic, interaction/polynomial-aware descriptions of fit, joint significance, ordinary associations, diagnostics, multicollinearity, influence and interval interpretation. Text export is available.

P-values below 0.001 display as `< 0.001`. The reporting significance level is 0.05; the configurable removal threshold applies only to elimination. Neither diagnostics nor a non-significant test prove assumptions. Associations are not causal claims.

## Guardrails and limits

Invalid selections, constant variables, single-level factors, invalid powers/interactions, singular matrices, nonfinite transformations and insufficient residual degrees of freedom raise UI validation errors. Factors above 100 observed levels and models above 300 coefficients are rejected as resource/usability safeguards. Sparse levels and small samples warn. Perfect/near-perfect fits suppress standard errors, tests and intervals rather than displaying misleading precision.

VIF ≥5, VIF >10, |internally studentized residual| >2/3, leverage >2k/n, and Cook’s distance >4/n are review heuristics; k includes the intercept. Polynomial/interaction correlation can persist after centering. Influence is not computed by deleting observations. Partial residual plots for predictors in interactions are omitted to avoid a misleading single-curve interpretation.

The fit is cached per session and keyed by selected data values, row identifiers, types and the full specification. Plot-only and prediction changes reuse it. There is no automatic imputation, robust covariance adjustment, causal identification or correction for model-selection uncertainty. No Shapiro–Wilk, condition number/index, train/test evaluation or cross-validation is added to this regression feature.

## Implementation and validation

- `utils/regression_design.py`: fitted specification, levels/centers, hierarchy dependencies, reusable design matrix.
- `utils/regression_analysis.py`: OLS, grouped backward elimination, VIF, influence, prediction.
- `utils/regression_statistics.py`: extracted and reused existing coefficient/residual diagnostics.
- `utils/regression_reporting.py`: deterministic findings and formatting.
- `utils/regression_plots.py`: model-based plots using the prediction pipeline.
- `utils/regression_ui.py`: controls, session cache and seven result areas.
- `utils/regression_validation_ui.py`: preserved diagnostic interface with consistent p-value display.
- `utils/relationship_ui.py`: routes Linear regression to the expanded feature; correlation and other methods remain separate.

Run `python -m unittest discover -s tests -p "test_*.py"`. The extended tests compare statistics, covariance-based intervals and predictions with statsmodels; cover each interaction type, degree 2/3 polynomials, hierarchy-protected elimination, categorical encoding, missing data, high VIF, artificial influence, invalid specifications, and live Streamlit controls/cache/prediction. Existing correlation, ANOVA and other analytics tests remain included.

Statistical references: [statsmodels prediction](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_prediction.html), [joint F-tests](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.f_test.html), [VIF](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html).


### Retained interaction and polynomial plots

Regression Visual Analysis shows interaction plots only for interactions retained in the final model, and polynomial plots only for retained powers. Numeric moderators use their complete-case 25th, 50th and 75th percentiles; categorical moderators use separate curves by level. Categorical-by-categorical plots use category axes with confidence bars. Other predictors are held at displayed complete-case means or reference categories. Predictions reuse the fitted design and show pointwise 95% mean-response confidence intervals. Numeric axes stay within observed ranges, including category-specific ranges where applicable. These conditional displays do not adjust for model-selection uncertainty. Polynomial predictors with retained interactions show curves by an interacting predictor. Single-predictor polynomial plots preserve observed points and both confidence and prediction bands.

Column Overview now supplies effective Variable Type and Variable Subtype metadata. Discrete and Continuous predictors remain numeric; Binary, Nominal and Ordinal predictors use categorical handling. Ordinal order is preserved as metadata and in frequency displays; regression still uses categorical indicators. See [VARIABLE_CLASSIFICATION.md](VARIABLE_CLASSIFICATION.md).

Regression response selection is labeled **Response / dependent variable (Y)**. Predictors and Interaction terms each provide Select all and Clear all buttons; the response is excluded from predictors and clearing predictors removes dependent terms. The summary is a two-column table, followed by the **Fitted regression equation** heading and equation. Checkbox controls retain their existing layout.

Relationship provides independent Select all / Clear all controls for Numerical and Categorical columns. Polynomial terms use the same concise button labels and have no extra Select powers subheading.


## Shared Regression entry point

`utils/regression_router.py` routes from effective response metadata: Numerical remains OLS (including numerical 0/1), while Categorical → Binary selects classical logistic regression. Both reuse `regression_controls.py`, `regression_design.py`, the Selected terms reporting and adjusted-curve grid construction. Logistic fitting, diagnostics and UI are separate so OLS assumptions and tests are not reused for binary responses. See [LOGISTIC_REGRESSION.md](LOGISTIC_REGRESSION.md).
