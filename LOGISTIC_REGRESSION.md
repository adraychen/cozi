# Binary logistic regression in Relationship → Regression

## Scope and setup

This is classical statistical regression for relationships, inference, model adequacy and probability prediction. It is separate from Predictive Analytics. The shared Regression response selector uses effective Column Overview metadata: Numerical → OLS; Categorical → Binary → binomial-logit. Numeric 0/1 remains numeric until reclassified. Both observed outcome levels must survive complete-case filtering. The event is explicitly selected; no assumption that numeric 1 is the event. Outcome mapping is internal and original labels are displayed.

Predictors use the effective numerical/categorical groups. Categorical Binary/Nominal/Ordinal predictors have k−1 indicators and user-selected reference levels. Ordinal predictors are categorical; equal-spacing numeric treatment is not implemented. Unprocessed dates and complex objects are excluded. More than 100 factor levels is rejected; text with median length >100 characters or >15 words is treated as unsupported free-text-like data. These are usability heuristics, not semantic identification. Derive meaningful categories before analysis. Numeric identifiers are not meaningful merely because they are eligible.

Missing/nonfinite values in the response and selected predictors are excluded with row counts and percentage reported. Selected reference levels must exist among complete cases. The original complete-case sample stays fixed for all model comparisons. Source row numbers and index labels are preserved. No imputation or automatic row deletion is added.

Confidence defaults to 95% and is configurable from 51% to 99.9%. The selected level applies to coefficient/OR intervals, probability intervals, curve bands and observed-proportion calibration intervals. Exploratory diagnostic significance screens retain α=.05; the confidence selector does not change the diagnostic testing rule.

## Shared terms and model fitting

The existing main-effect, two-way interaction and quadratic/cubic controls are reused, including Select all/Clear all. Centering uses the original complete-case means whenever powers or interactions are specified. Prediction reuses those means, levels and retained design columns exactly. Required main effects and lower powers cannot be removed while their dependent higher terms remain. The shared 300-coefficient design limit remains; more complete observations than estimated parameters and full design rank are required.

`statsmodels.GLM(..., family=Binomial())` uses its default logit link with maximum-likelihood IRLS estimation. There is no regularization or alternative ML fit. Coefficients, standard errors, covariance intervals, Wald tests, likelihood, deviance and influence use statsmodels. Odds ratios and their bounds exponentiate the coefficients and covariance-based confidence limits. Individual Wald statistics are squared z statistics (χ², one df); whole factor/interaction tests are joint Wald χ². Overall LR compares with an intercept-only model on exactly the same cases. BIC uses the log-likelihood definition, not the alternative GLM deviance convention. McFadden Pseudo-R² is not variance explained.

No elimination is default. Optional backward AIC evaluates each hierarchy-eligible whole-term removal and chooses the lowest AIC only if it improves by more than numerical tolerance. It stops when no eligible improvement remains. Failed candidate fits are reported, not silently treated as successful comparisons. This is a greedy exploratory procedure, not exhaustive model search. p-values and intervals are conditional on the final model and omit selection uncertainty.

## Seven tabs

1. **Model Overview:** model summary, event/reference and row counts, equation, fit statistics, Selected terms/centering/final status, category references and elimination history.
2. **Predictor Analysis:** coefficient/OR table with selected-level confidence intervals, individual Wald and joint term tests, CSV download.
3. **Diagnostics:** convergence, separation, event information, logit linearity, category/outcome counts, deviance and Pearson residual plots. No normality or constant-variance tests.
4. **Multicollinearity & Influence:** VIF/tolerance, original-row influence table, leverage, Cook’s distance, DFBETAs and diagnostic download. No condition number/index.
5. **Visual Analysis:** calibration with observed proportions and Wilson intervals; odds-ratio plot; adjusted probability, retained-interaction and retained-polynomial curves. Optional collapsed ROC/AUC is discrimination only, not an acceptance criterion.
6. **Prediction:** complete-case fitted probabilities and CSV, manual original-unit/category inputs and event-probability confidence intervals. No threshold/classification output and no linear-regression-style individual prediction interval.
7. **Regression Findings:** event-specific association interpretations and Model Adequacy with explained concerns/actions, plus user visual/study-design review and downloadable findings.

## Diagnostics and practical limits

Separation uses bounded-time linear programming on column-scaled signed design rows. Feasibility of all margins ≥1 detects complete separation; a positive total margin with all margins ≥0 and bounded direction detects quasi-complete separation. Confirmed separation blocks ordinary ML inference. Solver failures are reported as unavailable diagnostics. Statsmodels separation warnings, non-convergence and nonfinite estimates/covariance also prevent reliable-result output. Near-separation symptoms (large scaled coefficients/SE >20, extreme probabilities within 1e−8 of a boundary, overflowed OR intervals) produce explicit concerns; symptoms alone are not mathematical proof of separation.

Outcome information is screened using the smaller outcome count relative to estimated parameters, including the intercept. Below 10 observations per parameter is a review warning only, not a hard 10-events-per-variable rule. Rare levels are <5 cases or <1% of complete cases; outcome cells <5 and zero-event/non-event cells are flagged. Pre-fit category counts remain available even when separation prevents estimation. No categories are combined automatically.

VIF bands: <5 generally acceptable, 5–10 investigate, >10 strong concern. Influence review thresholds: |Pearson/deviance residual|>3, hat value>2k/n, Cook’s distance>4/n, or maximum |DFBETA|>2/√n. These are heuristics; no observations are removed. Infeasible influence computations are reported as unavailable.

Logit-linearity checks apply to retained additive Continuous predictors: compare the final model with added centered/scaled quadratic and cubic terms using a 2-df LR test on the same rows, with Holm adjustment across available tests. Zero/negative values require no shift or logarithm. This detects some polynomial departures, not every possible misspecification. Retained powers are identified as an explicitly nonlinear specification; interaction predictors need conditional visual review. Singular, separated or underpowered auxiliary fits are reported as unavailable. No automatic transformation is applied.

Calibration uses up to ten predicted-probability quantile groups (fewer for small samples/ties), observed proportions, Wilson intervals and Brier score. Groups below 20 cases trigger a precision warning. Two or more sufficiently populated groups with mean predictions outside the observed intervals trigger an exploratory review warning, not a formal goodness-of-fit test. All calibration is in-sample; no generalization claims are made. Calibration intercept/slope are deliberately omitted because the fitting-sample values can be ideal by construction. Hosmer–Lemeshow is not implemented.

Model Adequacy reports No major issues detected, Review recommended or Significant model concerns, with evidence/actions. This is not certification. Independence and visual calibration require study-design/user review; unassessed diagnostics are not passes. No arbitrary numerical score, no automatic alternative model and no causal claims are produced.

## Files and verification

- `utils/regression_router.py`: metadata-driven shared entry point.
- `utils/logistic_analysis.py`: event mapping, estimation, hierarchy-preserving AIC, inference and probability prediction.
- `utils/logistic_diagnostics.py`: separation-related stability, sample/sparse/VIF/linearity/influence/calibration summaries.
- `utils/logistic_ui.py`: setup, caching, seven tabs, graphics and findings.
- Shared regression design, controls, reporting and adjusted-curve helpers preserve existing OLS behavior.
- `tests/test_logistic.py`: independent statsmodels Logit/formula comparisons, reference/event direction, confidence intervals, categories/ordinal/interactions/powers, missing data, hierarchy, nonlinearity, VIF, sparse/imbalanced outcomes, separation and non-convergence.
- `tests/test_logistic_ui.py`: metadata routing, categorical-only eligibility, seven tabs, controls, cached prediction and switching back to linear regression.

Run `python -m unittest discover -s tests -p "test_*.py"`. Full-app Streamlit checks mock database tracking to avoid external writes.

Statistical library references: [GLM](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html), [prediction](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.get_prediction.html), [influence](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.get_influence.html).
