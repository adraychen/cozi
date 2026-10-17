# Guided Factorial ANOVA interactions

## Workflow and implementation

New analyses default to **All interactions**, jointly fitting every two-way combination of the selected factors. The **Interaction effects (optional)** checkbox grid starts with all pairs selected and provides **Select all** and **Clear all** buttons. Clearing every pair fits only the main effects. No higher-order terms are generated. Expand **Which interaction effects should I select?** for selection guidance. Settings update results automatically; no Apply button is required.

The same selector works for fixed, random and mixed models. Fixed × Fixed is a fixed interaction; any pair containing a random factor is a random interaction in the intended crossed-factor variance-components formulation. Random/mixed fitting requires study-design clarification in the **Review advised** expander above the result tabs. Once confirmed, results appear automatically and the expander is labeled **Study Design**. Diagnostics contains model checks rather than assignment controls. Interaction and significance-level changes retain study-design answers; changes to data, factors or effect types invalidate them. Nested ownership expands grouping identities through the existing engine; it is not silently interpreted as crossing. Redundant child × parent components, cyclic ownership and confounded structures are blocked.

| File | Responsibility |
| --- | --- |
| `utils/factorial_interactions.py` | Pair generation/classification, support tables, Holm assessments, response-unit interaction contrasts, nested OLS comparisons, mixed fixed-interaction Wald tests |
| `utils/factorial_interactions_ui.py` | Shared selection, Interaction Insights, optional AI, confirmed refinement |
| `utils/factorial_random_ui.py` | Random/mixed results within the existing five result tabs; study-design clarification and guided diagnostics |
| `utils/factorial_anova_ui.py` | Fixed workflow integration, synchronized assessments/plots/follow-up, reports |
| `utils/anova_effects.py` | Existing REML engine, bounded fixed design allocation, validated coefficient mapping and additional rank/support metadata |
| `utils/ai_interpreter.py` | Optional explanation of formatted validated summaries using the existing model/client |
| `tests/test_factorial_interactions.py` | New statistical reference, selection/state, refinement, report and AI tests |
| Existing factorial/comparison tests | Explicit manual-mode scenarios and current response/factor display expectations |

The earlier **Factor Effect Type** presentation change remains: normal-sized bold label and an expandable Fixed/Random definition box.

## Statistical methods

### Fixed factors

The original hierarchical OLS engine and Type II sums of squares remain. Main effects are retained. The ANOVA table preserves raw p-values and adds adjusted interaction p-values. Holm adjustment covers **all fixed interaction tests in the jointly fitted model**, in either selection mode. Interaction significance decisions in findings, plots and simple-effects eligibility use those adjusted values. Main-effect tests retain their established interpretation. Adjustment does not cover later models, outcomes, or data-driven refinement.

Interaction Insights includes DF, F, raw/adjusted p-values, partial η², estimability and pair support. Partial η² = SS_effect / (SS_effect + SS_residual); it is conditional on the model and is not labeled a percentage of explained variation. A difference of differences for the first/last observed levels provides a response-unit direction/magnitude illustration. Other factors are averaged equally; Bonferroni simultaneous 95% intervals cover one such contrast per selected interaction. This illustration is not the full omnibus interaction for factors with more than two levels.

Existing adjusted-mean plots, pointwise mean intervals, simple-effects comparisons, Holm pairwise adjustment and Bonferroni simultaneous comparison intervals remain available. Games–Howell is not substituted for factorial comparisons. HC3 remains an explicit separate sensitivity option.

### Random factors

The existing Gaussian independent variance-components engine uses REML, with one overarching MixedLM group and explicit component design matrices for supported crossed or explicitly nested structures. It reports factor, interaction and residual variances, standard deviations, total modeled variance and component percentages. Covariance-kernel rank is checked jointly after projecting out fixed effects, including the residual kernel, before fitting.

Random interaction assessments remain **Inconclusive** about practical importance/uncertainty. A nonzero variance does not establish importance. Variance-component confidence intervals, bootstrap tests and calibrated boundary likelihood-ratio tests are **not implemented**. No ordinary chi-square reference distribution or fixed-effects F test is used for these components. Recommendations concern reviewing magnitudes, combination distributions, replication and design, rather than eliminating components by p-value.

### Mixed factors

Fixed coefficients use the existing asymptotic normal Wald inference only when the existing health/group-count checks allow it. Selected fixed × fixed interactions additionally receive joint asymptotic Wald chi-square tests using the fitted fixed-coefficient covariance, with Holm adjustment across the fixed-interaction family. Coefficients are matched by name to the validated design matrix before testing. Fixed × random interactions remain independent Gaussian variance components, not OLS interaction tests.

No small-sample correction, mixed-model adjusted means/simple-effects inference, calibrated random-component test, or ML/REML likelihood comparison is introduced. REML likelihood comparisons across different fixed structures are not performed. These limits are stated in the interface; unsupported covariance structures are withheld.

## Feasibility and diagnostics

The existing fixed engine checks complete observations, factor levels, joint combinations, empty/singleton cells, parameter count, residual DF, design rank and finite Type II inference. Random/mixed checks include fixed-design rank, residual-subspace dimension, covariance-component rank, distinct groups, group count/size limits, coefficient count, optimizer convergence, positive-definite information, boundary variances and fixed-covariance health. Dense fixed-design construction is bounded before allocation.

Interaction support tables show possible/observed pairs, empty pairs, single-observation pairs and minimum counts without allocating a full Cartesian product. Pair counts pool other factors and cannot establish full-model estimability. Full-model concerns are explained in Diagnostics as what was found, why it matters, recommended action and how to act. Failures list the requested interactions as unavailable **in the requested model**; they do not claim to identify a uniquely culpable term when only a joint confounding failure is known. No terms are dropped and no partial fit is labeled a successful full model.

Overview carries a compact diagnostic status and only critical model alerts. Study-design questions, sparse support, boundary/convergence concerns and limitations live in Diagnostics. Significant interactions are findings, not diagnostic problems. Random/mixed study-design answers reset for relevant dataset/data/model changes; assumption-only changes reuse the fit. See [plain-language study-design diagnostics](FACTORIAL_STUDY_DESIGN.md) for the updated confirmation workflow and Productivity walkthrough.

Current random/mixed resource limits remain 4–1,500 complete observations, 2–200 groups per component, at most 12 random components and 500 random coefficients. The fixed part is additionally limited to 1,000 parameters and one million design-matrix entries before construction. Fixed models retain their existing resource limits.

## Interaction Insights, AI and reports

**Interaction Insights** appears within Model Overview and ANOVA Findings, using **What We Found**, **What It Means**, and **What to Do Next**. Assessments, uncertainty, practical-importance qualifications and executable follow-up guidance come from deterministic code. Fixed findings remain conditional on classical assumptions; random findings remain descriptive. Observational associations are not presented as causes.

AI is optional and runs only on an explicit button press. It receives formatted result tables, effect types, selection mode, actual model structure, variance components/fixed coefficients where applicable, diagnostics, available follow-up and any existing analytical-objective state. It receives no fitted objects or raw records. Prompts prohibit invented results, unsupported significance, causal claims and automatic term changes. The existing numerical-literal validator rejects invented/re-rounded numeric strings; this checks numbers, **not the truth of every AI sentence**. Advisory AI prose never controls calculations or assessments. Failed/unavailable AI leaves all deterministic findings available. Context and numerical-formatting digests prevent stale AI output.

The existing report snapshot flow captures insights, formatted tables/charts, model settings/effect types and material limitations. Interactive controls use Streamlit directly and are excluded from captured report content. Refit/refinement never changes existing snapshots or user-edited report text. No report redesign was made.

## Confirmed model refinement

Users choose interactions to consider removing. The model remains intact until **Confirm removal and refit**. All main effects remain, and the result is a new manual-mode model. Fixed models display a classical nested OLS F comparison on identical complete-case rows. This relies on independent homoscedastic errors and is explicitly exploratory after selection; it is not a universal elimination rule. When a comparison is unavailable, that limitation is shown rather than fabricating evidence.

Random/mixed refinement offers an explicit scientifically justified revision with no invalid universal comparison test. Descriptive variances and diagnostics provide context; no automatic removal rule or uncalibrated boundary test is supplied. Changing the fixed structure does not trigger a REML likelihood comparison.

## Verification

New references independently verify fixed interaction F statistics using least-squares projections; random REML variances using balanced expected-mean-square identities; mixed fixed estimates using direct generalized least squares; and joint fixed-interaction Wald tests using the GLS covariance and chi-square survival function. Existing balanced/unbalanced ANOVA, profile-REML, boundary/nonconvergence, nested-confounding, follow-up/HC3 and formatting tests remain in the regression set.

UI tests cover two-way generation, default exploration, manual-mode preservation, factor changes, stale-result withholding, confirmed refinement, main-effect retention, random study-design gating, confirmed group-structure changes, failed-fit suppression, immutable report snapshots, AI failure fallback and invented-number rejection. Workspace tests cover persistence of mode and exclusion of ephemeral AI/refinement buttons. A pre-allocation test verifies rejection of excessive mixed-model fixed designs without building a dense matrix.

Verification results: the combined run passed **115 tests**, and the subsequently added allocation-guard test passed separately (**116 distinct tests**). The final group-structure/workspace checks passed in a focused three-test rerun after that last change. Logs: `tmp/interaction-final-tests.log` and `tmp/interaction-structure-tests.log`. No live AI service was called; tests use deterministic mocked responses and failure cases.

The initial sandboxed browser preview could not connect. Restarting the preview outside the sandbox resolved it. The fixed-model controls and rendered results were inspected in the browser; interaction controls were visually checked at desktop, 390-pixel mobile and 768-pixel tablet widths. Preview images are in `tmp/factorial-interactions-preview.png`, `tmp/factorial-interactions-mobile.png` and `tmp/factorial-interactions-tablet.png`. Random/mixed interactive behavior was checked through Streamlit AppTest, rather than a full manual browser session for every model family.

Run the focused set with:

```powershell
.venv\Scripts\python.exe -B -m unittest tests.test_factorial_interactions tests.test_factorial_anova tests.test_factorial_guided_workflow tests.test_factorial_workspace_actions tests.test_anova_effects tests.test_anova tests.test_comparison_ui tests.test_report tests.test_numerical_formatting tests.test_factorial_guidance
```

Method references: [statsmodels ANOVA implementation](https://www.statsmodels.org/stable/_modules/statsmodels/stats/anova.html), [MixedLM implementation and supported covariance formulation](https://www.statsmodels.org/stable/_modules/statsmodels/regression/mixed_linear_model.html).
