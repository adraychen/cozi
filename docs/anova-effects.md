# ANOVA effect settings

Each selected categorical factor has a horizontal Fixed / Random radio control. Newly added factors default to Fixed. Factorial model-family labels update immediately. One-way Fixed identifies Welch in the Model summary Analysis method row and Statistical Results heading; its explanation is in Help. One-way Random shows its model name without an automatic-selection prefix. The comparison selection prunes removed factors, including when that selection makes ANOVA temporarily unavailable. Dataset and response changes reset unrelated settings; ordinary reruns retain choices. Saved report blocks are copied snapshots.

## Routing

| Settings | Engine |
|---|---|
| One factor, Fixed | Existing Welch ANOVA and Games–Howell |
| Multiple factors, all Fixed | Existing OLS Type II factorial analysis |
| One or more Random factors | Gaussian MixedLM variance components, REML |

The fixed engines and their numerical calculations are unchanged. Mixed models include selected Fixed main effects and selected Fixed × Fixed interactions in the mean. Random main effects and explicitly selected interactions involving Random factors define independent variance components. No interaction is automatically added or dropped.

## Group identity and dependence

For every random factor in a factorial model, a dropdown asks whether a label identifies the same real group everywhere or a group local to a selected parent. Local labels are encoded with the entire parent chain; crossed labels retain their global identity. Cyclic ownership is rejected. Observed one-parent mappings do not automatically establish nesting. Counts, observed/possible pairs, empty combinations and balance are shown; empty cells in a nested design can be structural.

A separate plain-language dropdown asks what dependence remains after the selected grouping structure. Unresolved designs, correlated slopes and additional serial/repeated-measure dependence block fitting. The existing identifier and chronological screening provides supporting evidence, not proof of independence. The one-way workflow does not pretend to model additional within-entity effects. Users can select additional grouping factors in Comparison when an identifiable factorial structure is suitable.

Fixed/Random guidance is deterministic and does not depend on an AI service. No objective metadata currently establishes a defensible automatic effect recommendation; category counts and identifier names never change a selection. Existing fixed-model AI explanations remain available as before.

## Numerical validation and inference

- Missing and nonfinite selected values are excluded and counted. Constant responses, insufficient levels and unusable numerical scales are rejected.
- Fixed design matrices must have full rank and residual information.
- Covariance kernels are projected into the fixed-effect residual subspace. A normalized Gram-matrix rank check rejects confounded components, aliases with residual variance, and redundant nested parent × child components.
- L-BFGS and Powell fit the same structure; the converged finite-likelihood candidate with the highest REML likelihood is retained. Optimizer failure never removes a term. Non-positive-definite information, invalid estimates and unusable fitted values are rejected even after reported convergence.
- Near-zero components receive boundary warnings. Fixed inference is withheld near a boundary or with fewer than five groups in any component. This threshold is a conservative UI safeguard, not a guarantee of accurate inference above five groups.
- When supported, fixed coefficients use statsmodels covariance standard errors, asymptotic normal Wald confidence intervals and two-sided z tests. These are large-sample coefficient tests, not Type II F tests, Satterthwaite tests or Kenward–Roger corrections. Reference groups are shown, and coefficients use actual labels.
- No random-component p-values or variance confidence intervals are reported. No ordinary chi-square boundary test is used. There are no likelihood comparisons of different fixed-effect models; any future implementation must use ML for those comparisons.
- Variance, SD, proportion of total conditional variance and residual variance are reported separately from fixed coefficients. A single-component model reports the shared-group ICC. Multiple-component models explain that pairwise correlation depends on which components two observations share.

## Output

Native content-sized tables follow the Dataset Preview standard. Charts use the shared presentation wrapper: variance bars, descriptive distributions, fixed coefficient intervals when available, conditional residuals versus fitted responses, and a residual Q–Q plot. The distribution-factor dropdown changes only the descriptive plot. All fitted structure settings and results are included in report snapshots.

## Supported scope and limits

Supported: identifiable crossed and explicitly nested independent Gaussian random intercepts, selected independent interaction components, and balanced/unbalanced data with a common residual variance. Ordinary fixed workflows retain their existing scope.

Unsupported: correlated random slopes, serial correlation, heterogeneous residual variances, general repeated-measures covariance structures, non-Gaussian generalized mixed models, and small-sample corrected fixed-effect inference. Complex designs requiring these structures are blocked through design clarification rather than silently approximated.

Resource limits: 4–1,500 complete observations, 2–200 groups per component, at most 12 random components and 500 total random group coefficients. These are implementation bounds, not statistical sufficiency rules. Users should not drop necessary terms merely to meet a limit.

## Validation

`tests/test_anova_effects.py` covers routing/defaults, factor removal/readdition, dataset changes, random-to-fixed switching, immutable report snapshots, crossed/nested/reused labels, unbalanced data, confounding, nonfinite observations, singular/nonconvergent fits, and fixed-effect Wald calculations.

Reference fixtures are reproducible in that test file. Balanced one-way, crossed-with-interaction and nested fits are compared with independent expected-mean-square estimates at interior solutions. The unbalanced one-way reference directly profiles the dense Gaussian REML likelihood using SciPy, without MixedLM. Existing ANOVA suites verify unchanged Welch/Games–Howell and Type II results. `tests/anova_browser_fixture.py` provides a local synthetic UI fixture for real-browser checks without user data or AI calls.

Statistical implementation references:

- [statsmodels variance-component examples: nested and crossed designs](https://www.statsmodels.org/stable/examples/notebooks/generated/variance_components.html)
- [statsmodels linear mixed-effects model documentation](https://www.statsmodels.org/stable/mixed_linear.html)
- [MixedLM result covariance and inference methods](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.html)

### Verification outcome (2026-10-15)

- All 11 focused ANOVA effect tests pass, including independent numerical references and Streamlit state/report checks.
- Complete suite: 361 tests run; 359 pass. Two existing failures reproduce with the original ANOVA renderers in an isolated copy: `test_fill_missing.FillMissingUITests.test_group_navigation_preview_apply_and_create` expects an obsolete dropdown placeholder, and `test_temporal_target_guidance.GuidanceUITests.test_workspace_preserves_controls_without_persisting_action` cannot find its target checkbox. Neither failure originates in the ANOVA changes.
- Real-browser checks use the synthetic fixture: one-way Fixed/Random routing, factorial Fixed/Mixed routing, group-identity and dependence clarification, result tables, diagnostics and charts. Desktop and mobile-width layouts were inspected; tables retain horizontal scrolling, and long chart titles were shortened for mobile.

### One-way effect-size explanation

The result shows Cohen f² (Welch) and its value. Interpretation is in Help → Effect Size (Cohen's f²). The results table labels are df numerator and Effect Size (Cohen's f²); stored calculation keys remain unchanged for compatibility. Technical method: the existing unequal-variance plug-in estimate uses group-size/variance weights (`effectsize_oneway`, `use_var="unequal"`, `ddof_between=0`). It is not bias corrected, is not a percentage of explained variation, and is not computed from the descriptive SS table or Welch-corrected F. Calculations are unchanged.
