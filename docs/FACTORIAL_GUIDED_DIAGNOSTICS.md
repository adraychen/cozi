# Guided Factorial ANOVA diagnostics

The fixed-effects Factorial ANOVA uses the existing Type II calculation engine.
No fitting, significance, marginality, adjusted-mean, simple-effect or multiplicity
calculations were changed. Guidance runs offline without an AI service.

Response and factor selections are displayed as plain text, using the variables
selected in Comparison. There are no duplicate Response (Y) or Factors (X)
dropdowns within Factorial ANOVA. Effect types and interaction choices remain
editable because they specify the model.

## Interface and message placement

- **Model Overview:** one diagnostic status, response/factor model information,
  Type II method information, observation and exclusion counts, ANOVA results,
  and the existing optional HC3 sensitivity comparison. Detailed Type II
  marginality guidance is in a contextual expander. Excluded source rows can be
  inspected with their exclusion reason.
- **Diagnostics:** structured items for Cell Coverage, Model Complexity, Model
  Estimability, Independence of Observations, Residual Diagnostics and Variance
  Assessment. Each has an observed result, impact, recommendation and supported
  action. Items without detected concerns start collapsed; review items open.
- **Effect plots / Post-hoc comparisons / ANOVA Findings:** significant
  interactions are analytical findings. Existing interaction plots, adjusted
  means, Holm comparisons and Bonferroni simultaneous intervals remain available.
  Main effects remain visible with contextual interpretation, rather than being
  automatically suppressed. Games–Howell is not used for factorial follow-ups.

The former general warning collection above the results is removed. Empty or
singleton cells and limited model support are covered by the specific design
items. The old observations-per-parameter heuristic is replaced in the UI by
actual counts, model parameters, residual degrees of freedom and validated rank.
No universal minimum sample rule or automatic model reduction is imposed.
Blocking fitting/estimability failures have a prominent Model issue status in
Overview and explanations/actions in Diagnostics. Invalid inference and report
actions for that failed fit are withheld.

## Supported actions

- Open Diagnostics selects the Diagnostics tab.
- Review Cell Coverage expands the count details. Two-factor matrices use actual
  counts and muted sparse-cell shading. The full-factor table can show all
  combinations or filter empty, singleton and two-observation combinations.
- Review Interactions / Model Terms links to the real configuration controls.
- Study-design answers record independent observations, repeated/clustered
  observations or uncertainty. Identifiers, grouping labels and temporal metadata
  are indicators only. Repeated labels are not automatically treated as repeated
  experimental units. Absence of indicators never proves independence.
- Reported repetition exposes guidance and navigation to the existing fixed/random
  effect controls. It does not automatically select or fit a mixed model.
- View residual plots links to the existing plots. Review HC3 sensitivity option
  opens Model Overview without enabling HC3 automatically.
- Changes to response, factors, effect types, interactions or alpha require
  **Apply changes and rerun** after an initial configuration. Pending changes
  withhold results rather than display findings under mismatched settings.

## State, report and limits

Study-design answers are scoped to a fingerprint of source data, variable
metadata and analysis configuration. Changed data, variables or model settings
reset those answers. Existing dataset-scoped `fac_` state is preserved. Random
model structure confirmations also reset when the source context changes.

Factorial action buttons (`fac_open_diagnostics`, `fac_apply_model`, and
`fac_action_*`) are excluded from workspace state retention, dataset snapshots
and restoration. Streamlit forbids assigning button values through session
state, including assigning `False`. Old saved datasets containing these action
keys are filtered when restored. Integration tests exercise actual workspace
reruns, navigation actions, model confirmation and dataset switching.

The existing `factorial` report component identifier is preserved. Report
snapshots include the structured diagnostic records and study-design
qualification. Configuration and navigation controls are not captured as report
content. A later rerun does not update a saved snapshot or user-edited report
text. Pending or invalid fits clear the latest unsaved factorial result while
leaving saved report blocks untouched.

The existing mixed/random-effects estimation workflow is retained. Its supported
Gaussian group intercepts and independent interaction components do not provide
correlated random slopes, serial correlation or heterogeneous residual variance.
This change does not implement any new statistical procedure or redesign One-way
ANOVA. Unknown study design permits qualified review of an otherwise valid fit;
it does not establish validity. Follow-up contrasts retain their separate
estimability guards.

Full Cartesian cell tables are capped at 5,000 possible combinations; above
that limit the table lists observed cells and the summary reports the empty
count. Pairwise matrices are shown for manageable designs (up to 400 cells).
Counts are based on observed factor levels after complete-case filtering;
entirely absent planned levels cannot be inferred from the data.

## Files and verification

Implementation: `utils/guided_diagnostics.py`, `utils/factorial_guidance.py`,
`utils/factorial_guidance_ui.py`, `utils/factorial_anova_ui.py`.

Tests: `tests/test_factorial_guided_workflow.py`,
`tests/test_factorial_guidance.py`, `tests/test_factorial_anova.py`.

Focused coverage includes counts and bounded expansion, sparse but estimable
additive models, blocked rank-deficient models, configuration confirmation,
navigation, dependence indicators, answer persistence/reset, immutable report
snapshots, report control exclusion, existing Type II calculations, HC3,
adjusted means, plot swaps and simple effects. Browser verification uses synthetic
data at desktop, 768 × 1024 tablet and 390 × 844 mobile sizes; mobile document
width remained within the viewport and cell matrices remained contained.

Validation results: 45 focused factorial/effect tests passed. The final full
suite ran 393 tests: 391 passed, with the same two pre-existing failures in
`test_fill_missing.FillMissingUITests.test_group_navigation_preview_apply_and_create`
(obsolete menu placeholder expectation) and
`test_temporal_target_guidance.GuidanceUITests.test_workspace_preserves_controls_without_persisting_action`
(missing temporal target checkbox). These are outside the factorial changes.
Final workflow-specific navigation and report tests also passed independently.
