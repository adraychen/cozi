# Guided study-design diagnostics for Factorial ANOVA

## Scope and files

Random/mixed Diagnostics now asks how people or units were assigned and how measurements were collected. Observed data informs suggestions; explicit user confirmations configure the model. Fixed-effects methods, interaction selection/results, the REML engine and the Report editor are retained.

| File | Change |
| --- | --- |
| `utils/factorial_study_design.py` | Complete-case design evidence, crossed/partial/confined-label detection, contextual wording, time-column checks, deterministic eligibility and inference suppression |
| `utils/factorial_study_design_ui.py` | Suggested assignments, explicit confirmations, progressive collection questions and confirmed shared-group updates |
| `utils/factorial_random_ui.py` | Integrates decisions with fitting, statuses, reports and existing result tabs |
| `utils/factorial_anova_ui.py` | Invalidates design state before pending-model and invalid-selection early returns |
| `tests/test_factorial_study_design.py` | Productivity and synthetic evidence, mapping, state, assumption, report and offline-guidance tests |
| `tests/test_factorial_interactions.py` | Existing random workflow test now uses plain-language confirmations |
| `tests/test_factorial_workspace_actions.py` | New action buttons excluded from persisted dataset state |
| `tests/productivity_browser_fixture.py` | Reproducible browser walkthrough of the actual bundled CSV |
| `README.md`, `docs/ui-controls.md`, `docs/FACTORIAL_INTERACTIONS.md` | Updated workflow references |

## Observable facts and suggestions

Detection uses the selected response/factors' complete, finite response rows. For each random factor and each other selected factor, it counts labels, observed/possible pairs, labels spanning groups, empty/single-observation pairs and replication. Full-design counts are also recorded. Cartesian products are counted mathematically, not allocated.

- Every label at every other-factor level: suggest globally consistent identities.
- Each label restricted to exactly one level of one unambiguous parent: suggest local ownership within that parent.
- Partial coverage, multiple plausible parents or insufficient levels: no single recommendation.

These are descriptions of labels, **not identified people or a proven experimental design**. Reused local IDs can look identical to globally consistent IDs. Globally unique IDs and missing combinations do not prove nesting. A partial pattern is retained as partial; it is not silently relabeled fully crossed or nested.

Clear suggestions are preselected with **Suggested from data**. They do not authorize a fit. Each random factor requires its own **Confirm … assignment** action. Changing its answer invalidates that confirmation immediately. **I'm not sure** leaves identity unresolved and fitting withheld, with Review advised rather than a critical error.

Confirmed global identities map to the existing crossed formulation with no parent entry. Confirmed local ownership maps to `parents[child] = parent`. Actual covariance construction, nested expansion, cycle checks and component identifiability remain the existing engine's responsibility. A nested child × parent interaction that duplicates the child component remains blocked; no term is removed to obtain a fit.

## Measurement collection and eligibility

The collection question appears only after every random-factor assignment is confirmed. No collection answer is preselected. Examples use deterministic manufacturing, healthcare, education or general vocabulary based on the selected column names. Unknown names use the actual column names and generic people/units wording.

| Answer | Additional controls and behavior |
| --- | --- |
| Separate production runs / measurements | Records the user's expectation of no additional relationship beyond the modeled factors/combinations. Fits the otherwise valid model; does not claim independence was proven. |
| Sequence or time | Shows possible time/sequence columns and asks whether nearby measurements may share additional changing conditions. CSV row order is never used as measurement order. |
| Sequence, additional connection **Yes** | Model Issue: no fit/inference is presented as accounting for serial correlation. Explains the needed time/order and unit information and an appropriate time-correlated/repeated-measures model, which is unavailable here. |
| Sequence, **No additional connection expected** | Conditional eligibility based on the user's collection-process knowledge. This is an assumption, not a data test. |
| Sequence, connection **I'm not sure** | Review advised; conditional estimates may be explored, but fixed-effect p-values, intervals and fixed-interaction tests are withheld. |
| Shared common source | Asks which existing column identifies it and requires confirmation that identical labels identify the same source across factors, with no other connection expected. |
| Unknown collection | Review advised. The dataset is not declared invalid. Shows conditional estimates while withholding fixed-effect p-values/intervals and inferential interaction conclusions. |

Time/sequence suggestions come from date/time types, existing metadata and relevant column-name tokens. A selected column must contain at least two usable, distinct numeric or date/time values to be described as usable. Missing/invalid counts are reported. A column's presence never proves serial dependence, supplies a series identity, or changes the covariance model.

For an additional shared source, the existing `prepare_random` checks whether an additive random grouping can be represented. A structurally eligible proposal is included **only** after **Add … as a shared-source grouping and refit**. The selected factor effect types and interaction terms are retained; no new interactions are silently added. Fitting can still fail its subsequent convergence or information checks, in which case all fit results are withheld.

An already modeled random grouping can satisfy the shared-source answer after confirmation. A currently fixed factor cannot silently substitute for random residual clustering. Contradictory local/global identity answers are blocked. Confounded, unique-per-row or otherwise unidentifiable additional groupings are blocked with the engine's reason. Additional sources with locally reused IDs require reviewing the explicit factor/ownership configuration; this shortcut does not guess their parent structure.

## Statuses and actions

Every study-design item uses the existing Diagnostic schema and explains **What Cozy found**, **Why it matters**, **Recommended action**, and **Take action**. Relevant selectors and confirmation buttons are placed immediately with the explanation. Conditional controls remain hidden until needed. Advanced covariance terminology is in an expandable explanation.

- **Checks passed:** applicable study-design confirmations are consistent with the model. Independence is still an assumption.
- **Review advised:** identity is unconfirmed, the collection process is unknown, or information is incomplete. Whether a fit can be explored depends on whether the ambiguity changes group identity or only an assumption.
- **Model issue:** confirmed additional dependence is unsupported, answers imply contradictory groupings, or numerical/model specification checks fail.

Diagnostics shows a separate study-design status; the existing overall model status can still advise review for other issues, including unavailable variance-component uncertainty. Only critical issues get a prominent Overview alert. Interaction significance is not made a diagnostic problem.

## State, fitting, AI and reports

Design state is scoped by dataset identity, data content, response, selected factors/effect types/interactions and relevant metadata. Data/factor/model changes invalidate confirmations even if the page returns early while model changes await application. Unrelated datasets cannot reuse confirmations. Valid dataset-specific state remains compatible with the existing workspace save/restore mechanism.

Alpha, numerical formatting and selection-mode labels do not by themselves change who or what was measured, so they do not discard design confirmations. A changed parent/group structure withholds stale results until refitting is confirmed. A collection-assumption-only change reuses the same cached numerical fit: unknown collection strips inference from a copy, while a confirmed compatible assumption restores the validated cached inference. The cached fit is never mutated. Known unsupported dependence prevents output rather than being ignored.

All assignment suggestions, selected answers, confirmations, observed evidence, extra grouping and material limitations are stored with the analysis/report configuration. Report snapshots and user-edited interpretations are not rewritten by changed answers or refits. New action events remain excluded from saved dataset controls.

No AI call is needed for design detection, wording, eligibility or model configuration. The existing optional Interaction Insights AI receives the deterministic design diagnostics and limitations with validated results. It cannot determine identity, infer the collection process, override checks, change terms or claim independence verified. Unknown collection supplies no fixed-effect p-values or intervals to that explanatory layer. Offline tests verify the complete design workflow without AI access.

## Productivity verification walkthrough

Use `data/samples/Productivity.csv` with **Score** as the response, **Machine: Fixed**, **Operator: Random**, and **Machine × Operator** selected. The file remains unmodified.

1. Open Diagnostics. Cozy detects **54 observations**, **3 machines**, **6 operator labels**, **18 observed combinations**, **3 observations per combination**, and no empty/single-observation pairs.
2. It preselects **The same operators work on different machines**, marked Suggested from data. There are no estimates yet and the collection question is hidden.
3. Confirm that identical Operator labels identify the same actual people using **Confirm Operator assignment**. The collection question now appears with no default answer.
4. Choose **Separate production runs** only if that describes the real collection process. The existing mixed model fits with Machine fixed, Operator random and their random interaction retained. Study-design checks pass conditionally; the overall model can still flag uncertainty limitations.
5. Selecting **I'm not sure** about collection permits conditional exploration but removes fixed-effect p-values/intervals. No new numerical fit is needed for that assumption-only change.
6. Selecting sequence/time with a known additional connection produces a Model Issue explaining that the CSV has no identified order/time column and the current covariance engine does not model serial correlation.
7. Changing to local Operator ownership requires a new confirmation and **Apply group structure and refit**. With the original interaction retained, the existing engine detects redundant/confounded nested components and withholds results. It does not drop that interaction.
8. Existing report snapshots retain the original confirmations and limitations after these changes.

For browser verification:

```powershell
.venv\Scripts\python.exe -B -m streamlit run tests/productivity_browser_fixture.py --server.address 127.0.0.1 --server.port 8511 --server.headless true
```

## Tests and limits

The new tests cover full/partial/reused/confined label patterns; ambiguous parents; empty and sparse combinations; all four collection paths; usable/missing order columns; unsupported serial dependence; valid extra grouping through the unchanged engine; contradictory identities; explicit confirmation/refit; dataset/data/factor invalidation; cached assumption changes; immutable reports; and offline guidance. Existing fixed/random/mixed numerical references remain in the full suite.

Verification results:

- All **20 new study-design tests** pass. The focused new/affected set passes **23 tests**. The nine evidence tests also passed after adding nullable-response complete-case coverage.
- Full discovery ran **429 tests: 427 passed, one failure and one error**. The two non-passing tests reproduce with the committed `HEAD` factorial implementations substituted in memory, leaving working files untouched. They are outside this change: `test_fill_missing.FillMissingUITests.test_group_navigation_preview_apply_and_create` expects an old operation-placeholder option; `test_temporal_target_guidance.GuidanceUITests.test_workspace_preserves_controls_without_persisting_action` cannot find its expected target checkbox.
- Logs: `tmp/study-design-full-suite.log`, `tmp/study-design-focused-tests.log`, and `tmp/study-design-baseline-failures.log`. The baseline reproduction script is `tmp/verify_study_baseline_failures.py`.
- `node tests/test_dataset_scroll.cjs` passes. `git diff --check` passes.
- The actual Productivity browser flow was verified: unconfirmed assignment → confirmation → collection question → Separate production runs → study-design **Checks passed**, with the mixed model retained. Desktop, 390-pixel mobile and 768-pixel tablet layouts were checked; mobile/tablet page width matched viewport width. Preview images: `tmp/productivity-study-design.png`, `tmp/productivity-study-design-mobile.png`, `tmp/productivity-study-design-tablet.png`.

Full Python discovery uses the repository's namespace-style tests directory:

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_*.py'
node tests/test_dataset_scroll.cjs
```

Remaining limits: labels cannot establish real identity or collection history; time columns do not establish a series; general serial/repeated covariance models are unavailable; the additional-group shortcut requires globally consistent source IDs. No statistical engine, significance test or covariance approximation was added. Existing random-component uncertainty and mixed-model follow-up limitations continue to apply.
