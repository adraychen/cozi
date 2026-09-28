# Cozy UI control inventory and decisions

Status: current implementation inventory and decision register, 2026-09-28. See [UI standard](ui-standard.md) for agreed Inspect, Transform and Analyze rules. The open decision register below remains pending except where the approved updates explicitly establish a rule.

## Scope and terminology

Reviewed the active route from `app.py` through `utils/dataset_workspace.py`, `utils/dataset_views.py`, and their UI helpers, including Enterprise. Versioned snapshots (`appV*.py`, `*V*.py`) are excluded from the current UI inventory. Predictive Analytics in `dataset_views.py` is active and included. This is a source-code audit, not a browser/accessibility certification; conditional controls appear only for eligible data, selections, or results.

| Term | Meaning |
| --- | --- |
| Dropdown / pull-down menu | `selectbox`: one selected option |
| Multiselect dropdown | `multiselect`: multiple selected options shown as removable items |
| Single-choice checkbox grid | Checkboxes synchronized so selecting one clears the previous selection; visually checkboxes, behavior similar to radio buttons |
| Multiple-choice checkbox grid | Independent variable/term checkboxes; multiple selections allowed |
| Boolean checkbox | On/off setting or acknowledgement |
| Radio buttons | One choice from visible alternatives |
| Selection buttons | Ordinary buttons styled to show the current selection |
| Tabs | Navigation among datasets, workflows, analysis families, or result panels |
| Expander | Show/hide supplementary content, not a choice of analysis |

## Navigation, input and inspection

| Area / task | Current controls | Source |
| --- | --- | --- |
| Dataset navigation | Dataset tabs plus Add dataset tab | `utils/dataset_workspace.py` |
| Dataset workflow | Inspect / Transform / Analyze tabs | `utils/dataset_workspace.py` |
| Analyze families | Descriptive Statistics / Group Statistics / Visual Analysis / Relationship / Comparison tabs; Predictive Analytics below these panels | `utils/dataset_views.py` |
| File import | Multiple-file uploader; Excel sheets use multiselect dropdowns; Add uploaded datasets button | `utils/dataset_workspace.py` |
| Sample datasets | Individual sample buttons in an expander | `utils/dataset_workspace.py` |
| Manual input | Entry method uses horizontal radio buttons (Editable table / CSV text); editable grid or text area; header handling dropdown; dataset-name text field; action buttons | `utils/manual_input.py` |
| Inspection | Preview, column/row overview and quality sections use expanders; quality panels use variable-type tabs | `utils/dataset_views.py` |
| Edit variable classification | Column to edit, Variable Type and Variable Subtype are dropdowns; ordered-category sequence is a text input; Apply classification button is disabled without changes | `utils/variable_metadata_ui.py` |
| Enterprise / Community | Learn more and Back buttons; edition labels are noninteractive badges | `utils/enterprise_ui.py`, `utils/branding.py` |
| Enterprise enquiry | Deployment Interest and Number of Users dropdowns; contact fields are text inputs; requirements text area; Submit Enquiry button | `utils/enterprise_ui.py` |
| Feedback / privacy | Like/dislike buttons open a dialog with optional comment text area and Submit/Skip; information button opens privacy dialog | `app.py` |

## Transform

Five **side-by-side dropdowns**, wrapping on narrow screens, select one active operation across all groups. Choosing an operation clears the other groups' displayed selection. These are not checkboxes and do not configure several operations simultaneously. Applied changes can accumulate separately.

| Group | Dropdown operations |
| --- | --- |
| Combine datasets | Join, Append |
| Restructure data | Aggregate, Pivot, Unpivot (clickable ⓘ explanations below the dropdown) |
| Manage columns | Rename column, Remove column, Add column |
| Clean values | Fill missing values, Replace Values, Trim Spaces, Remove Characters |
| Convert values | Convert Currency to Numeric |

| Operation | Dropdowns | Multiselect dropdowns | Boolean checkboxes / other inputs |
| --- | --- | --- | --- |
| Join | Right dataset; join type | Left keys and right keys, in matching order | Match missing keys |
| Append | Columns to keep | Additional datasets, in output order | Include source dataset column; source-name text input |
| Aggregate | — | Group by; calculations; measure columns | Keep missing grouping values; output-name text inputs |
| Pivot | New columns from; values from; repeated-combination handling | Row identifiers | Fill missing result cells with zero |
| Unpivot | — | Keep as identifiers; columns to turn into rows | Drop rows with missing values; output-name text inputs |
| Rename / Remove column | Column to rename | Columns to remove | New-name text input |
| Fill missing values | Column; replacement type; True/False replacement when applicable | — | Replacement value otherwise uses a text input |
| Replace Values | Column or All columns; find type; replacement type; True/False values when applicable | — | Text inputs for other values; ignore surrounding spaces checkbox when finding text |
| Remove Characters | Column or All columns | — | Characters-to-remove text input |
| Trim Spaces | Column or All columns; spaces to remove | — | — |
| Convert Currency to Numeric | Column or All columns | Columns to convert when applicable | — |
| Add column | Value method/type; Boolean value; source column; calculation and operands as applicable | — | Name and constant-value text inputs |

Preview result, Apply and continue, Create dataset and Discard accumulated changes are action buttons. Dataset names are text fields; previews/validation use expanders. Source: `utils/transform_ui.py`. Ordered join keys and append inputs are meaningful exceptions to consider before replacing multiselects with checkbox grids.

## Analytical variable and method selection

| Screen | Variable selection | Method/settings controls | Source |
| --- | --- | --- | --- |
| Descriptive Statistics | Single-choice checkbox grid across Numerical, Categorical and Temporal; initially first available variable; can clear selection | Category-count slider; distribution type dropdown; include-binomial checkbox and trial-count numeric input | `utils/descriptive_controls.py`, `utils/distribution_fit_ui.py`, `utils/dataset_views.py` |
| Group Statistics | Single-choice numerical measure; multiple-choice categorical dimensions; single-choice calendar temporal variable; checkbox grids | Analysis type, statistic, period frequency, comparison/running options and reference segment dropdowns; chart segments multiselect; rolling-window numeric input; zero-fill/share-within-parent checkboxes | `utils/group_statistics_ui.py` |
| Visual Analysis | Multiple-choice variable grids by type; single-choice checkbox grids for chart roles (axes, grouping, measure, color, bubble size, heatmap roles); multiple-choice line measures and pair-plot variables | Chart selection uses a button grid; aggregation, correlation method and heatmap type use dropdowns; bins/bandwidth use sliders; standardize uses checkbox | `utils/visual_analysis.py`, `utils/variable_selection.py` |
| Relationship | Multiple-choice variable grids by type | Analysis method dropdown; supplementary Chart type uses horizontal radio buttons | `utils/dataset_views.py`, `utils/relationship_ui.py` |
| Correlation | Uses numerical variables already selected in Relationship | Pearson/Spearman dropdown | `utils/correlation_ui.py` |
| Cross-tabulation / chi-square | Row and column variables use dropdowns from selected candidates | Significance-level dropdown; include-missing-categories checkbox | `utils/contingency_ui.py` |
| Trend analysis | Time variable and measure dropdowns | Time period and aggregation dropdowns | `utils/relationship_ui.py` |
| Comparison | Multiple-choice variable grids by type; available analyses match the complete selection | Analysis method dropdown; selection guidance expander | `utils/comparison_ui.py` |
| One-way ANOVA | Grouping variable and numerical outcome dropdowns (currently one candidate each under Comparison eligibility) | Welch/Classical dropdown; independence confirmation checkbox; Q–Q group dropdown | `utils/comparison_ui.py`, `utils/anova_ui.py` |
| Factorial ANOVA | Response dropdown; factors multiselect; interaction effects multiselect | Alpha numeric input; follow-up factor dropdowns; show pairwise/simple-effects, include non-significant plots, swap factors and HC3 checkboxes | `utils/factorial_anova_ui.py` |
| t-Test | Response/sample columns, grouping variable and group levels use dropdowns | Test type, data structure and alternative hypothesis dropdowns; equal-variances checkbox; null mean and confidence (%) numeric inputs | `utils/ttest_ui.py` |
| Segment / Period-over-Period Comparison | After initial Comparison checkboxes, numerical measure uses dropdown, categorical dimensions multiselect, date dropdown where applicable | Shared Group Statistics settings described above | `utils/comparison_ui.py`, `utils/group_statistics_ui.py` |
| Linear / binary logistic regression | Single-choice response checkbox grid; predictors, interactions and polynomial terms use multiple-choice checkbox grids | Model-building method dropdown; numeric elimination threshold or confidence inputs as applicable | `utils/regression_controls.py`, `utils/regression_ui.py`, `utils/logistic_ui.py` |
| Predictive Analytics | Target dropdown; predictors in multiple-choice grids grouped by type | Run button; result class/SHAP class dropdowns. Model family follows target classification rather than a model-selection dropdown | `utils/dataset_views.py` |

Regression follow-up controls: selected effect/interaction/axis/moderator and partial-residual predictor use dropdowns; categories to display use multiselects. Categorical prediction inputs use dropdowns and numeric predictions use number inputs. Logistic event/reference categories are dropdowns; ROC display and meaningful-row-order acknowledgement are Boolean checkboxes in their respective analyses. Diagnostic review concerns use multiselects. Sources: `utils/regression_effect_plots.py`, `utils/regression_validation_ui.py`, `utils/regression_ui.py`, `utils/logistic_ui.py`.

## Selection behavior and layout details

- Most general variable grids have four columns. Regression response/predictor/term grids have three columns.
- Descriptive selection is exclusive across all type groups. Group Statistics exclusivity is per numerical/temporal role. Regression keeps one response and changes it when another is checked.
- Shared single-choice variable grids allow clearing a choice; required Visual Analysis roles then prompt for a selection. Many dropdowns select the first candidate automatically. Enterprise dropdowns explicitly start empty.
- Group Statistics and Visual Analysis shared multiple-choice grids provide Select all / Clear all. Comparison provides these for every nonempty type group. Relationship provides them for Numerical and Categorical but not Temporal.
- Regression predictors, interactions and polynomial terms provide Select all / Clear all. Cubic terms require the corresponding square, whose checkbox is disabled while required.
- Predictive Analytics uses per-type Select all / Clear all buttons matching the Relationship reference.
- Analysis and chart options are conditional on variable eligibility. A second role selector may therefore contain only one option. Such selectors still render as dropdowns in several Comparison analyses.
- Most exploratory calculations update with selection changes. Predictive Analytics explicitly uses Run; Transform uses preview/apply/create actions; classification uses Apply; Enterprise uses Submit. These different execution models may be intentional.

## Conflicts and decisions for the product owner

These are inconsistencies or potential ambiguities, **not approved changes**. Different task needs may justify exceptions. Decision status for every row: **Pending**.

| ID | Observed conflict | Decision needed |
| --- | --- | --- |
| UI-01 | One-variable roles use single-choice checkboxes in Descriptive, Group Statistics, Visual Analysis and regression, but dropdowns in ANOVA, t-Test, trend, cross-tabulation and Predictive Analytics. Checkboxes can imply multiple selection even when exclusive. | Choose the default for a single variable: checkbox grid, radio group or dropdown; define any list-length or task exceptions. |
| UI-02 | Multiple variables/terms use checkboxes in EDA/regression, but multiselect dropdowns for factorial factors/interactions and Transform column lists. | Choose a default for unordered multiple selection; decide whether ordered join keys/append inputs remain multiselects. |
| UI-03 | Group Statistics uses checkbox roles; Segment/Period Comparison uses dropdowns/multiselects for the same numerical, categorical and temporal roles. | Use the same role controls in both locations, or explicitly retain a compact Comparison exception. |
| UI-04 | Visual Analysis chart choice is a button grid; Relationship supplementary chart choice is a radio group. Analysis-method choice elsewhere is a dropdown. | Decide whether chart selection should use buttons, radio buttons or dropdowns consistently, and whether analysis methods follow that rule. |
| UI-05 | Comparison first selects variables using checkboxes, then asks for some of the same roles again using dropdowns/multiselects. One-way ANOVA's role dropdowns have only one candidate each. | Decide whether fixed roles should be shown as labels, kept as disabled controls or remain selectable; retain explicit role assignment where it resolves ambiguity. |
| UI-06 | Bulk-selection labels, placement and availability differ: Select all/Clear all per group vs global Select All/Deselect All; Relationship Temporal lacks bulk controls. | Choose wording, capitalization/icons, group/global scope and when bulk controls appear. |
| UI-07 | Variable checkbox grids use four columns in most screens, three in regression. Transform uses wrapping fixed-width menus. | Decide a common responsive grid rule, with exceptions for long term labels and grouped operation menus. |
| UI-08 | Single-choice grids may allow no selection, while many dropdowns automatically choose the first option; Enterprise starts blank; Factorial factors initially include the selected categories. | Specify defaults and whether users must explicitly choose required analytical roles. |
| UI-09 | Significance is a dropdown in chi-square, decimal alpha input in Factorial ANOVA, and percentage confidence input in t-Test/logistic regression. Allowed confidence ranges also differ. | Decide a consistent presentation and units while preserving each method's statistical meaning and valid ranges. |
| UI-10 | Model/statistical choices appear as both checkboxes and dropdowns: equal-variances t-test checkbox versus Welch/Classical ANOVA dropdown. | Decide whether binary method choices should use named dropdown/radio alternatives or explanatory Boolean checkboxes. |

## How to maintain this document

After each approved decision, record the chosen rule and exceptions against its ID, update the inventory and only then implement the corresponding UI changes. Update the inventory whenever adding a control. Keep versioned source snapshots out of active-screen conclusions. Do not infer statistical equivalence solely because two controls have similar labels.

Useful source search: `rg -n 'selectbox|multiselect|checkbox|radio|variable_checkboxes|render_checkbox_specification|render_term_checkboxes' app.py utils` (then follow active imports and callbacks). Shared wrappers matter: for example, `visual_analysis.choose` renders checkboxes for variable roles and dropdowns for other settings.

## Approved Analyze presentation updates (2026-09-28)

- Group Statistics and Visual Analysis use the heading **Temporal**. Eligible temporal subtypes and selection behavior are unchanged.
- Descriptive Statistics summary tables use the interactive dataframe renderer used in Group Statistics. Numerical statistic explanations are retained in an expander below the summary. Existing frequency/date tables already use this renderer; Time/Duration summaries now use it in Descriptive Statistics as well.
- Relationship linear/logistic regression summary tables use that same interactive renderer, replacing static tables while preserving values and statistical precision. Existing correlation/trend dataframes remain interactive. The specialized chi-square contingency layout retains its observed/expected/residual structure.
- Inspect was subsequently aligned with Dataset Preview; see the approved table standard below. The broader Analyze control-type decisions UI-01 through UI-10 remain pending.

## Transform clarification (2026-09-28)

The product owner considers the current Transform control pattern consistent; retain its grouped dropdowns and operation settings. Cross-screen decisions above do not imply that Transform needs redesign.

Restructure data uses short operation names. Below its dropdown, Aggregate, Pivot and Unpivot each have a clickable, keyboard-accessible ⓘ disclosure. Clicking expands respectively “Group rows and calculate summaries”, “Turn values into columns”, and “Turn columns into values”; clicking again closes it. Native dropdown options cannot contain separate clickable controls, so these disclosures sit immediately below the menu. Opening help does not select an operation.

## Approved table standard (2026-09-28)

All shared tables use the Dataset Preview native interactive dataframe format. Width is content-sized, capped by the available parent container rather than a fixed 1,200 pixels. Columns use native automatic content sizing; overflowing tables scroll horizontally. Small tables remain compact. Existing formatting, hidden columns, keys and selection behavior are preserved. The specialized stacked-cell contingency table retains its layout with content width and horizontal overflow.

Inspect Column Overview now uses the native table, including missing-percentage colors and A/M classification markers. Column-header help explains the markers. Unique value/range details remain available in an expandable section below the table with a column selector. Data Quality statistic panels, temporal summaries and classification category-order tables use the same native format.

## Selection-control clarification (2026-09-28)

Visual Analysis's existing chart-selection button grid is the standard **for Visual Analysis**. It is currently the only chart-selection button grid in the active app. Relationship's chart-type radio buttons and other analyses' chart-content dropdowns/multiselects are unchanged; applying buttons there requires a separate decision.

Relationship's plain, side-by-side **Select all / Clear all** buttons are the reference for bulk selection. Predictive Analytics now uses the same wording and per-variable-group layout, replacing its global icon-prefixed Select All / Deselect All controls. Existing matching controls elsewhere are retained. Relationship's current Temporal group remains unchanged.
