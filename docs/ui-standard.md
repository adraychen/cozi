# Cozy UI standard: Inspect, Transform and Analyze

Updated: 2026-09-28.

This document records the agreed presentation standards and the existing workflow controls to retain. It does not approve a universal replacement of dropdowns with checkboxes, or vice versa. The [control inventory and decision register](ui-controls.md) contains the detailed screen-by-screen audit and unresolved choices.

## Shared table standard — approved

Dataset Preview is the reference format for tables across Inspect, Transform and Analyze.

- Use the shared native interactive table renderer in `utils/presentation.py`; do not introduce a different HTML or static table style for ordinary summaries.
- Use consistent native headers, fonts, row spacing, borders and interaction behavior.
- Columns size automatically to their headers and displayed content. Do not squeeze columns or force equal widths to make a wide table fit.
- Small tables remain compact. Wide tables expand up to the full available **container** width. A table inside a panel or column is bounded by that panel, rather than by the entire page.
- When the combined columns exceed the available width, retain horizontal scrolling. A 16-column table is not capped at 1,200 pixels; it uses the available width when its content requires it.
- Keep existing numerical formats and statistical precision. Standardizing appearance must not change calculations, turn small p-values into zero, or alter missing-value meaning.
- Preserve meaningful index labels, column labels, conditional colors, sorting, downloads and selection behavior. Hide purely decorative row numbers where appropriate to the existing table.
- Keep explanatory text outside ordinary cells, using captions, header help or expandable sections.

Implementation: `presentation.dataframe(...)` uses native `width="content"` and automatic column widths. The compatibility `presentation.table(...)` also renders an interactive dataframe. Callers should not impose fixed widths; the shared renderer removes column width overrides while preserving other column configuration.

Existing exception: Relationship's specialized contingency report retains stacked observed/expected/residual cells, totals and its legend. It fits its content and scrolls horizontally. This exception does not authorize new custom table designs elsewhere. Editable input grids are form inputs, rather than read-only result tables.

## Inspect

### Layout and tables

Retain the Dataset Preview, Column Overview, Row Overview and Data Quality Summary sections. All their ordinary tables follow the shared table standard, including duplicate lists, quality-statistic panels, temporal summaries and category-order reference tables.

### Column Overview and classification

- Display storage data type, analytical variable type/subtype, unique count, total count, missing count and missing percentage.
- Retain missing-percentage colors and the A/M classification markers. A means Automatic classification; M means Manual Override. Header help explains these markers.
- Show unique values or ranges in the **Unique value details** expander below the table, with a column dropdown.
- Keep **Edit variable classification** collapsed initially. Use dropdowns for Column to edit, Variable Type and Variable Subtype; retain dependent subtype choices.
- Use **Apply classification** as the explicit action. Disable it when there are no unsaved classification changes.
- Retain the ordinal-order input and its category reference table where applicable.

### Duplicate and quality information

- “No duplicated columns found.” and “No duplicated rows found.” use ordinary text, without blue information banners.
- When duplicates exist, use the shared table format and retain the explicit action that creates a cleaned dataset.
- Data Quality summary values use native Statistic / Value tables alongside their charts. Keep the underlying quality checks and original data unchanged.

Sources: `utils/dataset_views.py`, `utils/variable_metadata_ui.py`, `utils/variable_overview_display.py`, `utils/duplicate_overview.py`.

## Transform

The product owner reviewed the existing Transform controls and found no inconsistency requiring redesign. Retain the grouped dropdown arrangement and operation-specific settings.

### Operation selection

Place these dropdowns next to one another, wrapping on narrower screens:

| Dropdown group | Operations |
| --- | --- |
| Combine datasets | Join; Append |
| Restructure data | Aggregate; Pivot; Unpivot |
| Manage columns | Rename column; Remove column; Add column |
| Clean values | Fill missing values; Replace Values; Trim Spaces; Remove Characters |
| Convert values | Convert Currency to Numeric |

Only one operation is active across the groups. Selecting an operation clears the other groups' displayed selections. Do not replace this with a dropdown for selecting the group followed by another dropdown for its operation.

### Restructure information badges

Keep the operation names short. Immediately below Restructure data, show each name with a clickable, keyboard-accessible ⓘ disclosure:

| Operation | Text revealed on click |
| --- | --- |
| Aggregate | Group rows and calculate summaries |
| Pivot | Turn values into columns |
| Unpivot | Turn columns into values |

Click again to close the explanation. Opening help must not change the selected operation. Disclosures are below the menu because native dropdown options cannot contain separate clickable controls.

### Settings and actions

Retain dropdowns for single settings, multiselect dropdowns for multiple columns/datasets/calculations, checkboxes for Boolean settings, and text inputs for names and values as currently implemented. Preserve selection order where meaningful, especially matching join keys and appended datasets.

Retain Preview result, Apply and continue, Create dataset and Discard accumulated changes as explicit actions. Pending changes use one summary and one applicable save action; do not duplicate the pending-change message or Create dataset control for the same draft. Preview and validation tables use the shared table standard.

Source: `utils/transform_ui.py`. The inventory lists controls for each operation.

## Analyze

### Agreed headings and table presentation

- Group Statistics and Visual Analysis label their temporal variable group **Temporal**. Do not append “(Date / DateTime / Year-Month)” to that heading. This label change does not expand which temporal subtypes an analysis supports.
- Descriptive Statistics summary tables use the same native interactive format as Group Statistics. Keep statistic explanations in **How to interpret the statistics** below the numerical summary.
- Relationship regression model/performance summaries and other ordinary result tables use the same native interactive format as Group Statistics. Preserve the specialized contingency-report exception described above.
- Preserve statistical precision, eligibility rules and analysis behavior when standardizing presentation.

### Current controls to retain until further decisions

These are the current patterns, not a newly approved universal control policy:

| Analyze area | Variables / roles | Analysis or chart selection |
| --- | --- | --- |
| Descriptive Statistics | Single-choice checkbox grid across variable types | Selected variable determines the available descriptive views |
| Group Statistics | Single-choice numerical and temporal checkbox grids; multiple categorical checkboxes | Analysis and statistic dropdowns |
| Visual Analysis | Multiple variable checkboxes; single-choice role checkboxes; multiple line measures/pair-plot variables | Chart-selection buttons; settings use dropdowns, sliders and Boolean checkboxes |
| Relationship | Multiple variable checkboxes; regression response is exclusive and predictors/terms are multiple checkboxes | Analysis dropdown; supplementary chart radio buttons; other analyses retain role dropdowns |
| Comparison | Multiple initial variable checkboxes; One-way ANOVA displays its grouping variable and numerical outcome as labels; other downstream role controls remain as implemented | Eligible-analysis dropdown; method-specific controls |
| Predictive Analytics | Target dropdown and predictor checkboxes | Explicit Run action; result-class dropdowns |

Single-choice checkbox groups must continue to enforce exclusivity. Multiple-choice groups retain their current bulk-selection actions. Analysis choices remain dependent on eligible variables. Do not standardize controls by exposing unsupported analyses or removing required study-design decisions.

Sources: `utils/dataset_views.py`, `utils/group_statistics_ui.py`, `utils/visual_analysis.py`, `utils/comparison_ui.py`, `utils/relationship_ui.py`, and their analysis UI helpers.

## Decisions still open

The owner has not yet selected one universal rule for:

- Single-variable checkbox grids versus dropdowns or radio groups.
- Unordered multiple-variable checkboxes versus multiselects in Analyze.
- Chart-selection buttons versus radio groups.
- Repeated role selection after initial Comparison selection.
- Bulk-selection wording/scope, grid column counts and initial defaults.
- Significance/confidence input presentation and binary statistical-method controls.

See UI-01 through UI-10 in the [decision register](ui-controls.md). The accepted Transform pattern is an explicit existing workflow; cross-screen differences do not require changing it. New decisions should specify their scope and exceptions before implementation.

## Maintenance and review

When a standard changes, update this document, the inventory and relevant implementation together. Check compact and wide tables, narrow containers, long headers/values, hidden columns, numerical formatting and retained help. UI-only changes must preserve calculations, source data, classification, navigation state and pending transformations. Record verification honestly; source audits and automated UI checks are not a substitute for claiming unperformed browser checks.

## Selection-control clarification (2026-09-28)

Visual Analysis's existing chart-selection button grid is the standard **for Visual Analysis**. It is currently the only chart-selection button grid in the active app. Relationship's chart-type radio buttons and other analyses' chart-content dropdowns/multiselects are unchanged; applying buttons there requires a separate decision.

Relationship's plain, side-by-side **Select all / Clear all** buttons are the reference for bulk selection. Predictive Analytics now uses the same wording and per-variable-group layout, replacing its global icon-prefixed Select All / Deselect All controls. Existing matching controls elsewhere are retained. Relationship's current Temporal group remains unchanged.

## Analyze loading behavior (2026-09-28)

Analyze has five exploratory tabs: Descriptive Statistics, Group Statistics, Visual Analysis, Relationship and Comparison. Only the open exploratory tab renders its analysis. Predictive Analytics remains below these tabs in Analyze; it is not a separate tab. Existing dataset-specific variable selections and results remain available when switching panels.

AI Interpretation requires **Generate interpretation** or **Refresh interpretation**. Opening a panel or a collapsed interpretation section does not initiate a request. Cached interpretations display without another request. Predictive configuration uses the lightweight `utils/modelling_config.py`; model-fitting dependencies load on Run, and SHAP chart imports load when prediction results are displayed. Training calculations and model grids are unchanged. The first training run may still spend time loading libraries.

### ANOVA factor effects

One-way and Factorial ANOVA use an independent horizontal **Fixed / Random** radio control for each factor; Fixed is the default. The model-family label is automatic. For random models, dropdowns clarify real-world group identity and remaining dependence before fitting. These are design questions, not a model-family selector. Distribution plots retain a factor dropdown; Visual Analysis's chart-selection buttons remain its own standard. See [ANOVA effect settings](anova-effects.md) for statistical scope, validation and report behavior.
