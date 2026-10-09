# Date components and guided temporal selection

## Create a reusable temporal variable

Open Transform → Manage columns → Add column. Choose **Combine Date Components** in Column values. Existing creation methods remain available.

| Combination | Subtype | Stored representation | Display formats |
|---|---|---|---|
| Year + Month | Year-Month | Month-start timestamp as an internal anchor | YYYY-MM, MMM YYYY, MMMM YYYY |
| Year + Quarter | Year-Quarter | Pandas quarterly Period | YYYY-Qn, Qn YYYY |
| Year + Month + Day | Date | Calendar date timestamp | YYYY-MM-DD |

Select source columns, the new column name, and a display format. Preview uses actual rows. Apply and continue retains the new column in the existing transformation draft; Create dataset saves a reusable result. Source columns and source datasets remain unchanged. Existing names cannot be overwritten. Chained edits and the existing draft discard mechanism remain available.

Year accepts four-digit values from 1000–9999, including whole-number storage. Month accepts names/abbreviations, case differences, whitespace, and integers or zero-padded text from 1–12. Quarter accepts Q1–Q4 or 1–4. Day accepts integers or zero-padded text from 1–31; complete dates are then checked against the calendar, including leap years. Boolean values, fractional components, invalid ranges, and unintended numeric date conversions are rejected.

The default policy stops on invalid or missing components, showing affected-row counts and representative input/reason rows. **Leave affected values missing** must be explicitly selected to save such rows as missing. No artificial year or valid-looking date replaces bad inputs. Month/date anchors must fit pandas' supported timestamp storage range; unsupported dates are reported as affected rows. Quarterly Periods do not require a particular observed day. These are calendar components; fiscal calendar conversion is not implemented.

Derived metadata records the temporal subtype, display format, source components, and explicit creation origin. It is preserved by the existing metadata inheritance rules. Inspect and transformation previews show declared precision; dataset CSV exports use canonical YYYY-MM or YYYY-Qn for these derived periods. Full original source columns retain their stored values. Analytical month parsing never treats the first-day anchor as an observed event date.

## Analyze checks

Shared deterministic validation is integrated into Period-over-Period Comparison, its calculation entry point, running/rolling Group Statistics, and temporal Relationship selections. It can also be reused by future forecasting code.

- **One relevant year:** month ordering is valid and no multiple-year warning is shown.
- **Multiple years:** Month alone can pool repeated months or incorrectly order December followed by January. Chronological analysis is blocked until a dated reporting variable is selected or created.
- **Suitable existing variable:** recommend a Year-Month or full-date variable only when its observed month/year values align row by row with the chosen components. Names alone are not enough. Declared monthly variables are preferred to full dates. A conflicting or unrelated similarly named column is not recommended.
- **No year information:** explain that these are month-of-year comparisons without dated chronology; no year is invented.
- **Intentional seasonality:** explicitly choose Monthly seasonal profile (pool years). Values aggregate by month of year, and chart titles, comparison status, interpretation, and report settings disclose that this is not dated month-over-month movement.
- **Ambiguous year sources:** choose the corresponding Year column. Sources with conflicting values or different missing coverage are not silently combined. If conflicting full-date variables are the available sources, select the intended dated variable explicitly.

Year candidates require valid values as well as temporal metadata or an appropriate component signal. Numeric measurements and identifiers are not assumed to be years merely because they have four digits. Explicit categorical/numerical overrides are respected when recommending temporal variables. A month-year can repeat across transaction rows; those rows are preserved and the existing statistic aggregates them within the reporting period.

Checks consider each selected segment's relevant year coverage. Partial missing year information is disclosed and blocks unverifiable chronology. Month-only multi-year safety is enforced in the calculation backend as well as the UI; pooled calculations require an explicit seasonal objective. Quarter-only multi-year guidance is not expanded by this change; its existing period comparison warnings still apply.

## Navigation and confirmation

**Use an existing variable** changes the temporal selection explicitly, without transforming data. **Create Month-Year Variable** opens Transform and the existing Add Column method. It preselects unambiguous components and a nonconflicting Month-Year name; uncertain components remain for review. It does not create a column or discard pending transformation steps.

After Create dataset, Inspect opens as usual. **Return to Analysis** restores the previous analysis tab. **Continue with the derived variable** explicitly restores available numerical/categorical selections and chooses the new dated column. Returning alone does not execute the comparison. Missing or renamed previous selections are not guessed. Dataset-scoped state keeps the source and result independent.

Example: November 2025 → December 2025 → January 2026 → February 2026 produces four consecutive monthly periods. January observations from separate years remain separate when the combined variable is selected. Missing-period policy, charts, CSV results, and report snapshots continue to use the existing validated period comparison implementation.

## Components and tests

- `utils/date_components.py`: construction, component validation, metadata, precision-aware display.
- `utils/table_editing.py` / `utils/transform_ui.py`: existing Add Column integration and confirmation.
- `utils/temporal_validation.py` / `utils/temporal_validation_ui.py`: shared assessment and contextual recommendations.
- `utils/temporal_navigation.py` / `utils/dataset_workspace.py`: explicit workspace handoffs and return selections.
- `utils/transformation_metadata.py`, `utils/dataset_views.py`, `utils/group_statistics_ui.py`, `utils/period_comparison.py`, `utils/period_comparison_ui.py`: integration with metadata, Inspect, and analysis.
- `tests/test_date_components.py`: component parsing, invalid/missing handling, source preservation, metadata, chronological aggregation, guidance, and full workspace interactions.

Run `python -B -m unittest discover -s tests`. Browser visual verification is separate from Streamlit AppTest interaction checks.
