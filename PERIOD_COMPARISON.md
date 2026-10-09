# Period-over-Period Comparison

Select one Temporal column, at most one numerical metric, and optional categorical segments in Exploratory Data Analysis → Comparison. Leave the metric empty to compare row counts. The existing comparison route and Group Statistics calculation entry point share one implementation.

## Temporal meaning and frequency

Dates and timestamps retain calendar chronology. Month names and abbreviations are month components; quarter labels are quarter components. Neither acquires an invented year. Repeated component labels are pooled within a segment because the source cannot identify their years. Months display their source names or abbreviations; unobserved months use full names. Internal period ordinals control chronological ordering independently of display labels.

| Input | Meaning | Available frequencies |
|---|---|---|
| Full date/timestamp | Dated observations | Daily, weekly, monthly, quarterly, yearly |
| January / Jan | Month without year | Monthly; quarterly with clarification |
| 2026-01 / Jan 2026 / January 2026 | Dated month | Monthly, quarterly, yearly |
| Q1 | Quarter without year | Quarterly |
| 2026-Q2 / Q2 2026 | Dated quarter | Quarterly, yearly |
| 2026 | Calendar year | Yearly |

Four-digit year text and numeric columns named Year or Calendar Year are automatically recognized. Other numeric columns retain their normal classification to avoid treating identifiers as years; apply a Temporal → Year override where necessary. Inspect also supports Quarter and Year-Quarter overrides. Existing manual choices are respected. Original data is never changed by parsing.

Frequency compatibility distinguishes Supported, Warning, and Unavailable. Unavailable frequencies are omitted from the selector and explained in the compatibility panel. A compatible native frequency is the default; users retain control of coarser reporting. Full dates are not assumed to be complete reporting coverage. Mixed month-and-year and full-date text requires a consistent reporting column; other invalid values follow the existing 80% parsing threshold, with exclusions disclosed.

Quarterly month-only reporting requires one clarification: monthly observations, or quarterly reports labelled by month. Monthly observations are aggregated using the chosen statistic and partial quarters are disclosed. Quarterly-report labels cannot include multiple month labels in the same quarter within a segment. Observations in January, April, July, and October alone do not establish quarterly totals. No metric names determine aggregation automatically.

## Aggregation and missingness

Sum, Mean, Median, Minimum, Maximum, sample Standard deviation, Non-missing count, and Row count retain their existing definitions. Period-end value additionally requires full dates/timestamps. It takes the value at the final timestamp, including a blank final value. Conflicting values tied at that timestamp require resolution; no implicit ordering of monthly or quarterly reports is assumed.

Each segment's observed first-to-last period defines its reporting range. Missing periods are periods with no records. An observed period with only blank metric values is different from a missing period and from an observed zero. Sum uses `min_count=1`; count statistics legitimately count rows or valid values and can return zero.

When gaps exist, the app shows the count and a preview of missing labels, then offers:

- Keep missing periods empty (recommended).
- Treat missing periods as zero, for this analysis only.

Zero filling changes only absent periods. It never fills blank measurements. When no gaps exist, the question disappears and any earlier zero-fill choice is ignored. Changing temporal column, metric, aggregation, dimensions, or frequency revalidates the policy. Each segment has its own observed range, so periods outside that range are not invented as gaps.

Previous uses the preceding calendar period, or the same period last year when a year is available. Gaps are never silently skipped to create successive-available-observation changes. Change is Value minus Previous; Change % is Change divided by the absolute Previous value times 100. A zero or missing baseline makes Change % N/A. Absolute changes remain defined when the baseline is zero. Negative metrics include an explanation of the absolute-denominator convention.

## Outputs and reporting

The actual-value line chart uses chronological categorical labels and does not connect missing values. The percentage-change bar chart uses positive/negative bars, a visible zero line, consistent segment colors, and N/A markers for undefined changes. Tooltips include values or changes, period, segment, and comparison status.

The table and CSV include Period, Value, Previous, Change, Change %, Segment, Previous period, Comparison, Missing period, Missing value, and Inserted zero. Tables format percentages and missing values explicitly; exported percentages remain numeric, with unavailable fields marked N/A.

Deterministic interpretation describes observed endpoint movement, largest defined increases/decreases, gaps, and limitations. Endpoint comparisons spanning gaps are explicitly described as nonconsecutive. No causal or forecast claims are generated. No AI call is needed.

Add to Report captures temporal column, metric, segments, frequency, aggregation, clarification, baseline, gap policy, results, both charts, interpretation, and warnings. The existing report architecture copies these outputs; rerunning cannot mutate a previously added result.

The engine guards against more than 250,000 segment-period combinations before timeline allocation. Charts display a contiguous final window of at most 2,000 points; tables, CSVs, interpretation, and report tables retain all results. Narrative output shows at most 20 segments and discloses truncation. Daily reporting is calendar based; weeks run Monday–Sunday. Quarters are calendar quarters, not fiscal quarters. Month-only data cannot establish December-to-January chronology across years; supply Year-Month or full dates for that analysis.

## Components and checks

- `utils/temporal.py`: format inference, parsing, component display.
- `utils/period_comparison.py`: compatibility, aggregation, timelines, changes, interpretation.
- `utils/period_comparison_ui.py`: guided controls, charts, table, CSV, report capture.
- Existing `utils/group_statistics.py`, `utils/group_statistics_ui.py`, and `utils/comparison_ui.py`: shared calculation and UI routing.
- `tests/test_period_comparison.py`: numerical, temporal, missingness, chart, UI, and report snapshot checks.

Run `python -B -m unittest discover -s tests`. Streamlit AppTest and Plotly-data checks verify interactions and outputs; native browser checks are separately required for desktop/mobile visual verification.
