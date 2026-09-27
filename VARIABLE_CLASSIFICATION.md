# Variable classification

Column Overview retains storage dtype, total/missing counts and missing percentages, and shows Variable Type followed immediately by Variable Subtype. Unique values include an expandable info control. A/M badges beside both classification fields show Automatic classification / Manual Override on hover and keyboard focus. Classification source and category order remain metadata but are not separate table columns. The associated column editor uses dependent dropdowns and an Apply button because subtype choices differ by row; the summary table is read-only, with inline HTML details controls. All dataset labels and values are HTML-escaped.

Continuous and Discrete (>10 unique) details show min/max; Discrete (≤10) and Binary show all values; Nominal shows the first five unique values in observed order; Ordinal shows all values in configured order. Missing values are excluded. Date/time details show the parsed date range.

## Metadata and validation

`utils/variable_metadata.py` separates inference, validation, overrides and analytical conversion from rendering. `variable_metadata` in session state stores variable_type, variable_subtype, order and source by original column name. `variable_dataset_id` scopes it to the active dataset. Existing `col_types` groups remain as a compatibility interface. New uploads, sample loads and manual replacements reset metadata; deleted columns are pruned. Upload identity includes SHA-256 content, not just name and size. Raw data stays in session unchanged; safe conversions apply to an analysis copy.

Integer numerical columns default to Discrete, floating columns to Continuous. Numeric 0/1 remains numerical. Categorical columns with two observed values default to Binary; others default to Nominal. Recognized complete Low/Medium/High, Small/Medium/Large and Poor/Fair/Good/Excellent sequences, plus explicitly ordered pandas categoricals, infer Ordinal. Other ordinal orders must be confirmed by the user; the editor starts with observed order. Every observed category must appear exactly once. Missing values are excluded from category counts.

Invalid type/subtype pairs, Binary with anything other than two categories, nonnumeric Numerical selections, and noninteger Discrete selections are rejected before saving. Date/time behavior retains the existing date parsing threshold and reports unparseable values as missing in analysis; raw values remain visible. Numerical date timestamps require explicit prior conversion. Nonfinite numeric values remain subject to existing analysis-specific exclusions.

Temporal subtypes include Date, Time, DateTime, Duration, Year, Month, and Year-Month. Explicit Year overrides accept four-digit years from 1000–9999; Month accepts English full/abbreviated names or integer values 1–12. Numeric year/month columns retain their Numerical default until manually reclassified. Year-Month accepts YYYY-MM or YYYY/MM and uses the first day solely as an internal period anchor. Full dates are not silently truncated into components by these overrides. At least 80% of non-missing values must parse, consistent with other temporal overrides; remaining invalid values become missing in the analysis copy. Raw values remain unchanged.

Standalone Year and Month use ordered frequency displays in inspection and descriptive statistics, without invented dates. They are excluded from date-based period comparisons. Year-Month is available for dated charts and monthly, quarterly, or yearly period comparisons; daily/weekly aggregation is disabled. It displays YYYY-MM in value details and frequency summaries, without implying an observed day.

## Downstream use

Effective groups drive quality reports, EDA, cross-tabs, charts and regression selectors. Numeric categorical overrides become categorical dtype in the analysis copy so predictive preprocessors also one-hot encode them. Numeric Discrete and Continuous variables remain eligible for linear regression. Distribution-fit Auto uses metadata; explicit distribution-family selection remains available. Ordinal frequencies and ordinary category charts respect the configured order; Pareto charts intentionally sort by contribution. No ordinal regression or new statistical module is introduced.

Predictive targets marked Numerical use existing regression models. Categorical Binary targets use logistic/random-forest classification independent of their storage labels. Binary labels are encoded to 0/1 for scoring with original display labels retained. The second displayed class is the positive class. Existing target-detection callers without subtype retain legacy behavior. Existing ID warnings remain unchanged.

## Verification

`tests/test_variable_metadata.py` covers integer/decimal/text/boolean defaults, 0/1 overrides, invalid Binary/numeric/type combinations, integer-valued floats, ordinal order/frequencies, persistence, replacement, removed columns, dependent dropdowns, atomic validation, linear-regression categorical handling, and logistic training/scoring for numeric, boolean and text binary targets. Full-app Streamlit checks run with database tracking mocked; no model API calls or external writes are required.
