# Variable classification

Column Overview retains storage dtype, total/missing counts and missing percentages, and shows Variable Type followed immediately by Variable Subtype. Unique values, classification source and ordinal order are also visible. The associated column editor uses dependent dropdowns and an Apply button because subtype choices differ by row; the summary grid is read-only.

## Metadata and validation

`utils/variable_metadata.py` separates inference, validation, overrides and analytical conversion from rendering. `variable_metadata` in session state stores variable_type, variable_subtype, order and source by original column name. `variable_dataset_id` scopes it to the active dataset. Existing `col_types` groups remain as a compatibility interface. New uploads, sample loads and manual replacements reset metadata; deleted columns are pruned. Upload identity includes SHA-256 content, not just name and size. Raw data stays in session unchanged; safe conversions apply to an analysis copy.

Integer numerical columns default to Discrete, floating columns to Continuous. Numeric 0/1 remains numerical. Categorical columns with two observed values default to Binary; others default to Nominal. Recognized complete Low/Medium/High, Small/Medium/Large and Poor/Fair/Good/Excellent sequences, plus explicitly ordered pandas categoricals, infer Ordinal. Other ordinal orders must be confirmed by the user; the editor starts with observed order. Every observed category must appear exactly once. Missing values are excluded from category counts.

Invalid type/subtype pairs, Binary with anything other than two categories, nonnumeric Numerical selections, and noninteger Discrete selections are rejected before saving. Date/time behavior retains the existing date parsing threshold and reports unparseable values as missing in analysis; raw values remain visible. Numerical date timestamps require explicit prior conversion. Nonfinite numeric values remain subject to existing analysis-specific exclusions.

## Downstream use

Effective groups drive quality reports, EDA, cross-tabs, charts and regression selectors. Numeric categorical overrides become categorical dtype in the analysis copy so predictive preprocessors also one-hot encode them. Numeric Discrete and Continuous variables remain eligible for linear regression. Distribution-fit Auto uses metadata; explicit distribution-family selection remains available. Ordinal frequencies and ordinary category charts respect the configured order; Pareto charts intentionally sort by contribution. No ordinal regression or new statistical module is introduced.

Predictive targets marked Numerical use existing regression models. Categorical Binary targets use logistic/random-forest classification independent of their storage labels. Binary labels are encoded to 0/1 for scoring with original display labels retained. The second displayed class is the positive class. Existing target-detection callers without subtype retain legacy behavior. Existing ID warnings remain unchanged.

## Verification

`tests/test_variable_metadata.py` covers integer/decimal/text/boolean defaults, 0/1 overrides, invalid Binary/numeric/type combinations, integer-valued floats, ordinal order/frequencies, persistence, replacement, removed columns, dependent dropdowns, atomic validation, linear-regression categorical handling, and logistic training/scoring for numeric, boolean and text binary targets. Full-app Streamlit checks run with database tracking mocked; no model API calls or external writes are required.
