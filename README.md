# Cozy · Community Edition

*Automated Data Analytics*

A multi-dataset analytics workspace built with Streamlit. Add CSV or Excel files, input data, or load sample datasets, then use **Inspect → Transform → Analyze** within full-width dataset tabs.

## Dataset workspace

The dataset workspace runs as a Streamlit fragment. Tab and control interactions rerun the workspace without rerunning the page header's database counters or visit tracking; explicit dataset-creation reruns still refresh the full page.

Initial column classification infers each subtype once and reuses parsing results for repeated text within each temporal parsing operation. Inference still evaluates all rows with their original frequency and keeps the existing validity thresholds. Inspect displays a preparation indicator while building Column Overview.

Adding datasets opens Inspect without requesting an AI interpretation. Use **Generate interpretation** when needed, or **Refresh interpretation** to update an existing summary. AI input summaries are calculated only on request. Excel sheet discovery is cached by file contents to avoid reopening the workbook when upload controls rerun.

Convert Currency to Numeric offers smart **All columns** selection: columns containing at least one valid amount marked with $, £, ¥, or € are preselected. Review the **Columns to convert** multiselect to add or remove columns before previewing. Ordinary text, dates, existing numeric columns, and plain numeric text are left unchanged by default. Every value in each selected column must validate; invalid values block the whole operation and are listed by column with counts. Validation shows converted and unchanged columns. Detection only suggests a selection; conversion remains explicitly user-triggered.

Currency conversion also recognizes accounting parentheses: `$(3000)`, `($3,000)`, and `(3000)` become `-3000`. Parenthesized decimals remain negative decimals. Unbalanced/nested parentheses or parentheses combined with explicit signs are rejected for manual correction.

**Convert Currency to Numeric** is an explicit Transform operation for a selected text column. It supports `$`, `£`, `¥`, `€`, comma thousands separators, decimal points, boundary whitespace, currency dashes as zero, and negative amounts such as `-$300` and `$-300`. Blank and genuinely missing cells stay missing; literal `None` is invalid. Any invalid value blocks the whole conversion and is listed with its occurrence count for manual cleanup. The standard preview, validation, Apply and continue, and Create dataset workflow is used. Successful outputs use nullable Int64 when all amounts are whole numbers, or Float64 for fractional amounts, and receive the normal numerical classification when the result dataset is created. It does not perform exchange-rate conversion; whole numbers outside Int64 range and nonfinite amounts are rejected.

Categorical Inspect panels show **Leading/Trailing Whitespace** directly below Special Characters, formatted to two decimal places. It counts non-missing text cells that differ from `str.strip()`, divided by all non-missing text cells. Internal spaces and non-text values are excluded from detection; missing and non-text cells are excluded from the denominator. With no eligible text cells the metric displays 0.00%. This check uses original stored values and never modifies the dataset; use Transform → Trim Spaces to clean it manually.

**Trim Spaces** removes leading whitespace, trailing whitespace, or both from one column or All columns. Spaces between words are preserved exactly, including repeated internal spaces. Edge tabs and line breaks are also trimmed; missing and non-text cells stay unchanged. Preview, Apply and continue, and Create dataset work as with other transformations. This is separate from Replace Values' optional Ignore surrounding spaces setting, which affects matching only and is enabled by default.

**Replace Values** and **Remove Characters** each offer **All columns** in the column selector. This applies the same rule across every column and reports total and per-column match/change counts. Exact replacement respects the selected value type; character removal only changes text cells. Existing missing values are preserved.

**Editing multiple columns:** Preview a transformation and select **Apply and continue** to retain it as a pending change. Subsequent operations use the accumulated result, so you can clean another column or combine different transformations before creating a dataset. The Pending changes section lists applied steps and offers a combined preview, **Create dataset from accumulated changes**, and **Discard accumulated changes**. Changing the current controls only invalidates the current preview; applied steps remain. Pending changes are scoped to the source dataset and retained when switching tabs during the session. You can also select **Create dataset** on a new preview to save that step together with all accumulated changes. Source data remains unchanged.

- **Dataset tabs** select the active dataset; **＋ Add datasets** contains multi-file upload, Excel sheet selection, the existing editable-table/CSV input, and sample datasets. Imports add tabs without replacing existing tables. Duplicate names receive a numeric suffix.
- **Inspect** contains Dataset Preview, Column Overview (including classification editing and duplicated-column removal), Row Overview, and Data Quality Summary as expanded, collapsible sections. **Download dataset (CSV)** exports every stored row and column, including joined and other created tables; the table preview itself shows only the first 100 rows.
- **Transform** shows five operation dropdowns side by side, wrapping on narrower screens. Selecting an operation clears the other dropdowns and displays its settings below; pending changes are preserved. Groups are **Combine datasets** (Join, Append), **Restructure data** (Aggregate — Group rows and calculate summaries; Pivot — Turn values into columns; Unpivot — Turn columns into rows), **Manage columns** (Rename, Remove, Add), **Clean values** (Fill missing values, Replace Values, Trim Spaces, Remove Characters), and **Convert values** (Convert Currency to Numeric). Configure an operation, select **Preview result**, review validation and the first 100 output rows, then name and **Create dataset**. Use **Apply and continue** to accumulate edits before creating a dataset. Changing settings invalidates the current preview without discarding applied changes. Outputs open in a new tab with Inspect selected.
- **Analyze** contains the existing EDA and predictive analytics workflows. Dataset-specific classifications, keyed controls, AI summaries, and model results remain separate when switching datasets. The selected workflow is retained when switching existing dataset tabs.
- **Help**, next to Analyze, explains Aggregate, Pivot, and Unpivot in expandable sections.
- Only the active dataset/workflow executes. **Create dataset** retains a result in session memory; this is not persistent project storage. Download the CSV to keep a local copy. Join validation uses row-count metrics and a left/right comparison table, with Yes/No for many-to-many matches.
- Sources are preserved. Created tables record their source dataset IDs and transformation settings, and infer their own variable classifications. Operations use stored data values, not analytical classification conversions. Created tables are snapshots; later source classification changes do not recompute them.

| Operation | Configuration and validation |
|---|---|
| Join | Left, inner, right, or full outer; one or multiple ordered key pairs; optional missing-key matching (off by default). Reports unmatched rows, duplicate/missing keys, and many-to-many matches. Conflicting column names use `_left`/`_right`. |
| Append | Two or more tables, aligned by column name; all or common columns; optional source dataset column. Keeps duplicate rows and warns about incompatible types without forcing conversion. |
| Aggregate | Optional grouping columns, multiple measures/calculations, named outputs, and a missing-group policy. Row count differs from nonmissing count; all-missing sums stay missing. |
| Pivot | Row identifiers, a column-label variable, a value variable, and optional aggregation. Repeated combinations require an aggregation choice; missing cells remain missing unless zero fill is selected. |
| Unpivot | Identifier columns, columns to turn into rows, variable/value output names, and optional removal of missing values. |
| Rename column | Select a column and enter a unique, non-empty new name. Values and column order are preserved. |
| Remove column | Select one or more columns to remove; at least one column must remain. |
| Add column | Blank values, a constant (text, number, or True/False), a copy of an existing column, row numbers starting at 1, or arithmetic between numerical columns. Division by zero is rejected. |
| Replace Values | Match an entire text, number, or boolean value and replace with a missing value, text, number, or boolean. Text matches are case-sensitive. Ignore surrounding spaces is enabled by default; turn it off for whitespace-sensitive matching. Replacement text is preserved exactly. Existing missing cells remain missing. Select Missing value for `$-` → missing; text `None` remains literal text. |
| Fill missing values | Fill actual missing cells in one column with an explicit, compatible value. Empty strings, whitespace, and other non-missing values remain unchanged. Supports text, numbers, booleans, dates/timestamps, and durations according to stored column type. Preview reports filled cells and remaining missing cells. Invalid values, fractional replacements in integer columns, and values outside the storage range are rejected; source data is preserved. |
| Remove Characters | Remove each supplied literal character from text cells. For `$3,000`, `$` gives `3,000`, `,` gives `$3000`, and `$,` gives `3000`. Results remain text; missing and non-text cells are preserved. |

**Duplicate cleanup belongs to Inspect.** Column Overview lists duplicated columns alongside the first equivalent column that will be kept. Columns must match every stored value, data type, and missing-value position; column names may differ. Empty datasets are not treated as evidence of duplicated columns. Row Overview appears immediately below Column Overview and shows the total row count and the number of extra duplicate copies after the first occurrence. For example, three identical rows count as two duplicates. Row comparisons use all stored columns, treat matching missing values as equal, and ignore the row index. Both removal actions create a named dataset, retain the first occurrence in source order, preserve the source, and open the result in Inspect. Duplicate checks use original stored values before analytical type conversions.

Imports and transformation results are bounded to 1,000,000 rows and 10,000,000 cells per table. Join, pivot, and unpivot size checks run before allocating the expanded output. Use Streamlit 1.64 or newer for stateful lazy tabs; install the updated requirements, including `openpyxl` and `xlrd` for Excel support.

Workspace and transformation tests: `python -m unittest discover -s tests -p "test_dataset_workspace.py"` and `python -m unittest discover -s tests -p "test_table_transforms.py"`.

Descriptive Statistics displays clickable ⓘ controls alongside all 15 numerical summary measures, including quartiles and IQR outlier counts. Each control expands an interpretation in place and supports keyboard use. Kurtosis is pandas' excess kurtosis (normal reference 0); the help text and numerical AI interpretation use that convention. Numerical calculations are unchanged.

---

## Sections

| Section | What it does |
|---|---|
| **Dataset Overview** | Row/column counts, scrollable data preview, boolean columns shown as True/False |
| **Column Overview** | Per-column table: data type, variable type/subtype, totals, missing values, unique values with expandable details and A/M classification badges; classification editor |
| **Row Overview** | Total rows, extra duplicated rows, preview of duplicates, and removal into a new dataset |
| **Data Quality Summary** | Tabbed quality panels with charts and stats for Numerical, Categorical, and Temporal columns + AI interpretation |
| **Exploratory Data Analysis** | Five tabs: Descriptive Statistics, Group Statistics, Visual Analysis, Relationship, and Comparison |
| **Visual Analysis (within EDA)** | Column-driven selection of distribution, comparison, relationship, time-series and matrix charts |
| **Predictive Analytics** | Automated model comparison and selection for regression and classification problems + AI interpretation |

---

## App Header

The top of the app shows three live metrics pulled from Supabase:

- 👁 **Total Visits** — unique sessions recorded since launch
- 👍 **Likes** — cumulative likes submitted across all feedback sections
- 👎 **Dislikes** — cumulative dislikes submitted across all feedback sections

---

## Sample Data

An expandable **Try with sample data** section sits below the file uploader. Clicking one of the four buttons fetches the file directly from Google Drive and loads it into the app without any manual download.

| File | Problem type | Rows |
|---|---|---|
| heart_disease.csv | Binary classification | 303 |
| BostonHousing.csv | Regression | 506 |
| iris.csv | Multi-class classification | 150 |
| CustomerSatisfaction.csv | Statistical Analysis | 120 |

---

## User Feedback

At the end of each of the three main sections — Data Quality Summary, Exploratory Data Analysis, and Predictive Analytics — a 👍 / 👎 row appears. Clicking either opens a popup dialog (Streamlit `@st.dialog`) titled **"Thank you for your feedback"** with an optional comment text area, a **Submit** button, and a **Skip** button. All responses are stored in Supabase with the section name, feedback type, optional comment, and timestamp.

---

## Dataset Overview

- Accepts CSV (`.csv`) and Excel (`.xlsx`, `.xls`) files
- Displays **Rows** and **Columns** as metric tiles immediately after upload
- Shows the first 100 rows in a scrollable table
- Boolean columns rendered as `True` / `False` text (not checkboxes)

---

## Column Overview

One row per column showing:

| Column | Description |
|---|---|
| Column Name | Original column name |
| Data Type | Plain-language dtype: Whole Number, Decimal Number, Text, True/False, Date/Time |
| Variable Type | Effective Numerical, Categorical, or Temporal classification |
| Variable Subtype | Discrete / Continuous; Binary / Nominal / Ordinal; Date / Time / DateTime / Duration |
| Total Values | Total row count |
| Missing Values | Count of missing entries |
| Missing % | Percentage missing, formatted to 2 decimal places |

**Colour coding:** Red = 20%+ missing. Yellow = 5–19% missing.

**Classification editor:** Select a column below the overview table, choose Variable Type and its dependent Variable Subtype, then Apply classification. Binary requires exactly two non-missing values. Numeric classifications require safe numeric conversion; Discrete requires integer-valued finite numbers. Temporal retains the existing 80% parsing threshold for date-like values; numeric timestamps require prior explicit conversion. The original dataset is preserved.

Integers default to Discrete, floats to Continuous. Categorical pairs default to Binary; other categories default to Nominal unless an explicit ordered dtype or recognized complete semantic sequence identifies Ordinal. Numeric 0/1 values remain numerical until overridden. Ordinal order is editable using category numbers from lowest to highest, preserving the original labels. Automatic and user-overridden classifications are shown as A/M badges beside both Variable Type and Variable Subtype. Hover text reads Automatic classification or Manual Override. Overrides persist in session metadata and reset when the dataset is replaced. Upload identity includes a content hash, preventing equal-name/equal-size replacements from retaining stale metadata.

The overview shows Unique Values, with a column selector in the expandable value-details section below the table: Continuous and Discrete with more than 10 unique values show min/max; Discrete with at most 10 and Binary show all values; Nominal shows the first five observed unique values; Ordinal shows all values in configured order. Missing values are excluded. Classification Source and Category Order are no longer separate columns; ordinal order remains editable. Missing percentages retain their color coding. See [VARIABLE_CLASSIFICATION.md](VARIABLE_CLASSIFICATION.md) for implementation and validation details.

---

## Variable Type Detection

| Type | Detection logic |
|---|---|
| **Numerical** | Pandas `int` or `float` dtype, excluding `bool` |
| **Categorical** | `object` dtype, `bool` dtype, or columns that cannot be reliably parsed as dates |
| **Temporal** | `datetime64` dtype, or `object` columns where 80% or more of non-missing values parse as dates |

> **Boolean columns** are classified as Categorical — True/False values represent discrete categories, not quantities.

---

## Data Quality Summary

Three tabbed panels, each rendered as bordered cards (two per row):

### Numerical Columns

**Continuous** (effective Variable Subtype):
- Left: histogram stacked above a boxplot
- Right: Missing Values, Missing %, Unique Values, Unique %, Minimum, Maximum, Average (all to 2 dp)

**Discrete** (effective Variable Subtype):
- Left: vertical bar chart sorted descending, capped at 10 bars
- Right: same stats as continuous

> Charts use the effective Variable Subtype. The theoretical distribution fitter’s Auto option also uses this subtype; its explicit family selector remains available.

### Categorical Columns
- Left: vertical bar chart, top 10 by frequency, sorted descending
- Right: Missing Values, Missing %, Unique Values, Unique %, Case Inconsistency, Special Characters
- Unique Values reports cardinality and excludes missing values.

### Temporal Columns
- Left: histogram stacked above a boxplot
- Right: Missing Values, Missing %, Unique Values, Unique %, Earliest Date, Latest Date, Date Range, Future Dates

> **Note on date parsing:** dates that cannot be parsed — such as text values, impossible dates (e.g. April 31), or unrecognised formats — are automatically converted to `NaT` and counted under **Missing Values**. The **Future Dates** field counts valid dates that fall after today. Invalid day numbers such as September 32 may be silently rolled over to the next valid date by the parser rather than treated as missing.

### AI Interpretation
After all quality panels load, the app generates a 4–5 sentence plain-English summary of the dataset's type, quality issues, and column composition using Groq (OpenAI GPT-OSS 120B (`openai/gpt-oss-120b`)). Cached in `st.session_state` and refreshable via a **🔄 Refresh** button.

---

## Exploratory Data Analysis

Tab order: **Descriptive Statistics → Group Statistics → Visual Analysis → Relationship → Comparison**.

### Descriptive Statistics (single column)

Select any column from a dropdown for a detailed deep-dive.

**Numerical columns:**
- Distribution chart (bar chart for discrete, histogram + boxplot for continuous) alongside a violin plot
- Summary statistics as a **Statistic / Value table**: Mean, Median, Std Dev, Standard Error (Mean), Coefficient of Variation (%), 95% Confidence Interval (Mean), Skewness, Kurtosis, Minimum, Quartile 1/2/3, Maximum, Outliers Found (IQR), Outlier %
- AI interpretation (cached per column, expandable)

**Categorical columns:**
- Frequency bar chart with configurable bar count slider (5 to 50 bars)
- Frequency table with percentage; values below 1% highlighted in yellow
- **Mode** directly below the frequency table, showing all tied modes and their frequency. Missing values are excluded; columns without non-missing values show an explanatory message.
- AI interpretation (cached per column, expandable)

**Temporal columns:**
- Histogram stacked above a boxplot (left) and violin plot (right) — showing the distribution of date values
- Summary statistics as a **Statistic / Value table**: Count, Missing Values, Earliest Date, Latest Date, Date Range, Most Common Year/Month
- AI interpretation (cached per column, expandable)

---

#### Numerical summary calculation notes

The summary table includes standard error of the mean (sample SD / sqrt(n)), coefficient of variation (sample SD / mean × 100%, positive means only), and a two-sided 95% Student-t confidence interval for the population mean (n−1 degrees of freedom). These use finite, non-missing observations and require at least two values; unavailable results show N/A. CV is meaningful for ratio-scale data with a true zero. Confidence intervals assume independent observations and approximately normal data or a sufficiently large sample. Constant samples produce zero standard error and a collapsed interval.

#### Theoretical distribution fitting

Below the numeric distribution chart, only these three results are shown by default:

- **Best theoretical fit**
- **Goodness of fit**
- **Conclusion**

The collapsed **Fit details** expander contains the type selector and detected type, optional known binomial trials, SSE, AIC, KS statistic, candidate comparison table, bootstrap p-values, sampling information, and assumptions. Lowest AIC identifies the best candidate among tested families; it does not establish adequate fit. Auto uses Variable Subtype from Column Overview; direct calls without metadata retain the integer-valued fallback. Users can explicitly select the family for rounded continuous measurements. Integers/ordinal values do not establish a count process.

Continuous candidates: Normal, Uniform, and strictly positive Lognormal/Exponential (location fixed at zero). Discrete candidates: Poisson and zero-based geometric, plus binomial only when the user supplies a known trial count. Unsupported discrete support and insufficient/constant data get explanatory messages.

Fit assessment uses 199 reproducible parametric bootstrap simulations, refitting each sample. A bootstrap p-value below 0.05 is Poor; otherwise Not rejected, never proof of a theoretical distribution. Discrete distance checks both sides of CDF jumps, without an ordinary continuous KS p-value. If all candidates fail, the conclusion states no suitable theoretical distribution was identified among tested families. Tests are exploratory, unadjusted across families, and assume independent observations.

Requires 20 finite observations and two distinct values. Above 3,000 observations a reproducible random sample is used and disclosed. Computations are cached. SSE uses common histogram probability-mass bins with tails; metrics and ranking refer to the same data sample.

### Group Statistics

Exploratory Data Analysis now has five tabs: **Descriptive Statistics → Group Statistics → Visual Analysis → Relationship → Comparison**. Use the type-grouped checkboxes to select a numerical measure (or leave all numerical variables unchecked for row counts), optional categorical dimensions, and a calendar variable when needed; then choose an analysis type.

- **Group Summary:** sum, mean, median, min/max, sample standard deviation, non-missing numerical count, or row count.
- **Running & Cumulative Statistics:** cumulative sum, expanding mean, and observed-period count of the selected period statistic.
- **Rolling Averages:** moving mean over a complete window of calendar periods.
- **Contribution to Total:** share of sum or count, using the dataset total or parent group as denominator.

Time analysis requires a detected Temporal column (or Column Overview override). Weeks run Monday–Sunday. Missing calendar periods are retained as gaps, with optional zero filling. Invalid dates are excluded and reported. Zero denominators produce N/A. Current-period and boundary coverage cautions are displayed. Expanding means weight periods equally, not original rows. Missing categories are retained; negative contributions are flagged. Chart segment selection does not change table results or contribution denominators. Results support CSV download.

Calculations live in `utils/group_statistics.py`; Streamlit rendering lives in `utils/group_statistics_ui.py`. Run calculation checks with `python -m unittest discover -s tests -p "test_group_statistics.py"`.


### Comparison

The analysis menu matches the full checked-variable selection. One-way ANOVA requires one numerical response and one categorical factor; Factorial ANOVA requires one quantitative numerical response and at least two factors. Segment Comparison accepts categorical variables and at most one numerical measure. Period comparison requires exactly one temporal variable and at most one numerical measure, with optional categorical segments. Temporal selections exclude non-period analyses. t-Test accepts one quantitative numerical variable alone, two quantitative numerical variables alone, or one quantitative numerical variable plus one categorical variable with exactly two observed groups. Its test-type menu follows that selection while leaving the choice between independent and paired samples explicit. Unsupported selections show guidance instead of analyses that would silently use a subset of the variables.

The **Comparison** tab sits immediately after **Relationship** and contains:

Factorial ANOVA's **Effect plots** tab displays every significant main effect and selected interaction at the user-selected α (default 0.05). An optional checkbox includes non-significant terms. Plots show model-adjusted estimated marginal means (EMMs) with pointwise 95% confidence intervals, equally weighted over other-factor levels. Interaction plots appear first and support swapping their axes; associated main effects are flagged for joint interpretation.

- **t-Test:** one-sample, independent samples (stacked or separate columns), and paired samples. Welch is the independent-sample default; pooled Student is optional. Directional hypotheses, confidence intervals, effect sizes, graphical diagnostics and CSV exports are included. See [TTEST.md](TTEST.md).
- **One-way ANOVA:** categorical grouping and numerical outcome, group summaries and box plots, normality/variance diagnostics, and Welch or classical ANOVA. The existing independence confirmation and validation are preserved.
- **Factorial ANOVA:** one numerical response and two or more metadata-classified categorical factors. Choose domain-relevant two-way interactions explicitly; all main effects remain. Classical OLS retains Type II sums of squares and partial η². Type II tests respect marginality: a main-effect test does not adjust for interactions containing that factor. Missing/nonfinite counts, empty/singleton cells, rank checks, residual error/degree-of-freedom checks, and complexity limits remain enforced. No transformation, model reduction, or method switch is automatic.

Factorial follow-up is organized as Diagnostics → Effect plots → Post-hoc comparisons → Findings:

- **Model validity:** normality, equal variance, independence, cell coverage, replication, rank, and overall reliability use OK/Caution/Problem/Cannot assess. OK means no issue was detected, not proof of an assumption. Independence remains a study-design question. Q–Q and residual-vs-fitted plots, Jarque–Bera and median-centered Levene remain available. Explicitly labeled exploratory screens flag spread trends, binned residual patterns, and extreme studentized residuals; their thresholds are disclosed in the UI.
- **Adjusted means and comparisons:** EMMs average predictions equally across a full reference grid of other-factor levels, which can differ from raw means in unbalanced designs. Mean SEs and pointwise 95% t intervals use fitted classical OLS covariance. Global pairs are available for 2–30-level factors without a significant selected interaction. For significant interactions, simple effects hold every significant partner fixed, support either comparison direction, and withhold strata missing a target level. Holm-adjusted p-values cover all displayed pairs (all supported strata for simple effects); Bonferroni simultaneous 95% intervals cover the same family. These intervals can be more conservative than Holm decisions. Repeated exploration across directions/factors/models is not jointly adjusted.
- **Safeguards:** the full-rank model requirement remains; follow-up vectors additionally undergo a row-space estimability check. Large reference grids (>100,000 cells or >2 million design entries), >60 simple-effect strata, or >2,000 conditional pairs are withheld with an explanation. Empty-cell additive predictions are explicitly identified as model-based extrapolation. Unsupported simple-effect strata are withheld rather than invented.
- **Guidance:** response-domain checks govern optional log/square-root suggestions; recommendations require a user-performed transformation/refit. Non-significant interactions prompt optional simplification, never automatic deletion. Findings combine reliability, evidence, partial η², supported adjusted-mean direction, requested comparisons, and next steps without claiming causation or universal effects in significant interactions.
- **HC3 sensitivity comparison:** explicitly requested robust Type II F tests use the same OLS model and appear separately; classical SS, partial η², EMM uncertainty, and findings are unchanged. Unit-leverage observations or invalid robust tests block HC3 output. HC3 does not repair dependence or missing design support. Welch is only suggested as a one-way alternative, not a general factorial solution.

Statistical references: [statsmodels Type II/HC3 ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html), [EMM reference grids and weighting](https://rvlenth.github.io/emmeans/reference/emmeans.html), [interaction/simple comparisons](https://rvlenth.github.io/emmeans/articles/AQuickStart.html). Verification includes balanced, unbalanced, sparse, main-effect, interaction, heteroskedastic and skewed datasets; direct reference comparisons for ANOVA, EMMs, intervals, contrasts and HC3; and browser/AppTest workflow checks. See `tests/test_factorial_anova.py` and `tests/test_factorial_guidance.py`.
- **Segment Comparison:** compare groups with a reference segment using absolute and percentage differences.
- **Period-over-Period Comparison:** previous calendar period or same period last year for monthly/quarterly/yearly data.

Segment and period comparisons reuse the existing calculations, missing-value/calendar behavior, charts and CSV downloads. Their controls have independent state from Group Statistics. ANOVA is no longer listed under Relationship. Regression, correlation, cross-tabulation/chi-square and time–numerical trends remain there. The new renderer is `utils/comparison_ui.py`.

#### Categorical group statistics and contingency tables

The numerical selector supports **None**, including datasets with no numerical columns. With None selected, group summaries, segment comparisons, contribution analysis, and time analyses use **Row count**, including rows with missing numerical values. Numerical measures retain the original statistics; Row count is also available alongside non-missing count.

In **EDA → Relationship**, select at least two categorical columns and choose **Categorical–categorical: cross-tabulation and chi-square**, then choose distinct row and column variables. Extra dimensions are not included as controls. This analysis always uses unweighted row counts, regardless of the numerical selection. Missing categories are excluded by default, with an option to include them explicitly.

The cross-tab report shows three stacked lines per cell: observed count, expected count (2 decimals), and standardized/Pearson residual (4 decimals). Row totals, column totals, grand total, and a cell-content legend follow the reference report format; no heatmap or view selector is shown. Download observed/expected counts, differences, and residuals together as CSV. Expected counts use row total × column total / grand total.

The Pearson chi-square test reports statistic, degrees of freedom, p-value, and interpretation at the selected significance level (0.05 default), without continuity correction. It requires at least two observed categories per variable and independent observations. Expected counts below 5 trigger a sparse-table warning. P-values are unadjusted for repeated testing; a non-significant result does not prove independence and association does not establish causation. Unused category levels are omitted. Tables exceeding 10,000 cells are rejected; wide tables scroll horizontally without truncating calculations.

Method: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html

Run calculation tests: `python -m unittest discover -s tests -p "test_*statistics.py"`.

---

### Relationship (multi-column)

Select variables with the existing checkboxes, then choose an available **Analysis method**. Existing chart choices remain below the statistical analysis.

| Variables | Added methods |
|---|---|
| Numerical–numerical | Scatterplot with OLS line; Pearson correlation and p-value; Spearman correlation; simple regression coefficients, standard errors, 95% intervals, and in-sample R² |
| Categorical–categorical | Stacked cross-tabulation report and Pearson chi-square, available only in Relationship |
| Time–numerical | Daily/weekly/monthly/quarterly/yearly mean or sum, aggregated time-series plot and descriptive linear trend per elapsed day |
| Multiple variables | Numerical outcome with two or more numerical predictors; multivariable OLS coefficients, intervals, p-values, adjusted R², and observed-versus-fitted plot |

Missing/nonfinite pairs are excluded and reported. Regression rejects insufficient samples, constant outcomes, and rank-deficient predictor matrices. Classical coefficient inference assumes independent, homoscedastic errors and approximately normal errors for small samples. ANOVA requires two groups with at least two observations each; independence, equal variances and approximately normal errors are assumptions. In-sample R² is not predictive validation; association is not causation. Time trends omit missing periods without treating them as zero, use actual elapsed time, and do not model seasonality/autocorrelation or provide forecasts.


Select columns using checkboxes grouped by type. Available chart types appear automatically based on the combination selected:

| Column selection | Available charts |
|---|---|
| 2 categorical + 1 numerical | Heatmap — Numerical Values |
| 1 numerical + 1 categorical | Boxplot — Distribution by Category |
| 1 datetime + 1 or more numerical | Line Chart — Trend |
| Exactly 2 numerical | Scatter Plot — Correlation, Interaction Chart — Density |
| More than 2 numerical | Pearson or Spearman matrix within the Correlation analysis |
| Exactly 2 categorical | Heatmap — Occurrence, Interaction Chart — Grouped Bar |

---

## Predictive Analytics

### Flow

1. **Select Target Column** — the column to predict
2. **Predictor Columns** — checkboxes grouped by variable type; **✅ Select All** and **☐ Deselect All** buttons for quick selection; columns suspected of being identifiers are flagged with a warning
3. **Model** — the app detects the problem type, explains the model choice, and shows a hyperparameter configuration table with an estimated training time
4. **▶ Run** — trains models, displays results, and generates an AI interpretation

### Target Type Detection

| Condition | Type | Models run |
|---|---|---|
| Categorical → Binary | Binary | Logistic Regression vs RF Classifier |
| Numerical → Discrete or Continuous | Numerical regression | Lasso vs RF Regressor |
| Categorical → Nominal or Ordinal | Classification | Multinomial LR vs RF Classifier |

> For binary classification of numeric 0/1 values, set Categorical → Binary. Two-label targets are encoded internally for logistic F1/ROC scoring; results preserve the original class labels. Ordinal predictors currently use categorical encoding, not an ordinal-specific regression model.

### Regression Comparison (Lasso vs Random Forest)

Both models trained simultaneously. Winner determined by Test R².

**Lasso Regression** — StandardScaler applied automatically to numerical features. Alpha tuned via GridSearchCV.

**Random Forest Regressor** — no scaling required. Depth, split, and leaf parameters tuned.

**Hyperparameter grid:**

| Model | Parameter | Values tested | Fits |
|---|---|---|---|
| Lasso | Alpha | 0.001, 0.01, 0.1, 1, 10 | 25 |
| Random Forest | Max depth | 10, 15 | — |
| Random Forest | Min samples to split | 5, 10 | — |
| Random Forest | Min samples per leaf | 3, 5 | 40 |
| Combined | CV folds | 5 | **65 total** |

**Results:** comparison table (green = better value), winner announcement, metric cards (R², RMSE, MAE, CV Mean R²), learning curve (R²), feature importance (RF) or coefficient chart (Lasso), SHAP summary plot, AI interpretation.

### Binary Classification Comparison (Logistic Regression vs Random Forest)

Both models trained simultaneously. Winner determined by Test F1 Score.

**Logistic Regression** — StandardScaler applied automatically. Penalty and C tuned. Coefficients converted to odds ratios when LR wins. Class weights balanced.

**Random Forest Classifier** — no scaling required. Class weights balanced.

**Hyperparameter grid:**

| Model | Parameter | Values tested | Fits |
|---|---|---|---|
| Logistic Regression | Penalty | l1, l2 | — |
| Logistic Regression | C | 0.01, 0.1, 1, 10 | 40 |
| Random Forest | Max depth | 10, 15 | — |
| Random Forest | Min samples to split | 5, 10 | — |
| Random Forest | Min samples per leaf | 3, 5 | — |
| Random Forest | Max features | sqrt, log2 | 80 |
| Combined | CV folds | 5 | **120 total** |

**Results:** comparison table, winner announcement, metric cards (F1, Accuracy, Precision, Recall, ROC-AUC), confusion matrix + ROC curve, learning curve (F1), feature importance (RF) or coefficient chart + odds ratio table (LR), SHAP summary plot, AI interpretation.

### Multi-Class Classification Comparison (Multinomial LR vs Random Forest)

Both models trained simultaneously. Winner determined by weighted Test F1 Score.

**Multinomial Logistic Regression** — StandardScaler applied. `saga` solver. Per-class coefficients and odds ratios shown when LR wins. Class weights balanced.

**Random Forest Classifier** — no scaling required. Class weights balanced.

**Hyperparameter grid:**

| Model | Parameter | Values tested | Fits |
|---|---|---|---|
| Multinomial LR | Penalty | l1, l2 | — |
| Multinomial LR | C | 0.01, 0.1, 1, 10 | 40 |
| Random Forest | Max depth | 10, 15 | — |
| Random Forest | Min samples to split | 5, 10 | — |
| Random Forest | Min samples per leaf | 3, 5 | — |
| Random Forest | Max features | sqrt, log2 | 80 |
| Combined | CV folds | 5 | **120 total** |

**Results:** comparison table (all metrics weighted), winner announcement, metric cards (F1 weighted, Accuracy, Precision weighted, Recall weighted, Weighted ROC-AUC), confusion matrix, learning curve (Weighted F1), feature importance (RF) or per-class coefficient chart + odds ratio table with class selector (LR), SHAP summary with class selector, AI interpretation.

### AI Interpretation (Predictive Analytics)

After results load, select **Generate interpretation** inside the **🤖 AI Interpretation** expander to generate a 4–5 sentence plain-English explanation covering:
1. Which model won and what it suggests about the data's complexity
2. The 2–3 most important features and what drives predictions
3. How top features push predictions up or down (based on SHAP)
4. Whether the model generalises well or shows overfitting signs (train vs test gap)
5. One practical recommendation

Cached per target + predictor combination. Invalidated automatically when the selection changes.

### ID Column Detection

Predictor columns are scanned for likely identifiers before training:
- Column name contains a whole-word match for: `id`, `key`, `index`, `code`, `uuid`, `ref`, `pk`
- Integer columns where 95%+ of values are unique

Float columns are excluded from uniqueness checks.

---

## Charts Reference (`utils/charts.py`)

All charts use a consistent visual style: blue primary colour (`#4C72B0`), white background, dark axis labels, light grey gridlines.

### Quality and Profile Charts

| Function | Signature | What it produces |
|---|---|---|
| `make_histogram_boxplot` | `(series, height=300)` | Histogram (top) stacked above boxplot (bottom) on a shared x-axis. Used for continuous numerical and date/time columns. |
| `make_bar_chart` | `(series, max_bars=10, height=300)` | Vertical bar chart of value counts, sorted descending, capped at `max_bars`. Returns `(figure, was_truncated, total_unique)`. |
| `make_violin` | `(series, height=300)` | Violin plot with embedded boxplot and mean line. Used in Descriptive Statistics alongside the distribution chart. |
| `make_time_series` | `(series, freq="M", height=360)` | Line chart of record counts aggregated by time period. `freq`: `"D"` day, `"W"` week, `"M"` month, `"Y"` year. |

### Relationship Charts

| Function | Signature | What it produces |
|---|---|---|
| `make_line_chart` | `(df, x_col, y_cols, height=400)` | Multi-series line chart. One line per column in `y_cols`, coloured from the `LINE_COLORS` palette (8 colours). |
| `make_heatmap` | `(corr_df, height=460)` | Correlation matrix heatmap. Red–Blue diverging scale (RdBu_r), values annotated in cells, scale fixed at –1 to +1. |
| `make_scatter` | `(x, y, x_name, y_name, height=400)` | Scatter/dot plot with 55% opacity markers. Correlation coefficient shown as caption below the chart. |
| `make_density_contour` | `(x, y, x_name, y_name, height=420)` | 2D density contour (Blues scale) with faint individual data points underneath for context. |
| `make_occurrence_heatmap` | `(df, col_a, col_b, max_cats=10, height=460)` | Pivot-table heatmap showing how often each combination of two categorical columns occurs. Blues scale, count annotated in cells. |
| `make_grouped_bar` | `(df, col_a, col_b, max_cats=10, height=420)` | Side-by-side bars grouped by `col_b` values, one group per `col_a` value. Multi-colour palette. |
| `make_boxplot_by_category` | `(df, num_col, cat_col, max_cats=15, height=420)` | One boxplot per category, sorted by median descending, mean line shown. Returns `(figure, was_truncated, total_cats)`. |
| `make_pivot_heatmap` | `(df, cat_a, cat_b, num_col, max_cats=10, height=460)` | Heatmap of the mean of a numerical column per categorical combination. Red–Yellow–Blue diverging scale. |

### Predictive Analytics Charts

| Function | Signature | What it produces |
|---|---|---|
| `make_feature_importance_chart` | `(feat_imp, height=450)` | Horizontal bar chart of top 20 feature importances sorted descending. `num__` / `cat__` prefixes stripped automatically. |
| `make_learning_curve_chart` | `(train_sizes, train_scores, val_scores, height=380, score_label="Score")` | Training vs validation score over training set size. ±1 std dev shaded bands. `score_label` sets the y-axis title — pass `"R² Score"`, `"F1 Score"`, `"Weighted F1"`, or `"Accuracy"`. |
| `make_coefficient_chart` | `(coef_series, height=450)` | Horizontal bar chart of top 20 coefficients by absolute value. Blue = positive, orange = negative. Dotted zero reference line. |
| `make_confusion_matrix` | `(cm, labels, height=360)` | Heatmap confusion matrix with count annotated in each cell. Blues colour scale. |
| `make_roc_curve` | `(fpr, tpr, auc_score, height=360)` | ROC curve with AUC score in the legend. Grey dashed diagonal baseline. |

### Constants

| Constant | Value | Purpose |
|---|---|---|
| `DISCRETE_THRESHOLD` | `15` | Legacy fallback for callers without classification metadata; app routing uses effective Variable Subtype. |
| `CHART_COLOR` | `#4C72B0` | Primary blue used for all single-series charts |
| `LINE_COLORS` | 8-colour palette | Cycle of colours for multi-series charts |

---

## Number Formatting

All percentage and numeric stats formatted to **2 decimal places**:
- Missing % and Unique % (Column Overview and quality panels)
- Minimum, Maximum, Average (numerical quality panels)
- All predictive analytics metrics (R², RMSE, MAE, F1, Accuracy, etc.)

---

## Installation

### Requirements

- Python 3.9 or later
- A [Groq API key](https://console.groq.com) (free tier available)
- A [Supabase](https://supabase.com) project (free tier available) with two tables created

### Supabase Setup

Run the following SQL in your Supabase SQL Editor:

```sql
CREATE TABLE data_analytics_visits (
    id         BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE data_analytics_feedback (
    id            BIGSERIAL PRIMARY KEY,
    section       TEXT        NOT NULL,
    feedback_type TEXT        NOT NULL,
    comment       TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### Setup

```bash
# 1. Clone or download the project
git clone https://github.com/your-username/data-analytics-assistant.git
cd data-analytics-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your secrets
# Edit .streamlit/secrets.toml:
# GROQ_API_KEY = "your-groq-api-key-here"
# SUPABASE_DB_URL = "postgresql://postgres.xxx:[password]@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
```

### Run locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Deploy to Streamlit Community Cloud

1. Push the project to a GitHub repository (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** and select your repository, branch, and `app.py` as the entry point
4. Open **Advanced settings** and add both secrets:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   SUPABASE_DB_URL = "postgresql://postgres.xxx:[password]@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
   ```
5. Click **Deploy**

> Do not commit `.streamlit/secrets.toml` to GitHub. Add it to `.gitignore` to keep your credentials private.

---

## Configuration

### `.streamlit/secrets.toml`

```toml
GROQ_API_KEY   = "your-groq-api-key-here"
SUPABASE_DB_URL = "postgresql://postgres.xxx:[password]@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).
Get a free Supabase project at [supabase.com](https://supabase.com) — copy the connection string from **Project Settings → Database → Connection string → URI**.

### `.gitignore` (recommended)

```
.streamlit/secrets.toml
__pycache__/
*.pyc
.env
```

---

## File Structure

```
data-analytics-assistant/
├── app.py                    # Streamlit entry point — all UI sections
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── tests/                    # Automated group, contingency, and distribution checks
├── .streamlit/
│   └── secrets.toml          # API keys and DB URL (do not commit)
└── utils/
    ├── __init__.py
    ├── loader.py             # CSV/Excel file loading; Column Overview table builder
    ├── detector.py           # Column type classification (Numerical/Categorical/Date)
    ├── charts.py             # All Plotly chart functions; DISCRETE_THRESHOLD constant
    ├── quality.py            # Quality check DataFrames for AI summary text
    ├── exploration.py        # Single-column deep-dive stats (descriptive stats,
    │                         # outlier detection, frequency tables, datetime stats)
    ├── visual_analysis.py   # Column-driven Visual Analysis charts and controls
    ├── manual_input.py       # Validated table/CSV entry and dataset activation
    ├── relationship_stats.py # OLS regression and one-way ANOVA calculations
    ├── relationship_ui.py    # Five relationship analysis methods
    ├── group_statistics.py   # Group, time-series, contribution, contingency calculations
    ├── group_statistics_ui.py # Streamlit Group Statistics controls and results
    ├── contingency_table.py  # Stacked observed/expected/residual HTML report
    ├── distribution_fit.py   # Candidate fitting and refitted bootstrap assessment
    ├── distribution_fit_ui.py # Three-result summary and collapsed Fit details
    ├── ai_interpreter.py     # LangChain + Groq chains: dataset overview,
    │                         # single-column, heatmap, and predictive analytics
    ├── modelling.py          # ML pipelines: target type detection, ID column detection,
    │                         # regression comparison (Lasso vs RF), binary comparison
    │                         # (LR vs RF), multiclass comparison (Multinomial LR vs RF),
    │                         # time estimates, result interpretation helpers
    └── database.py           # Supabase connection via psycopg2: visit tracking,
                              # stats retrieval (cached 60s), feedback submission
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `pandas` | Data loading and manipulation |
| `openpyxl` | Excel file support |
| `plotly` | Interactive charts |
| `scipy` | Statistical summaries, Student-t confidence intervals, chi-square testing, and theoretical distribution fitting |
| `scikit-learn` | ML pipelines, GridSearchCV, preprocessing, model classes |
| `shap` | SHAP explainability plots for all model types |
| `matplotlib` | SHAP summary plot rendering via `st.pyplot` |
| `psycopg2-binary` | PostgreSQL connection to Supabase for visit and feedback storage |
| `langchain` | LLM chain orchestration |
| `langchain-groq` | Groq API client for LangChain |

---

## AI Interpretations

Uses **OpenAI GPT-OSS 120B** (`openai/gpt-oss-120b`) via Groq through LangChain `ChatGroq`, with temperature **0.3**. The existing `GROQ_API_KEY` is used. Python performs the statistical calculations and model training; the LLM provides explanatory text.

| Function | When it runs | Output |
|---|---|---|
| `interpret_dataset_overview` | After data quality summary loads | 4–5 sentences on dataset type, missing values, and variable types |
| `interpret_single_column` | On Descriptive Statistics deep-dive | 3–4 sentences on quality, anomalies, and recommended action |
| `interpret_predictive_results` | After model training completes | 4–5 sentences on winning model, key features, SHAP impact, overfitting, and recommendation |

All interpretations are cached in `st.session_state`. Dataset-level and predictive analytics interpretations regenerate only when the data or model selection changes. Column-level interpretations have a **🔄 Refresh** button.

---

## Roadmap

| Feature | Status |
|---|---|
| File upload, dataset overview, column overview | ✅ Complete |
| Data quality summary (numerical, categorical, date/time) with AI | ✅ Complete |
| Descriptive Statistics — single column deep-dive with AI | ✅ Complete |
| Relationship — multi-column charts with AI | ✅ Complete |
| Predictive Analytics — Regression (Lasso vs RF) with AI | ✅ Complete |
| Predictive Analytics — Binary Classification (LR vs RF) with AI | ✅ Complete |
| Predictive Analytics — Multi-class Classification (Multinomial LR vs RF) with AI | ✅ Complete |
| Visit counter + like/dislike feedback with Supabase | ✅ Complete |
| Sample datasets (heart disease, Boston housing, iris, e-commerce sales) | ✅ Complete |
| Group Statistics summaries; contingency tables and chi-square in Relationship | ✅ Complete |
| Categorical mode and cardinality label | ✅ Complete |
| Standard error, coefficient of variation, and 95% mean confidence interval | ✅ Complete |
| Distribution fit summary and collapsed Fit details | ✅ Complete |
| Manual Input Data: editable table and CSV text pane | ✅ Complete |
| Data Cleaning | 🔜 Planned |


## Manual data entry — Input Data

**Status: implemented.** Existing file upload and **Try with sample data** remain available. Use the **Input Data** button for manual entry, offering:

- An editable table for entering or pasting data.
- A text pane for typed or pasted CSV-formatted data, with the first row used as column headers.

The editable grid is fixed at **500 rows × 25 columns**, with no dimension or column-name setup controls. Paste the table including headers into the top-left cell. Auto detection treats a populated text-only first row as headers; **Header detection and preview** provides first-row/no-header overrides for ambiguous categorical data or numeric headers. Unused trailing rows/columns are trimmed; interior missing rows/cells are retained. A pasted header occupies one grid row, leaving **499 data rows**; without headers the grid holds 500 data rows. CSV entry still accepts 500 data rows plus a header. Blank input, invalid headers and excess limits are rejected. Numeric inference preserves leading-zero identifiers.

Both manual entry methods are limited to **500 data rows and 25 columns**; the header row does not count toward the row limit. The manual submission button is also labelled **Input Data**, not Load Data. Validate the input and display clear errors before sending valid data through the existing analytics workflow. These limits apply to manual entry, not existing uploads or Google Drive samples.

## Automated checks

The `tests/` folder contains development checks; they do not run during normal app usage.

| File | Coverage |
|---|---|
| `tests/test_group_statistics.py` | Group totals, reference comparisons, contributions, missing periods, rolling/running statistics, and year-over-year calculations |
| `tests/test_categorical_statistics.py` | Known chi-square results, expected counts, residuals, missing/unused categories, row counts, and degenerate tables |
| `tests/test_input_relationship.py` | Manual CSV limits/validation, known regression coefficients, rank deficiency, and ANOVA |
| `tests/test_distribution_fit.py` | Poor count-family fits for bedroom-like data, lognormal/binomial examples, discrete CDF distance, invalid support, and insufficient data |

After installing the app dependencies, run from the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

These tests cover selected calculations. They do not verify live Groq or Google Drive availability, Supabase connections, or every browser layout. The production app remains Python/Streamlit, with Streamlit Community Cloud deployment instructions above; the earlier proposed Netlify frontend migration is not the active direction.


## Visual Analysis

A tab within Exploratory Data Analysis, between Group Statistics and Relationship. Select variable checkboxes in **Numerical**, **Categorical**, and **Temporal (Date / DateTime)** groups; only compatible graph buttons appear. Click a button to select a graph; the active button is highlighted. Buttons are arranged four per row, and changing columns resets an incompatible selection to the first available graph. Additional controls assign axes, grouping, measures, and aggregation.

| Graph | Selected columns needed |
|---|---|
| Histogram, dot plot, density, box plot, normal Q–Q plot | At least one numeric; optional categorical grouping for box plots |
| Bar, pie, Pareto | At least one categorical; optional numerical measure (otherwise counts) |
| Stacked-bar | At least two categorical; optional numerical measure |
| Line, area | Date/time + numeric, or at least two numeric |
| Scatter, correlation matrix, pair plot | At least two numeric |
| Heatmap | At least two categorical for counts/mean pivot, or two numeric for record-value heatmap |
| Bubble | At least three numeric for X, Y, and nonnegative size; optional categorical color |

Histogram bin count and density bandwidth are adjustable. Dot plots stack repeated values. Q–Q uses a normal reference. Bars offer sum/mean or counts; Pareto uses nonnegative counts/sums and a cumulative percentage against the full total. Line/area charts sort X and aggregate repeated values; areas stack the selected measures. Correlations support Pearson and Spearman with pairwise complete observations. Pair plots accept 2–6 variables. Numerical heatmaps optionally standardize columns. Categorical heatmaps show counts or mean measures.

Display limits are disclosed: dot plots sample 2,000 observations, density 10,000, scatter/bubble 5,000 and pair plots 2,000 when needed, all with a fixed seed. Numerical heatmaps show the first 500 rows. Categorical heatmaps reject more than 10,000 cells. Missing/nonfinite values are omitted as appropriate; invalid sizes, constant density inputs, and insufficient Q–Q data receive explanatory messages.

Implemented in `utils/visual_analysis.py`. Existing Descriptive Statistics, Group Statistics, Relationship, and Predictive Analytics remain available.


For exactly **one numeric + one categorical column** (and no date/time column), Visual Analysis offers only **Box plot, Bar chart, Pie chart, and Pareto chart**. Box plots group the selected measure by the category; bar and Pareto charts use the selected numerical measure rather than defaulting to row counts. Single-variable histogram, dot, density, and Q–Q plots are not offered for this combination.


Stacked-bar charts have an **Additive measure** selector: **Sum** or **Count**. Sum requires a selected numerical measure and adds its non-missing values. Count counts rows per category combination, including rows with missing numeric values. Missing grouping categories are excluded. With no numeric column selected, only Count is available and a hint explains how to enable Sum. Mean is not offered for stacked bars.


### Exact Visual Analysis selection rules

For these combinations without date/time columns, the following menus override the general chart-availability table above:

| Selected columns | Available graphs |
|---|---|
| One numeric + one categorical | Box plot, bar chart, pie chart, Pareto chart |
| Two numeric + one categorical | Scatter plot, with category color required |
| One numeric + two categorical | Grouped bar chart, stacked-bar chart, grouped box plot, heatmap |
| Two numeric | Scatter plot, correlation matrix, pair plot, heatmap, line chart, area chart |
| Two categorical | Grouped bar chart, stacked-bar chart, count heatmap |

Grouped bars use both categories (axis and color), with the numeric measure required when selected. Grouped boxes show the selected measure by category and subgroup. Numeric/category heatmaps require the selected numeric measure and display its mean by the two categories. Two-numeric line/area plots require a meaningful ordered X variable, as noted in the UI. Stacked bars retain the requested Sum/Count selector: Sum uses the selected measure, whereas Count intentionally shows category frequencies, including rows with missing numeric values. Other combinations retain their existing chart options.


## Relationship validation update

Cross-tabulation and chi-square are available only in **EDA → Relationship**. Group Statistics retains four descriptive methods; Segment Comparison and Period-over-Period Comparison are in Comparison. The renderer is `utils/contingency_ui.py`; existing counts, percentages/residual report, sparse-table warning and CSV download are preserved.

**Regression:** OLS equations, coefficient explanations, 95% mean-response bands for a single predictor, and an expandable Model validation panel. Checks include residual histogram, Q–Q plot, Omnibus and Jarque–Bera tests, residual/fitted plot with a smooth trend, Koenker Breusch–Pagan variance test, and Cook's distance/leverage screening. Durbin–Watson and the residual sequence plot require meaningful original row order. Missing/nonfinite values, rank-deficient predictors, small samples and perfect fits are handled explicitly. Perfect fits suppress inference. Diagnostics warn rather than certify assumptions; non-normality alone does not bias OLS under a correct conditional-mean model, and constant variance does not imply independence. Conventional intervals are not automatically replaced by robust intervals.

**ANOVA (EDA → Comparison → Group Comparison):** choose a categorical group and numeric outcome. Group summaries, box plots, Shapiro–Wilk within each nonconstant group (3–5,000 observations), group Q–Q plots and median-centered Levene/Brown–Forsythe variance diagnostics are provided. Welch is the default; classical is explicitly selectable. At least two groups with two usable observations each are required. Welch needs positive variance in every group; classical needs positive pooled within-group variance. Missing pairs and lost groups are reported. Small groups, imbalance and outliers trigger warnings without automatic removal. Users must confirm independent observations before results appear; confirmation resets with the dataset or selected columns. Repeated/clustered measurements need a suitable alternative model. The result reports F, degrees of freedom and p-value; no automatic post-hoc tests are run.

Install with the project's `requirements.txt`, including statsmodels. Run checks with `python -m unittest discover -s tests -p "test_*.py"`.

References: [Ray Chen's regression validation article](https://medium.com/@chenchungwai/a-comprehensive-guide-to-regression-model-validation-with-visual-tools-and-statistical-tests-in-f86d4ff6efbb), [statsmodels ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html), [SciPy Levene](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html), [SciPy Shapiro–Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html).


### Linear regression with numeric and categorical predictors

In EDA → Relationship, select the columns you want to use and choose **Regression**. Choose a numeric **Response / dependent variable (Y)** and one or more **Predictors / independent variables (X)**, which can be numeric, categorical, or a mixture. One unified interface supports simple and multiple regression. Categorical predictors use reference-level indicator coding; the coding table reports the reference and comparison levels. Coefficients compare each level with its reference while holding other predictors constant. Missing selected values are excluded before encoding. Constant categories, redundant encoded predictors, and insufficient residual degrees of freedom are blocked; sparse categories warn. More than 100 observed levels in a predictor are blocked as a usability safeguard. Numeric category codes use the categorical classification from Column Overview. Multiple/categorical models display observed versus fitted values; one numeric predictor retains the fitted line and mean-response confidence band. The existing residual diagnostics apply to the encoded model.


### Separate correlation analysis

In EDA → Relationship, select two or more numeric columns and choose **Correlation**. Choose Pearson or Spearman. Two variables display their correlation, two-sided p-value and scatter plot; more than two display a correlation matrix heatmap and coefficient table, with usable pair counts. Missing/nonfinite values are omitted pairwise, with at least three usable observations and variation in both variables required. Undefined entries remain blank/NaN. Correlation results are no longer included under Linear regression.


## Extended Linear Regression

Linear regression now supports explicitly selected numeric/categorical interactions, centered quadratic/cubic terms, hierarchy-preserving backward elimination, detailed fit/coefficient analysis, VIF/tolerance, studentized residuals and influence plots, confidence and prediction bands, original-unit prediction with both intervals, and deterministic Regression Findings. Results are organized into seven areas and fitted models are cached per session. See [the regression guide](REGRESSION.md) for methods, model hierarchy, validation rules, files and tests. Correlation remains a separate Relationship analysis.


### Retained interaction and polynomial plots

Regression Visual Analysis shows interaction plots only for interactions retained in the final model, and polynomial plots only for retained powers. Numeric moderators use their complete-case 25th, 50th and 75th percentiles; categorical moderators use separate curves by level. Categorical-by-categorical plots use category axes with confidence bars. Other predictors are held at displayed complete-case means or reference categories. Predictions reuse the fitted design and show pointwise 95% mean-response confidence intervals. Numeric axes stay within observed ranges, including category-specific ranges where applicable. These conditional displays do not adjust for model-selection uncertainty. Polynomial predictors with retained interactions show curves by an interacting predictor. Single-predictor polynomial plots preserve observed points and both confidence and prediction bands.

Regression Model Overview combines model performance with a two-column Model detail / Value summary table and the full fitted equation. Selected terms shows Term, Type, Final status, Centering - Mean Subtracted and Centering - Transformation. Separate specified-predictor, centering and equation-component tables are removed. Elimination history remains expandable.

Linear Regression uses checkbox controls for its target (exactly one), predictors, interaction pairs, and individual quadratic/cubic terms. Polynomial controls include Select all and Clear; cubic terms automatically include their squares. Unavailable dependent terms are cleared when predictors change.

Regression response selection is labeled **Response / dependent variable (Y)**. Predictors and Interaction terms each provide Select all and Clear all buttons; the response is excluded from predictors and clearing predictors removes dependent terms. The summary is a two-column table, followed by the **Fitted regression equation** heading and equation. Checkbox controls retain their existing layout.

Relationship provides independent Select all / Clear all controls for Numerical and Categorical columns. Polynomial terms use the same concise button labels and have no extra Select powers subheading.


## Classical regression in Relationship

**Regression** is the shared entry point. A Numerical response (Discrete or Continuous, including numeric 0/1) uses the existing linear regression. A Categorical → Binary response with two non-missing levels uses classical binary logistic regression; to treat numeric 0/1 as binary, change its classification in Column Overview. Regression is available with a valid response and at least one other selected numerical/categorical variable, including categorical-only selections with a binary response.

Both models use the same response/predictor checkboxes, interaction/polynomial controls, hierarchy rules and seven result tabs. Logistic regression requires an explicit modeled event and supports categorical reference choices and a configurable confidence level. It uses statsmodels binomial-logit maximum likelihood, coefficient/odds-ratio inference, LR model tests, AIC/BIC, deviance and McFadden Pseudo-R². No elimination is the default; optional backward AIC selection preserves hierarchy.

Logistic diagnostics cover separation, convergence, outcome information, sparse categories, VIF, logit functional form, residuals, influence and in-sample calibration. Predictions are probabilities with confidence intervals. ROC/AUC is optional and secondary. There is no train/test split, cross-validation, threshold optimization or automatic alternative model in this Relationship workflow. See [LOGISTIC_REGRESSION.md](LOGISTIC_REGRESSION.md) for implementation, limitations and verification.

### Comparison selection and ANOVA results

Comparison uses grouped Numerical, Categorical and Temporal checkboxes with Select all/Clear all, matching Relationship. Only selected variables are offered in the analysis controls. ANOVA requires numerical and categorical selections; Segment Comparison requires a categorical selection; Period-over-Period Comparison requires a date/time selection and supports row counts without a numerical measure. Unavailable analysis selections reset automatically.

ANOVA has five result tabs: Model Overview (summary table and overall test), Group Analysis (descriptive statistics), Diagnostics (normality, variance and warnings), Visual Analysis (group box plots), and ANOVA Findings (interpretation). Method selection and independence confirmation remain above the tabs; descriptive results remain available before confirmation. Existing statistical validation is unchanged.

### t-Test in Comparison

Select at least one quantitative Numerical variable to enable **t-Test**. The selected variables constrain the response, group and sample controls. Classification overrides are respected, including Categorical → Binary columns stored as integers. Automatically classified identifier-like names are excluded unless deliberately overridden to Numerical.

The five result tabs are **Test Configuration**, **Descriptive Statistics**, **Diagnostics**, **t-Test Results**, and **Interpretation**. All three modes support two-sided/greater/less hypotheses, a configurable confidence level (95% default), and matching confidence bounds for the signed difference. Significance alpha is 1 minus confidence. Diagnostics use the existing histogram/box plot and IQR outlier helpers, Q–Q plots and skewness; no Shapiro–Wilk or other formal normality test is run, and outliers are retained. Paired calculations and diagnostics use complete-pair differences. Statistical implementation, assumptions, effect-size definitions and verification are documented in [TTEST.md](TTEST.md).

Descriptive Statistics uses checkboxes grouped by **Numerical**, **Categorical**, and **Temporal** according to the effective Column Overview classifications. Only one variable can be selected at a time; selecting another automatically clears the previous selection. Unchecking the active variable shows a selection prompt. Existing statistics, charts and distribution analysis are preserved.

Temporal is the visible analytical variable-type label throughout the app; the internal `Date / Time` metadata key is retained for compatibility with saved classifications.

### Temporal subtypes

Temporal now offers **Date**, **Time**, **DateTime**, and **Duration** in Column Overview, with automatic inference and manual overrides. Legacy generic Temporal subtype metadata is upgraded on load while retaining its classification source.

- Date: calendar dates; explicit timestamps (even textual midnight timestamps) are DateTime. Native midnight-only datetime columns default to Date.
- Time: clock times such as `08:30`, `17:45:15`, or `8:30 PM`, stored for analysis as elapsed time since midnight without inventing a calendar date.
- DateTime: date and time, including timestamps with zones; timezone-aware inputs are normalized to UTC for analysis. Original input values remain intact.
- Duration: timedeltas or explicitly unit-bearing values such as `2 hours`, `1 day`, or `PT2H`. Clock notation defaults to Time; a manual Duration override interprets `08:30` as eight hours and thirty minutes. Unitless numbers are not silently treated as nanoseconds. Calendar months/years are not fixed-length durations.

Overrides require at least 80% valid nonmissing values for the selected subtype; invalid values become missing in the analysis copy and are reported. Date overrides reject timestamps with nonzero time components instead of truncating them. Column Overview value details display subtype-specific ranges. Descriptive Statistics and Temporal quality panels show clock-time/duration summaries and distributions. Calendar-based relationship/trend and period-comparison controls offer only Date/DateTime columns; Time and Duration are not silently treated as calendar timestamps.

Parsing and inference are in `utils/temporal.py`; clock/duration descriptive displays are in `utils/temporal_ui.py`. Temporal metadata includes all four subtypes; the `datetime` analysis group remains restricted to calendar-compatible columns for backward compatibility.

### EDA variable checkbox controls

Group Statistics uses type-grouped checkbox grids: at most one Numerical measure, any Categorical grouping variables, and at most one Temporal (Date/DateTime) column. Checking a different single-role variable clears its predecessor. No numerical selection means row counts; time analyses prompt for a selected calendar variable. Categorical selections provide Select all/Clear all. Comparison's existing shared calculation paths are unchanged.

Visual Analysis uses grouped variable checkboxes with Select all/Clear all. Chart roles (axes, grouping, color, size and measures) use single-selection checkboxes; line-series measures and pair-plot variables use multiple-selection checkboxes. Aggregation and other chart settings retain their existing controls. The compatible chart buttons remain selection-driven.

All requested chart families are available: **pie, bar, stacked-bar, scatter, bubble, heatmap and pair plot**. Pie charts use one categorical variable with row counts by default or sums of an optional numerical variable. Category totals must be nonnegative and sum to a positive value. Missing categories or selected measures are excluded; no absolute-value conversion is performed. Bubble sizes require nonnegative values, and pair plots support two to six numerical variables. Existing exact-combination chart restrictions still apply as documented above.

Temporal grouping is backward-compatible with older classification dictionaries that contain `datetime` but no `temporal` key. The app and Column Overview derive the aggregate from available `datetime`, `time`, and `duration` groups (or an empty list). Deploy the app and utility modules together when updating classification features.

The **Edit variable classification** panel is collapsed by default. A message above it explains how to expand the editor, change a variable type or subtype, and apply the classification.

The regression Model validation panel omits the introductory article attribution and general diagnostic disclaimer. Its summary warning names tests with review flags (p < 0.05) and describes possible departures from the tested assumptions, rather than claiming that assumptions are proved or disproved. Diagnostic calculations are unchanged.

Group Statistics and Visual Analysis hide variable-type headings and controls when there are no eligible variables in that group, matching Descriptive Statistics. Empty groups also clear their saved selections.

### Content-aware result sizing

The Cozy title uses muted terracotta (`#A65F46`). Shared renderers in `utils/presentation.py` use native content-sized interactive tables. Columns fit their content; short tables remain compact, while wide tables fill the available parent width and scroll horizontally without a fixed 1,200-pixel cap. Inspect tables use the Dataset Preview format, with Column Overview value details in an expander below the table. Small previews shrink vertically instead of reserving empty rows. Existing table scrolling, formatting, selection keys, and downloads remain available.

Plotly charts size by categories, label length, series/legend complexity, heatmap dimensions, and subplot domains rather than scatter-point count. Simple charts begin at 640 × 360 pixels, with larger plots capped at 1,200 × 900; widths shrink to the screen/container on mobile. Existing deliberate chart heights are preserved within that bound. Matplotlib figures use their native aspect and an appropriate bounded display width. The specialized HTML contingency report fits its content and scrolls horizontally; Inspect overview tables use the native interactive renderer. Calculation and statistical behavior are unchanged.

### Database table naming

All database tables for this app must use the `data_analytics_` prefix and lowercase snake_case names. Existing tables are `data_analytics_visits` and `data_analytics_feedback`; Enterprise enquiries use `data_analytics_enterprise_enquiries`.

### Cozy Enterprise

Community keeps its existing analytics workspace. An understated footer links to the in-app Enterprise page (`?page=enterprise`); **Back to Cozy Community** returns to the same session and retains dataset selections, analytical controls, and pending Transform changes. The Enterprise title uses slate teal (`#285D63`), with responsive deployment cards for Azure, AWS, Google Cloud, and on-premises as options for one product. Private deployment, customized analytical capabilities, organizational access requirements, and implementation/support are described without claiming unimplemented identity features.

The enquiry form validates required contact/organization fields, basic email syntax, deployment choice, and field lengths. Optional job title, user range, and analytical requirements are included. Failed validation and failed saves retain entered values. `utils/enterprise_enquiries.py` separates normalization/validation and submission handling from `utils/enterprise_ui.py`.

The form now persists through a server-only PostgreSQL adapter using `SUPABASE_ENQUIRY_DB_URL` and a restricted submission login. A confirmed database commit clears the form and shows success; failed saves retain inputs. Stable submission IDs and a database uniqueness constraint protect retries and reruns. Supabase INSERT webhooks invoke the protected `notify-enterprise-enquiry` Edge Function to send administrator email through Resend independently of the form. Notification failure never rolls back an enquiry.

Apply the migration, configure the restricted login, deploy the function and set up the authenticated webhook and email secrets using [Enterprise backend setup](docs/enterprise-enquiries.md). This code does not configure a live project or establish live email delivery. Python form/persistence tests and offline Edge Function tests cover validation, failures, success, duplicate protection and preserved analytics state.

### UI control inventory

See [UI controls and pending standardization decisions](docs/ui-controls.md) for the current dropdown, checkbox, radio, button and navigation patterns, including inconsistencies awaiting a product decision. This documents current behavior; it does not impose a new standard.

### UI standard

[Inspect, Transform and Analyze UI standard](docs/ui-standard.md) documents the agreed table format and sizing, classification controls, Transform menus and information badges, and Analyze presentation rules. It distinguishes approved standards from control choices still awaiting a decision.

### Analyze loading performance

Analyze renders only its open exploratory tab, with Predictive Analytics remaining below the five exploratory tabs in the original layout. AI interpretations are generated/refreshed explicitly by button, not automatically on page load. Prediction configuration is separated from training dependencies so sklearn/SHAP do not load merely to display prediction controls. Run still performs the existing model training; displaying saved results loads the necessary plotting libraries. Selections and cached results persist across panel switches.
