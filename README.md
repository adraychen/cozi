# 📊 Data Analytics Assistant

A multi-section data analytics tool built with Streamlit. Upload a CSV or Excel file — or load a sample dataset — and the assistant will inspect your data, run a quality audit, visualise every column, explore relationships, and run machine learning models, all explained in plain English.

---

## Sections

| Section | What it does |
|---|---|
| **Dataset Overview** | Row/column counts, scrollable data preview, boolean columns shown as True/False |
| **Column Overview** | Per-column table: data type, column type, total values, missing values, missing % (colour-coded) |
| **Data Quality Summary** | Tabbed quality panels with charts and stats for Numerical, Categorical, and Date/Time columns + AI interpretation |
| **Data Exploratory Analysis** | Three tabs: Data Profile, Group Statistics, and Relationship |
| **Visual Analysis** | Column-driven selection of distribution, comparison, relationship, time-series and matrix charts |
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
| ecommerceSalesAnalytics.csv | E-commerce sales analytics | 5,000 |

---

## User Feedback

At the end of each of the three main sections — Data Quality Summary, Data Exploratory Analysis, and Predictive Analytics — a 👍 / 👎 row appears. Clicking either opens a popup dialog (Streamlit `@st.dialog`) titled **"Thank you for your feedback"** with an optional comment text area, a **Submit** button, and a **Skip** button. All responses are stored in Supabase with the section name, feedback type, optional comment, and timestamp.

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
| Column Type | Classified as Numerical, Categorical, or Date/Time |
| Total Values | Total row count |
| Missing Values | Count of missing entries |
| Missing % | Percentage missing, formatted to 2 decimal places |

**Colour coding:** Red = 20%+ missing. Yellow = 5–19% missing.

**Date/Time override:** A multiselect below the table lets users reclassify any column as Date/Time. Columns where fewer than 80% of values parse as valid dates are rejected with a warning message.

---

## Column Type Detection

| Type | Detection logic |
|---|---|
| **Numerical** | Pandas `int` or `float` dtype, excluding `bool` |
| **Categorical** | `object` dtype, `bool` dtype, or columns that cannot be reliably parsed as dates |
| **Date / Time** | `datetime64` dtype, or `object` columns where 80% or more of non-missing values parse as dates |

> **Boolean columns** are classified as Categorical — True/False values represent discrete categories, not quantities.

---

## Data Quality Summary

Three tabbed panels, each rendered as bordered cards (two per row):

### Numerical Columns

**Continuous** (15 or more unique values):
- Left: histogram stacked above a boxplot
- Right: Missing Values, Missing %, Unique Values, Unique %, Minimum, Maximum, Average (all to 2 dp)

**Discrete** (fewer than 15 unique values):
- Left: vertical bar chart sorted descending, capped at 10 bars
- Right: same stats as continuous

> The threshold of 15 unique values is defined as `DISCRETE_THRESHOLD = 15` in `utils/charts.py` used for chart selection and target type detection in Predictive Analytics. The separate theoretical distribution fitter uses integer-valued observations for automatic discrete detection, with a user override.

### Categorical Columns
- Left: vertical bar chart, top 10 by frequency, sorted descending
- Right: Missing Values, Missing %, Unique Values, Unique %, Case Inconsistency, Special Characters
- **(Cardinality)** appears directly beneath the Unique Values count; the count excludes missing values.

### Date/Time Columns
- Left: histogram stacked above a boxplot
- Right: Missing Values, Missing %, Unique Values, Unique %, Earliest Date, Latest Date, Date Range, Future Dates

> **Note on date parsing:** dates that cannot be parsed — such as text values, impossible dates (e.g. April 31), or unrecognised formats — are automatically converted to `NaT` and counted under **Missing Values**. The **Future Dates** field counts valid dates that fall after today. Invalid day numbers such as September 32 may be silently rolled over to the next valid date by the parser rather than treated as missing.

### AI Interpretation
After all quality panels load, the app generates a 4–5 sentence plain-English summary of the dataset's type, quality issues, and column composition using Groq (OpenAI GPT-OSS 120B (`openai/gpt-oss-120b`)). Cached in `st.session_state` and refreshable via a **🔄 Refresh** button.

---

## Data Exploratory Analysis

Tab order: **Data Profile → Group Statistics → Relationship**.

### Data Profile (single column)

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

**Date/Time columns:**
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

The collapsed **Fit details** expander contains the type selector and detected type, optional known binomial trials, SSE, AIC, KS statistic, candidate comparison table, bootstrap p-values, sampling information, and assumptions. Lowest AIC identifies the best candidate among tested families; it does not establish adequate fit. Automatic type detection uses integer-valued observations, independently of the chart's unique-value threshold; users can override rounded continuous measurements. Integers/ordinal values do not establish a count process.

Continuous candidates: Normal, Uniform, and strictly positive Lognormal/Exponential (location fixed at zero). Discrete candidates: Poisson and zero-based geometric, plus binomial only when the user supplies a known trial count. Unsupported discrete support and insufficient/constant data get explanatory messages.

Fit assessment uses 199 reproducible parametric bootstrap simulations, refitting each sample. A bootstrap p-value below 0.05 is Poor; otherwise Not rejected, never proof of a theoretical distribution. Discrete distance checks both sides of CDF jumps, without an ordinary continuous KS p-value. If all candidates fail, the conclusion states no suitable theoretical distribution was identified among tested families. Tests are exploratory, unadjusted across families, and assume independent observations.

Requires 20 finite observations and two distinct values. Above 3,000 observations a reproducible random sample is used and disclosed. Computations are cached. SSE uses common histogram probability-mass bins with tails; metrics and ranking refer to the same data sample.

### Group Statistics

Data Exploratory Analysis now has three tabs: **Data Profile → Group Statistics → Relationship**. Select a numerical measure (or **None** for categorical data / row counts), optional categorical dimensions, then an analysis type.

- **Group Summary:** sum, mean, median, min/max, sample standard deviation, non-missing numerical count, or row count.
- **Segment Comparison:** compare groups with a reference segment using absolute and percentage differences.
- **Period-over-Period Comparison:** previous calendar period, or same period last year for monthly/quarterly/yearly data.
- **Running & Cumulative Statistics:** cumulative sum, expanding mean, and observed-period count of the selected period statistic.
- **Rolling Averages:** moving mean over a complete window of calendar periods.
- **Contribution to Total:** share of sum or count, using the dataset total or parent group as denominator.
- **Cross Tabulation / Chi-square:** categorical contingency report with observed/expected counts, residuals, totals, and association testing.

Time analysis requires a detected Date/Time column (or Column Overview override). Weeks run Monday–Sunday. Missing calendar periods are retained as gaps, with optional zero filling. Invalid dates are excluded and reported. Zero denominators produce N/A. Current-period and boundary coverage cautions are displayed. Expanding means weight periods equally, not original rows. Missing categories are retained; negative contributions are flagged. Chart segment selection does not change table results or contribution denominators. Results support CSV download.

Calculations live in `utils/group_statistics.py`; Streamlit rendering lives in `utils/group_statistics_ui.py`. Run calculation checks with `python -m unittest discover -s tests -p "test_group_statistics.py"`.


#### Categorical group statistics and contingency tables

The numerical selector supports **None**, including datasets with no numerical columns. With None selected, group summaries, segment comparisons, contribution analysis, and time analyses use **Row count**, including rows with missing numerical values. Numerical measures retain the original statistics; Row count is also available alongside non-missing count.

Choose **Cross Tabulation / Chi-square** and select at least two categorical dimensions, then choose distinct row and column variables. Extra dimensions are not included as controls. This analysis always uses unweighted row counts, regardless of the numerical selection. Missing categories are excluded by default, with an option to include them explicitly.

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
| Categorical–numerical | Per-group count/mean/median/SD/min/max, boxplot, classical one-way ANOVA |
| Categorical–categorical | Stacked cross-tabulation report and Pearson chi-square using the same calculations as Group Statistics |
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
| 2 or more numerical | Heatmap — Multiple Correlations (with AI interpretation) |
| Exactly 2 categorical | Heatmap — Occurrence, Interaction Chart — Grouped Bar |

---

## Predictive Analytics

### Flow

1. **Select Target Column** — the column to predict
2. **Predictor Columns** — checkboxes grouped by column type; **✅ Select All** and **☐ Deselect All** buttons for quick selection; columns suspected of being identifiers are flagged with a warning
3. **Model** — the app detects the problem type, explains the model choice, and shows a hyperparameter configuration table with an estimated training time
4. **▶ Run** — trains models, displays results, and generates an AI interpretation

### Target Type Detection

| Condition | Type | Models run |
|---|---|---|
| Exactly 2 unique values (any dtype including 0/1) | Binary | Logistic Regression vs RF Classifier |
| Numerical, 3–14 unique values | Discrete (multi-class) | Multinomial LR vs RF Classifier |
| Numerical, 15 or more unique values | Continuous (regression) | Lasso vs RF Regressor |
| Categorical, 3 or more unique values | Multi-class | Multinomial LR vs RF Classifier |

> Numerical columns with exactly 2 unique values (e.g. 0/1) are treated as binary classification, not regression. The binary check runs before the numerical type check.

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

After results load, an **🤖 AI Interpretation** expander generates a 4–5 sentence plain-English explanation covering:
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
| `make_violin` | `(series, height=300)` | Violin plot with embedded boxplot and mean line. Used in Data Profile alongside the distribution chart. |
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
| `DISCRETE_THRESHOLD` | `15` | Columns with fewer unique values are treated as discrete (bar chart). 15 or more treated as continuous (histogram + boxplot). Also used by target type detection. |
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
| `interpret_dataset_overview` | After data quality summary loads | 4–5 sentences on dataset type, missing values, and column types |
| `interpret_single_column` | On Data Profile deep-dive | 3–4 sentences on quality, anomalies, and recommended action |
| `interpret_heatmap` | On Heatmap — Multiple Correlations | 3–5 sentences on strongest correlations and unexpected relationships |
| `interpret_predictive_results` | After model training completes | 4–5 sentences on winning model, key features, SHAP impact, overfitting, and recommendation |

All interpretations are cached in `st.session_state`. Dataset-level and predictive analytics interpretations regenerate only when the data or model selection changes. Column-level interpretations have a **🔄 Refresh** button.

---

## Roadmap

| Feature | Status |
|---|---|
| File upload, dataset overview, column overview | ✅ Complete |
| Data quality summary (numerical, categorical, date/time) with AI | ✅ Complete |
| Data Profile — single column deep-dive with AI | ✅ Complete |
| Relationship — multi-column charts with AI | ✅ Complete |
| Predictive Analytics — Regression (Lasso vs RF) with AI | ✅ Complete |
| Predictive Analytics — Binary Classification (LR vs RF) with AI | ✅ Complete |
| Predictive Analytics — Multi-class Classification (Multinomial LR vs RF) with AI | ✅ Complete |
| Visit counter + like/dislike feedback with Supabase | ✅ Complete |
| Sample datasets (heart disease, Boston housing, iris, e-commerce sales) | ✅ Complete |
| Group Statistics, including categorical contingency tables and chi-square | ✅ Complete |
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

A standalone section between Data Exploratory Analysis and Predictive Analytics. Select columns in **Numeric**, **Categorical**, and **Date/time** groups; only compatible graphs appear. Additional controls assign axes, grouping, measures, and aggregation.

| Graph | Selected columns needed |
|---|---|
| Histogram, dot plot, density, box plot, normal Q–Q plot | At least one numeric; optional categorical grouping for box plots |
| Bar, Pareto | At least one categorical; optional numerical measure (otherwise counts) |
| Stacked-bar | At least two categorical; optional numerical measure |
| Line, area | Date/time + numeric, or at least two numeric |
| Scatter, correlation matrix, pair plot | At least two numeric |
| Heatmap | At least two categorical for counts/mean pivot, or two numeric for record-value heatmap |
| Bubble | At least three numeric for X, Y, and nonnegative size; optional categorical color |

Histogram bin count and density bandwidth are adjustable. Dot plots stack repeated values. Q–Q uses a normal reference. Bars offer sum/mean or counts; Pareto uses nonnegative counts/sums and a cumulative percentage against the full total. Line/area charts sort X and aggregate repeated values; areas stack the selected measures. Correlations support Pearson and Spearman with pairwise complete observations. Pair plots accept 2–6 variables. Numerical heatmaps optionally standardize columns. Categorical heatmaps show counts or mean measures.

Display limits are disclosed: dot plots sample 2,000 observations, density 10,000, scatter/bubble 5,000 and pair plots 2,000 when needed, all with a fixed seed. Numerical heatmaps show the first 500 rows. Categorical heatmaps reject more than 10,000 cells. Missing/nonfinite values are omitted as appropriate; invalid sizes, constant density inputs, and insufficient Q–Q data receive explanatory messages.

Implemented in `utils/visual_analysis.py`. Existing Data Profile, Group Statistics, Relationship, and Predictive Analytics remain available.
