import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.3,
    )


# ── Dataset overview ──────────────────────────────────────────────────────────

def interpret_dataset_overview(
    shape: tuple,
    col_types: Dict[str, List[str]],
    num_summary: str,
    cat_summary: str,
    dt_summary: str,
) -> str:
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(
        """You are explaining a data file to someone with no technical background.

Dataset details:
- Number of rows: {rows}
- Number of columns: {cols}
- Numerical columns ({num_count}): {numerical}
- Categorical columns ({cat_count}): {categorical}
- Date/Time columns ({dt_count}): {datetime_cols}

Numerical column quality:
{num_summary}

Categorical column quality:
{cat_summary}

Date/Time column quality:
{dt_summary}

Write 4 to 5 plain sentences. Follow these rules strictly:
- Be direct. Do not use phrases like "appears to be", "seems to be", "may need", or "could be"
- One idea per sentence. Keep sentences short
- Name specific columns when discussing problems
- Use friendly, plain language
- Do not use the words "null", "dtype", "NaN", or "dataframe"
- Say "missing values" instead of "null" or "NaN"
- Do not use the word "dataset" more than once

Cover these points in order:
1. What kind of data this is (one sentence)
2. Which specific columns have missing values and how serious the problem is
3. What the column types tell us about what the data contains
"""
    )
    chain = prompt | llm
    response = chain.invoke({
        "rows": shape[0],
        "cols": shape[1],
        "num_count": len(col_types["numerical"]),
        "numerical": ", ".join(col_types["numerical"]) or "None",
        "cat_count": len(col_types["categorical"]),
        "categorical": ", ".join(col_types["categorical"]) or "None",
        "dt_count": len(col_types["datetime"]),
        "datetime_cols": ", ".join(col_types["datetime"]) or "None",
        "num_summary": num_summary or "No numerical columns.",
        "cat_summary": cat_summary or "No categorical columns.",
        "dt_summary": dt_summary or "No date/time columns.",
    })
    return response.content


# ── Single column deep dive ───────────────────────────────────────────────────

def interpret_single_column(
    col_name: str,
    col_type: str,
    stats: dict,
    extra: dict = None,
) -> str:
    """
    Plain-English interpretation of a single column.
    extra: optional dict with additional context (e.g. outlier info, rare categories).
    """
    llm = _get_llm()

    stats_text = "\n".join(f"- {k}: {v}" for k, v in stats.items())
    extra_text = (
        "\nAdditional information:\n" + "\n".join(f"- {k}: {v}" for k, v in extra.items())
        if extra else ""
    )

    prompt = ChatPromptTemplate.from_template(
        """You are explaining a data column to someone with no technical background.

Column name: {col_name}
Column type: {col_type}

Statistics:
{stats_text}
{extra_text}

Write 3 to 4 plain sentences:
1. Describe what this column contains and its overall quality
2. Highlight anything unusual or worth attention — missing values, outliers, skewness, dominant values, rare categories, date gaps, etc.
3. Give one practical recommendation if any action is needed. If nothing needs attention, say so clearly.

Rules:
- Be direct. No phrases like "appears to be" or "seems to"
- Short sentences. One idea per sentence
- Refer to the column by name
- Do not repeat raw numbers — explain what they mean
- Plain language only. No jargon
"""
    )
    chain = prompt | llm
    response = chain.invoke({
        "col_name": col_name,
        "col_type": col_type,
        "stats_text": stats_text,
        "extra_text": extra_text,
    })
    return response.content


# ── Correlation heatmap ───────────────────────────────────────────────────────

def interpret_heatmap(corr_pairs: str, col_names: List[str]) -> str:
    """
    Plain-English interpretation of a correlation heatmap.
    corr_pairs: upper-triangle pairs formatted as 'col_a × col_b: 0.85'
    """
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_template(
        """You are explaining relationships between data columns to someone with no technical background.

Columns analysed: {columns}

Relationship strength between column pairs (scale: -1 to +1):
- Close to +1: both columns increase together
- Close to -1: one increases as the other decreases
- Close to 0: little or no relationship

Values:
{corr_pairs}

Write 3 to 5 plain sentences:
1. Name the strongest positive relationship and explain what it means practically
2. Name the strongest negative relationship (if any) and explain what it means
3. Flag any surprising or counterintuitive relationships worth investigating
4. Note if any columns appear unrelated to the rest

Rules:
- Be direct. No hedging phrases
- Short sentences. One idea per sentence
- Name specific column pairs
- Explain relationships in plain terms, e.g. "when X goes up, Y tends to go up too"
- Do not use the word "correlation" more than twice
- Plain language only
"""
    )
    chain = prompt | llm
    response = chain.invoke({
        "columns": ", ".join(col_names),
        "corr_pairs": corr_pairs,
    })
    return response.content


# ── Predictive analytics interpretation ──────────────────────────────────────

def interpret_predictive_results(
    model_type: str,
    winner: str,
    target_col: str,
    n_rows: int,
    metrics_text: str,
    features_text: str,
    shap_text: str,
) -> str:
    """
    Plain-English interpretation of model comparison results.
    Covers: winning model, key features, SHAP impact, overfitting, recommendation.
    """
    llm = _get_llm()

    problem_type = {
        "regression_comparison":  "regression (predicting a continuous numerical value)",
        "binary_comparison":      "binary classification (predicting one of two outcomes)",
        "multiclass_comparison":  "multi-class classification (predicting one of several categories)",
    }.get(model_type, "predictive modelling")

    prompt = ChatPromptTemplate.from_template(
        """You are explaining machine learning results to someone with no technical background.

Problem type: {problem_type}
Column being predicted: {target_col}
Training rows: {n_rows}
Winning model: {winner}

Performance metrics (includes train vs test gap — a large gap indicates overfitting):
{metrics_text}

Top predictors by model importance score:
{features_text}

Top predictors by SHAP impact (mean absolute effect on predictions):
{shap_text}

Write 4 to 5 plain sentences:
1. State which model performed better and what this suggests — does the data follow simple linear patterns or complex non-linear ones?
2. Name the 2 to 3 most important features by importance score and explain in plain terms what drives the predictions
3. Using the SHAP values, explain how the top features actually push predictions up or down
4. Comment on whether the model generalises well or shows signs of overfitting, based on the train vs test gap
5. Give one practical recommendation — what action should someone take based on these results?

Rules:
- Be direct. No phrases like "appears to be", "seems to", or "it is worth noting"
- Plain language only — no jargon like "hyperparameter", "regularisation", "ensemble", or "SHAP"
- Refer to columns by their actual name, not as "the top feature" or "feature 1"
- Short sentences. One idea per sentence
- Do not repeat raw numbers — explain what they mean in context
"""
    )

    chain = prompt | llm
    response = chain.invoke({
        "problem_type":  problem_type,
        "target_col":    target_col,
        "n_rows":        n_rows,
        "winner":        winner,
        "metrics_text":  metrics_text,
        "features_text": features_text,
        "shap_text":     shap_text,
    })
    return response.content
