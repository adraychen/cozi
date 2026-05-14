import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.3,
    )


def interpret_dataset_overview(
    shape: tuple,
    col_types: Dict[str, List[str]],
    num_summary: str,
    cat_summary: str,
    dt_summary: str,
) -> str:
    """
    Send a structured summary of the dataset to Groq and return
    a plain-English paragraph for non-technical users.
    """
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_template(
        """You are explaining data analysis results to someone with no technical background.

Dataset details:
- Number of rows: {rows}
- Number of columns: {cols}
- Numerical columns ({num_count}): {numerical}
- Text/Categorical columns ({cat_count}): {categorical}
- Date/Time columns ({dt_count}): {datetime_cols}

Numerical column quality:
{num_summary}

Text column quality:
{cat_summary}

Date/Time column quality:
{dt_summary}

Write 3 to 5 plain sentences that:
1. Describe what kind of dataset this appears to be
2. Highlight which columns may need attention (focus on missing values, inconsistencies)
3. Explain what the column types suggest about the data

Rules:
- Use simple, plain English — no technical jargon
- Do not use the words "null", "dtype", "NaN", or "dataframe"
- Say "missing value" instead of "null"
- Be specific about column names where relevant
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
        "cat_summary": cat_summary or "No text columns.",
        "dt_summary": dt_summary or "No date/time columns.",
    })

    return response.content
