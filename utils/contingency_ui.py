"""Cross-tabulation and chi-square interface for Relationship."""
import pandas as pd
import streamlit as st
from utils.contingency_table import contingency_table_html
from utils.group_statistics import contingency


def render_contingency(df, dimensions, key_prefix="rel_ct"):
    st.caption("Counts rows for two categorical variables; the numerical selection is not used.")
    if len(dimensions) < 2:
        st.info("Select at least two categorical columns above to build a contingency table.")
        return
    row = st.selectbox("Row variable", dimensions, key=f"{key_prefix}_row")
    column = st.selectbox("Column variable", [c for c in dimensions if c != row], key=f"{key_prefix}_column")
    if len(dimensions) > 2:
        st.caption("Only the chosen row and column variables enter this test; other dimensions are not used as controls.")
    missing = st.checkbox("Include missing values as categories", key=f"{key_prefix}_missing")
    alpha = st.selectbox("Significance level", [0.05, 0.01, 0.10], key=f"{key_prefix}_alpha")
    try:
        result = contingency(df, row, column, missing)
    except (ValueError, TypeError) as exc:
        st.warning(str(exc))
        return
    st.caption(f"Included {result['n']:,} rows; excluded {result['excluded']:,} rows with missing categories. Expected counts = row total × column total ÷ grand total.")
    table = result["observed"]
    st.markdown(contingency_table_html(result, row, column), unsafe_allow_html=True)
    st.caption("Standardized residual = (observed − expected) / √expected (Pearson residual), matching the reference layout. Positive values mean more observations than expected under independence; these are not separate significance tests.")
    # Positional cell records preserve the original category names, even when they match output labels.
    records = [{"Row category": r, "Column category": c, "Observed": int(result['observed'].loc[r, c]),
                "Expected": float(result['expected'].loc[r, c]), "Observed - expected": float(result['difference'].loc[r, c]),
                "Pearson residual": float(result['residual'].loc[r, c])}
               for r in table.index for c in table.columns]
    st.download_button("Download observed and expected counts (CSV)", pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig"),
                       file_name="contingency_comparison.csv", mime="text/csv", key=f"{key_prefix}_download")
    st.markdown("##### Chi-square test of association")
    st.caption("Pearson chi-square test without continuity correction. Null hypothesis: the two variables are independent. Rows must represent independent observations; association does not imply causation.")
    if result["chi2"] is None:
        st.info("The test requires at least two observed categories in each variable.")
        return
    a, b, c = st.columns(3)
    a.metric("Chi-square", f"{result['chi2']:.4f}")
    b.metric("Degrees of freedom", result["dof"])
    c.metric("p-value", f"{result['p']:.4g}")
    if result["low_expected"]:
        st.warning(f"{result['low_expected']} of {table.size} cells have expected counts below 5. The asymptotic p-value may be unreliable; interpret cautiously.")
    if result["p"] < alpha:
        st.write(f"At α = {alpha}, the test provides evidence of an association, subject to the assumptions above.")
    else:
        st.write(f"At α = {alpha}, the test does not provide sufficient evidence of an association. This does not prove independence.")
    st.caption("Repeatedly testing different pairs increases the chance of false positives; these p-values are unadjusted.")
    st.markdown("[Chi-square method documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html)")


