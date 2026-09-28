import streamlit as st
from utils.database import record_visit, get_stats, submit_feedback
from utils.branding import render_brand
from utils.enterprise_ui import (enterprise_requested, preserve_page_state,
                                 render_enterprise_page, render_footer)

# ── Sample datasets ───────────────────────────────────────────────────────────
_SAMPLE_DATASETS = [
    {
        "name": "heart_disease.csv",
        "id":   "1dcRKYhrdo0AlA0AAOIZAPnxWc3XjuTZC",
        "desc": "Binary classification — predict heart disease (303 rows)",
    },
    {
        "name": "BostonHousing.csv",
        "id":   "1rX_UtOz92mVjbJFrB3pv12kjwjROOluv",
        "desc": "Regression — predict housing prices (506 rows)",
    },
    {
        "name": "iris.csv",
        "id":   "1vviVuBIZvizhTTLyldzkD5wQiaFUbDXP",
        "desc": "Multi-class classification — predict iris species (150 rows)",
    },
    {
        "name": "CustomerSatisfaction.csv",
        "id":   "1tv1KDUNDixgPrhI2w6ixti1EKZtJHg_N",
        "desc": "Statistical Analysis (120 rows)",
        "separator": "-",
    },
    {
        "name": "UCLA_admission.csv",
        "id":   "1AKC6ShNS0Z0YaIECjl25s4fILPpUb7jL",
        "desc": "binary logistic regression analysis",
        "separator": "-",
    },
    {
        "name": "PlantGrowth.csv",
        "id":   "19w_ELwkTJ_5fKWUi0Rh7YuLz8rNoFeB7",
        "desc": "One-way ANOVA",
        "separator": "-",
    },
    {
        "name": "Galton_Family_Heights.csv",
        "id":   "18k9ocHuBFGFb9YTY64tk6Ug5iRlUdArh",
        "desc": "linear regression analysis",
        "separator": "-",
    },
    {
        "name": "AirPassengers.csv",
        "id":   "1mHxCpRF7iWmSCKuIqyea4lo8XgxsNkYc",
        "desc": "time series analysis",
        "separator": "-",
    },
]


# ── Feedback dialog (must be at module level for @st.dialog) ──────────────────
@st.dialog("Thank you for your feedback")
def _feedback_dialog(section: str, feedback_type: str) -> None:
    comment = st.text_area(
        "Add a comment (optional)",
        placeholder="Tell us what you think…",
        key="dlg_comment",
    )
    c1, c2 = st.columns(2)
    if c1.button("Submit", type="primary", key="dlg_submit"):
        try:
            submit_feedback(section, feedback_type, comment)
        except Exception:
            pass
        st.rerun()
    if c2.button("Skip", key="dlg_skip"):
        try:
            submit_feedback(section, feedback_type, "")
        except Exception:
            pass
        st.rerun()


def _show_feedback(section: str, key: str) -> None:
    """Render 👍 / 👎 buttons that open the feedback dialog."""
    st.divider()
    st.markdown("**Was this section helpful?**")
    fb1, fb2, _ = st.columns([1, 1, 8])
    if fb1.button("👍", key=f"like_{key}", help="Like this section"):
        _feedback_dialog(section, "like")
    if fb2.button("👎", key=f"dislike_{key}", help="Dislike this section"):
        _feedback_dialog(section, "dislike")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cozy — Community Edition", page_icon="📊", layout="wide")
preserve_page_state()
if enterprise_requested():
    st.set_page_config(page_title="Cozy — Enterprise Edition")
    render_enterprise_page()
    st.stop()

# ── App stats banner (above title) ───────────────────────────────────────────
_b1, _b2, _b3 = st.columns(3)
try:
    _app_stats = get_stats()
    _b1.metric("👁 Total Visits", f"{_app_stats['visits']:,}")
    _b2.metric("👍 Likes",        f"{_app_stats['likes']:,}")
    _b3.metric("👎 Dislikes",     f"{_app_stats['dislikes']:,}")
except Exception:
    _b1.metric("👁 Total Visits", "—")
    _b2.metric("👍 Likes",        "—")
    _b3.metric("👎 Dislikes",     "—")

render_brand()
st.caption(
    "Upload or input your data and analyze it through a broad range of guided analytical workflows and visualizations. "
    "Designed to make data analysis more accessible, structured, and efficient."
)

@st.dialog("Data and Privacy")
def _show_data_privacy():
    st.write(
        "Uploaded files and input data are temporarily processed in server memory to perform the requested analysis. "
        "When AI interpretation is used, dataset structure, column names, statistics, and analysis summaries may be "
        "transmitted to Groq. The original uploaded file is not intentionally sent to the AI model."
    )

st.html("""
<style>
    .st-key-data_privacy_info button {
        min-height: 1.5rem;
        height: 1.5rem;
        min-width: 1.5rem;
        padding: 0;
    }
</style>
""")
with st.container(horizontal=True, vertical_alignment="center", gap="xxsmall"):
    st.markdown("Data and Privacy", width="content")
    if st.button("ⓘ", key="data_privacy_info", type="tertiary", help="Read Data and Privacy information"):
        _show_data_privacy()

# ── Visit tracking ────────────────────────────────────────────────────────────
if not st.session_state.get("visit_recorded"):
    try:
        record_visit()
        st.session_state["visit_recorded"] = True
    except Exception:
        pass
st.divider()

from utils.dataset_workspace import render_workspace

render_workspace(_SAMPLE_DATASETS, _show_feedback)
render_footer()
