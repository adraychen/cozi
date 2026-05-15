import streamlit as st
from supabase import create_client, Client

_VISITS_TABLE   = "data_analytics_visits"
_FEEDBACK_TABLE = "data_analytics_feedbacks"


@st.cache_resource
def _get_client() -> Client:
    """Cached Supabase client (one instance per app process)."""
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


def record_visit() -> None:
    """Insert one visit row."""
    _get_client().table(_VISITS_TABLE).insert({}).execute()


@st.cache_data(ttl=60)
def get_stats() -> dict:
    """
    Return visit count and like/dislike counts.
    Result cached for 60 seconds to reduce Supabase reads.
    """
    client = _get_client()

    visits = (
        client.table(_VISITS_TABLE)
        .select("id", count="exact")
        .execute()
        .count or 0
    )
    likes = (
        client.table(_FEEDBACK_TABLE)
        .select("id", count="exact")
        .eq("feedback_type", "like")
        .execute()
        .count or 0
    )
    dislikes = (
        client.table(_FEEDBACK_TABLE)
        .select("id", count="exact")
        .eq("feedback_type", "dislike")
        .execute()
        .count or 0
    )

    return {"visits": visits, "likes": likes, "dislikes": dislikes}


def submit_feedback(section: str, feedback_type: str, comment: str = "") -> None:
    """Insert one feedback row and clear the stats cache."""
    _get_client().table(_FEEDBACK_TABLE).insert({
        "section":       section,
        "feedback_type": feedback_type,
        "comment":       comment or None,
    }).execute()
    get_stats.clear()   # force fresh counts on next load
