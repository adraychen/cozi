import streamlit as st
import psycopg2
import os
from streamlit.errors import StreamlitSecretNotFoundError

_VISITS_TABLE   = "data_analytics_visits"
_FEEDBACK_TABLE = "data_analytics_feedback"


def configured_value(name):
    """Server-only configuration, using the existing Streamlit secrets convention."""
    try:
        value = st.secrets.get(name)
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        value = None
    return value or os.environ.get(name)


def _connect(secret_name="SUPABASE_DB_URL", **kwargs):
    """Open a new PostgreSQL connection using the pooler URL."""
    value = configured_value(secret_name)
    if not value:
        raise RuntimeError('Database configuration is unavailable.')
    return psycopg2.connect(value, **kwargs)


def record_visit() -> None:
    """Insert one visit row."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {_VISITS_TABLE} DEFAULT VALUES;")
        conn.commit()


@st.cache_data(ttl=60)
def get_stats() -> dict:
    """
    Return visit count and like/dislike counts.
    Result cached for 60 seconds to reduce database reads.
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {_VISITS_TABLE};")
            visits = cur.fetchone()[0]

            cur.execute(
                f"SELECT COUNT(*) FROM {_FEEDBACK_TABLE} WHERE feedback_type = 'like';"
            )
            likes = cur.fetchone()[0]

            cur.execute(
                f"SELECT COUNT(*) FROM {_FEEDBACK_TABLE} WHERE feedback_type = 'dislike';"
            )
            dislikes = cur.fetchone()[0]

    return {"visits": visits, "likes": likes, "dislikes": dislikes}


def submit_feedback(section: str, feedback_type: str, comment: str = "") -> None:
    """Insert one feedback row and clear the stats cache."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {_FEEDBACK_TABLE} (section, feedback_type, comment) "
                f"VALUES (%s, %s, %s);",
                (section, feedback_type, comment or None),
            )
        conn.commit()
    get_stats.clear()
