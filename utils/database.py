import streamlit as st
import psycopg2

_VISITS_TABLE   = "data_analytics_visits"
_FEEDBACK_TABLE = "data_analytics_feedbacks"


def _connect():
    """Open a new PostgreSQL connection using the pooler URL."""
    return psycopg2.connect(st.secrets["SUPABASE_DB_URL"])


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
