"""Shared title, edition badge, and subtitle for both Cozy editions."""
import streamlit as st


def render_brand(*, enterprise=False, show_subtitle=True):
    color = '#D4AF37' if enterprise else '#A65F46'
    badge_style = ('background: #000000; color: #FFFFFF; border: 1px solid #000000;'
                   if enterprise else
                   'background: #15803d; color: #FFFFFF; border: 1px solid #15803d;')
    edition = 'Enterprise Edition' if enterprise else 'Community Edition'
    subtitle = 'Analytics designed around your organization.' if enterprise else 'Automated Data Analytics'
    subtitle_html = f'<p class="cozy-subtitle">{subtitle}</p>' if show_subtitle else ''
    st.html(f"""
<style>
    .cozy-brand {{ margin: 0.5rem 0 1rem; }}
    .cozy-brand-heading {{ display: flex; align-items: center; flex-wrap: wrap; gap: 0.75rem; }}
    .cozy-brand-heading h1 {{ margin: 0; padding: 0; font-size: 2.75rem; font-weight: 750; line-height: 1.2; }}
    .cozy-edition {{ display: inline-block; padding: 0.3rem 0.75rem; border-radius: 999px;
                    font-size: 0.8rem; font-weight: 650; letter-spacing: 0.025em; }}
    .cozy-subtitle {{ margin: 0.4rem 0 0; font-size: 1.1rem; font-style: italic; }}
</style>
<header class="cozy-brand">
    <div class="cozy-brand-heading"><h1 style="color:{color}">Cozy</h1><span class="cozy-edition" style="{badge_style}">{edition}</span></div>
    {subtitle_html}
</header>
""")
