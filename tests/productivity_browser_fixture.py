"""Manual browser verification of the bundled Productivity mixed model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from utils.factorial_anova_ui import render_factorial_anova

st.set_page_config(layout='wide')
st.title('Productivity — study-design verification')
df = pd.read_csv(Path(__file__).resolve().parents[1] / 'data/samples/Productivity.csv')
st.session_state['file_id'] = 'productivity-study-design-verification'
st.session_state.setdefault('fac_effects_context', (st.session_state['file_id'], 'Score'))
st.session_state.setdefault('fac_effects', {'Machine': 'Fixed', 'Operator': 'Random'})
render_factorial_anova(df, ['Score'], ['Machine', 'Operator'])
