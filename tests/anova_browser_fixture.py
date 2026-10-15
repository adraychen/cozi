import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import streamlit as st
from utils.anova_ui import render_anova
from utils.factorial_anova_ui import render_factorial_anova
st.set_page_config(layout='wide')
rng=np.random.default_rng(18)
a=rng.normal(0,3,6); b=rng.normal(0,2,5)
df=pd.DataFrame([(str(i),str(j),a[i]+b[j]+rng.normal()) for i in range(6) for j in range(5) for k in range(4)],columns=['Factory','Machine','Response'])
st.session_state['file_id']='synthetic-anova-verification'
st.title('ANOVA verification — synthetic data')
mode=st.selectbox('Analysis',['One-way ANOVA','Factorial ANOVA'])
if mode=='One-way ANOVA': render_anova(df,'Machine','Response')
else: render_factorial_anova(df,['Response'],['Factory','Machine'])
