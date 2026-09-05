import pandas as pd
import streamlit as st
from utils.distribution_fit import fit_distributions


@st.cache_data(show_spinner=False, max_entries=32)
def cached_fit(series, kind, trials):
    return fit_distributions(series, kind, trials)


def render_distribution_fit(series, column):
    st.markdown('**Theoretical distribution fit**')
    summary = st.container()
    details = st.expander('Fit details', expanded=False)
    with details:
        kind = st.selectbox('Distribution type', ['Auto', 'Discrete', 'Continuous'], key=f'dist_kind_{column}',
                            help='Auto treats integer-valued observations as discrete. Override for rounded continuous measurements. This is a data representation rule, not a claim about its generating process.')
        trials = None
        if kind != 'Continuous' and st.checkbox('Include binomial with known number of trials', key=f'dist_binom_{column}'):
            trials = int(st.number_input('Known trials per observation', min_value=1, value=10, key=f'dist_trials_{column}'))
            st.caption('Use only for successes out of a fixed known number of trials; the observed maximum is not an estimate of this number.')
    with st.spinner('Comparing distributions and checking fit…'):
        try:
            result = cached_fit(series, kind, trials)
        except (ValueError, TypeError) as exc:
            st.info(str(exc))
            return
    best = result['best']
    with summary:
        st.write("Best theoretical fit: " + (best['Distribution'] if best else "Not identified"))
        st.write("Goodness of fit: " + result['quality'])
        st.write('Conclusion: ' + result['conclusion'])
    with details:
        st.write(f"Distribution type: {result['type']}")
        if best:
            st.text(f"SSE:          {best['SSE']:.3f}\nAIC:          {best['AIC']:.0f}\nKS statistic: {best['KS statistic']:.3f}")
        st.caption(f"Uses {result['used']:,} of {result['n']:,} finite observations. Above 3,000, a reproducible random sample is used; metrics refer to that sample. Best candidate is ranked by lowest AIC within the tested family set.")
        st.caption('SSE compares observed and fitted probability masses in common histogram bins including tails; it depends on binning. KS measures the maximum CDF distance, including both sides of discrete jumps. No ordinary continuous KS p-value is used.')
        st.caption('Fit quality uses 199 parametric bootstrap simulations with parameters refitted each time: p < 0.05 is Poor; otherwise Not rejected. Tests assume independent observations. These are exploratory, unadjusted checks across candidates, not proof of a distribution. Small samples have limited power and borderline p-values have Monte Carlo uncertainty.')
        st.caption('Integer or ordinal data, such as Bedroom, do not imply a count process. Poisson and geometric are only candidates. Binomial requires a known trial count. Continuous candidates: Normal, Uniform, and (for strictly positive data) Lognormal and Exponential; positive families have location fixed at zero.')
        if result['candidates']:
            st.dataframe(pd.DataFrame(result['candidates']), hide_index=True, use_container_width=True)
        st.markdown('[Why refit parameters during goodness-of-fit testing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html)')
