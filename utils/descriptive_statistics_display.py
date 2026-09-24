"""Descriptive-statistics table with keyboard-accessible interpretation details."""
from html import escape


STATISTIC_HELP = {
    'Mean': 'The average value. It is calculated by adding all values and dividing by the number of values. The mean gives a general idea of the center of the data, but it can be affected by unusually high or low values.',
    'Median': 'The middle value when the data is arranged from lowest to highest. Half of the values are below the median and half are above it. It is less affected by extreme values than the mean.',
    'Std Dev': 'Standard deviation measures how spread out the values are around the mean. A small value means the data is relatively consistent; a large value means the values vary more widely.',
    'Standard Error (Mean)': 'Measures the uncertainty in the estimated mean. A smaller standard error means the sample mean is likely to be a more precise estimate of the true population mean. It is different from standard deviation, which measures variation among individual observations.',
    'Coefficient of Variation': 'Measures the amount of variation relative to the mean, expressed as a percentage. It is useful for comparing variability between datasets that have different units or different average values. A higher percentage means greater relative variation. This comparison is meaningful for data with a true zero. The app reports it only for a positive mean; it can be unstable when the mean is close to zero.',
    '95% Confidence Interval (Mean)': 'A range of values representing the uncertainty around the estimated mean. In repeated sampling, about 95% of confidence intervals calculated this way would contain the true population mean. A narrower interval indicates a more precise estimate.',
    'Skewness': 'Describes whether the distribution is balanced or stretched more toward one side. A value near 0 indicates approximate symmetry. Positive skewness indicates a longer right tail, while negative skewness indicates a longer left tail.',
    'Kurtosis': 'Describes how strongly a distribution tends to have extreme values or heavy tails compared with the normal distribution. This app reports excess kurtosis (the pandas/Fisher definition): the normal-distribution reference is 0, not 3. Positive values indicate heavier tails and a greater tendency toward extreme observations; negative values indicate lighter tails. A value near 0 alone does not establish that the data follows a normal distribution.',
    'Minimum': 'The smallest observed value in the data.',
    'Quartile 1 (25%)': 'The value below which approximately 25% of the observations fall. It is also called the first quartile (Q1).',
    'Quartile 2 (50%)': 'The value below which approximately 50% of the observations fall. It is the same as the median (Q2).',
    'Quartile 3 (75%)': 'The value below which approximately 75% of the observations fall. It is also called the third quartile (Q3).',
    'Maximum': 'The largest observed value in the data.',
    'Outliers Found (IQR)': 'The number of observations identified as unusually low or high using the Interquartile Range (IQR) method. IQR = Q3 − Q1. Values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR are flagged as potential outliers. They are not necessarily errors and may represent valid unusual observations.',
    'Outlier %': 'The percentage of observations identified as potential outliers using the IQR method. A higher percentage indicates that unusual values make up a larger part of the dataset. This percentage is calculated over non-missing observations.',
}


def statistics_html(statistics):
    rows = []
    for label, value in statistics.items():
        safe_label = escape(str(label), quote=True)
        explanation = STATISTIC_HELP.get(label)
        if explanation:
            term = (
                '<details class="ds-stat-details">'
                f'<summary aria-label="How to interpret {safe_label}" title="Show interpretation">'
                f'<span>{safe_label}</span> <span class="ds-stat-info" aria-hidden="true">ⓘ</span>'
                f'</summary><p>{escape(explanation)}</p></details>'
            )
        else:
            term = safe_label
        rows.append(f'<tr><th scope="row">{term}</th><td>{escape(str(value))}</td></tr>')
    return '''<style>
    .ds-statistics {border:1px solid #8885; border-radius:8px; overflow:hidden;}
    .ds-statistics table {width:100%; border-collapse:collapse; font-size:14px; text-align:left;}
    .ds-statistics th,.ds-statistics td {padding:10px 12px; border-bottom:1px solid #8883; vertical-align:top; overflow-wrap:anywhere;}
    .ds-statistics thead th {font-weight:600;}
    .ds-statistics tbody th {font-weight:400; width:70%;}
    .ds-statistics td {font-variant-numeric:tabular-nums;}
    .ds-statistics tbody tr:last-child th,.ds-statistics tbody tr:last-child td {border-bottom:0;}
    .ds-stat-details summary {display:list-item; list-style:none; width:fit-content; cursor:pointer; border-radius:3px;}
    .ds-stat-details summary::-webkit-details-marker {display:none;}
    .ds-stat-details summary:focus-visible {outline:2px solid #4c72b0; outline-offset:3px;}
    .ds-stat-info {color:#4c72b0; font-size:16px; padding:3px;}
    .ds-stat-details[open] summary {font-weight:600;}
    .ds-stat-details p {margin:8px 0 0; padding:10px 12px; border-left:3px solid #4c72b0; background:#4c72b012; font-size:14px; line-height:1.6; font-weight:400;}
    @media (pointer:coarse) {.ds-stat-details summary {min-height:44px; align-content:center;}}
    </style><div class="ds-statistics"><table aria-label="Descriptive summary statistics">
    <thead><tr><th scope="col">Statistic</th><th scope="col">Value</th></tr></thead>
    <tbody>''' + ''.join(rows) + '</tbody></table></div>'
