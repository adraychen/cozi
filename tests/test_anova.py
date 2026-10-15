import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import stats
from streamlit.testing.v1 import AppTest
from utils.anova_analysis import prepare_anova, run_anova, games_howell, variation_decomposition, EFFECT_SIZE
from utils.anova_independence import assess_independence
from utils.anova_charts import pairwise_plot, group_mean_plot


def frame_for(*groups):
    return pd.DataFrame({'group': np.repeat(np.arange(len(groups)), list(map(len, groups))),
                         'value': np.concatenate(groups)})


def prepare(*groups):
    return prepare_anova(frame_for(*groups), 'group', 'value')


GROUPS = ([6.,7,9,8,10,11], [0.,2,3,1,4,5], [6.1,7.2,8.3,9.4,10.5,11.6])


class AnovaTests(unittest.TestCase):
    def test_welch_known_result_and_classical_rejected(self):
        prepared = prepare([1,2,3], [7,8,9])
        result = run_anova(prepared)
        self.assertAlmostEqual(result['F'], 54)
        self.assertAlmostEqual(result['p-value'], stats.f.sf(54, 1, 4))
        self.assertAlmostEqual(result['df denominator'], 4)
        with self.assertRaises(ValueError):
            run_anova(prepared, 'Classical')
        with self.assertRaises(ValueError):
            run_anova(prepared, independent=False)

    def test_welch_balanced_unequal_sizes_variances_and_df(self):
        rng = np.random.default_rng(34)
        for sizes, scales in [([30]*3,[1]*3), ([9,23,41],[1]*3), ([30]*3,[1,3,7]), ([9,23,41],[1,3,7])]:
            groups = [rng.normal(i, s, n) for i,(n,s) in enumerate(zip(sizes, scales))]
            with self.subTest(sizes=sizes, scales=scales):
                prepared = prepare(*groups)
                result = run_anova(prepared)
                reference = stats.f_oneway(*groups, equal_var=False)
                self.assertAlmostEqual(result['F'], reference.statistic)
                self.assertAlmostEqual(result['p-value'], reference.pvalue)
                # Independently calculate Welch's adjusted denominator df.
                n = np.array(sizes)
                variance = np.array([np.var(g, ddof=1) for g in groups])
                weights = n/variance
                term = np.sum((1-weights/weights.sum())**2/(n-1))
                self.assertAlmostEqual(result['df denominator'], (len(groups)**2-1)/(3*term))
                means = np.array([np.mean(g) for g in groups])
                weighted_mean = np.average(means, weights=weights)
                self.assertAlmostEqual(result[EFFECT_SIZE], np.sum(weights*(means-weighted_mean)**2)/sum(n))

    def test_two_sample_welch_t(self):
        a, b = [1.,2,2,3,4], [2.,8,12,13,19,24,30]
        result = run_anova(prepare(a,b))
        reference = stats.ttest_ind(a,b,equal_var=False)
        self.assertAlmostEqual(result['F'], reference.statistic**2)
        self.assertAlmostEqual(result['p-value'], reference.pvalue)
        self.assertTrue(games_howell(prepare(a,b))['table'].empty)

    def test_variation_known_and_independent_total(self):
        table = variation_decomposition(prepare([1,2,3], [7,8,9]))
        np.testing.assert_allclose(table['Sum of Squares (SS)'], [54,4,58])
        np.testing.assert_allclose(table.DF, [1,4,5])
        self.assertTrue(np.isnan(table['Mean Square (MS)'].iloc[2]))
        groups = ([1.,4,2,3], [2.,8,12,16,18], [3.,5,4,6,8,5])
        table = variation_decomposition(prepare(*groups))
        ss = table['Sum of Squares (SS)'].to_numpy()
        self.assertAlmostEqual(ss[0]+ss[1], ss[2])
        all_values = np.concatenate(groups)
        self.assertAlmostEqual(ss[2], np.sum((all_values-all_values.mean())**2))
        welch = run_anova(prepare(*groups))
        self.assertNotAlmostEqual(welch['F'], table['Mean Square (MS)'].iloc[0]/table['Mean Square (MS)'].iloc[1])

    def test_games_howell_matches_scipy_p_and_ci(self):
        rng = np.random.default_rng(17)
        for groups in [GROUPS, (rng.normal(0,1,9), rng.normal(2,4,17), rng.normal(6,2,31), rng.normal(0,1,11))]:
            for alpha in [.05,.01,.1]:
                with self.subTest(alpha=alpha, k=len(groups)):
                    result = games_howell(prepare(*groups), alpha)
                    reference = stats.tukey_hsd(*groups, equal_var=False)  # SciPy's Games–Howell path.
                    ci = reference.confidence_interval(1-alpha)
                    for _,row in result['table'].iterrows():
                        i,j = int(row['Group A']), int(row['Group B'])
                        self.assertAlmostEqual(row['Mean Difference'], reference.statistic[i,j])
                        self.assertAlmostEqual(row['Adjusted p-value'], reference.pvalue[i,j], places=10)
                        self.assertAlmostEqual(row[result['lower']], ci.low[i,j], places=8)
                        self.assertAlmostEqual(row[result['upper']], ci.high[i,j], places=8)
                        self.assertEqual(row.Significant, row['Adjusted p-value'] < alpha)
                        self.assertEqual(row.Significant, row[result['lower']] > 0 or row[result['upper']] < 0)

    def test_pairwise_signs_nonsignificance_and_plot_agreement(self):
        result = games_howell(prepare(*GROUPS))
        table = result['table']
        self.assertTrue(table['Mean Difference'].gt(0).any())
        self.assertTrue(table['Mean Difference'].lt(0).any())
        self.assertTrue(table.Significant.any())
        self.assertTrue((~table.Significant).any())
        fig = pairwise_plot(result)
        for trace in fig.data:
            rows = table.set_index('Comparison').loc[list(trace.y)]
            np.testing.assert_allclose(trace.x, rows['Mean Difference'])
            np.testing.assert_allclose(np.asarray(trace.x)-trace.error_x.arrayminus, rows[result['lower']])
            np.testing.assert_allclose(np.asarray(trace.x)+trace.error_x.array, rows[result['upper']])
            np.testing.assert_allclose(np.asarray(trace.customdata)[:,2], rows['Adjusted p-value'])
        self.assertEqual(fig.layout.shapes[0].x0, 0)
        self.assertIsNone(pairwise_plot(games_howell(prepare([1,2],[3,4]))))

    def test_no_omnibus_gate_and_alpha_intervals(self):
        groups = ([1.,2,3,4], [1.1,2.1,3.1,4.1], [.9,1.9,2.9,3.9])
        self.assertGreater(run_anova(prepare(*groups))['p-value'], .05)
        wide = games_howell(prepare(*groups), .01)
        narrow = games_howell(prepare(*groups), .1)
        self.assertEqual(len(wide['table']), 3)
        self.assertTrue((wide['table'][wide['upper']]-wide['table'][wide['lower']] >
                         narrow['table'][narrow['upper']]-narrow['table'][narrow['lower']]).all())
        for alpha in [0,1,np.nan]:
            with self.assertRaises(ValueError):
                games_howell(prepare(*groups), alpha)

    def test_missing_nonfinite_and_lost_group_report(self):
        frame = pd.DataFrame({'g':['a','a','b','b','c',None,'b'], 'y':[1,2,3,4,np.inf,9,-np.inf]})
        prepared = prepare_anova(frame,'g','y')
        self.assertEqual(prepared['excluded'],3)
        self.assertEqual(prepared['n'],4)
        self.assertTrue(any('excluded entirely' in n for n in prepared['warnings']))
        self.assertTrue(np.isfinite(run_anova(prepared)['F']))
        prepared = prepare_anova(pd.DataFrame({'g':['a','a','b','b'],'y':[1,np.nan,2,3]}),'g','y')
        with self.assertRaises(ValueError):
            run_anova(prepared)
        prepared = prepare_anova(pd.DataFrame({'g':[1.,1.,2.,2.,np.inf],'y':[1.,2.,3.,4.,5.]}),'g','y')
        self.assertEqual(prepared['excluded'], 1)
        self.assertEqual(prepared['missing_factor'], 1)

    def test_invalid_and_extreme_data_blocked(self):
        for groups in [([1,2,3],), ([1],[2,3]), ([1,1],[2,2]), ([1,1,1],[2,3,4]),
                       ([1e308,-1e308],[2e307,-2e307])]:
            with self.subTest(groups=groups):
                prepared = prepare(*groups)
                with self.assertRaises(ValueError):
                    run_anova(prepared)
                self.assertTrue(games_howell(prepared)['table'].empty)
        with self.assertRaises(ValueError):
            prepare_anova(pd.DataFrame({'g':['a','b'],'v':['1','2']}),'g','v')
        with self.assertRaises(ValueError):
            variation_decomposition(prepare([1e308,-1e308],[1e307,-1e307]))

    def test_distribution_no_shapiro_and_group_mean_ci(self):
        with patch('scipy.stats.shapiro', side_effect=AssertionError('No Shapiro')):
            prepared = prepare([1.,2,3,4], [2.,3,5,8,10])
        self.assertIn('Skewness', prepared['distribution'])
        for g, (_,row) in zip(prepared['groups'], prepared['summary'].iterrows()):
            half = stats.t.ppf(.975,len(g)-1)*np.std(g,ddof=1)/np.sqrt(len(g))
            self.assertAlmostEqual(row['95% CI Lower'], np.mean(g)-half)
        trace = group_mean_plot(prepared).data[0]
        np.testing.assert_allclose(np.asarray(trace.y)-trace.error_y.arrayminus, prepared['summary']['95% CI Lower'])
        constant = prepare([1.,1,1], [2.,3,5])
        self.assertTrue(np.isnan(constant['summary']['95% CI Lower'].iloc[0]))
        self.assertEqual(len(group_mean_plot(constant).data[0].x), 1)

    def test_unused_categories_empty_and_many_groups(self):
        frame = pd.DataFrame({'g':pd.Categorical(['a','a','b','b'],categories=['a','b','c']), 'v':[1.,2,3,5]})
        self.assertEqual(len(prepare_anova(frame,'g','v')['groups']),2)
        self.assertTrue(prepare_anova(frame.iloc[:0],'g','v')['errors'])
        too_many = games_howell(prepare(*([1.,2,3] for _ in range(31))))
        self.assertTrue(too_many['table'].empty)
        self.assertIn('limited',too_many['reason'])


class IndependenceTests(unittest.TestCase):
    def screen(self, frame):
        prepared = prepare_anova(frame,'group','value')
        return prepared, assess_independence(frame,prepared)

    def test_repeated_categories_do_not_prove_dependence(self):
        prepared, screen = self.screen(frame_for(*GROUPS))
        self.assertEqual(screen['state'],'Insufficient information')
        self.assertFalse(screen['requires_clarification'])
        self.assertTrue(np.isfinite(run_anova(prepared,independence=screen)['F']))
        self.assertIn('No diagnostic can prove', screen['qualification'])

    def test_unique_identifier_and_duplicate_measurements(self):
        frame = frame_for(*GROUPS)
        frame['subject_id'] = range(len(frame))
        _, screen = self.screen(frame)
        self.assertEqual(screen['state'],'No dependence indicators detected')
        self.assertNotIn('verified',screen['state'])
        self.assertFalse(screen['requires_clarification'])

    def test_repeated_subject_and_cluster_require_clarification(self):
        for name in ['subject_id','batch']:
            frame = frame_for(*GROUPS)
            frame[name] = np.tile(range(6),3)
            prepared, screen = self.screen(frame)
            self.assertEqual(screen['state'],'Potential dependence detected')
            with self.assertRaises(ValueError):
                run_anova(prepared,independence=screen)
            self.assertTrue(np.isfinite(run_anova(prepared,independence=screen,design='independent')['F']))

    def test_serial_only_on_meaningful_regular_time_and_shuffle_invariance(self):
        n = 50
        rng = np.random.default_rng(1)
        g = np.cumsum(rng.normal(size=n))
        frame = frame_for(g, g+10)
        frame['when'] = list(pd.date_range('2024-01-01',periods=n))*2
        prepared, screen = self.screen(frame)
        self.assertTrue(screen['requires_clarification'])
        self.assertTrue(screen['serial']['Ljung–Box p-value'].notna().all())
        shuffled = frame.sample(frac=1,random_state=2)
        _, other = self.screen(shuffled)
        np.testing.assert_allclose(sorted(screen['serial']['Durbin–Watson']),sorted(other['serial']['Durbin–Watson']))
        del frame['when']
        _, screen = self.screen(frame)
        self.assertTrue(screen['serial'].empty)
        self.assertFalse(screen['requires_clarification'])
        frame['date'] = pd.Timestamp('2024-01-01')
        _, screen = self.screen(frame)
        self.assertNotIn('Ljung–Box p-value', screen['serial'])

    def test_nonsignificant_serial_diagnostic_never_verifies_independence(self):
        rng = np.random.default_rng(33)
        frame = frame_for(rng.normal(size=60),rng.normal(size=60))
        frame['date'] = list(pd.date_range('2024-01-01',periods=60))*2
        _, screen = self.screen(frame)
        self.assertEqual(screen['state'],'Insufficient information')
        self.assertTrue(screen['serial'].Assessment.str.contains('does not establish independence').all())

    def test_incomplete_ids_and_irregular_time(self):
        frame = frame_for(*GROUPS)
        frame['subject_id'] = range(len(frame))
        frame.loc[0,'subject_id'] = np.nan
        _, screen = self.screen(frame)
        self.assertEqual(screen['state'],'Insufficient information')
        self.assertTrue(screen['evidence'].Indicator.str.contains('Incomplete').any())


UI_FIXTURE = '''
import pandas as pd
import streamlit as st
from utils.anova_ui import render_anova
dataset = st.selectbox('Dataset', ['one','two'], key='fixture_dataset')
st.session_state['file_id'] = dataset
df = pd.DataFrame({'g':['A']*6+['B']*6+['C']*6,
                   'v':[6.,7,9,8,10,11,0.,2,3,1,4,5,6.1,7.2,8.3,9.4,10.5,11.6]})
render_anova(df,'g','v')
'''


class AnovaUITests(unittest.TestCase):
    def test_automatic_welch_tables_charts_alpha_and_report(self):
        app = AppTest.from_string(UI_FIXTURE).run(timeout=45)
        self.assertFalse(app.exception)
        self.assertFalse(any(s.key == 'rel_anova_method' for s in app.selectbox))
        self.assertFalse(any(c.key == 'rel_anova_independent' for c in app.checkbox))
        self.assertTrue(any('Welch One-way ANOVA' in m.value for m in app.markdown))
        self.assertTrue(any('Sum of Squares (SS)' in d.value for d in app.dataframe))
        pair = next(d.value for d in app.dataframe if 'Adjusted p-value' in d.value and 'Mean Difference' in d.value)
        self.assertEqual(len(pair),3)
        app.number_input(key='rel_anova_alpha').set_value(.01).run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('99% CI Lower' in d.value for d in app.dataframe))
        app.button(key='report_add_anova').click().run(timeout=45)
        self.assertFalse(app.exception)
        snapshot = app.session_state['cozy_report']['blocks'][0]['snapshot']
        self.assertEqual(snapshot['analysis_type'],'One-way ANOVA')
        self.assertEqual(snapshot['results']['method'], 'Welch')
        self.assertTrue(np.isfinite(snapshot['results']['F']))
        self.assertTrue(any('Adjusted p-value' in table['columns'] for table in snapshot['tables']))
        self.assertGreaterEqual(len(snapshot['charts']),4)
        app.selectbox(key='fixture_dataset').select('two').run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('Welch One-way ANOVA' in m.value for m in app.markdown))

    def test_result_navigation_and_pairwise_download_removed(self):
        app = AppTest.from_string(UI_FIXTURE).run(timeout=45)
        self.assertFalse(app.exception)
        self.assertEqual([tab.label for tab in app.tabs], ['Overview', 'Group Comparisons', 'Diagnostics', 'ANOVA Findings'])
        overview, groups, diagnostics, findings = app.tabs
        self.assertFalse(any('Adjusted p-value' in table.value for table in overview.dataframe))
        result = next(table.value for table in overview.dataframe if 'F' in table.value)
        self.assertEqual(list(result.columns), ['Method', 'F', 'p-value', 'df numerator', 'df denominator', "Effect Size (Cohen's f²)", 'α'])
        variation = next(table.value for table in overview.dataframe if 'Sum of Squares (SS)' in table.value)
        self.assertEqual(list(variation.columns), ['Source', 'DF', 'Sum of Squares (SS)', 'Variation (%)'])
        self.assertAlmostEqual(variation['Variation (%)'].iloc[:2].sum(), 100)
        self.assertEqual(len(groups.get('plotly_chart')), 3)
        self.assertTrue(any('Adjusted p-value' in table.value for table in groups.dataframe))
        self.assertGreaterEqual(len(diagnostics.get('plotly_chart')), 3)
        self.assertFalse(app.get('download_button'))
        self.assertEqual(app.button(key='report_add_anova').label, 'Add to Report')
        self.assertFalse(any(widget.key == 'rel_anova_pair_filter' for widget in app.selectbox))
        self.assertFalse(any('Student-t' in caption.value or 'filtered comparisons' in caption.value or 'simultaneous Games–Howell intervals.' in caption.value or 'Mean Difference = Mean(Group A)' in caption.value for caption in groups.caption))
        content = app.session_state['report_latest']['content']
        heading = next(i for i, item in enumerate(content) if item == ('Subheading', 'Games–Howell Pairwise Comparisons'))
        self.assertEqual(content[heading + 1][0], 'Table')
        self.assertEqual(content[heading + 2], ('Subheading', 'Games–Howell Pairwise Comparison Plot'))


    def test_invalid_groups_keep_summary(self):
        app = AppTest.from_string("""
import pandas as pd
from utils.anova_ui import render_anova
render_anova(pd.DataFrame({'g':['a','b','b'],'v':[1.,2,4]}),'g','v')
""").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertTrue(app.dataframe)
        self.assertFalse(app.checkbox)

    def test_repeated_design_guard_reset_and_ai_fallback(self):
        fixture = UI_FIXTURE.replace("render_anova(df,'g','v')", "df['subject_id'] = list(range(6))*3\nrender_anova(df,'g','v')")
        app = AppTest.from_string(fixture).run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('Inference withheld' in w.value for w in app.warning))
        self.assertFalse(any('F' in d.value for d in app.dataframe))
        app.selectbox(key='rel_anova_design').select('Independent observations; these labels/order do not identify dependent units').run(timeout=45)
        self.assertTrue(any('F' in d.value for d in app.dataframe))
        with patch('utils.ai_interpreter.interpret_anova', side_effect=RuntimeError('AI unavailable')):
            app.button(key='rel_anova_ai_generate').click().run(timeout=45)
        self.assertFalse(app.exception)
        self.assertTrue(any('AI guidance is unavailable' in m.value for m in app.info))
        self.assertTrue(any('F' in d.value for d in app.dataframe))
        app.selectbox(key='fixture_dataset').select('two').run(timeout=45)
        self.assertEqual(app.selectbox(key='rel_anova_design').value,'Unresolved study design')

    def test_ai_context_has_no_raw_ids_and_refreshes(self):
        fixture = UI_FIXTURE.replace("render_anova(df,'g','v')", "df['subject_id'] = ['private-id-'+str(i) for i in range(len(df))]\nrender_anova(df,'g','v')")
        app = AppTest.from_string(fixture).run(timeout=45)
        with patch('utils.ai_interpreter.interpret_anova', return_value='Validated summary') as ai:
            app.button(key='rel_anova_ai_generate').click().run(timeout=45)
        self.assertFalse(app.exception)
        context = ai.call_args.args[0]
        text = json.dumps(context,allow_nan=False)
        self.assertNotIn('private-id-',text)
        self.assertNotIn('clean',context)
        self.assertTrue(any(m.value == 'Validated summary' for m in app.markdown))
        app.number_input(key='rel_anova_alpha').set_value(.1).run(timeout=45)
        self.assertFalse(any(m.value == 'Validated summary' for m in app.markdown))

    def test_many_comparison_plot_pages_use_full_family(self):
        app = AppTest.from_string("""
import pandas as pd
from utils.anova_ui import render_anova
df = pd.DataFrame([(str(i),float(i+j)) for i in range(8) for j in range(4)],columns=['g','v'])
render_anova(df,'g','v')
""").run(timeout=60)
        self.assertFalse(app.exception)
        table = next(d.value for d in app.dataframe if 'Adjusted p-value' in d.value and 'Mean Difference' in d.value)
        self.assertEqual(len(table),28)
        self.assertEqual(app.selectbox(key='rel_anova_pair_page').options,['1 of 2','2 of 2'])
        app.selectbox(key='rel_anova_pair_page').select(1).run(timeout=60)
        self.assertFalse(app.exception)
        chart = next(c for c in app.get('plotly_chart') if 'rel_anova_pair_plot' in c.proto.id)
        spec = json.loads(chart.proto.spec)
        self.assertEqual(sum(len(t['y']) for t in spec['data']),8)


if __name__ == '__main__':
    unittest.main()
