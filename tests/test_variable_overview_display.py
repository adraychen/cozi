import unittest
import pandas as pd
from utils.variable_overview_display import value_details, overview_html
from utils.variable_metadata import infer_metadata

class ValueDetailsTests(unittest.TestCase):
    def details(self,values,subtype,order=None):
        return value_details(pd.Series(values),dict(variable_subtype=subtype,order=order or []))[1]

    def test_ranges_and_discrete_boundary(self):
        self.assertEqual(self.details([1.2,8.9,None,3.5],'Continuous'),[('Minimum',1.2),('Maximum',8.9)])
        self.assertEqual(len(self.details(list(range(10))+[None,0],'Discrete')),10)
        self.assertEqual(self.details(list(range(11)),'Discrete'),[('Minimum',0),('Maximum',10)])
        self.assertEqual(self.details(['10','2','3'],'Continuous'),[('Minimum',2),('Maximum',10)])

    def test_categories_missing_and_order(self):
        self.assertEqual(self.details(['Yes',None,'No','Yes'],'Binary'),[('Value','Yes'),('Value','No')])
        self.assertEqual(self.details(list('CABDEFGAC'),'Nominal'),[('Value',v) for v in 'CABDE'])
        self.assertEqual(self.details(['High','Low','Medium'],'Ordinal',['Low','Medium','High']),[('Value',v) for v in ['Low','Medium','High']])
        self.assertEqual(self.details([None,None],'Continuous'),[])

    def test_badges_details_and_escaping(self):
        d=pd.DataFrame({'<script>':['<img src=x onerror=alert(1)>','B']})
        m=infer_metadata(d)
        html=overview_html(d,m)
        self.assertEqual(html.count('title="Automatic classification"'),2)
        self.assertIn('<details class="vm-details">',html)
        self.assertIn('aria-label="Value details for &lt;script&gt;"',html)
        self.assertNotIn('<script>',html)
        self.assertNotIn('<img',html)
        self.assertNotIn('Classification Source',html)
        self.assertNotIn('Category Order',html)
        m['<script>']['source']='User override'
        self.assertEqual(overview_html(d,m).count('title="Manual Override"'),2)

if __name__=='__main__': unittest.main()
