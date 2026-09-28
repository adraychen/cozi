import unittest
from dataclasses import asdict
from unittest.mock import Mock

import pandas as pd
from streamlit.testing.v1 import AppTest

from utils.enterprise_enquiries import (DEPLOYMENTS, USER_COUNTS, UnconfiguredStore,
                                        submit_enquiry, validate_enquiry)
from utils.dataset_workspace import add_dataset


VALID = dict(name='Alex Example', email='alex@example.org', organization='Example Organization',
             deployment='Microsoft Azure', job_title='', number_of_users='', requirements='')


class EnquiryTests(unittest.TestCase):
    def test_required_fields_and_basic_email_validation(self):
        enquiry, errors = validate_enquiry({})
        self.assertIsNone(enquiry)
        self.assertEqual(len(errors),4)
        for field in ['name','email','organization']:
            self.assertIsNone(validate_enquiry(dict(VALID, **{field:'  '}))[0])
        for email in ['alex', '@example.org', 'alex@', 'alex@example', 'alex @example.org',
                      'alex..name@example.org', '.alex@example.org', 'alex.@example.org',
                      'alex@example..org', 'alex@-example.org', 'alex@example.org\nBCC:bad@example.org']:
            with self.subTest(email=email):
                self.assertIsNone(validate_enquiry(dict(VALID,email=email))[0])
        for email in ['alex@example.org', 'alex+cozy@division.example.org']:
            self.assertFalse(validate_enquiry(dict(VALID,email=email))[1])

    def test_options_limits_and_normalization(self):
        for deployment in DEPLOYMENTS:
            enquiry, errors=validate_enquiry(dict(VALID, deployment=deployment, name=' Alex Example '))
            self.assertFalse(errors)
            self.assertEqual(enquiry.name,'Alex Example')
        for count in USER_COUNTS:
            self.assertFalse(validate_enquiry(dict(VALID,number_of_users=count))[1])
        for update in [dict(deployment='invalid'),dict(number_of_users='invalid'),dict(requirements='x'*5001)]:
            self.assertIsNone(validate_enquiry(dict(VALID,**update))[0])

    def test_pending_store_does_not_report_success(self):
        result=submit_enquiry(VALID,{},UnconfiguredStore())
        self.assertEqual(result.status,'unavailable')
        self.assertIn("couldn't submit",result.messages[0])

    def test_success_deduplication_and_changed_enquiry(self):
        store=Mock(configured=True)
        store.save.return_value='confirmed-receipt'
        state={}
        self.assertEqual(submit_enquiry(VALID,state,store).status,'submitted')
        self.assertEqual(asdict(store.save.call_args.args[0]),VALID)
        self.assertEqual(submit_enquiry(VALID,state,store).status,'duplicate')
        store.save.assert_called_once()
        self.assertEqual(submit_enquiry(dict(VALID,requirements='Custom methods'),state,store).status,'submitted')
        self.assertEqual(store.save.call_count,2)
        self.assertNotEqual(store.save.call_args_list[0].args[1],store.save.call_args_list[1].args[1])

    def test_failed_or_uncertain_save_retains_retry_id_and_hides_details(self):
        store=Mock(configured=True)
        store.save.side_effect=[RuntimeError('SECRET connection password'), 'confirmed-receipt']
        state={}
        failed=submit_enquiry(VALID,state,store)
        self.assertEqual(failed.status,'failed')
        self.assertNotIn('SECRET',str(failed))
        self.assertEqual(submit_enquiry(VALID,state,store).status,'submitted')
        self.assertEqual(store.save.call_args_list[0].args[1],store.save.call_args_list[1].args[1])
        store.save.side_effect=None
        store.save.return_value=None
        self.assertEqual(submit_enquiry(VALID,{},store).status,'failed')
        before=store.save.call_count
        self.assertEqual(submit_enquiry({},state,store).status,'invalid')
        self.assertEqual(before,store.save.call_count)


APP = '''
import runpy
from unittest.mock import patch
from utils.enterprise_enquiries import UnconfiguredStore
with patch('utils.database.record_visit'), patch('utils.database.get_stats', return_value={'visits':0,'likes':0,'dislikes':0}), patch('utils.enterprise_ui.get_enquiry_store',return_value=UnconfiguredStore()):
    runpy.run_path('app.py',run_name='__main__')
'''


class EnterpriseUITests(unittest.TestCase):
    def test_footer_navigation_and_analytics_state_roundtrip(self):
        app=AppTest.from_string(APP)
        state={}
        add_dataset(state,pd.DataFrame({'A':['x','y','z'],'Value':[1,2,3]}),'Working dataset','Test')
        for key,value in state.items():
            app.session_state[key]=value
        app.run(timeout=60)
        self.assertFalse(app.exception)
        app.session_state['ws_workflow']='Transform'
        app.run(timeout=60)
        app.selectbox(key='tf_menu_clean_values').select('Trim Spaces').run(timeout=60)
        # A draft and model preferences must survive hidden-page widget cleanup.
        draft=pd.DataFrame({'A':['x','y','z'],'Value':[1,2,3]})
        app.session_state['tf_draft']={'frame':draft,'steps':[{'operation':'Trim Spaces','column':0}],
                                       'parents':[app.session_state['ws_active']]}
        app.session_state['fac_alpha']=.01
        app.button(key='enterprise_learn_more').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.query_params['page'],['enterprise'])
        self.assertTrue(any('© 2026 Cozy. All rights reserved.'==c.value for c in app.caption))
        self.assertFalse(any(b.key=='enterprise_learn_more' for b in app.button))
        app.button(key='enterprise_back').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['ws_workflow'],'Transform')
        self.assertEqual(app.selectbox(key='tf_menu_clean_values').value,'Trim Spaces')
        self.assertEqual(app.session_state['fac_alpha'],.01)
        pd.testing.assert_frame_equal(app.session_state['tf_draft']['frame'],draft)

    def test_form_validation_preserves_values_and_unconfigured_message(self):
        app=AppTest.from_string(APP)
        app.query_params['page']='enterprise'
        app.run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='enterprise_form_deployment').value,None)
        self.assertEqual(app.selectbox(key='enterprise_form_number_of_users').value,None)
        app.text_input(key='enterprise_form_name').set_value('Alex Example')
        app.text_input(key='enterprise_form_email').set_value('not-an-email')
        app.text_area(key='enterprise_form_requirements').set_value('Specialized statistical reports')
        app.button(key='FormSubmitter:enterprise_enquiry-Submit Enquiry').click().run(timeout=60)
        self.assertGreaterEqual(len(app.warning),3)
        self.assertEqual(app.text_input(key='enterprise_form_name').value,'Alex Example')
        self.assertEqual(app.text_area(key='enterprise_form_requirements').value,'Specialized statistical reports')
        app.text_input(key='enterprise_form_email').set_value('alex@example.org')
        app.text_input(key='enterprise_form_organization').set_value('Example Organization')
        app.selectbox(key='enterprise_form_deployment').select('Not sure yet')
        app.button(key='FormSubmitter:enterprise_enquiry-Submit Enquiry').click().run(timeout=60)
        self.assertFalse(app.success)
        self.assertTrue(any("couldn't submit" in m.value for m in app.warning))
        app.button(key='enterprise_back').click().run(timeout=60)
        app.button(key='enterprise_learn_more').click().run(timeout=60)
        self.assertEqual(app.text_input(key='enterprise_form_name').value,'Alex Example')
        self.assertEqual(app.selectbox(key='enterprise_form_deployment').value,'Not sure yet')

    def test_ui_confirmed_success_duplicate_and_safe_failure(self):
        script = '''
import streamlit as st
from unittest.mock import patch
from utils.enterprise_ui import render_enquiry_form
class TestStore:
    configured=True
    def save(self,enquiry,submission_id):
        st.session_state['save_count']=st.session_state.get('save_count',0)+1
        if st.session_state.get('fail'):
            raise RuntimeError('SECRET')
        return submission_id
with patch('utils.enterprise_ui.get_enquiry_store',return_value=TestStore()):
    render_enquiry_form()
'''
        app=AppTest.from_string(script).run(timeout=30)
        for field in ['name','email','organization']:
            app.text_input(key='enterprise_form_'+field).set_value(VALID[field])
        app.selectbox(key='enterprise_form_deployment').select('AWS')
        app.session_state['fail']=True
        app.button(key='FormSubmitter:enterprise_enquiry-Submit Enquiry').click().run()
        self.assertTrue(app.warning)
        self.assertNotIn('SECRET',app.warning[0].value)
        self.assertEqual(app.text_input(key='enterprise_form_email').value,VALID['email'])
        app.session_state['fail']=False
        app.button(key='FormSubmitter:enterprise_enquiry-Submit Enquiry').click().run()
        self.assertTrue(app.success)
        self.assertEqual(app.text_input(key='enterprise_form_name').value,'')
        self.assertIsNone(app.selectbox(key='enterprise_form_deployment').value)
        count=app.session_state['save_count']
        app.run()
        self.assertEqual(count,app.session_state['save_count'])
        for field in ['name','email','organization']:
            app.text_input(key='enterprise_form_'+field).set_value(VALID[field])
        app.selectbox(key='enterprise_form_deployment').select('AWS')
        app.button(key='FormSubmitter:enterprise_enquiry-Submit Enquiry').click().run()
        self.assertIn('already been submitted',app.success[0].value)
        self.assertEqual(count,app.session_state['save_count'])


if __name__=='__main__':
    unittest.main()
