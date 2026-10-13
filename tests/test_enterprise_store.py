import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from utils.enterprise_enquiries import Enquiry, submit_enquiry
from utils.enterprise_store import PostgresEnquiryStore
from utils.database import configured_value


class PostgresStoreTests(unittest.TestCase):
    def test_connection_diagnostics_classify_without_disclosing_error_text(self):
        from psycopg2 import OperationalError
        cases = [
            ('password authentication failed for user', 'password_authentication_failed'),
            ('Tenant or user not found', 'pooler_tenant_or_user_not_found'),
            ('unsupported startup parameter: options', 'unsupported_startup_parameter'),
            ('could not translate host name', 'dns_resolution_failed'),
            ('connection timed out', 'connection_timeout'),
            ('connection refused', 'connection_refused'),
            ('SSL error: certificate verify failed', 'tls_certificate_verification_failed'),
            ('Circuit breaker open', 'pooler_circuit_breaker_open'),
            ('unrecognized error', 'unclassified_connection_failure'),
        ]
        for message, reason in cases:
            with self.subTest(reason=reason):
                error = OperationalError(message + ' SECRET_PASSWORD postgresql://private-host alex@example.org')
                with patch('utils.enterprise_store.configured_value', return_value='dbname=postgres'), patch('utils.enterprise_store._connect', side_effect=error), self.assertLogs('utils.enterprise_store', level='ERROR') as logs:
                    with self.assertRaises(OperationalError):
                        PostgresEnquiryStore().save(Enquiry(name='Alex', email='alex@example.org', organization='Example', deployment='AWS'), str(uuid4()))
                output = ' '.join(logs.output)
                self.assertIn('reason=' + reason, output)
                self.assertIn('phase=connect', output)
                for sensitive in ('SECRET_PASSWORD', 'private-host', 'alex@example.org', 'postgresql://'):
                    self.assertNotIn(sensitive, output)

    def test_parameterized_save_receipt_and_closed_connection(self):
        connection=MagicMock()
        cursor=connection.cursor.return_value.__enter__.return_value
        receipt=str(uuid4())
        cursor.fetchone.return_value=(receipt,)
        values=dict(name="O'Brien",email='alex@example.org',organization='Example',deployment='AWS')
        enquiry=Enquiry(**values)
        submission_id=str(uuid4())
        with patch('utils.enterprise_store.configured_value',return_value='dbname=postgres host=localhost'), patch('utils.enterprise_store._connect',return_value=connection) as connect:
            self.assertEqual(PostgresEnquiryStore().save(enquiry,submission_id),receipt)
        query,params=cursor.execute.call_args.args
        self.assertNotIn(enquiry.name,query)
        self.assertIn('submit_enterprise_enquiry',query)
        self.assertEqual(params,(submission_id,"O'Brien",'alex@example.org','Example','AWS',None,None,None))
        self.assertEqual(connect.call_args.args,('SUPABASE_ENQUIRY_DB_URL',))
        self.assertEqual(connect.call_args.kwargs['sslmode'],'require')
        connection.__exit__.assert_called_once()
        connection.close.assert_called_once()

    def test_commit_failure_is_not_success_and_logs_no_sensitive_details(self):
        connection=MagicMock()
        connection.cursor.return_value.__enter__.return_value.fetchone.return_value=(str(uuid4()),)
        connection.__exit__.side_effect=RuntimeError('SECRET password alex@example.org SQL')
        values=dict(name='Alex',email='alex@example.org',organization='Example',deployment='AWS')
        state={}
        with patch('utils.enterprise_store.configured_value',return_value='dbname=postgres'), patch('utils.enterprise_store._connect',return_value=connection), self.assertLogs(level='ERROR') as logs:
            result=submit_enquiry(values,state,PostgresEnquiryStore())
        self.assertEqual(result.status,'failed')
        self.assertEqual(state['enterprise_receipts'],{})
        self.assertNotIn('SECRET',' '.join(logs.output))
        self.assertNotIn('alex@example.org',' '.join(logs.output))
        self.assertIn('phase=commit',' '.join(logs.output))
        connection.close.assert_called_once()

    def test_stricter_tls_preserved_and_no_notification_dependency(self):
        connection=MagicMock()
        connection.cursor.return_value.__enter__.return_value.fetchone.return_value=(str(uuid4()),)
        with patch('utils.enterprise_store.configured_value',return_value='dbname=postgres sslmode=verify-full'), patch('utils.enterprise_store._connect',return_value=connection) as connect:
            result=submit_enquiry(dict(name='Alex',email='alex@example.org',organization='Example',deployment='AWS'),{},PostgresEnquiryStore())
        self.assertEqual(result.status,'submitted')
        self.assertEqual(connect.call_args.kwargs['sslmode'],'verify-full')
        self.assertEqual(connection.cursor.return_value.__enter__.return_value.execute.call_count,1)

    def test_server_configuration_prefers_secrets_and_falls_back_to_environment(self):
        with patch('utils.database.st.secrets',{'KEY':'secret-value'}), patch.dict('os.environ',{'KEY':'environment-value'}):
            self.assertEqual(configured_value('KEY'),'secret-value')
        with patch('utils.database.st.secrets',{}), patch.dict('os.environ',{'KEY':'environment-value'}):
            self.assertEqual(configured_value('KEY'),'environment-value')


if __name__=='__main__':
    unittest.main()
