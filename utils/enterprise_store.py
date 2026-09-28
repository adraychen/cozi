"""Server-only Supabase/PostgreSQL adapter; no browser client or email sending."""
from contextlib import closing
import logging
from uuid import UUID

from psycopg2.extensions import parse_dsn
from utils.database import _connect, configured_value


logger = logging.getLogger(__name__)
CONFIG_KEY = 'SUPABASE_ENQUIRY_DB_URL'


class PostgresEnquiryStore:
    @property
    def configured(self):
        return bool(configured_value(CONFIG_KEY))

    def save(self, enquiry, submission_id):
        UUID(submission_id)
        phase = 'connect'
        try:
            settings = parse_dsn(configured_value(CONFIG_KEY) or '')
            sslmode = settings.get('sslmode', 'require')
            if sslmode not in ('require', 'verify-ca', 'verify-full'):
                sslmode = 'require'
            # Same connection mechanism as visit/feedback storage, different role.
            # The login may only execute this narrow SECURITY DEFINER function.
            with closing(_connect(CONFIG_KEY, sslmode=sslmode, connect_timeout=5,
                                  options='-c statement_timeout=10000',
                                  application_name='cozy_enterprise')) as conn:
                with conn:
                    with conn.cursor() as cursor:
                        phase = 'insert'
                        cursor.execute('SELECT public.submit_enterprise_enquiry(%s::uuid,%s::text,%s::text,%s::text,%s::text,%s::text,%s::text,%s::text)',
                                       (submission_id, enquiry.name, enquiry.email, enquiry.organization,
                                        enquiry.deployment, enquiry.job_title or None,
                                        enquiry.number_of_users or None, enquiry.requirements or None))
                        row = cursor.fetchone()
                        if not row or not row[0]:
                            raise ValueError('Missing persistence receipt')
                        receipt = str(UUID(str(row[0])))
                    phase = 'commit'
                # A receipt is returned only after the transaction commits.
                return receipt
        except Exception as exc:
            logger.error('Enterprise persistence failed: phase=%s type=%s sqlstate=%s submission_id=%s',
                         phase, type(exc).__name__, getattr(exc, 'pgcode', None), submission_id)
            raise
