"""Server-only Supabase/PostgreSQL adapter; no browser client or email sending."""
from contextlib import closing
import logging
import re
from uuid import UUID

from psycopg2.extensions import parse_dsn
from utils.database import _connect, configured_value


logger = logging.getLogger(__name__)
CONFIG_KEY = 'SUPABASE_ENQUIRY_DB_URL'


def safe_failure_reason(exc, phase):
    """Classify privately; emit only fixed labels, never the exception text."""
    message = str(exc).lower()
    rules = [
        ('password authentication failed', 'password_authentication_failed'),
        ('authentication failed', 'authentication_failed'),
        ('unsupported or invalid secret format', 'pooler_invalid_password_format'),
        ('tenant or user not found', 'pooler_tenant_or_user_not_found'),
        ('role', 'role_error'),
    ]
    # Specific transport/pooler messages must take precedence over generic role errors.
    rules[4:4] = [
        ('circuit breaker', 'pooler_circuit_breaker_open'),
        ('unsupported startup parameter', 'unsupported_startup_parameter'),
        ('could not translate host name', 'dns_resolution_failed'),
        ('name or service not known', 'dns_resolution_failed'),
        ('connection refused', 'connection_refused'),
        ('timeout', 'connection_timeout' if phase == 'connect' else 'database_timeout'),
        ('timed out', 'connection_timeout' if phase == 'connect' else 'database_timeout'),
        ('network is unreachable', 'network_unreachable'),
        ('no route to host', 'network_unreachable'),
        ('certificate verify failed', 'tls_certificate_verification_failed'),
        ('ssl', 'tls_connection_failed'),
        ('too many connections', 'connection_limit_reached'),
        ('remaining connection slots', 'connection_limit_reached'),
        ('permission denied', 'permission_denied'),
    ]
    for pattern, reason in rules:
        if pattern in message:
            return reason
    if phase == 'configuration':
        return 'invalid_connection_configuration'
    return 'unclassified_connection_failure' if phase == 'connect' else 'unclassified_database_failure'


class PostgresEnquiryStore:
    @property
    def configured(self):
        return bool(configured_value(CONFIG_KEY))

    def save(self, enquiry, submission_id):
        UUID(submission_id)
        phase = 'configuration'
        try:
            settings = parse_dsn(configured_value(CONFIG_KEY) or '')
            sslmode = settings.get('sslmode', 'require')
            if sslmode not in ('require', 'verify-ca', 'verify-full'):
                sslmode = 'require'
            phase = 'connect'
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
            code = getattr(exc, 'pgcode', None)
            code = code if isinstance(code, str) and re.fullmatch(r'[A-Z0-9]{5}', code) else None
            logger.error('Enterprise persistence failed: phase=%s type=%s sqlstate=%s reason=%s submission_id=%s',
                         phase, type(exc).__name__, code, safe_failure_reason(exc, phase), submission_id)
            raise
