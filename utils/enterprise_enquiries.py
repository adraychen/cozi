"""Validated, idempotent enterprise submissions through server-side storage."""
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import logging
from typing import Protocol
from uuid import uuid4


DEPLOYMENTS = ('Microsoft Azure', 'AWS', 'Google Cloud', 'On-Premises', 'Not sure yet')
USER_COUNTS = ('1–10', '11–50', '51–200', '200+', 'Not sure')
FIELD_LIMITS = {'name': 200, 'email': 254, 'organization': 300,
                'job_title': 200, 'requirements': 5000}


@dataclass(frozen=True)
class Enquiry:
    name: str
    email: str
    organization: str
    deployment: str
    job_title: str = ''
    number_of_users: str = ''
    requirements: str = ''


def validate_enquiry(values):
    """Normalize only for persistence; validation never edits form widget values."""
    clean = {field: str(values.get(field) or '').strip() for field in Enquiry.__dataclass_fields__}
    errors = []
    for field, label in [('name', 'your name'), ('email', 'your work email'),
                         ('organization', 'your company or organization')]:
        if not clean[field]:
            errors.append(f'Please enter {label}.')
    email = clean['email']
    # Basic address syntax, not a claim that a mailbox exists or is a work domain.
    if email and (not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+", email)
                  or email.startswith('.') or '..' in email.split('@')[0] or '.@' in email):
        errors.append('Please enter a valid work email address, such as name@company.com.')
    if clean['deployment'] not in DEPLOYMENTS:
        errors.append('Please select a deployment interest, or choose “Not sure yet”.')
    if clean['number_of_users'] and clean['number_of_users'] not in USER_COUNTS:
        errors.append('Please choose one of the listed user ranges, or leave it blank.')
    labels = {'name': 'Name', 'email': 'Work Email', 'organization': 'Company / Organization',
              'job_title': 'Job Title', 'requirements': 'Customization Needs / Requirements'}
    for field, limit in FIELD_LIMITS.items():
        if len(clean[field]) > limit:
            errors.append(f'{labels[field]} must be {limit:,} characters or fewer.')
    return (None, errors) if errors else (Enquiry(**clean), [])


class EnquiryStore(Protocol):
    configured: bool

    def save(self, enquiry: Enquiry, submission_id: str) -> str:
        """Persist idempotently by submission_id and return a confirmed receipt.

        The production store must enforce a unique submission_id atomically and
        return the same receipt on retries after an uncertain network response.
        """
        ...


class UnconfiguredStore:
    configured = False

    def save(self, enquiry, submission_id):
        raise RuntimeError('Enterprise enquiry storage has not been configured.')


def get_enquiry_store() -> EnquiryStore:
    from utils.enterprise_store import PostgresEnquiryStore
    return PostgresEnquiryStore()


@dataclass(frozen=True)
class SubmissionResult:
    status: str
    messages: tuple[str, ...]


def submit_enquiry(values, state, store=None):
    enquiry, errors = validate_enquiry(values)
    if errors:
        return SubmissionResult('invalid', tuple(errors))
    fingerprint = hashlib.sha256(json.dumps(asdict(enquiry), sort_keys=True).encode()).hexdigest()
    receipts = state.setdefault('enterprise_receipts', {})
    if fingerprint in receipts:
        return SubmissionResult('duplicate', ('This enquiry has already been submitted. Thank you—we’ll contact you to discuss your requirements.',))
    store = store if store is not None else get_enquiry_store()
    if not store.configured:
        logging.getLogger(__name__).error('Enterprise persistence configuration is missing')
        return SubmissionResult('unavailable', ("We couldn't submit your enquiry at the moment. Please try again later.",))
    attempts = state.setdefault('enterprise_attempt_ids', {})
    submission_id = attempts.setdefault(fingerprint, str(uuid4()))
    try:
        receipt = store.save(enquiry, submission_id)
        if not isinstance(receipt, str) or not receipt.strip():
            raise ValueError('Persistence did not confirm a receipt.')
    except Exception as exc:
        # Do not reveal database errors, credentials, or contact details in the UI.
        logging.getLogger(__name__).error('Enterprise submission failed: type=%s', type(exc).__name__)
        return SubmissionResult('failed', ("We couldn't submit your enquiry at the moment. Please try again later.",))
    receipts[fingerprint] = receipt
    return SubmissionResult('submitted', ("Thank you for your interest in Cozy Enterprise. Your enquiry has been received. We will contact you to discuss your organization's requirements.",))
