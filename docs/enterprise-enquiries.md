# Enterprise enquiry backend setup

The Streamlit server uses the existing psycopg2/PostgreSQL architecture. It commits an enquiry before returning success. A Supabase **INSERT database webhook** then invokes `notify-enterprise-enquiry`, which reads the canonical record and sends a plain-text administrator email through Resend. Email failures cannot roll back the committed enquiry. The page and analytics session are unchanged.

This repository prepares the workflow; deployment and a controlled live test are still required. Do not use production contact data in tests.

## 1. Apply the migration and create a restricted login

Apply `supabase/migrations/202609280001_enterprise_enquiries.sql` once using the Supabase SQL Editor as the database administrator. Alternatively, use your normal reviewed Supabase CLI migration deployment process. Check the target project before applying it. The migration is additive; it does not change the existing visits or feedback tables.

The table has RLS enabled and **no public access policies**. `anon`, `authenticated`, and the Streamlit submission role cannot read, update, delete, or directly insert rows. A constrained `SECURITY DEFINER` function validates table constraints and returns only a receipt UUID. All functions use an empty search path and fully qualified table names. The submission function accepts a stable submission UUID, so a retry after an uncertain commit returns the same receipt without another insert/webhook.

Create a dedicated server login in the SQL Editor, substituting a newly generated password privately (never save the filled-in SQL in this repository):

```sql
create role cozy_enterprise_app login password '<GENERATE-PRIVATE-PASSWORD>'
  nosuperuser nocreatedb nocreaterole noreplication nobypassrls inherit;
grant cozy_enquiry_submitter to cozy_enterprise_app;
```

Do not grant this login other roles or table privileges. The NOLOGIN group has only schema usage and execute permission on `submit_enterprise_enquiry`. The function owner must remain the trusted migration administrator. The Edge Function's service-role credential has execute permission on the notification claim/completion functions; this migration revokes its direct access to this table. Service-role credentials remain powerful elsewhere in Supabase and must stay exclusively in backend secrets. Administrators can manage enquiries privately through the SQL Editor; no public administration interface is added.

## 2. Configure Streamlit

Set **`SUPABASE_ENQUIRY_DB_URL`** in the deployment's Streamlit secrets or server environment. Streamlit secrets take precedence. Use the dedicated login above, a Supabase direct connection or compatible session pooler, database `postgres`, and SSL. Follow the connection panel's host/port and custom-role username format; pooler usernames may include the project reference.

```toml
SUPABASE_ENQUIRY_DB_URL = "postgresql://cozy_enterprise_app:<URL-ENCODED-PASSWORD>@<DATABASE-HOST>:5432/postgres?sslmode=require"
```

`verify-full` with the appropriate CA is also supported and preserved. The adapter enforces at least `require`, a five-second connection timeout, and a ten-second statement timeout. Existing `SUPABASE_DB_URL` remains for existing application features; do not replace it with this restricted login or use it as an enquiry fallback. Do not expose the DSN in browser settings, query parameters, logs, or source control.

## 3. Configure the email provider and deploy the function

In Resend, add and verify a sending domain using its required DNS records. Choose a sender on that verified domain and a controlled administrator inbox. Provision a sending API key with the narrowest available scope.

In Supabase Dashboard → Edge Functions → Secrets, configure:

| Secret | Value |
| --- | --- |
| `ENTERPRISE_WEBHOOK_SECRET` | A random secret of at least 32 characters; use the same value in the webhook header |
| `RESEND_API_KEY` | Private Resend sending API key |
| `ENTERPRISE_NOTIFICATION_FROM` | Verified sender, optionally `Cozy Enterprise <sender@your-domain>` |
| `ENTERPRISE_NOTIFICATION_TO` | Private administrator notification address |

Supabase supplies `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to deployed Edge Functions. The service-role key is never configured in Streamlit or browser code for this feature. Missing configuration fails closed without sending email.

From the repository root, after authenticating the CLI and choosing your project:

```powershell
supabase functions deploy notify-enterprise-enquiry --project-ref <PROJECT-REF>
```

`supabase/config.toml` disables gateway JWT verification **only for this function** because database webhooks use a custom shared-secret header. The handler independently verifies `X-Cozy-Webhook-Secret` before parsing events or accessing the database. It accepts only POST/INSERT/public.data_analytics_enterprise_enquiries events and uses only their record UUID. Contact fields supplied by a caller are not trusted as email content. The sender and recipient always come from backend configuration.

## 4. Create the database webhook

In Supabase Dashboard → Database → Webhooks, enable the feature if needed, then create:

- Table: `public.data_analytics_enterprise_enquiries`
- Event: **INSERT only** (never UPDATE, which would loop on notification status changes)
- Method: POST
- URL: `https://<PROJECT-REF>.supabase.co/functions/v1/notify-enterprise-enquiry`
- Headers: `Content-Type: application/json` and `X-Cozy-Webhook-Secret: <SAME-PRIVATE-SECRET>`
- Timeout: 5,000 ms or the maximum supported value appropriate for your project

Use Supabase's standard row event JSON. Database webhooks use asynchronous `pg_net` requests after commit; do not add a synchronous email trigger. Store webhook configuration privately: exported webhook SQL may contain the header secret. Rotate the header and Edge secret together. Review webhook request/response history in the database's `net` schema and Edge logs if a notification stays pending. No scheduled retry worker is installed by this change.

## 5. Safe verification and failure recovery

First use a development Supabase project and a controlled recipient inbox. Run the automated tests below before deployment. Then:

1. Submit an incomplete/invalid form: validation appears and nothing is inserted.
2. Fill the Enterprise form with clearly labeled test contact details, leaving optional fields blank. Submit: one row is committed, the specified success message appears, and fields clear. Reruns and an identical resubmission in the same session must not insert another row.
3. In the private SQL Editor, verify the receipt, default `status = 'new'`, and notification progression from `pending` to `sent`. Check the actual inbox and provider dashboard for the professional email with the enquiry ID. `sent` and `notification_sent_at` mean **provider acceptance**, not guaranteed inbox delivery.
4. Temporarily set an invalid enquiry DSN in the development app: submission must retain the form and show a friendly failure, with no success. Restore the DSN and retry the same form.
5. Temporarily configure an invalid Resend key in the development Edge Function: the form must still succeed and the row remain saved; the notification must become `failed`. Restore the key, then replay the authenticated INSERT event for that same UUID and confirm `sent`. Never test outages against production configuration.
6. Replay the same event again: no second email should be produced. Without the private header, requests must return 401 and do no database work.

For a private retry, POST the following JSON with the same protected header using a trusted backend/API client. Keep secrets out of shell history and screenshots:

```json
{"type":"INSERT","schema":"public","table":"data_analytics_enterprise_enquiries","record":{"id":"<EXISTING-ENQUIRY-UUID>"}}
```

The database claim is atomic. Active claims expire after five minutes, and already-sent records are skipped. Retries use the same Resend idempotency key (`cozy-enterprise/<enquiry UUID>`). Resend retains these keys for 24 hours; automated claims stop 22 hours after the first attempt and switch to `manual_review`. This prevents uncertain sends from being repeated after the provider's deduplication window expires. If a status update fails after provider acceptance, leave the lease intact and retry after five minutes **within this window**. If the webhook never reached the function, the row remains `pending` and can be replayed later.

For `manual_review`, reconcile the enquiry ID with Resend's delivery history first. If accepted, an administrator can record the provider receipt and mark it sent. Only if non-delivery is established should an administrator reset it to pending and clear the attempt/start/claim fields before replaying. Do not blindly reset old uncertain deliveries. Transient failures are logged using event codes/internal UUIDs, never contact fields, credentials, provider bodies, or exception messages.

Session deduplication does not identify a person across browsers. A new session can create a new enquiry; there is no new visitor tracking. If public abuse becomes an issue, deploy ingress rate limits or a challenge separately rather than making the enquiry table public.

## Automated checks (no production services)

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests
deno check supabase/functions/notify-enterprise-enquiry/index.ts
deno test supabase/functions/notify-enterprise-enquiry/handler_test.ts
npm ci --prefix supabase/tests --ignore-scripts
npm test --prefix supabase/tests
```

Python tests exercise the existing form with Streamlit AppTest and mocked persistence, including commit failures, input retention, successful reset, duplicate reruns, validation and analytics state. Edge tests inject database/email doubles, including concurrent delivery, safe failures, fixed recipients and stable provider idempotency keys. Database tests run the actual migration in an in-memory PGlite PostgreSQL engine: role access denials, constraints, duplicate receipts/INSERT events, atomic leases, failed notifications, stale attempts and manual review. PGlite is a test-only dependency; these tests do not require a Supabase project or send email. These checks do not establish that a live webhook, Supabase deployment or email delivery is operational.

References: [Supabase database webhooks](https://supabase.com/docs/guides/database/webhooks), [Edge Function configuration](https://supabase.com/docs/guides/functions/function-configuration), [Resend idempotency](https://resend.com/docs/dashboard/emails/idempotency-keys).
