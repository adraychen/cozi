-- Run as the Supabase database administrator. No credentials belong in this file.
begin;

do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'cozy_enquiry_submitter') then
    create role cozy_enquiry_submitter nologin nosuperuser nobypassrls;
  end if;
end $$;

create table public.data_analytics_enterprise_enquiries (
  id uuid primary key default gen_random_uuid(),
  submission_id uuid not null unique,
  created_at timestamptz not null default now(),
  name text not null check (length(btrim(name)) between 1 and 200),
  work_email text not null check (length(work_email) <= 254 and work_email ~ '^[^[:space:]@]+@[^[:space:]@.]+(\.[^[:space:]@.]+)+$'),
  organization text not null check (length(btrim(organization)) between 1 and 300),
  job_title text check (length(job_title) <= 200),
  deployment_interest text not null check (deployment_interest in ('Microsoft Azure','AWS','Google Cloud','On-Premises','Not sure yet')),
  number_of_users text check (number_of_users in ('1–10','11–50','51–200','200+','Not sure')),
  customization_needs text check (length(customization_needs) <= 5000),
  status text not null default 'new' check (status in ('new','contacted','qualified','closed')),
  notification_status text not null default 'pending' check (notification_status in ('pending','sending','sent','failed','manual_review')),
  notification_sent_at timestamptz,
  -- Operational idempotency/lease fields, not visitor tracking.
  notification_attempt_id uuid,
  notification_started_at timestamptz,
  notification_claimed_at timestamptz,
  notification_provider_id text
);
create index data_analytics_enterprise_enquiries_status_created_idx on public.data_analytics_enterprise_enquiries(status, created_at);
alter table public.data_analytics_enterprise_enquiries enable row level security;
-- Deliberately no public policies, not even public INSERT. Submission goes through
-- a constrained function and a dedicated server login, never a browser client.
revoke all on public.data_analytics_enterprise_enquiries from public, anon, authenticated, service_role, cozy_enquiry_submitter;
grant usage on schema public to cozy_enquiry_submitter;

create function public.submit_enterprise_enquiry(
  p_submission_id uuid, p_name text, p_work_email text, p_organization text,
  p_deployment_interest text, p_job_title text default null,
  p_number_of_users text default null, p_customization_needs text default null
) returns uuid language plpgsql security definer set search_path = '' as $$
declare saved public.data_analytics_enterprise_enquiries;
begin
  insert into public.data_analytics_enterprise_enquiries(submission_id,name,work_email,organization, deployment_interest,job_title,number_of_users,customization_needs)
    values (p_submission_id,p_name,p_work_email,p_organization,p_deployment_interest,p_job_title,p_number_of_users,p_customization_needs)
    on conflict (submission_id) do nothing returning * into saved;
  if saved.id is null then
    select * into saved from public.data_analytics_enterprise_enquiries where submission_id = p_submission_id;
    if saved.id is null or row(saved.name,saved.work_email,saved.organization,saved.deployment_interest,saved.job_title,saved.number_of_users,saved.customization_needs)
      is distinct from row(p_name,p_work_email,p_organization,p_deployment_interest,p_job_title,p_number_of_users,p_customization_needs) then
      raise exception 'Submission reference conflict' using errcode = '22023';
    end if;
  end if;
  -- Return only the receipt, never contact data. Retries do not cause INSERT events.
  return saved.id;
end $$;
revoke all on function public.submit_enterprise_enquiry(uuid,text,text,text,text,text,text,text) from public,anon,authenticated,service_role;
grant execute on function public.submit_enterprise_enquiry(uuid,text,text,text,text,text,text,text) to cozy_enquiry_submitter;

create function public.claim_enterprise_notification(p_id uuid, p_attempt_id uuid)
returns setof public.data_analytics_enterprise_enquiries language plpgsql security definer set search_path = '' as $$
begin
  -- Resend retains idempotency keys for 24h. Stop automatic sends after 22h
  -- from the first attempt to avoid duplicates after an ambiguous acceptance.
  update public.data_analytics_enterprise_enquiries set notification_status = 'manual_review'
    where id=p_id and notification_status in ('pending','failed','sending')
      and notification_started_at < now()-interval '22 hours';
  return query update public.data_analytics_enterprise_enquiries
    set notification_status='sending', notification_attempt_id=p_attempt_id,
        notification_started_at=coalesce(notification_started_at,now()), notification_claimed_at=now()
    where id=p_id and (notification_status in ('pending','failed') or
      (notification_status='sending' and notification_claimed_at < now()-interval '5 minutes'))
    returning *;
end $$;

create function public.finish_enterprise_notification(p_id uuid, p_attempt_id uuid,
    p_sent boolean, p_provider_id text default null)
returns boolean language plpgsql security definer set search_path = '' as $$
declare changed integer;
begin
  if p_sent and (p_provider_id is null or length(p_provider_id)=0) then
    raise exception 'A provider receipt is required' using errcode='22023';
  end if;
  update public.data_analytics_enterprise_enquiries
    set notification_status=case when p_sent then 'sent' else 'failed' end,
        notification_sent_at=case when p_sent then now() else null end,
        notification_provider_id=case when p_sent then p_provider_id else null end
    where id=p_id and notification_status='sending' and notification_attempt_id=p_attempt_id;
  get diagnostics changed = row_count;
  return changed=1;
end $$;
revoke all on function public.claim_enterprise_notification(uuid,uuid) from public,anon,authenticated,cozy_enquiry_submitter;
revoke all on function public.finish_enterprise_notification(uuid,uuid,boolean,text) from public,anon,authenticated,cozy_enquiry_submitter;
grant execute on function public.claim_enterprise_notification(uuid,uuid) to service_role;
grant execute on function public.finish_enterprise_notification(uuid,uuid,boolean,text) to service_role;
commit;
