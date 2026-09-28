// Dependencies are injected for offline testing. Webhook data is never trusted as
// email content: only the UUID is accepted, and the record is loaded from the DB.
export type EnquiryRecord = {
  id: string; created_at: string; name: string; work_email: string;
  organization: string; deployment_interest: string; job_title?: string | null;
  number_of_users?: string | null; customization_needs?: string | null;
};
export interface NotificationStore {
  claim(id: string, attempt: string): Promise<EnquiryRecord | null>;
  finish(id: string, attempt: string, sent: boolean, receipt?: string): Promise<boolean>;
}
export interface EmailProvider {
  send(record: EnquiryRecord): Promise<string>;
}
export function emailText(record: EnquiryRecord): string {
  const fields = [
    ['Submitted', record.created_at], ['Name', record.name],
    ['Work Email', record.work_email], ['Organization', record.organization],
    ['Job Title', record.job_title], ['Deployment Interest', record.deployment_interest],
    ['Number of Users', record.number_of_users],
    ['Customization Needs / Requirements', record.customization_needs], ['Enquiry ID', record.id],
  ];
  return 'New Cozy Enterprise Enquiry\n\n' + fields.filter(([,value])=>value)
    .map(([label,value])=>`${label}:\n${value}`).join('\n\n');
}
async function validSecret(actual: string, expected: string): Promise<boolean> {
  if (expected.length < 32 || !actual) return false;
  const digest = (value: string) => crypto.subtle.digest('SHA-256', new TextEncoder().encode(value));
  const [a,b] = await Promise.all([digest(actual),digest(expected)]);
  const aa = new Uint8Array(a), bb = new Uint8Array(b);
  let difference = 0;
  for (let i=0;i<aa.length;i++) difference |= aa[i]^bb[i];
  return difference===0;
}
const response = (status: number, state: string) => Response.json({status:state}, {status});

async function fetchJson(fetcher: typeof fetch, url: string, init: RequestInit): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(()=>controller.abort(),10000);
  try {
    const result = await fetcher(url,{...init,signal:controller.signal});
    if (!result.ok) {
      await result.body?.cancel();
      throw new Error('External notification request failed');
    }
    return await result.json();
  } finally { clearTimeout(timer); }
}

export function createHandler(dependencies: {
  secret: string; store: NotificationStore; email: EmailProvider;
  log?: (event: string, id?: string)=>void;
}) {
  const log = dependencies.log || (()=>{});
  return async (request: Request): Promise<Response> => {
    if (request.method !== 'POST') return response(405,'method_not_allowed');
    if (!await validSecret(request.headers.get('X-Cozy-Webhook-Secret') || '', dependencies.secret))
      return response(401,'unauthorized');
    let id: string;
    try {
      const body = await request.text();
      if (body.length > 32768) return response(413,'payload_too_large');
      const payload = JSON.parse(body);
      if (payload.type !== 'INSERT' || payload.schema !== 'public' || payload.table !== 'enterprise_enquiries'
          || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(payload.record?.id || ''))
        return response(400,'invalid_event');
      id = payload.record.id;
    } catch { return response(400,'invalid_event'); }
    const attempt = crypto.randomUUID();
    let record: EnquiryRecord | null;
    try { record = await dependencies.store.claim(id,attempt); }
    catch { log('claim_failed',id); return response(503,'retry_later'); }
    if (!record) return response(200,'not_claimed'); // already sent, leased, or needs review
    let receipt: string;
    try {
      receipt = await dependencies.email.send(record);
      if (!receipt) throw new Error('Missing email receipt');
    } catch {
      log('email_failed',id);
      try { await dependencies.store.finish(id,attempt,false); }
      catch { log('failure_status_update_failed',id); }
      return response(502,'notification_failed');
    }
    try {
      if (!await dependencies.store.finish(id,attempt,true,receipt)) {
        log('success_status_not_updated',id);
        return response(503,'retry_later');
      }
    } catch {
      // Leave the lease intact on an uncertain completion. A subsequent attempt
      // reuses the provider idempotency key, rather than sending a new email.
      log('success_status_update_failed',id);
      return response(503,'retry_later');
    }
    return response(200,'sent');
  };
}

export class SupabaseNotificationStore implements NotificationStore {
  constructor(private url: string, private key: string, private fetcher: typeof fetch = fetch) {}
  private async rpc(name: string, body: object): Promise<unknown> {
    return await fetchJson(this.fetcher,`${this.url}/rest/v1/rpc/${name}`, {
      method:'POST', headers:{'Content-Type':'application/json', apikey:this.key, Authorization:`Bearer ${this.key}`},
      body:JSON.stringify(body),
    });
  }
  async claim(id: string, attempt: string): Promise<EnquiryRecord | null> {
    const rows = await this.rpc('claim_enterprise_notification',{p_id:id,p_attempt_id:attempt});
    if (!Array.isArray(rows)) throw new Error('Invalid claim response');
    return rows[0] || null;
  }
  async finish(id: string, attempt: string, sent: boolean, receipt?: string): Promise<boolean> {
    return await this.rpc('finish_enterprise_notification',{
      p_id:id,p_attempt_id:attempt,p_sent:sent,p_provider_id:receipt || null,
    }) === true;
  }
}

export class ResendProvider implements EmailProvider {
  constructor(private key: string, private from: string, private to: string, private fetcher: typeof fetch = fetch) {}
  async send(record: EnquiryRecord): Promise<string> {
    const data = await fetchJson(this.fetcher,'https://api.resend.com/emails', {
      method:'POST', headers:{Authorization:`Bearer ${this.key}`, 'Content-Type':'application/json',
                             'Idempotency-Key':`cozy-enterprise/${record.id}`},
      body:JSON.stringify({from:this.from,to:[this.to],subject:'New Cozy Enterprise Enquiry',text:emailText(record)}),
    }) as {id?: unknown};
    if (typeof data.id !== 'string' || !data.id) throw new Error('Missing provider receipt');
    return data.id;
  }
}
