import {createHandler, emailText, ResendProvider, SupabaseNotificationStore, type EnquiryRecord, type NotificationStore} from './handler.ts';

function equal(actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}
const id = '10000000-0000-4000-8000-000000000001';
const secret = 'offline-test-webhook-secret-32-characters';
const record: EnquiryRecord = {id,created_at:'2026-09-28T12:00:00Z',name:'Example Person',work_email:'person@example.org',organization:'Example Org',deployment_interest:'AWS'};
function request(body: unknown = {type:'INSERT',schema:'public',table:'data_analytics_enterprise_enquiries',record:{id,name:'Untrusted content'}}, token = secret) {
  return new Request('https://example.invalid/webhook',{method:'POST',headers:{'X-Cozy-Webhook-Secret':token},body:JSON.stringify(body)});
}
class Store implements NotificationStore {
  status = 'pending'; calls = 0; attempt = ''; failFinish = false;
  async claim(_id: string, attempt: string) {
    this.calls++;
    if (this.status === 'sent' || this.status === 'sending') return null;
    this.status = 'sending'; this.attempt = attempt; return record;
  }
  async finish(_id: string, attempt: string, sent: boolean) {
    if (this.failFinish) throw new Error('sensitive database error');
    if (attempt !== this.attempt) return false;
    this.status = sent ? 'sent' : 'failed'; return true;
  }
}

Deno.test('authentication and event validation prevent email and database access', async () => {
  const store = new Store(); let emails = 0;
  const handler = createHandler({secret,store,email:{send:async()=>{emails++;return 'receipt';}}});
  equal((await handler(request(undefined,'wrong'))).status,401);
  equal((await handler(request({type:'UPDATE'}))).status,400);
  equal((await handler(request({type:'INSERT',schema:'public',table:'data_analytics_enterprise_enquiries',record:{id:'invalid'}}))).status,400);
  equal((await handler(new Request('https://example.invalid'))).status,405);
  equal(store.calls,0); equal(emails,0);
});

Deno.test('canonical saved record is emailed once across concurrent deliveries and reruns', async () => {
  const store = new Store(); const delivered: EnquiryRecord[] = [];
  const handler = createHandler({secret,store,email:{send:async value=>{delivered.push(value);return 'receipt';}}});
  const responses = await Promise.all([handler(request()),handler(request())]);
  equal(responses.map(value=>value.status),[200,200]);
  equal((await handler(request())).status,200);
  equal(delivered,[record]); equal(store.status,'sent');
});

Deno.test('email failure preserves the saved record and records failure without leaking details', async () => {
  const store = new Store(); const logs: unknown[] = [];
  const handler = createHandler({secret,store,email:{send:async()=>{throw new Error('SECRET person@example.org');}},log:(...args)=>logs.push(args)});
  const result = await handler(request());
  equal(result.status,502); equal(store.status,'failed');
  equal(logs,[['email_failed',id]]);
  equal(await result.json(),{status:'notification_failed'});
  equal((await store.claim(id,crypto.randomUUID()))?.id,id);
});

Deno.test('uncertain status update retains lease for safe retry; claim failures never send', async () => {
  const store = new Store(); store.failFinish = true; let emails = 0;
  const handler = createHandler({secret,store,email:{send:async()=>{emails++;return 'receipt';}}});
  equal((await handler(request())).status,503); equal(store.status,'sending');
  equal((await handler(request())).status,200); equal(emails,1);
  store.claim = async()=>{throw new Error('private connection string');};
  equal((await handler(request())).status,503); equal(emails,1);
});

Deno.test('email contains contact fields, optional fields only when present, and enquiry receipt', () => {
  const text = emailText(record);
  for (const value of [record.name,record.work_email,record.organization,record.created_at,id,'AWS']) equal(text.includes(value),true);
  equal(text.includes('Job Title'),false);
  const full = emailText({...record,job_title:'Director',number_of_users:'11–50',customization_needs:'Private deployment'});
  for (const value of ['Director','11–50','Private deployment']) equal(full.includes(value),true);
});

Deno.test('Resend uses fixed addresses, plain text, and stable idempotency key', async () => {
  const calls: RequestInit[] = [];
  const fetcher: typeof fetch = async (_url, init)=>{calls.push(init!);return Response.json({id:'provider-receipt'});};
  const provider = new ResendProvider('test-key','sender@example.org','admin@example.org',fetcher);
  equal(await provider.send(record),'provider-receipt'); await provider.send(record);
  const headers = new Headers(calls[0].headers);
  equal(headers.get('Idempotency-Key'),`cozy-enterprise/${id}`);
  equal(new Headers(calls[1].headers).get('Idempotency-Key'),headers.get('Idempotency-Key'));
  const body = JSON.parse(calls[0].body as string);
  equal(body.to,['admin@example.org']); equal(body.from,'sender@example.org'); equal(body.html,undefined);
});

Deno.test('Supabase adapter uses narrow RPCs and never exposes upstream error bodies', async () => {
  const calls: {url:string,body:unknown}[] = [];
  const fetcher: typeof fetch = async (url,init)=>{
    calls.push({url:String(url),body:JSON.parse(init!.body as string)});
    return Response.json(String(url).endsWith('claim_enterprise_notification') ? [record] : true);
  };
  const store = new SupabaseNotificationStore('https://project.example','test-key',fetcher);
  equal(await store.claim(id,'attempt'),record); equal(await store.finish(id,'attempt',true,'receipt'),true);
  equal(calls[1].body,{p_id:id,p_attempt_id:'attempt',p_sent:true,p_provider_id:'receipt'});
  const failing = new SupabaseNotificationStore('https://project.example','test-key',async()=>new Response('SECRET',{status:500}));
  try { await failing.claim(id,'attempt'); throw new Error('Expected failure'); }
  catch (error) { equal((error as Error).message,'External notification request failed'); }
});
