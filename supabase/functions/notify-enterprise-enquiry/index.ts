import {createHandler, ResendProvider, SupabaseNotificationStore} from './handler.ts';

const secret = Deno.env.get('ENTERPRISE_WEBHOOK_SECRET') || '';
const url = Deno.env.get('SUPABASE_URL') || '';
const databaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || '';
const emailKey = Deno.env.get('RESEND_API_KEY') || '';
const from = Deno.env.get('ENTERPRISE_NOTIFICATION_FROM') || '';
const to = Deno.env.get('ENTERPRISE_NOTIFICATION_TO') || '';
const configured = secret.length >= 32 && url.startsWith('https://') && databaseKey && emailKey && from && to;

const handler = createHandler({
  secret, store:new SupabaseNotificationStore(url,databaseKey),
  email:new ResendProvider(emailKey,from,to),
  // Event codes and internal UUIDs only. Never log bodies, emails, headers,
  // provider responses, credentials, or exception messages containing PII.
  log:(event,id)=>console.error(JSON.stringify({event,enquiry_id:id})),
});
Deno.serve((request: Request) => {
  if (!configured) {
    console.error('enterprise_notification_configuration_missing');
    return Response.json({status:'temporarily_unavailable'},{status:503});
  }
  return handler(request);
});
