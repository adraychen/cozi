import {PGlite} from '@electric-sql/pglite';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';

test('migration, least privilege, idempotent submission and notification lifecycle', async (t) => {
  const db = new PGlite();
  try {
    await db.exec('create role anon; create role authenticated; create role service_role; grant usage on schema public to anon,authenticated,service_role;');
    await db.exec(await readFile(new URL('../migrations/202609280001_enterprise_enquiries.sql',import.meta.url),'utf8'));
    // Model the INSERT-only webhook boundary without making network requests.
    await db.exec(`create table public.test_insert_events (id uuid);
      create function public.test_capture_insert() returns trigger language plpgsql security definer set search_path='' as $$
      begin insert into public.test_insert_events values (new.id); return new; end $$;
      create trigger test_insert after insert on public.enterprise_enquiries for each row execute function public.test_capture_insert();`);
    const query = async (sql, params=[]) => (await db.query(sql,params)).rows;
    const asRole = async (role, fn) => {
      await db.exec(`set role ${role}`);
      try { return await fn(); } finally { await db.exec('reset role'); }
    };
    const submitSql = 'select public.submit_enterprise_enquiry($1::uuid,$2::text,$3::text,$4::text,$5::text,$6::text,$7::text,$8::text) as id';
    const values = [randomUUID(),'Test Person','test@example.org','Test Organization','AWS',null,null,null];
    let id;
    await t.test('RLS is enabled without public policies; public CRUD and RPC access denied', async () => {
      assert.equal((await query("select relrowsecurity from pg_class where oid='public.enterprise_enquiries'::regclass"))[0].relrowsecurity,true);
      assert.equal((await query("select * from pg_policies where tablename='enterprise_enquiries'")).length,0);
      for (const role of ['anon','authenticated','cozy_enquiry_submitter','service_role']) {
        await asRole(role, async()=>{
          for (const sql of ['select * from public.enterprise_enquiries',"update public.enterprise_enquiries set status='closed'",'delete from public.enterprise_enquiries','insert into public.enterprise_enquiries default values']) {
            await assert.rejects(query(sql),{code:'42501'});
          }
          if (role!=='cozy_enquiry_submitter') await assert.rejects(query(submitSql,values),{code:'42501'});
          if (role!=='service_role') await assert.rejects(query('select * from public.claim_enterprise_notification($1,$2)',[randomUUID(),randomUUID()]),{code:'42501'});
        });
      }
    });
    await t.test('restricted server submits, retries once, and cannot repurpose a receipt', async () => {
      await asRole('cozy_enquiry_submitter',async()=>{
        id=(await query(submitSql,values))[0].id;
        assert.equal((await query(submitSql,values))[0].id,id);
        const changed=[...values]; changed[1]='Changed';
        await assert.rejects(query(submitSql,changed),{code:'22023'});
      });
      const rows=await query('select * from public.enterprise_enquiries');
      assert.equal(rows.length,1); assert.equal(rows[0].notification_status,'pending');
      assert.equal(rows[0].status,'new'); assert.equal(rows[0].job_title,null);
      assert.deepEqual(await query('select * from public.test_insert_events'),[{id}]);
    });
    await t.test('database constraints reject missing fields and invalid choices', async()=>{
      for (const [index,value] of [[1,' '],[2,'invalid-email'],[3,''],[4,'Invalid cloud'],[6,'500'],[7,'x'.repeat(5001)]]) {
        const bad=[...values];bad[0]=randomUUID();bad[index]=value;
        await asRole('cozy_enquiry_submitter',()=>assert.rejects(query(submitSql,bad),{code:'23514'}));
      }
    });
    await t.test('one notification lease; failure preserves saved enquiry, retry marks sent', async()=>{
      await asRole('service_role',async()=>{
        const attempt=randomUUID();
        assert.equal((await query('select * from public.claim_enterprise_notification($1,$2)',[id,attempt])).length,1);
        assert.equal((await query('select * from public.claim_enterprise_notification($1,$2)',[id,randomUUID()])).length,0);
        assert.equal((await query('select public.finish_enterprise_notification($1,$2,false) as done',[id,randomUUID()]))[0].done,false);
        assert.equal((await query('select public.finish_enterprise_notification($1,$2,false) as done',[id,attempt]))[0].done,true);
      });
      assert.equal((await query('select notification_status from public.enterprise_enquiries where id=$1',[id]))[0].notification_status,'failed');
      await asRole('service_role',async()=>{
        const attempt=randomUUID();
        assert.equal((await query('select * from public.claim_enterprise_notification($1,$2)',[id,attempt])).length,1);
        await assert.rejects(query('select public.finish_enterprise_notification($1,$2,true)',[id,attempt]),{code:'22023'});
        assert.equal((await query('select public.finish_enterprise_notification($1,$2,true,$3) as done',[id,attempt,'test-provider-receipt']))[0].done,true);
        assert.equal((await query('select * from public.claim_enterprise_notification($1,$2)',[id,randomUUID()])).length,0);
      });
      const saved=(await query('select * from public.enterprise_enquiries where id=$1',[id]))[0];
      assert.equal(saved.notification_status,'sent');assert.ok(saved.notification_sent_at);
      assert.equal(saved.name,values[1]);assert.equal(saved.notification_provider_id,'test-provider-receipt');
    });
    await t.test('stale leases recover but expired provider deduplication requires manual review', async()=>{
      const second=[...values];second[0]=randomUUID();
      const next=(await asRole('cozy_enquiry_submitter',()=>query(submitSql,second)))[0].id;
      const oldAttempt=randomUUID();
      await asRole('service_role',()=>query('select * from public.claim_enterprise_notification($1,$2)',[next,oldAttempt]));
      await query("update public.enterprise_enquiries set notification_claimed_at=now()-interval '6 minutes' where id=$1",[next]);
      await asRole('service_role',async()=>{
        assert.equal((await query('select * from public.claim_enterprise_notification($1,$2)',[next,randomUUID()])).length,1);
        assert.equal((await query('select public.finish_enterprise_notification($1,$2,true,$3) as done',[next,oldAttempt,'stale']))[0].done,false);
      });
      await query("update public.enterprise_enquiries set notification_started_at=now()-interval '23 hours' where id=$1",[next]);
      await asRole('service_role',async()=>assert.equal((await query('select * from public.claim_enterprise_notification($1,$2)',[next,randomUUID()])).length,0));
      assert.equal((await query('select notification_status from public.enterprise_enquiries where id=$1',[next]))[0].notification_status,'manual_review');
    });
  } finally { await db.close(); }
});
