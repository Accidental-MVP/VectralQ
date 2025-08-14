CREATE OR REPLACE FUNCTION app.apply_tenant_rls(schema_name text, table_name text, tenant_column text)
RETURNS void AS $BODY$
BEGIN
  EXECUTE format('ALTER TABLE %I.%I ENABLE ROW LEVEL SECURITY', schema_name, table_name);
  EXECUTE format('ALTER TABLE %I.%I FORCE ROW LEVEL SECURITY', schema_name, table_name);
  EXECUTE format('DROP POLICY IF EXISTS %I_isolate ON %I.%I', table_name, schema_name, table_name);
  EXECUTE format(
    'CREATE POLICY %2$I_isolate ON %1$I.%2$I USING ((%3$I)::text = current_setting(''app.tenant_id'', true)) WITH CHECK ((%3$I)::text = current_setting(''app.tenant_id'', true))',
    schema_name, table_name, tenant_column
  );
END;
$BODY$ LANGUAGE plpgsql;

-- Policies are applied in Alembic after tables are created

