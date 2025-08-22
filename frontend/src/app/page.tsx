"use client";

import { useEffect, useMemo, useState } from "react";

type ConnectorState = {
  lastSyncAt?: string;
  startPageToken?: string;
  pageToken?: string;
};

type BeginResp = { redirect: string };

export default function Home() {
  const [tenantId, setTenantId] = useState<string>(
    typeof window !== "undefined"
      ? localStorage.getItem("tenantId") || ""
      : ""
  );
  const [status, setStatus] = useState<string>("Unknown");
  const [state, setState] = useState<ConnectorState | null>(null);
  const backend = useMemo(() => process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000", []);

  useEffect(() => {
    if (tenantId) {
      localStorage.setItem("tenantId", tenantId);
    }
  }, [tenantId]);

  // If redirected back from backend with ?tenant=... restore the tenant
  useEffect(() => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    const t = url.searchParams.get("tenant");
    if (t && t !== tenantId) {
      setTenantId(t);
      localStorage.setItem("tenantId", t);
      // clean the URL
      url.searchParams.delete("tenant");
      window.history.replaceState({}, "", url.toString());
    }
  }, []);

  async function fetchConnector() {
    if (!tenantId) return;
    try {
      const res = await fetch(`${backend}/api/integrations/google/oauth/begin`, {
        headers: { "X-Tenant-ID": tenantId },
      });
      if (res.ok) {
        setStatus("Not connected or ready to connect");
      }
      // We don’t expose a dedicated status API yet; show lastSyncAt after a sync
    } catch (e) {
      console.error(e);
    }
  }

  async function beginOAuth() {
    if (!tenantId) return alert("Enter a Tenant ID first");
    const res = await fetch(`${backend}/api/integrations/google/oauth/begin`, {
      headers: { "X-Tenant-ID": tenantId },
    });
    const data = (await res.json()) as BeginResp;
    window.location.href = data.redirect;
  }

  async function syncNow() {
    if (!tenantId) return alert("Enter a Tenant ID first");
    setStatus("Syncing...");
    const res = await fetch(`${backend}/api/integrations/google/sync`, {
      method: "POST",
      headers: { "X-Tenant-ID": tenantId },
    });
    if (!res.ok) {
      setStatus(`Sync failed (${res.status})`);
      return;
    }
    const js = await res.json();
    setStatus(
      `Synced: processed ${js.processed}, skipped ${js.skipped}, deleted ${js.deleted}`
    );
    setState((prev) => ({ ...(prev || {}), lastSyncAt: new Date().toISOString() }));
  }

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900">
      <div className="max-w-3xl mx-auto p-6">
        <header className="py-8">
          <h1 className="text-3xl font-semibold tracking-tight">VectralQ Admin</h1>
          <p className="text-gray-600 mt-1">Google Drive Connector</p>
        </header>

        <section className="bg-white border border-gray-200 rounded-2xl shadow-sm p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span
                className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${state?.lastSyncAt ? "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200" : "bg-amber-50 text-amber-700 ring-1 ring-amber-200"}`}
              >
                {state?.lastSyncAt ? "Connected" : "Needs connection"}
              </span>
              <span className="text-sm text-gray-500">
                {state?.lastSyncAt ? `Synced ${new Date(state.lastSyncAt).toLocaleString()}` : "Not synced yet"}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <input
                className="border rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                placeholder="Tenant ID"
                value={tenantId}
                onChange={(e) => setTenantId(e.target.value)}
              />
              <button
                onClick={beginOAuth}
                className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                Connect Google Drive
              </button>
              <button
                onClick={syncNow}
                className="inline-flex items-center gap-2 rounded-md bg-gray-900 px-3 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-black/90 focus:outline-none focus:ring-2 focus:ring-gray-900"
              >
                Sync now
              </button>
            </div>
          </div>

          <div className="text-sm text-gray-600">{status}</div>
        </section>

        <section className="mt-6 bg-white border border-gray-200 rounded-2xl shadow-sm p-6">
          <h2 className="text-base font-medium mb-3">Try a query</h2>
          <QueryBox tenantId={tenantId} backend={backend} />
        </section>
      </div>
    </main>
  );
}

function QueryBox({ tenantId, backend }: { tenantId: string; backend: string }) {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function runQuery() {
    if (!tenantId) return alert("Enter a Tenant ID first");
    setLoading(true);
    setAnswer(null);
    const body = {
      question: q,
      options: { top_k: 6, sources: ["google_drive"] },
    };
    const res = await fetch(`${backend}/api/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-ID": tenantId,
      },
      body: JSON.stringify(body),
    });
    const js = await res.json();
    setAnswer(JSON.stringify(js, null, 2));
    setLoading(false);
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask about your Drive docs..."
          className="flex-1 border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <button
          onClick={runQuery}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? "Asking..." : "Ask (Drive only)"}
        </button>
      </div>
      {answer && (
        <pre className="text-xs bg-gray-50 border rounded-lg p-3 overflow-auto max-h-80">
          {answer}
        </pre>
      )}
    </div>
  );
}

// Removed scaffolded default export to avoid duplicate exports
