"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

/**
 * /settings — runtime LLM provider config (Phase 13).
 *
 * Lets the user pick a provider, type a model id, set a base URL
 * (NIM only), and paste an API key. POSTs to /settings; the backend
 * persists to data/external/api/llm_settings.json and mirrors the
 * values into os.environ so the very next /ask call uses them —
 * no restart required.
 *
 * Mirrors the chat page's client-component shape:
 *   - single useState bag for the form
 *   - useEffect on mount for the GET
 *   - plain fetch() against NEXT_PUBLIC_API_URL
 */

type Settings = {
  provider: string;
  model:    string;
  base_url: string;
  api_key:  string;
  has_key:  boolean;
};

type Status = "idle" | "saving" | "saved" | "error";

const PROVIDERS: { id: string; label: string }[] = [
  { id: "stub",       label: "Stub (no network, deterministic)" },
  { id: "openrouter", label: "OpenRouter" },
  { id: "groq",       label: "Groq" },
  { id: "gemini",     label: "Google Gemini" },
  { id: "nim",        label: "NVIDIA NIM" },
];

const EMPTY: Settings = {
  provider: "stub", model: "", base_url: "", api_key: "", has_key: false,
};

export default function SettingsPage() {
  const api = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8765";
  const [form,    setForm]    = useState<Settings>(EMPTY);
  const [status,  setStatus]  = useState<Status>("idle");
  const [message, setMessage] = useState<string>("");

  // Prefill on mount — backend may be down; that's fine.
  useEffect(() => {
    let cancelled = false;
    fetch(`${api}/settings`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return (await r.json()) as Settings;
      })
      .then((d) => {
        if (cancelled) return;
        setForm((f) => ({ ...f, ...d }));
      })
      .catch(() => {
        /* keep defaults; backend may be offline */
      });
    return () => {
      cancelled = true;
    };
  }, [api]);

  function save() {
    setStatus("saving");
    setMessage("");

    // Send exactly what the user has typed. Do NOT mix in anything from
    // the previous GET — the backend masks the key in GET responses, so
    // a stale `form.api_key` (still showing dots from the placeholder)
    // would otherwise overwrite the real key with "**********bjkT".
    const payload: Settings = {
      provider: form.provider,
      model:    form.model,
      base_url: form.base_url,
      api_key:  form.api_key,
      has_key:  false,
    };

    fetch(`${api}/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(async (r) => {
        const body = await r.text();
        if (!r.ok) throw new Error(body || `HTTP ${r.status}`);
        return JSON.parse(body) as Settings;
      })
      .then((d) => {
        // Server returns the key MASKED. We must NOT splat `d.api_key`
        // back into form state — that would replace the user's typed
        // key with dots after every save and silently break subsequent
        // saves. Only update fields the user didn't touch.
        setForm((f) => ({
          provider: d.provider || f.provider,
          model:    d.model    || f.model,
          base_url: d.base_url || f.base_url,
          // api_key: keep what the user typed; mark has_key from server.
          api_key:  f.api_key,
          has_key:  d.has_key,
        }));
        setStatus("saved");
        setMessage("Saved. Next /ask will use this provider.");
      })
      .catch((err: unknown) => {
        setStatus("error");
        setMessage(String(err));
      });
  }

  function reset() {
    setStatus("saving");
    setMessage("");
    fetch(`${api}/settings`, { method: "DELETE" })
      .then(async (r) => {
        const body = await r.text();
        if (!r.ok) throw new Error(body || `HTTP ${r.status}`);
        return JSON.parse(body) as Settings;
      })
      .then((d) => {
        // Reset wipes everything including any in-progress typing.
        setForm({ ...EMPTY });
        setStatus("saved");
        setMessage("Reset. Next /ask will fall back to env / stub.");
      })
      .catch((err: unknown) => {
        setStatus("error");
        setMessage(String(err));
      });
  }

  const showBaseUrl = form.provider === "nim";

  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
      <p className="mt-1 text-sm text-gray-500">
        Configure which LLM the advisor uses. Values are saved to disk and
        applied immediately &mdash; no restart needed.
      </p>

      <div className="mt-6 space-y-4">
        <label className="block">
          <span className="text-sm font-medium">Provider</span>
          <select
            className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2"
            value={form.provider}
            onChange={(e) => setForm({ ...form, provider: e.target.value })}
          >
            {PROVIDERS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="text-sm font-medium">Model</span>
          <input
            type="text"
            className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 font-mono text-sm"
            placeholder="(provider default if blank)"
            value={form.model}
            onChange={(e) => setForm({ ...form, model: e.target.value })}
          />
        </label>

        {showBaseUrl && (
          <label className="block">
            <span className="text-sm font-medium">
              Base URL{" "}
              <span className="font-normal text-gray-500">
                (NIM only; blank uses integrated endpoint)
              </span>
            </span>
            <input
              type="url"
              className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 font-mono text-sm"
              placeholder="https://integrate.api.nvidia.com/v1"
              value={form.base_url}
              onChange={(e) => setForm({ ...form, base_url: e.target.value })}
            />
          </label>
        )}

        <label className="block">
          <span className="text-sm font-medium">API key</span>
          <input
            type="password"
            className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 font-mono text-sm"
            placeholder={
              form.has_key ? "(saved - type to replace)" : "paste key here"
            }
            value={form.api_key}
            onChange={(e) => setForm({ ...form, api_key: e.target.value })}
          />
          {form.has_key && (
            <span className="text-xs text-gray-500">
              A key is currently saved. Type a new one to replace it.
            </span>
          )}
        </label>

        <div className="flex gap-2">
          <button
            onClick={save}
            disabled={status === "saving"}
            className="rounded-lg bg-blue-600 px-4 py-2 text-white shadow-sm hover:bg-blue-700 disabled:opacity-50"
          >
            {status === "saving" ? "Saving…" : "Save"}
          </button>
          <button
            onClick={reset}
            className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-700 hover:bg-gray-50"
          >
            Reset
          </button>
        </div>

        {message && (
          <p
            className={
              status === "error"
                ? "rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800"
                : "rounded-lg border border-green-200 bg-green-50 p-3 text-sm text-green-800"
            }
          >
            {message}
          </p>
        )}
      </div>

      <p className="mt-8 text-sm">
        <Link className="underline text-gray-500" href="/">
          &larr; Back
        </Link>
      </p>
    </div>
  );
}
