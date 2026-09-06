"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import { ChatInput } from "@/components/ChatInput";
import { ChatReply } from "@/components/ChatReply";
import { ImpactList } from "@/components/ImpactList";

type ReplyState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok";    reply: string; latencyMs: number }
  | { kind: "error"; message: string };

/**
 * Next.js 15 requires useSearchParams() to be inside a <Suspense>
 * boundary so the page can pre-render with a fallback while the
 * query string is read on the client. See:
 *   https://nextjs.org/docs/messages/missing-suspense-with-csr-bailout
 *
 * We split the page into a server-friendly shell (this file) plus
 * an inner client component (ChatInner) that does the actual fetch.
 */
function ChatInner() {
  const sp = useSearchParams();
  const q = sp.get("q") ?? "";
  const [state, setState] = useState<ReplyState>({ kind: "idle" });

  useEffect(() => {
    if (!q) return;
    let cancelled = false;
    setState({ kind: "loading" });

    const api =
      process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

    fetch(`${api}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return (await r.json()) as { reply: string; latency_ms: number };
      })
      .then((d) => {
        if (cancelled) return;
        setState({
          kind: "ok",
          reply: d.reply,
          latencyMs: d.latency_ms,
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setState({
          kind: "error",
          message:
            "Could not reach the advisor service. " +
            `Is the backend running on ${api}? (${String(err)})`,
        });
      });

    return () => {
      cancelled = true;
    };
  }, [q]);

  return (
    <div>
      <h1 className="text-xl font-semibold">Ask the advisor</h1>
      <p className="mt-1 text-sm text-gray-500">
        You asked:{" "}
        <span className="font-mono">{q || "(empty)"}</span>
      </p>

      <div className="mt-6">
        {state.kind === "loading" && (
          <p className="text-gray-500">Thinking…</p>
        )}

        {state.kind === "ok" && (
          <>
            <ChatReply reply={state.reply} />
            <ImpactList reply={state.reply} />
            <p className="mt-2 text-right text-xs text-gray-400">
              {state.latencyMs} ms
            </p>
          </>
        )}

        {state.kind === "error" && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-800">
            {state.message}
          </div>
        )}

        {state.kind === "idle" && q && (
          <p className="text-gray-500">Send the question below.</p>
        )}
      </div>

      <div className="mt-10 border-t pt-6">
        <p className="mb-2 text-sm text-gray-500">
          Ask another question:
        </p>
        <ChatInput />
      </div>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense
      fallback={
        <p className="text-gray-500">Loading chat…</p>
      }
    >
      <ChatInner />
    </Suspense>
  );
}