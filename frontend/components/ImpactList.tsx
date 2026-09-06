/**
 * Best-effort parsing of the news-impact line from the orchestrator's reply.
 *
 * The orchestrator's stub synthesis (and likely the real LLM output too)
 * emits a single line like:
 *   "News (last 7d): 3 items, overall bullish,bearish (avg magnitude 0.65)."
 * We extract that and render a synthetic single-item card. Per-item reasons
 * aren't enumerated in the stub format; a real LLM reply would need richer
 * parsing which is out of scope for v1.
 */
export type Impact = {
  impact: "bullish" | "bearish" | "neutral";
  magnitude: number;
  horizon_days: number;
  reason: string;
  published_at: string;
};

export function parseAggregatedImpact(reply: string): Impact | null {
  const m = reply.match(
    /News \(last 7d\):\s*\d+\s*items?,\s*overall\s+([a-z,]+)\s*\(avg magnitude\s+([\d.]+)\)/i,
  );
  if (!m) return null;
  const labels = m[1]
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const priority = ["bullish", "bearish", "neutral"] as const;
  const impact = (priority.find((l) => labels.includes(l)) ??
    "neutral") as Impact["impact"];
  return {
    impact,
    magnitude: parseFloat(m[2]),
    horizon_days: 7,
    reason: `Aggregated across ${labels.length} label${
      labels.length === 1 ? "" : "s"
    } in last 7d`,
    published_at: new Date().toISOString(),
  };
}

export function ImpactList({ reply }: { reply: string }) {
  const item = parseAggregatedImpact(reply);
  if (!item) return null;

  const color =
    item.impact === "bullish"
      ? "text-green-700 bg-green-50 border-green-200"
      : item.impact === "bearish"
        ? "text-red-700 bg-red-50 border-red-200"
        : "text-gray-700 bg-gray-50 border-gray-200";

  return (
    <div className={`mt-4 rounded-lg border p-3 ${color}`}>
      <div className="flex items-baseline gap-2">
        <span className="font-medium capitalize">{item.impact}</span>
        <span className="text-sm opacity-75">
          magnitude {item.magnitude.toFixed(2)} · {item.horizon_days}d
        </span>
      </div>
      <p className="mt-1 text-sm opacity-80">{item.reason}</p>
    </div>
  );
}