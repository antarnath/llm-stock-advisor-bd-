// Server component — runs on every request. We hit the backend in
// parallel so the page renders as fast as the slowest fetch.

type Prediction = {
  ticker: string;
  available: boolean;
  as_of?: string;
  horizon?: string;
  current_price?: number;
  predicted_price?: number;
  predicted_return?: number;   // % over the horizon
  fusion?: string;
};

type NewsItem = {
  article_id: number;
  impact: "bullish" | "bearish" | "neutral";
  magnitude: number;
  horizon_days: number;
  reason: string;
  published_at: string;
};

type StockSummary = {
  code: string;
  name: string;
  sector: string;
};

type Freshness = {
  ran_at: string | null;
  status: string;
  age_hours: number | null;
  label: string;
};

async function getJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

export default async function TickerPage({
  params,
}: {
  params: Promise<{ code: string }>;
}) {
  // Next.js 15 makes `params` a Promise in server components.
  const { code } = await params;
  const upper = code.toUpperCase();
  const api =
    process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  const [stocks, pred, news, fresh] = await Promise.all([
    getJson<StockSummary[]>(`${api}/stocks`),
    getJson<Prediction>(`${api}/stocks/${upper}/prediction`),
    getJson<{ ticker: string; days: number; items: NewsItem[] }>(
      `${api}/stocks/${upper}/news?days=7`,
    ),
    getJson<Freshness>(`${api}/freshness`),
  ]);

  const meta = stocks?.find((s) => s.code === upper);

  const ret = pred?.predicted_return ?? 0;
  const arrow = ret > 0 ? "▲" : ret < 0 ? "▼" : "·";
  const retColor =
    ret > 0
      ? "text-green-700"
      : ret < 0
        ? "text-red-700"
        : "text-gray-700";

  return (
    <div>
      <h1 className="text-3xl font-bold tracking-tight">
        {meta?.name ?? upper}{" "}
        <span className="text-gray-400">({upper})</span>
      </h1>
      {meta?.sector && (
        <p className="mt-1 text-sm text-gray-500 capitalize">
          {meta.sector}
        </p>
      )}

      {/* Prediction card */}
      <section className="mt-6 rounded-xl border bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold">5-day prediction</h2>
        {!pred?.available ? (
          <p className="mt-2 text-gray-500">
            No prediction available.
          </p>
        ) : (
          <div className="mt-3 flex items-baseline gap-3">
            <span className={`text-4xl font-bold ${retColor}`}>
              {arrow}
            </span>
            <span className={`text-3xl font-semibold ${retColor}`}>
              {ret >= 0 ? "+" : ""}
              {ret.toFixed(2)}%
            </span>
            <span className="text-sm text-gray-500">
              (price {pred.current_price?.toFixed(2)} →{" "}
              {pred.predicted_price?.toFixed(2)})
            </span>
          </div>
        )}
        {pred?.as_of && (
          <p className="mt-2 text-xs text-gray-400">
            Model as of {pred.as_of} · horizon {pred.horizon} · fusion{" "}
            {pred.fusion}
          </p>
        )}
      </section>

      {/* News list */}
      <section className="mt-6">
        <h2 className="text-lg font-semibold">
          Last 7 days of news
        </h2>
        {!news || news.items.length === 0 ? (
          <p className="mt-2 text-gray-500">
            No tracked impact items.
          </p>
        ) : (
          <ul className="mt-3 space-y-2">
            {news.items.map((it) => {
              const color =
                it.impact === "bullish"
                  ? "border-green-200 bg-green-50"
                  : it.impact === "bearish"
                    ? "border-red-200 bg-red-50"
                    : "border-gray-200 bg-gray-50";
              const tColor =
                it.impact === "bullish"
                  ? "text-green-700"
                  : it.impact === "bearish"
                    ? "text-red-700"
                    : "text-gray-700";
              return (
                <li
                  key={it.article_id}
                  className={`rounded-lg border p-3 ${color}`}
                >
                  <div
                    className={`flex items-baseline gap-2 ${tColor}`}
                  >
                    <span className="font-medium capitalize">
                      {it.impact}
                    </span>
                    <span className="text-sm opacity-75">
                      magnitude {it.magnitude.toFixed(2)} ·{" "}
                      {it.horizon_days}d
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-gray-700">
                    {it.reason}
                  </p>
                  <p className="mt-1 text-xs text-gray-400">
                    {new Date(it.published_at).toLocaleString()}
                  </p>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <footer className="mt-10 border-t pt-4 text-xs text-gray-400">
        Data last refreshed: {fresh?.label ?? "unknown"}
      </footer>
    </div>
  );
}