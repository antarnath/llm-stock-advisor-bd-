import Link from "next/link";
import { ChatInput } from "@/components/ChatInput";

export default function Landing() {
  return (
    <div>
      <h1 className="text-3xl font-bold tracking-tight">
        DSE Advisor
      </h1>
      <p className="mt-2 text-gray-600">
        Bangladesh stock market in plain language. Ask anything —
        the model looks at the last 5 trading days and today&apos;s news.
      </p>

      <div className="mt-8">
        <ChatInput autoFocus />
      </div>

      <p className="mt-6 text-sm text-gray-500">
        Tip: try{" "}
        <Link className="underline" href="/ticker/GP">
          GP
        </Link>
        ,{" "}
        <Link className="underline" href="/ticker/BEXIMCO">
          BEXIMCO
        </Link>
        , or{" "}
        <Link className="underline" href="/ticker/SQURPHARMA">
          SQURPHARMA
        </Link>
        .
      </p>

      <p className="mt-2 text-sm text-gray-500">
        Or{" "}
        <Link className="underline" href="/settings">
          configure the LLM provider
        </Link>
        .
      </p>
    </div>
  );
}