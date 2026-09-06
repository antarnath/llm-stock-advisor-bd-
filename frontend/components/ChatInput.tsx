"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export function ChatInput({ autoFocus = false }: { autoFocus?: boolean }) {
  const [q, setQ] = useState("");
  const router = useRouter();

  function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const trimmed = q.trim();
    if (!trimmed) return;
    router.push(`/chat?q=${encodeURIComponent(trimmed)}`);
  }

  return (
    <form onSubmit={onSubmit} className="w-full">
      <input
        type="text"
        className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-lg shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        placeholder="Should I buy GP next week?"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        autoFocus={autoFocus}
        maxLength={2000}
      />
    </form>
  );
}