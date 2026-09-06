export function ChatReply({ reply }: { reply: string }) {
  return (
    <div className="prose prose-slate max-w-none whitespace-pre-wrap rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      {reply}
    </div>
  );
}