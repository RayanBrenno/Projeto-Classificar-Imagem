type Props = {
  label: string;
  value: number;
  rank: number;
};

export function ConfidenceBar({ label, value, rank }: Props) {
  const pct = Math.round(value * 100);
  const isTop = rank === 0;

  return (
    <div className="flex items-center gap-3">
      <span
        className={`w-16 text-right text-sm font-mono tracking-wide truncate ${
          isTop ? "text-slate-200" : "text-slate-500"
        }`}
      >
        {label}
      </span>

      <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 delay-100 ${
            isTop
              ? "bg-gradient-to-r from-violet-500 to-emerald-400"
              : "bg-slate-600"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <span
        className={`w-11 text-right text-sm font-mono tabular-nums ${
          isTop ? "text-violet-300" : "text-slate-600"
        }`}
      >
        {pct}%
      </span>
    </div>
  );
}
