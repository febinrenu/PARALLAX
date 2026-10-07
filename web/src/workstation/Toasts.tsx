import { useEffect, useState } from "react";

type Listener = (text: string) => void;
const listeners = new Set<Listener>();

export function toast(text: string) {
  listeners.forEach((fn) => fn(text));
}

/** One toast at a time, announced politely to screen readers. */
export function Toasts() {
  const [text, setText] = useState<string | null>(null);
  useEffect(() => {
    let timer = 0;
    const fn: Listener = (t) => {
      setText(t);
      window.clearTimeout(timer);
      timer = window.setTimeout(() => setText(null), 5200);
    };
    listeners.add(fn);
    return () => {
      listeners.delete(fn);
      window.clearTimeout(timer);
    };
  }, []);
  return (
    <div aria-live="polite" role="status" className="pointer-events-none fixed bottom-4 left-1/2 z-50 -translate-x-1/2">
      {text && <p className="max-w-[56ch] rounded-[var(--radius-panel)] bg-film-raised px-4 py-2.5 text-[13px] text-ink">{text}</p>}
    </div>
  );
}
