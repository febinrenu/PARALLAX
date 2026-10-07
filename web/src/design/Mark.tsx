/** Two offset circles: the same object seen from two lines of sight. */
export function Mark({ size = 20, className }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className={className}>
      <circle cx="13" cy="16" r="7.5" fill="none" stroke="currentColor" strokeWidth="2" />
      <circle cx="19" cy="16" r="7.5" fill="none" stroke="var(--color-pencil-yellow)" strokeWidth="2" />
    </svg>
  );
}
