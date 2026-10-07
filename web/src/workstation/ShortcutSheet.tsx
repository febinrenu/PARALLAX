import * as Dialog from "@radix-ui/react-dialog";

export const SHORTCUTS: [string, string][] = [
  ["Right-drag", "Window and level"],
  ["Wheel", "Zoom about the cursor"],
  ["Drag, or Space + drag", "Pan"],
  ["Double-click, 0", "Fit to view"],
  ["1 2 3 4", "Heatmap, mask, boxes, anatomy"],
  ["I", "Invert"],
  ["J / K", "Next or previous finding"],
  ["A / R", "Accept or reject finding"],
  ["Ctrl K", "Command palette"],
  ["?", "This sheet"],
];

export function ShortcutSheet({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-film-base/70" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[min(92vw,440px)] -translate-x-1/2 -translate-y-1/2 rounded-[var(--radius-panel)] bg-film-panel p-5 text-ink">
          <Dialog.Title className="text-[15px] font-medium">Keyboard and mouse</Dialog.Title>
          <Dialog.Description className="mt-1 text-[13px] text-ink-dim">Every action is reachable from the keyboard.</Dialog.Description>
          <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-[13px]">
            {SHORTCUTS.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="font-mono text-[12px] text-ink">{k}</dt>
                <dd className="text-ink-dim">{v}</dd>
              </div>
            ))}
          </dl>
          <Dialog.Close className="mt-5 rounded-[var(--radius-control)] border border-film-line px-3 py-1.5 text-[13px] text-ink">Close</Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
