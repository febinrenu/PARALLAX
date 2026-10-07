import { Command } from "cmdk";
import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { loadCaseIndex } from "../api/client";
import type { CaseIndexEntry } from "../api/types";
import { displayLabel } from "../lib/findings";
import { openSample, selectFinding, useSession } from "./store";

export interface PaletteActions {
  toggleLayer: (key: "heatmap" | "mask" | "boxes" | "anatomy") => void;
  fit: () => void;
  invert: () => void;
  decide: (d: "accept" | "reject") => void;
  shortcuts: () => void;
}

export function CommandPalette({ open, onOpenChange, actions }: { open: boolean; onOpenChange: (o: boolean) => void; actions: PaletteActions }) {
  const [cases, setCases] = useState<CaseIndexEntry[]>([]);
  const findings = useSession((s) => s.findings);
  const studyId = useSession((s) => s.studyId);
  const navigate = useNavigate();

  useEffect(() => {
    if (open && !cases.length) loadCaseIndex().then(setCases).catch(() => undefined);
  }, [open, cases.length]);

  const run = (fn: () => void) => () => {
    onOpenChange(false);
    fn();
  };

  const item = "flex cursor-pointer items-center rounded-[var(--radius-control)] px-2.5 py-2 text-[13px] text-ink aria-selected:bg-film-raised";
  const group = "px-1 pb-1 pt-3 text-[12px] text-ink-dim [&_[cmdk-group-heading]]:px-1.5 [&_[cmdk-group-heading]]:pb-1";

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-film-base/70" />
        <Dialog.Content className="fixed left-1/2 top-[14vh] z-50 w-[min(92vw,560px)] -translate-x-1/2 overflow-hidden rounded-[var(--radius-panel)] bg-film-panel">
          <Dialog.Title className="sr-only">Command palette</Dialog.Title>
          <Command label="Command palette" loop>
            <Command.Input autoFocus placeholder="Type a command or a finding" className="w-full border-b border-film-line bg-transparent px-4 py-3 text-[14px] text-ink outline-none placeholder:text-ink-dim" />
            <Command.List className="max-h-[52vh] overflow-y-auto p-1.5">
              <Command.Empty className="px-3 py-6 text-center text-[13px] text-ink-dim">No matching command.</Command.Empty>
              {findings.length > 0 && (
                <Command.Group heading="Findings" className={group}>
                  {findings.map((f) => (
                    <Command.Item key={f.finding_id} value={`finding ${f.label} ${f.finding_id}`} onSelect={run(() => selectFinding(f.finding_id))} className={item}>
                      {displayLabel(f.label)}
                      <span className="ml-auto font-mono text-[12px] text-ink-dim">{f.prob_calibrated.toFixed(2)}</span>
                    </Command.Item>
                  ))}
                </Command.Group>
              )}
              <Command.Group heading="Viewer" className={group}>
                {(["heatmap", "mask", "boxes", "anatomy"] as const).map((k, i) => (
                  <Command.Item key={k} value={`toggle ${k}`} onSelect={run(() => actions.toggleLayer(k))} className={item}>
                    Toggle {k}
                    <kbd className="ml-auto font-mono text-[11px] text-ink-dim">{i + 1}</kbd>
                  </Command.Item>
                ))}
                <Command.Item value="fit to view" onSelect={run(actions.fit)} className={item}>Fit to view<kbd className="ml-auto font-mono text-[11px] text-ink-dim">0</kbd></Command.Item>
                <Command.Item value="invert" onSelect={run(actions.invert)} className={item}>Invert<kbd className="ml-auto font-mono text-[11px] text-ink-dim">I</kbd></Command.Item>
              </Command.Group>
              <Command.Group heading="Decisions" className={group}>
                <Command.Item value="accept finding" onSelect={run(() => actions.decide("accept"))} className={item}>Accept finding<kbd className="ml-auto font-mono text-[11px] text-ink-dim">A</kbd></Command.Item>
                <Command.Item value="reject finding" onSelect={run(() => actions.decide("reject"))} className={item}>Reject finding<kbd className="ml-auto font-mono text-[11px] text-ink-dim">R</kbd></Command.Item>
              </Command.Group>
              <Command.Group heading="Open" className={group}>
                {cases.map((c) => (
                  <Command.Item key={c.id} value={`open sample ${c.title}`} onSelect={run(() => void openSample(c.id))} className={item}>Open sample: {c.title}</Command.Item>
                ))}
                {studyId && <Command.Item value="audit trail" onSelect={run(() => navigate(`/audit/${studyId}`))} className={item}>Open audit trail</Command.Item>}
                <Command.Item value="validation" onSelect={run(() => navigate("/validation"))} className={item}>Open validation</Command.Item>
                <Command.Item value="models" onSelect={run(() => navigate("/models"))} className={item}>Open model cards</Command.Item>
                <Command.Item value="keyboard shortcuts" onSelect={run(actions.shortcuts)} className={item}>Keyboard shortcuts<kbd className="ml-auto font-mono text-[11px] text-ink-dim">?</kbd></Command.Item>
              </Command.Group>
            </Command.List>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
