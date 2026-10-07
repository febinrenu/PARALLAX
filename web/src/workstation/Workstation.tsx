import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import type { Viewer } from "../gl/viewer";
import { ActionBar, runDecision } from "./ActionBar";
import { CommandPalette } from "./CommandPalette";
import { FindingsPanel } from "./FindingsPanel";
import { NotesPanel } from "./NotesPanel";
import { ShortcutSheet } from "./ShortcutSheet";
import { StudyRail } from "./StudyRail";
import { Toasts } from "./Toasts";
import { Viewport } from "./Viewport";
import { getSession, openSample, selectFinding, stepFinding } from "./store";

type Tab = "findings" | "notes" | "studies";

function isTyping(e: KeyboardEvent) {
  const t = e.target as HTMLElement | null;
  return !!t && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));
}

export function Workstation() {
  const viewerRef = useRef<Viewer | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [tab, setTab] = useState<Tab>("findings");
  const [params] = useSearchParams();
  const isMobile = useIsMobile();

  useEffect(() => {
    document.title = "Parallax workstation";
    const id = params.get("case");
    if (id && getSession().caseId !== id) void openSample(id);
  }, [params]);

  const toggleLayer = useCallback((key: "heatmap" | "mask" | "boxes" | "anatomy") => {
    const v = viewerRef.current;
    if (v) v.setLayers({ [key]: !v.getState().layers[key] });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
        return;
      }
      if (isTyping(e) || e.metaKey || e.ctrlKey || e.altKey || paletteOpen || sheetOpen) return;
      const v = viewerRef.current;
      switch (e.key) {
        case "1": toggleLayer("heatmap"); break;
        case "2": toggleLayer("mask"); break;
        case "3": toggleLayer("boxes"); break;
        case "4": toggleLayer("anatomy"); break;
        case "0": v?.fit(); break;
        case "i": case "I": v?.setInvert(!v.getState().invert); break;
        case "j": case "J": stepFinding(1); break;
        case "k": case "K": stepFinding(-1); break;
        case "a": case "A": void runDecision("accept"); break;
        case "r": case "R": void runDecision("reject"); break;
        case "?": setSheetOpen(true); break;
        case "Escape": selectFinding(null); break;
        default: return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [paletteOpen, sheetOpen, toggleLayer]);

  const onDropFile = (f: File) => {
    setPendingFile(f);
    setTab("studies");
  };

  const tabs: { id: Tab; label: string }[] = [
    { id: "findings", label: "Findings" },
    { id: "notes", label: "Notes" },
    { id: "studies", label: "Studies" },
  ];

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <h1 className="sr-only">Workstation</h1>

      {/* Desktop: rail | viewport | findings, notes under the viewport (plan.md section 10.4).
          Exactly one layout is mounted, so there is only ever one WebGL context. */}
      {!isMobile && (
      <div className="grid min-h-0 flex-1 md:grid-cols-[232px_minmax(0,1fr)_340px] md:grid-rows-[minmax(0,1fr)_auto] lg:grid-cols-[248px_minmax(0,1fr)_368px]" style={{ height: "calc(100svh - 48px)" }}>
        <div className="row-span-2 min-h-0">
          <StudyRail pendingFile={pendingFile} setPendingFile={setPendingFile} />
        </div>
        <div className="grid min-h-0 grid-rows-[minmax(0,1fr)_minmax(96px,22%)]">
          <Viewport viewerRef={viewerRef} onDropFile={onDropFile} />
          <NotesPanel />
        </div>
        <div className="row-span-2 grid min-h-0 grid-rows-[minmax(0,1fr)_auto]">
          <FindingsPanel />
          <ActionBar />
        </div>
      </div>
      )}

      {/* Mobile: viewport plus a bottom sheet with tabs. */}
      {isMobile && (
      <div className="flex min-h-0 flex-1 flex-col" style={{ height: "calc(100svh - 48px)" }}>
        <div className="grid h-[52svh] shrink-0">
          <Viewport viewerRef={viewerRef} onDropFile={onDropFile} />
        </div>
        <div role="tablist" aria-label="Panels" className="flex shrink-0 border-y border-film-line/60 bg-film-panel">
          {tabs.map((t) => (
            <button key={t.id} role="tab" type="button" aria-selected={tab === t.id} onClick={() => setTab(t.id)} className={`flex-1 py-2.5 text-[13px] ${tab === t.id ? "text-ink shadow-[inset_0_-2px_0_var(--color-pencil-yellow)]" : "text-ink-dim"}`}>
              {t.label}
            </button>
          ))}
        </div>
        <div role="tabpanel" className="grid min-h-0 flex-1">
          {tab === "findings" && (
            <div className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto]">
              <FindingsPanel />
              <ActionBar />
            </div>
          )}
          {tab === "notes" && <NotesPanel />}
          {tab === "studies" && <StudyRail pendingFile={pendingFile} setPendingFile={setPendingFile} />}
        </div>
      </div>
      )}

      <CommandPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        actions={{
          toggleLayer,
          fit: () => viewerRef.current?.fit(),
          invert: () => {
            const v = viewerRef.current;
            if (v) v.setInvert(!v.getState().invert);
          },
          decide: (d) => void runDecision(d),
          shortcuts: () => setSheetOpen(true),
        }}
      />
      <ShortcutSheet open={sheetOpen} onOpenChange={setSheetOpen} />
      <Toasts />
    </div>
  );
}

function useIsMobile() {
  const query = "(max-width: 767px)";
  const [isMobile, setIsMobile] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const fn = () => setIsMobile(mq.matches);
    mq.addEventListener("change", fn);
    return () => mq.removeEventListener("change", fn);
  }, []);
  return isMobile;
}
