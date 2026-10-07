import { NavLink, Outlet, useLocation } from "react-router";
import { Mark } from "../design/Mark";

const NAV = [
  { to: "/read", label: "Workstation" },
  { to: "/validation", label: "Validation" },
  { to: "/models", label: "Models" },
];

export function AppShell() {
  const { pathname } = useLocation();
  const isPrint = pathname.startsWith("/report/");
  return (
    <div className="flex min-h-svh flex-col">
      {!isPrint && (
        <header className="flex h-12 shrink-0 items-center gap-6 border-b border-film-line/60 bg-film-base px-4 print:hidden">
          {/* The landing is a separate document, so this is a full page load. */}
          <a href="/" className="flex items-center gap-2 text-ink" aria-label="Parallax home">
            <Mark size={20} />
            <span className="text-[15px] font-medium tracking-[-0.01em]">Parallax</span>
          </a>
          <nav aria-label="Main" className="flex items-center gap-1">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) =>
                  `rounded-[var(--radius-control)] px-2.5 py-1 text-[13px] transition-colors duration-150 ${
                    isActive ? "bg-film-raised text-ink" : "text-ink-dim hover:text-ink"
                  }`
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <p className="ml-auto hidden text-[12px] text-ink-dim md:block">Decision support only. Not a diagnosis.</p>
        </header>
      )}
      <main className="flex min-h-0 flex-1 flex-col">
        <Outlet />
      </main>
    </div>
  );
}
