import { createBrowserRouter } from "react-router";
import { AppShell } from "./AppShell";
import { RouteError } from "./RouteError";

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    errorElement: <RouteError />,
    hydrateFallbackElement: <div className="min-h-svh bg-film-base" />,
    children: [
      { path: "/read", lazy: async () => ({ Component: (await import("../workstation/Workstation")).Workstation }) },
      { path: "/validation", lazy: async () => ({ Component: (await import("../pages/Validation")).Validation }) },
      { path: "/models", lazy: async () => ({ Component: (await import("../pages/Models")).Models }) },
      { path: "/audit/:id", lazy: async () => ({ Component: (await import("../pages/Audit")).Audit }) },
      { path: "/report/:id", lazy: async () => ({ Component: (await import("../pages/Report")).Report }) },
      { path: "*", lazy: async () => ({ Component: (await import("../pages/NotFound")).NotFound }) },
    ],
  },
]);
