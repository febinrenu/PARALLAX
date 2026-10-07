import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "node:path";

const API_TARGET = process.env.PARALLAX_API ?? "http://127.0.0.1:8000";

// Two entries: index.html is the prerendered landing (no router), app.html is the client-rendered
// app. App routes like /read or /audit/abc must load app.html in dev and preview; the production
// server does the same rewrite (P4.8).
const APP_ROUTES = /^\/(read|validation|models|audit|report)(\/|$)/;

function appRoutes(): Plugin {
  const rewrite = (req: { url?: string }) => {
    const url = req.url ?? "/";
    const path = url.split("?")[0];
    if (APP_ROUTES.test(path)) req.url = "/app.html";
  };
  return {
    name: "parallax-app-routes",
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        rewrite(req);
        next();
      });
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, _res, next) => {
        rewrite(req);
        next();
      });
    },
  };
}

const proxy = {
  "/api": {
    target: API_TARGET,
    changeOrigin: true,
    rewrite: (p: string) => p.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react(), tailwindcss(), appRoutes()],
  server: { proxy },
  preview: { proxy },
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        landing: resolve(__dirname, "index.html"),
        app: resolve(__dirname, "app.html"),
      },
      output: {
        // React is shared by both entries; name its chunk for what it is.
        manualChunks: (id) => (/node_modules[/\\](react|react-dom|scheduler)[/\\]/.test(id) ? "react" : undefined),
      },
    },
  },
});
