import { isRouteErrorResponse, useRouteError } from "react-router";

export function RouteError() {
  const err = useRouteError();
  const message = isRouteErrorResponse(err) ? `${err.status} ${err.statusText}` : err instanceof Error ? err.message : "Unknown error";
  return (
    <div className="mx-auto max-w-xl px-6 py-24">
      <h1 className="text-2xl font-light">This page failed to load.</h1>
      <p className="mt-3 text-ink-dim">{message}</p>
      <p className="mt-6">
        <a className="text-pencil-yellow underline underline-offset-4" href="/read">
          Open the workstation
        </a>
      </p>
    </div>
  );
}
