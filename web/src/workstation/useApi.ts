import { useEffect, useState } from "react";
import { apiAvailable } from "../api/client";

/** null while checking, then whether an analysis server answers behind /api. */
export function useApiAvailable(): boolean | null {
  const [ok, setOk] = useState<boolean | null>(null);
  useEffect(() => {
    let live = true;
    void apiAvailable().then((v) => live && setOk(v));
    return () => {
      live = false;
    };
  }, []);
  return ok;
}

export const NO_API_NOTE = "Live analysis needs the analysis server, which this hosted site does not run. The sample cases work fully; to analyse your own images, run the project locally (see the README).";
