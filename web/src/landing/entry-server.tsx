import { StrictMode } from "react";
import { renderToString } from "react-dom/server";
import { Landing } from "./Landing";

/** Build-time render of the landing so it reads without JavaScript and paints text immediately. */
export function render(): string {
  return renderToString(
    <StrictMode>
      <Landing />
    </StrictMode>,
  );
}
