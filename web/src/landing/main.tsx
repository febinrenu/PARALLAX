import { StrictMode } from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import "../styles/base.css";
import "./landing.css";
import { Landing } from "./Landing";

const html = document.documentElement;
const root = document.getElementById("root")!;
const tree = (
  <StrictMode>
    <Landing />
  </StrictMode>
);

// The production build ships prerendered markup (scripts/prerender.mjs); dev serves an empty root.
if (root.firstElementChild) hydrateRoot(root, tree);
else createRoot(root).render(tree);

function liteIntro() {
  // Article mode (narrow screen, reduced motion or no WebGL2): reveal the hero sentences only.
  const sentences = document.querySelectorAll<HTMLElement>(".hero-sentence");
  sentences.forEach((el, i) => window.setTimeout(() => el.classList.add("is-in"), html.classList.contains("reduce-motion") ? 0 : 500 + i * 650));
}

function bootStory() {
  // Loaded after first paint so GSAP, Lenis and the GL engine never compete with LCP.
  import("../story/engine")
    .then((m) => m.startStory())
    .catch((err) => {
      console.error("story engine failed, falling back to the article layout", err);
      html.classList.remove("scrolly");
      html.classList.add("story-failed");
      liteIntro();
    });
}

if (html.classList.contains("scrolly")) {
  const idle = (window as Window & { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number }).requestIdleCallback;
  const schedule = () => {
    if (idle) idle(bootStory, { timeout: 900 });
    else setTimeout(bootStory, 120);
  };
  if (document.readyState === "complete") schedule();
  else window.addEventListener("load", schedule, { once: true });
} else {
  liteIntro();
}
