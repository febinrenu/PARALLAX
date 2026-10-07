import { Link } from "react-router";

export function NotFound() {
  return (
    <div className="mx-auto max-w-xl px-6 py-24">
      <h1 className="text-[36px] font-light">No page here.</h1>
      <p className="mt-3 text-ink-dim">The address may have a typo, or the study may have been analysed on a server that has since restarted.</p>
      <p className="mt-6 flex gap-5">
        <Link to="/read" className="text-pencil-yellow underline underline-offset-4">Open the workstation</Link>
        <a href="/" className="text-ink-dim underline underline-offset-4">Home</a>
      </p>
    </div>
  );
}
