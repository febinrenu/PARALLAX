"""Shared fixtures for the ml/ tests. No test touches the network except a loopback server."""

from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class _Handler(BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    etags: dict[str, str] = {}
    ignore_range = False
    fail_after: dict[str, int] = {}  # path -> bytes to send before dropping the connection
    hits: list[tuple[str, str | None]] = []

    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        data = self.files.get(self.path)
        rng = self.headers.get("Range")
        type(self).hits.append((self.path, rng))
        if data is None:
            self.send_error(404)
            return
        start = 0
        if rng and not self.ignore_range:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        if self.path in self.etags:
            self.send_header("ETag", f'"{self.etags[self.path]}"')
        self.end_headers()
        limit = self.fail_after.pop(self.path, None)
        if limit is not None:
            self.wfile.write(body[:limit])
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)


@pytest.fixture()
def server():
    handler = type("H", (_Handler,), {"files": {}, "etags": {}, "fail_after": {}, "hits": [], "ignore_range": False})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    handler.base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield handler
    httpd.shutdown()
