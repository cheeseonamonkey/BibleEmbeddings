"""Public Verse Lawyer API. Nginx serves static files; this process serves JSON.

One trained ModernBERT, BM25 candidates, and transparent NLI scores.
Claims are not retained in app logs.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from bible_embeddings.corpus import load_passages
from bible_embeddings.judge import NLIJudge, top_evidence
from bible_embeddings.retrieval import Retriever

HOST = "127.0.0.1"
PORT = int(os.environ.get("VERSE_PORT", "8794"))
CORPUS = os.environ.get("VERSE_CORPUS", "/var/lib/verse-lawyer/en_kjv.json")
CHECKPOINT = os.environ.get("VERSE_MODEL", "/var/lib/verse-lawyer/model")
MAX_CLAIM = 180
MAX_BODY = 2048

torch.set_num_threads(max(1, min(4, int(os.environ.get("VERSE_THREADS", "2")))))
print("Loading Bible passages…", flush=True)
retriever = Retriever(load_passages(CORPUS))
print("Loading shared ModernBERT checkpoint…", flush=True)
judge = NLIJudge(CHECKPOINT, batch_size=8, device="cpu")
semaphore = threading.BoundedSemaphore(value=1)
print(f"Ready; {len(retriever.passages)} passages, one neural model", flush=True)


class API(BaseHTTPRequestHandler):
    server_version = "VerseLawyer"
    sys_version = ""

    def log_message(self, fmt, *args):
        # No claim text, user agents, or IP addresses in application logs.
        pass

    def send_json(self, status: int, value: dict):
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/health":
            self.send_json(200, {"ready": True, "model": "ModernBERT 149.6M"})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/analyze":
            self.send_json(404, {"error": "Not found"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.send_json(415, {"error": "Expected application/json"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        if not 0 < n <= MAX_BODY:
            self.send_json(413, {"error": "Invalid request size"})
            return
        try:
            payload = json.loads(self.rfile.read(n))
            claim = payload.get("claim")
            if not isinstance(claim, str) or not 0 < len(claim.strip()) <= MAX_CLAIM:
                raise ValueError("Invalid proposition")
            claim = claim.strip()
        except (ValueError, UnicodeDecodeError, AttributeError, TypeError):
            self.send_json(400, {"error": "Enter a valid proposition up to 180 characters."})
            return
        if not semaphore.acquire(blocking=False):
            self.send_json(429, {"error": "The model is considering another argument. Retry shortly."})
            return
        try:
            started = time.monotonic()
            hits = retriever.search(claim, limit=14, remove_overlap=False)
            judged = judge.evaluate(hits, claim)
            rows = {
                "claim": claim, "inspected": len(judged),
                "for": [e.as_dict() for e in top_evidence(judged, "entailment", n=3, threshold=0.4)],
                "against": [e.as_dict() for e in top_evidence(judged, "contradiction", n=3, threshold=0.4)],
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "disclaimer": "Model evidence signals; not calibrated or historical fact.",
            }
            self.send_json(200, rows)
        except Exception as exc:
            print(f"Analysis error: {type(exc).__name__}", flush=True)
            self.send_json(500, {"error": "Inference failed. Try again."})
        finally:
            semaphore.release()


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), API)
    server.daemon_threads = True
    server.serve_forever(poll_interval=0.5)
