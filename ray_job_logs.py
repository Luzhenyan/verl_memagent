#!/usr/bin/env python3
"""Ray Dashboard Job log helper.

Why this exists:
- `ray job submit` stdout can be truncated depending on terminal/session.
- The Ray Dashboard API always has the authoritative driver logs.

This script pulls logs via the dashboard HTTP API:
- List jobs:     GET  /api/jobs/
- Get logs:      GET  /api/jobs/<submission_id>/logs
- Stop job:      POST /api/jobs/<submission_id>/stop

It uses only the Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def _normalize_dashboard_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("empty dashboard url")
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url
    return url


def _http_json(method: str, url: str, timeout_s: float = 30.0):
    req = urllib.request.Request(url=url, method=method)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8", errors="replace"))


def list_jobs(dashboard_url: str):
    payload = _http_json("GET", f"{dashboard_url}/api/jobs/")
    # Ray versions differ:
    # - some return a list of job dicts
    # - some return {"jobs": [...]}.
    if isinstance(payload, list):
        jobs = payload
    elif isinstance(payload, dict):
        jobs = payload.get("jobs", [])
    else:
        raise RuntimeError(f"unexpected /api/jobs/ response type: {type(payload)}")
    if not jobs:
        print("(no jobs)")
        return

    for job in jobs:
        sid = job.get("submission_id")
        status = job.get("status")
        entry = job.get("entrypoint", "")
        entry_short = entry.replace("\n", " ")
        if len(entry_short) > 140:
            entry_short = entry_short[:140] + "…"
        print(f"{sid}\t{status}\t{entry_short}")


def get_logs(dashboard_url: str, submission_id: str) -> str:
    payload = _http_json("GET", f"{dashboard_url}/api/jobs/{urllib.parse.quote(submission_id)}/logs")
    logs = payload.get("logs")
    if logs is None:
        raise RuntimeError(f"unexpected response keys: {sorted(payload.keys())}")
    return logs


def stop_job(dashboard_url: str, submission_id: str):
    try:
        payload = _http_json(
            "POST", f"{dashboard_url}/api/jobs/{urllib.parse.quote(submission_id)}/stop"
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"stop failed: HTTP {e.code}: {body}") from e
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def follow_logs(
    dashboard_url: str,
    submission_id: str,
    poll_s: float,
    out_path: str | None,
    only_newline_flush: bool,
):
    last_len = 0
    out_f = None
    try:
        if out_path:
            out_f = open(out_path, "a", encoding="utf-8")

        while True:
            logs = get_logs(dashboard_url, submission_id)
            if len(logs) > last_len:
                chunk = logs[last_len:]
                sys.stdout.write(chunk)
                if not only_newline_flush or chunk.endswith("\n"):
                    sys.stdout.flush()

                if out_f is not None:
                    out_f.write(chunk)
                    out_f.flush()

                last_len = len(logs)
            time.sleep(poll_s)
    except KeyboardInterrupt:
        return
    finally:
        if out_f is not None:
            out_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dashboard",
        required=True,
        help="Ray dashboard URL, e.g. http://172.16.0.46:8265",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List Ray Jobs")

    p_logs = sub.add_parser("logs", help="Fetch job logs")
    p_logs.add_argument("submission_id")
    p_logs.add_argument("--out", help="Write logs to a file")

    p_follow = sub.add_parser("follow", help="Follow job logs (polling dashboard API)")
    p_follow.add_argument("submission_id")
    p_follow.add_argument("--poll", type=float, default=2.0, help="Polling interval seconds")
    p_follow.add_argument("--out", help="Also append new logs to a file")
    p_follow.add_argument(
        "--only-newline-flush",
        action="store_true",
        help="Flush stdout only when a newline is seen (reduces overhead)",
    )

    p_stop = sub.add_parser("stop", help="Stop a running job")
    p_stop.add_argument("submission_id")

    args = ap.parse_args()
    dashboard_url = _normalize_dashboard_url(args.dashboard)

    if args.cmd == "list":
        list_jobs(dashboard_url)
        return

    if args.cmd == "logs":
        logs = get_logs(dashboard_url, args.submission_id)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(logs)
        else:
            sys.stdout.write(logs)
        return

    if args.cmd == "follow":
        follow_logs(
            dashboard_url,
            args.submission_id,
            poll_s=args.poll,
            out_path=args.out,
            only_newline_flush=args.only_newline_flush,
        )
        return

    if args.cmd == "stop":
        stop_job(dashboard_url, args.submission_id)
        return


if __name__ == "__main__":
    main()
