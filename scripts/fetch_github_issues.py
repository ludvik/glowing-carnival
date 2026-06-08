#!/usr/bin/env python3
"""Fetch a stable GitHub issues corpus for evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


API_ROOT = "https://api.github.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch non-PR GitHub issues and write a stable JSON corpus."
    )
    parser.add_argument("--repo", default="digitalocean/doctl", help="owner/repo")
    parser.add_argument(
        "--output",
        default="data/doctl_issues.json",
        help="Path to write the JSON corpus.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional seconds to sleep between paginated requests.",
    )
    return parser.parse_args()


def github_request(url: str, token: str | None) -> tuple[list[dict[str, Any]], str | None]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "glowing-carnival-eval-corpus-fetcher",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            link = response.headers.get("Link")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub API request failed: {exc}") from exc

    return json.loads(body), next_link(link)


def next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section:
            continue
        start = section.find("<")
        end = section.find(">")
        if start != -1 and end != -1 and end > start:
            return section[start + 1 : end]
    return None


def simplify_issue(issue: dict[str, Any]) -> dict[str, Any]:
    labels = [
        {
            "name": label.get("name"),
            "color": label.get("color"),
            "description": label.get("description"),
        }
        for label in issue.get("labels", [])
    ]
    labels.sort(key=lambda label: (label.get("name") or "").lower())

    return {
        "id": issue["id"],
        "number": issue["number"],
        "state": issue["state"],
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "labels": labels,
        "user_login": (issue.get("user") or {}).get("login"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "comments": issue.get("comments", 0),
        "html_url": issue.get("html_url"),
        "api_url": issue.get("url"),
    }


def fetch_issues(repo: str, token: str | None, sleep_seconds: float) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"state": "all", "per_page": "100"})
    url: str | None = f"{API_ROOT}/repos/{repo}/issues?{query}"
    issues: list[dict[str, Any]] = []
    page = 0

    while url:
        page += 1
        batch, url = github_request(url, token)
        page_issues = [item for item in batch if "pull_request" not in item]
        issues.extend(simplify_issue(issue) for issue in page_issues)
        print(
            f"Fetched page {page}: {len(page_issues)} issues "
            f"({len(batch) - len(page_issues)} PRs skipped)",
            file=sys.stderr,
        )
        if url and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    issues.sort(key=lambda issue: issue["number"])
    return issues


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    token = os.environ.get("GITHUB_TOKEN")

    issues = fetch_issues(args.repo, token, args.sleep)
    payload = {
        "source": {
            "repo": args.repo,
            "api": "GitHub REST issues endpoint",
            "state": "all",
            "pull_requests_excluded": True,
        },
        "issue_count": len(issues),
        "issues": issues,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(issues)} issues to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
