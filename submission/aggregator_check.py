"""
NHS EAD Forecasting Competition — pre-submission validation
Rafal Urbaniak, Basis Research Institute

Replicates the Forecast AggregatoR's submission checks against this fork.
Mirrors Forecast-AggregatoR/FAR_app/app.R lines 125-215 (validation block) and
adds an `else if(repo == "NHS-EAD-forecast")` branch with column / numeric /
sentinel checks appropriate for this competition. Re-port if upstream app.R
changes — see https://github.com/SPHERE-PPL/Forecast-AggregatoR.

The upstream AggregatoR fetches files via the GitHub Contents API, not git
clone. This script does the same so it catches the silent failure modes a
local-disk check would miss: a file that's gitignored, in Git LFS, or larger
than the API's 1 MB inline-content limit.

Usage:
    uv run python submission/aggregator_check.py rfl-urbaniak rafal-basis
    uv run python submission/aggregator_check.py --local
    uv run python submission/aggregator_check.py --local --skip-mse   # pre-20-June

Exits 0 on PASS, 1 on FAIL (errors written to stderr).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ── NHS-EAD competition contract ─────────────────────────────────────────────
# The upstream AggregatoR (app.R lines 142-178) hardcodes column lists and a
# numeric regex per competition. These are the equivalents for NHS-EAD-forecast,
# inferred from the rules in the contest README:
#   - 173 sliding 10-day forecast windows (Oct 2025 - Mar 2026)
#   - MSE evaluated separately for days 1-5 and days 6-10
#   - -9999 is the sentinel for missing values in the development dataset

PRED_MATRIX_FILE = "pred_matrix.csv"
PRED_MATRIX_COLS = ["forecast_id"] + [f"day_{i}" for i in range(1, 11)]
PRED_MATRIX_NUMERIC_COLS = [f"day_{i}" for i in range(1, 11)]
PRED_MATRIX_ROWS = 173

MSE_SUMMARY_FILE = "mse_summary.csv"
MSE_SUMMARY_COLS = ["forecast_id", "mse_1_5", "mse_6_10"]
MSE_SUMMARY_NUMERIC_COLS = ["mse_1_5", "mse_6_10"]

# Exact regex from app.R line 169: `^-?\d+(\.\d+)?$`
# Note: this rejects scientific notation (e.g. "1.5e-3"). pandas.to_csv writes
# plain decimals for predictions in our value range, so this is fine.
NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")

SENTINEL_VALUE = -9999


def _fetch_github(owner: str, repo: str, path: str) -> str:
    """Fetch submission/{path} from the GitHub Contents API and return decoded text."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/submission/{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    if token := os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(f"{path} not found at {url}") from e
        if e.code == 403 and "rate limit" in (e.read().decode(errors="ignore").lower()):
            raise RuntimeError("GitHub API rate-limited — set GITHUB_TOKEN env var") from e
        raise
    encoding = data.get("encoding")
    if encoding == "none":
        raise RuntimeError(
            f"{path} is too large for the Contents API (size={data.get('size')} bytes); "
            "AggregatoR will see this as truncated"
        )
    if encoding != "base64":
        raise RuntimeError(f"unexpected encoding {encoding!r} for {path}")
    return base64.b64decode(data["content"]).decode("utf-8")


def _read_local(path: str) -> str:
    p = Path(__file__).parent / path
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return p.read_text()


def _check_csv(
    content: str,
    file_label: str,
    expected_cols: list[str],
    numeric_cols: list[str],
    expected_rows: int,
) -> list[str]:
    """Return a list of error strings (empty if all checks pass)."""
    errors: list[str] = []

    # Mirror app.R lines 150-156: split on \n, strip \r, parse as CSV.
    lines = [line.rstrip("\r") for line in content.split("\n") if line.strip()]
    if not lines:
        return [f"{file_label}: empty file"]

    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    header = rows[0]
    body = rows[1:]

    # Column-name check (app.R line 164-166)
    missing = [c for c in expected_cols if c not in header]
    if missing:
        errors.append(f"{file_label}: incorrect column names — missing {missing}")
        return errors  # downstream checks unsafe without correct columns

    # Row-count check — NHS-EAD-specific addition (173 windows is in contest rules)
    if len(body) != expected_rows:
        errors.append(f"{file_label}: expected {expected_rows} rows, got {len(body)}")

    # Numeric + sentinel checks (app.R lines 169-176)
    col_idx = {c: header.index(c) for c in numeric_cols}
    non_numeric: list[tuple[int, str, str]] = []
    has_sentinel = False
    for r_idx, row in enumerate(body, start=2):  # 1-indexed plus header
        for c in numeric_cols:
            cell = row[col_idx[c]] if col_idx[c] < len(row) else ""
            if not NUMERIC_RE.match(cell):
                non_numeric.append((r_idx, c, cell))
            elif float(cell) == SENTINEL_VALUE:
                has_sentinel = True
    if non_numeric:
        head = ", ".join(f"row {r} col {c}={v!r}" for r, c, v in non_numeric[:3])
        errors.append(f"{file_label}: forecast column is not numeric — {head}")
    if has_sentinel:
        errors.append(f"{file_label}: forecast column contains -9999")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("owner", nargs="?", help="GitHub owner (omit with --local)")
    ap.add_argument("repo", nargs="?", help="GitHub repo (omit with --local)")
    ap.add_argument(
        "--local", action="store_true",
        help="read CSVs from the local submission/ folder instead of the GitHub API",
    )
    ap.add_argument(
        "--skip-mse", action="store_true",
        help="skip mse_summary.csv (deferred until assessment actuals are released)",
    )
    args = ap.parse_args()

    if not args.local and not (args.owner and args.repo):
        ap.error("owner and repo are required unless --local is used")

    if args.local:
        fetch = _read_local
        source = f"local submission/ at {Path(__file__).parent}"
    else:
        fetch = lambda p: _fetch_github(args.owner, args.repo, p)
        source = f"github.com/{args.owner}/{args.repo}/contents/submission"
    print(f"Aggregator check — source: {source}")

    all_errors: list[str] = []

    # pred_matrix.csv — required
    try:
        content = fetch(PRED_MATRIX_FILE)
        all_errors.extend(_check_csv(
            content, PRED_MATRIX_FILE,
            PRED_MATRIX_COLS, PRED_MATRIX_NUMERIC_COLS, PRED_MATRIX_ROWS,
        ))
    except FileNotFoundError as e:
        all_errors.append(f"{PRED_MATRIX_FILE}: {e}")

    # mse_summary.csv — written only after assessment actuals are released
    if not args.skip_mse:
        try:
            content = fetch(MSE_SUMMARY_FILE)
            all_errors.extend(_check_csv(
                content, MSE_SUMMARY_FILE,
                MSE_SUMMARY_COLS, MSE_SUMMARY_NUMERIC_COLS, PRED_MATRIX_ROWS,
            ))
        except FileNotFoundError:
            print(
                f"{MSE_SUMMARY_FILE}: not present — skipping. Pass --skip-mse to silence "
                "this notice pre-20-June.",
                file=sys.stderr,
            )

    if all_errors:
        print(f"\nAGGREGATOR CHECK: FAIL — {len(all_errors)} error(s)", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("\nAGGREGATOR CHECK: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
