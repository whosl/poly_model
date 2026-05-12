"""Refresh read-only live Polymarket market metadata for the shadow logger.

This script does not require or use any trading credentials. It queries the
public Gamma API for time-based BTC up/down markets and writes a compact parquet
with the columns consumed by run_pm_repricing_shadow.py.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl


UTC = timezone.utc
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-path", default="configs/shadow_market_meta.parquet")
    p.add_argument("--report-path", default="reports/shadow/live_market_meta_refresh.md")
    p.add_argument("--asset", default="btc", choices=["btc"])
    p.add_argument("--interval-minutes", type=int, default=15)
    p.add_argument("--lookback-minutes", type=int, default=30)
    p.add_argument("--lookahead-hours", type=float, default=24.0)
    p.add_argument("--sleep-seconds", type=float, default=0.05)
    p.add_argument("--timeout-seconds", type=float, default=15.0)
    return p.parse_args()


def utc_now() -> datetime:
    return datetime.now(UTC)


def floor_dt(dt: datetime, seconds: int) -> datetime:
    ts = int(dt.timestamp())
    return datetime.fromtimestamp((ts // seconds) * seconds, UTC)


def maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def gamma_get_market_by_slug(slug: str, timeout_seconds: float) -> dict[str, Any] | None:
    params = urllib.parse.urlencode({"slug": slug})
    url = f"{GAMMA_MARKETS_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "pm-shadow-refresh/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if isinstance(data, list) and data:
        return data[0]
    return None


def slugs_for_btc_15m(now: datetime, lookback_minutes: int, lookahead_hours: float) -> list[tuple[str, datetime, datetime]]:
    interval = 15 * 60
    start = floor_dt(now - timedelta(minutes=lookback_minutes), interval)
    stop = floor_dt(now + timedelta(hours=lookahead_hours), interval)
    rows: list[tuple[str, datetime, datetime]] = []
    cur = start
    while cur <= stop:
        start_ts = int(cur.timestamp())
        # Polymarket currently uses btc-updown-15m-<window_start_epoch>.
        rows.append((f"btc-updown-15m-{start_ts}", cur, cur + timedelta(minutes=15)))
        cur += timedelta(minutes=15)
    return rows


def market_to_row(market: dict[str, Any], default_start: datetime, default_end: datetime) -> dict[str, Any] | None:
    outcomes = maybe_json(market.get("outcomes"))
    clob = maybe_json(market.get("clobTokenIds"))
    if not isinstance(outcomes, list) or not isinstance(clob, list) or len(outcomes) != len(clob):
        return None

    lower = [str(x).strip().lower() for x in outcomes]
    up_idx = next((i for i, x in enumerate(lower) if x in {"up", "yes"}), None)
    down_idx = next((i for i, x in enumerate(lower) if x in {"down", "no"}), None)
    if up_idx is None or down_idx is None:
        return None

    end_raw = market.get("endDate") or market.get("end_date")
    start_raw = market.get("startDate") or market.get("start_date")
    try:
        end_ts = datetime.fromisoformat(str(end_raw).replace("Z", "+00:00")).astimezone(UTC) if end_raw else default_end
    except Exception:
        end_ts = default_end
    try:
        start_ts = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00")).astimezone(UTC) if start_raw else default_start
    except Exception:
        start_ts = default_start
    # Gamma startDate is often market creation time. For fixed 15m markets, the
    # trading/settlement window is encoded in the slug, so use default_start
    # when Gamma's startDate is far from the 15m window.
    if abs((start_ts - default_start).total_seconds()) > 3600:
        start_ts = default_start

    return {
        "market_id": str(market.get("conditionId") or market.get("condition_id") or market.get("id")),
        "yes_asset_id": str(clob[up_idx]),
        "no_asset_id": str(clob[down_idx]),
        "market_start_ts": start_ts,
        "market_end_ts": end_ts,
        "question": str(market.get("question") or ""),
        "slug": str(market.get("slug") or ""),
        "gamma_market_id": str(market.get("id") or ""),
        "outcomes": json.dumps(outcomes),
        "active": bool(market.get("active")),
        "closed": bool(market.get("closed")),
        "archived": bool(market.get("archived", False)),
        "enable_order_book": bool(market.get("enableOrderBook", True)),
    }


def write_report(path: str | Path, rows: list[dict[str, Any]], misses: list[str], started_at: datetime) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    lines = [
        "# Live PM Market Meta Refresh\n\n",
        "SHADOW MODE ONLY - NO ORDERS WILL BE PLACED\n\n",
        f"- started_at_utc: `{started_at.isoformat()}`\n",
        f"- finished_at_utc: `{now.isoformat()}`\n",
        f"- market_count: `{len(rows)}`\n",
        f"- missing_slug_count: `{len(misses)}`\n\n",
    ]
    if rows:
        df = pl.DataFrame(rows)
        lines.append("## Time range\n\n")
        lines.append(f"- min_start: `{df['market_start_ts'].min()}`\n")
        lines.append(f"- max_end: `{df['market_end_ts'].max()}`\n\n")
        lines.append("## Markets\n\n")
        lines.append("| start | end | slug | question |\n")
        lines.append("| --- | --- | --- | --- |\n")
        for r in rows[:80]:
            lines.append(f"| {r['market_start_ts']} | {r['market_end_ts']} | {r['slug']} | {r['question']} |\n")
    if misses:
        lines.append("\n## Missing slugs sample\n\n")
        for slug in misses[:50]:
            lines.append(f"- `{slug}`\n")
    p.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = utc_now()
    if args.asset != "btc" or args.interval_minutes != 15:
        raise RuntimeError("Only BTC 15m up/down live market refresh is currently implemented")

    rows: list[dict[str, Any]] = []
    misses: list[str] = []
    for slug, start_ts, end_ts in slugs_for_btc_15m(started, args.lookback_minutes, args.lookahead_hours):
        try:
            market = gamma_get_market_by_slug(slug, args.timeout_seconds)
        except Exception as exc:
            misses.append(f"{slug} ({type(exc).__name__})")
            continue
        if not market:
            misses.append(slug)
            continue
        row = market_to_row(market, start_ts, end_ts)
        if row is None:
            misses.append(f"{slug} (parse_failed)")
            continue
        if row["market_end_ts"] >= started - timedelta(minutes=args.lookback_minutes):
            rows.append(row)
        time.sleep(args.sleep_seconds)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pl.DataFrame(rows).unique(subset=["market_id"], keep="first").sort("market_start_ts")
        df.write_parquet(out)
    else:
        # Keep schema stable even if no markets are found.
        df = pl.DataFrame(
            schema={
                "market_id": pl.Utf8,
                "yes_asset_id": pl.Utf8,
                "no_asset_id": pl.Utf8,
                "market_start_ts": pl.Datetime(time_zone="UTC"),
                "market_end_ts": pl.Datetime(time_zone="UTC"),
                "question": pl.Utf8,
                "slug": pl.Utf8,
                "gamma_market_id": pl.Utf8,
                "outcomes": pl.Utf8,
                "active": pl.Boolean,
                "closed": pl.Boolean,
                "archived": pl.Boolean,
                "enable_order_book": pl.Boolean,
            }
        )
        df.write_parquet(out)
    write_report(args.report_path, rows, misses, started)
    print(f"Wrote {out} with {len(rows)} markets")
    print(f"Wrote {args.report_path}")


if __name__ == "__main__":
    main()
