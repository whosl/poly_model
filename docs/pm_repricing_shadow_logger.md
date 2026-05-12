# PM Repricing No-trade Shadow Logger Design

## Goal

Validate whether the offline PM repricing edge is observable in a real-time
system before any paper or live order placement.  The logger records signals,
quotes, latency, and simulated outcomes only.  It never submits orders.

## Inputs

### Binance streams

- `bookTicker`
- `aggTrade`

### Polymarket streams

- orderbook snapshots / deltas
- `price_change` updates

## Signal record

Write to:

`data/shadow/repricing_signals/date=YYYY-MM-DD/*.parquet`

Fields:

- `signal_id`
- `market_id`
- `sample_ts`
- `local_signal_ts`
- `model_version`
- `p_up`
- `p_down`
- `p_flat`
- `direction`
- `threshold`
- `yes_bid`
- `yes_ask`
- `no_bid`
- `no_ask`
- `yes_spread`
- `no_spread`
- `quote_age`
- `formula_p_yes`
- `yes_mid`
- `no_mid`
- `btc_mid`
- `btc_return_1s`
- `btc_return_5s`

## Latency instrumentation

Record all timestamps using monotonic local clock plus exchange event time when
available:

- `data_receive_ts`
- `feature_ready_ts`
- `model_infer_start_ts`
- `model_infer_end_ts`
- `signal_emit_ts`
- `quote_update_ts`
- `simulated_entry_ts`
- `simulated_exit_ts`

Derived fields:

- `feature_latency_ms = feature_ready_ts - data_receive_ts`
- `model_latency_ms = model_infer_end_ts - model_infer_start_ts`
- `emit_latency_ms = signal_emit_ts - sample_ts`
- `quote_age_ms = signal_emit_ts - quote_update_ts`

## Simulated outcomes

Write to:

`data/shadow/repricing_outcomes/date=YYYY-MM-DD/*.parquet`

For every signal, simulate entry latencies:

- 0 ms
- 250 ms
- 500 ms
- 1 s
- 2 s

For every entry, simulate exit horizons:

- 1 s
- 5 s
- 10 s
- 30 s

Execution convention:

- UP: buy YES at entry `yes_ask`, exit at future `yes_bid`
- DOWN: buy NO at entry `no_ask`, exit at future `no_bid`

Outcome fields:

- `signal_id`
- `market_id`
- `direction`
- `threshold`
- `entry_latency_ms`
- `exit_horizon_seconds`
- `entry_quote_ts`
- `entry_price`
- `exit_quote_ts`
- `exit_price`
- `pnl`
- `roi`
- `entry_quote_available`
- `exit_quote_available`
- `entry_quote_stale`
- `exit_quote_stale`
- `entry_crossed_quote`
- `exit_crossed_quote`

## Daily report

Write:

`reports/shadow/repricing_daily_YYYY-MM-DD.md`

Sections:

1. signal count
2. simulated PnL by latency
3. simulated PnL by threshold
4. quote availability
5. average quote lifetime after signal
6. percentage of signals where entry ask disappeared within:
   - 250 ms
   - 500 ms
   - 1 s
7. by-market summary
8. by-time-to-expiry summary
9. top drawdowns
10. data quality warnings

## Pass criteria before paper trading

No-trade shadow logging should run for multiple days.  Before paper trading,
the strategy should show:

- positive simulated PnL at 1 s latency after conservative cost
- stable results across dates
- no single market contributing most PnL
- acceptable quote availability
- no evidence that entry quotes disappear immediately after signal
- robust results under first-signal and cooldown modes

## Explicit non-goals

- No real orders
- No paper orders requiring exchange state
- No fill probability assumptions treated as facts
- No terminal model conclusions
- No btc_direction training
