# Privacy Policy

**We collect no data.**

ML Equity Intelligence is a local research tool. It is designed to run entirely on
your own machine, and it does not collect, store, transmit, or sell any personal or
usage data.

## What we do NOT do

- **No accounts, no login.** The dashboard is open locally; you are not asked to sign
  in or create an account.
- **No tracking or analytics.** There are no cookies, pixels, ad networks, or
  behavioural analytics. Streamlit's own usage statistics are disabled
  (`browser.gatherUsageStats = false`).
- **No data sold or shared.** Nothing you type or generate leaves your computer to us
  or to any third party.
- **No paid APIs required.** The default setup uses only free, public data sources.

## What data is used, and where it goes

- **Market data** is fetched on demand from public sources (Yahoo Finance via
  `yfinance`) only when you run an analysis. It is used in memory to compute rankings
  and backtests and is not persisted by the app beyond optional local report files you
  choose to export.
- **Optional news evidence** is fetched from public NYT/Economist RSS feeds only if you
  explicitly enable it.
- **Local secrets** (an optional API bearer token and, historically, a dashboard
  password) live only in a local, gitignored `.env` file on your machine. They are
  access controls for services *you* run — they are never sent to us.

## Network exposure

By default the dashboard and API bind to `127.0.0.1` (localhost) and are not reachable
from your network or the internet. You would have to intentionally override the bind
address to expose them.

## Contact

This is open-source software you run yourself. There is no operator collecting data on
the other end.
