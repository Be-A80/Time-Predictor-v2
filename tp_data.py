"""
Time Predictor – Datenschicht (übernommen aus pcp_data.py, ergänzt um 2x-Timeframes wie 6d, 32h, 4w, 4M
und die Hilfsfunktion double_tf()).
Coin-Universum (CoinGecko), Hybrid-Datenquellen und Timeframe-Resampling.

Quellen-Reihenfolge im Modus "auto":
  Binance (nur wenn erreichbar – auf Streamlit Cloud/US-Servern geblockt)
  → yfinance (Batch-Download, tiefe Historie)
  → Coinbase → Kraken (Fallback pro Coin)
Jede geladene Reihe wird gegen den CoinGecko-Preis plausibilisiert (falsche Ticker fallen so auf).
"""
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    import ccxt
except ImportError:  # ccxt optional – dann nur yfinance
    ccxt = None

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

# ==========================================
# TIMEFRAMES
# ==========================================
# Basis-Intervalle, aus denen alle Timeframes abgeleitet werden
BASE_SPECS = {
    "5m": {"dur": pd.Timedelta(minutes=5), "yf": "5m", "yf_max_days": 59, "cap": 16000},
    "15m": {"dur": pd.Timedelta(minutes=15), "yf": "15m", "yf_max_days": 59, "cap": 5600},
    "30m": {"dur": pd.Timedelta(minutes=30), "yf": "30m", "yf_max_days": 59, "cap": 2800},
    "1h": {"dur": pd.Timedelta(hours=1), "yf": "1h", "yf_max_days": 729, "cap": 17400},
    "1d": {"dur": pd.Timedelta(days=1), "yf": "1d", "yf_max_days": None, "cap": 4500},
}


def _tf(base, dur, tv, rule=None, monday=False, months=None):
    return {"base": base, "dur": dur, "tv": tv, "rule": rule, "monday": monday, "months": months}


TF_SPECS = {
    "5m": _tf("5m", pd.Timedelta(minutes=5), "5"),
    "10m": _tf("5m", pd.Timedelta(minutes=10), "10", rule="10min"),
    "15m": _tf("15m", pd.Timedelta(minutes=15), "15"),
    "30m": _tf("30m", pd.Timedelta(minutes=30), "30"),
    "1h": _tf("1h", pd.Timedelta(hours=1), "60"),
    "2h": _tf("1h", pd.Timedelta(hours=2), "120", rule="2h"),
    "4h": _tf("1h", pd.Timedelta(hours=4), "240", rule="4h"),
    "8h": _tf("1h", pd.Timedelta(hours=8), "480", rule="8h"),
    "12h": _tf("1h", pd.Timedelta(hours=12), "720", rule="12h"),
    "16h": _tf("1h", pd.Timedelta(hours=16), "960", rule="16h"),
    "32h": _tf("1h", pd.Timedelta(hours=32), "1920", rule="32h"),
    "1d": _tf("1d", pd.Timedelta(days=1), "D"),
    "2d": _tf("1d", pd.Timedelta(days=2), "2D", rule="2D"),
    "3d": _tf("1d", pd.Timedelta(days=3), "3D", rule="3D"),
    "4d": _tf("1d", pd.Timedelta(days=4), "4D", rule="4D"),
    "6d": _tf("1d", pd.Timedelta(days=6), "6D", rule="6D"),
    "8d": _tf("1d", pd.Timedelta(days=8), "8D", rule="8D"),
    "16d": _tf("1d", pd.Timedelta(days=16), "16D", rule="16D"),
    "32d": _tf("1d", pd.Timedelta(days=32), "32D", rule="32D"),
    "1w": _tf("1d", pd.Timedelta(days=7), "W", rule="7D", monday=True),
    "2w": _tf("1d", pd.Timedelta(days=14), "2W", rule="14D", monday=True),
    "4w": _tf("1d", pd.Timedelta(days=28), "4W", rule="28D", monday=True),
    "1M": _tf("1d", pd.Timedelta(days=30.44), "M", months=1),
    "2M": _tf("1d", pd.Timedelta(days=60.88), "2M", months=2),
    "4M": _tf("1d", pd.Timedelta(days=121.76), "4M", months=4),
}
TF_ORDER = sorted(TF_SPECS, key=lambda k: TF_SPECS[k]["dur"], reverse=True)  # groß → klein
MONDAY_ORIGIN = pd.Timestamp("1970-01-05")  # ein Montag – fester Anker für Wochenkerzen


def tf_dur(tf):
    return TF_SPECS[tf]["dur"]


def double_tf(tf):
    """Timeframe mit exakt doppelter Kerzendauer (1d → 2d, 3d → 6d, 1w → 2w …) oder None."""
    target = 2 * TF_SPECS[tf]["dur"]
    for k, s in TF_SPECS.items():
        if abs((s["dur"] - target) / target) < 0.01:
            return k
    return None


def tf_sort(tfs):
    return sorted(tfs, key=TF_ORDER.index)


def resample_ohlcv(df, tf):
    """
    Leitet einen Timeframe aus seinem Basis-Intervall ab.
    WICHTIG: fester Ursprung (Epoch bzw. Montag/Monatsanfang). Mit origin='start' würden sich die
    Kerzengrenzen bei jedem neuen Datenfenster verschieben – und damit Pivots und Phase.
    """
    spec = TF_SPECS[tf]
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)
    if spec["rule"] is None and spec["months"] is None:
        return df.reset_index(drop=True)

    d = df.set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if spec["months"]:
        m = spec["months"]
        key = (d.index.year * 12 + d.index.month - 1) // m
        g = d.groupby(key).agg(agg)
        starts = [pd.Timestamp(year=int(k * m // 12), month=int(k * m % 12) + 1, day=1) for k in g.index]
        g.index = pd.DatetimeIndex(starts)
    else:
        origin = MONDAY_ORIGIN if spec["monday"] else "epoch"
        g = d.resample(spec["rule"], origin=origin, label="left", closed="left").agg(agg)
    g = g.dropna(subset=["open", "high", "low", "close"])
    # erste Kerze ist meist unvollständig (Daten beginnen mitten in der Periode)
    if len(g) > 1 and g.index[0] < d.index[0]:
        g = g.iloc[1:]
    g.index.name = "timestamp"
    return g.reset_index()[OHLCV_COLS]


def base_needs(tfs, limit):
    """Wie viele Basis-Kerzen werden für die gewünschten TFs (je `limit` Kerzen) benötigt?"""
    needs = {}
    for tf in tfs:
        s = TF_SPECS[tf]
        b = s["base"]
        ratio = s["dur"] / BASE_SPECS[b]["dur"]
        n = int((limit + 10) * ratio) + 5
        needs[b] = min(BASE_SPECS[b]["cap"], max(needs.get(b, 0), n))
    return needs


# ==========================================
# COIN-UNIVERSUM (CoinGecko)
# ==========================================
CG_URL = "https://api.coingecko.com/api/v3/coins/markets"
STABLES = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "PYUSD", "USDD", "FRAX", "USDP", "GUSD", "BUSD",
    "USD0", "USDX", "EURC", "EURS", "RLUSD", "USDY", "USDTB", "BUIDL", "USD1", "USDF", "SUSDE", "SUSDS",
    "XAUT", "PAXG", "USDG", "USYC", "LUSD", "CRVUSD", "GHO", "USDB", "USDL", "BFUSD", "USDA",
}
EXCLUDE_SYMBOLS = {
    "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "CBBTC", "RETH", "METH", "EZETH", "WBETH", "LBTC", "SOLVBTC",
    "CLBTC", "BTCB", "JITOSOL", "MSOL", "BNSOL", "RSETH", "OSETH", "SWETH", "TBTC", "CBETH", "FBTC", "LSETH",
    "EETH", "BSC-USD", "WBNB", "WHYPE", "KHYPE", "STHYPE", "JUPSOL", "BBSOL", "WTRX", "SAVAX", "STKAAVE",
}
EXCLUDE_NAME_KEYWORDS = ("wrapped", "staked", "bridged", "restaked", "liquid staking", "binance-peg", "tokenized")

FALLBACK_COINS = [
    "BTC", "ETH", "XRP", "BNB", "SOL", "DOGE", "TRX", "ADA", "HYPE", "LINK", "AVAX", "SUI", "XLM", "BCH", "HBAR",
    "TON", "LTC", "DOT", "XMR", "UNI", "PEPE", "AAVE", "NEAR", "APT", "ICP", "ETC", "ONDO", "TAO", "POL", "ARB",
    "KAS", "ATOM", "ALGO", "RENDER", "FIL", "VET", "OP", "ENA", "WLD", "SEI", "INJ", "TIA", "IMX", "STX", "GRT",
    "JUP", "BONK", "FET", "QNT", "LDO", "XTZ", "THETA", "SAND", "MANA", "AXS", "FLOW", "EGLD", "CRV", "RUNE",
    "GALA", "CHZ", "APE", "PYTH", "WIF", "FLOKI", "JASMY", "SNX", "COMP", "DYDX", "ZEC", "IOTA", "NEO", "KAIA",
    "MINA", "CFX", "BEAM", "AR", "ROSE", "ZRO", "STRK", "PENDLE", "ENS", "1INCH", "CAKE", "RAY", "TRUMP",
    "VIRTUAL", "S", "BERA", "PENGU", "SHIB", "CRO", "OKB", "LEO", "BGB", "XDC", "FLR", "GT", "SKY",
]


def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _coingecko_markets(per_page=250):
    headers = {"accept": "application/json"}
    key = _secret("COINGECKO_API_KEY")  # optional (Demo-Key) in .streamlit/secrets.toml
    if key:
        headers["x-cg-demo-api-key"] = key
    r = requests.get(
        CG_URL,
        params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": per_page, "page": 1,
                "sparkline": "false"},
        headers=headers, timeout=15,
    )
    r.raise_for_status()
    return r.json()


def fetch_top_coins(max_n=100):
    """Top-N Coins nach Marktkapitalisierung (ohne Stablecoins/Wrapped/Staked). Fallback: statische Liste."""
    try:
        raw = _coingecko_markets()
        coins, seen = [], set()
        for c in raw:
            sym = str(c.get("symbol", "")).upper()
            name = str(c.get("name", ""))
            price = c.get("current_price")
            if not sym or sym in seen or sym in STABLES or sym in EXCLUDE_SYMBOLS:
                continue
            if not sym.replace("-", "").isalnum() or "_" in sym:  # z. B. tokenisierte Kredite (FIGR_HELOC)
                continue
            if any(k in name.lower() for k in EXCLUDE_NAME_KEYWORDS):
                continue
            if price and 0.97 <= float(price) <= 1.03 and "usd" in name.lower():
                continue
            seen.add(sym)
            coins.append({"symbol": sym, "name": name, "id": c.get("id"),
                          "rank": c.get("market_cap_rank"), "price": float(price) if price else None})
            if len(coins) >= max_n:
                break
        if len(coins) >= 10:
            return coins, "CoinGecko"
        raise ValueError("zu wenige Coins")
    except Exception as e:
        return ([{"symbol": s, "name": s, "id": None, "rank": i + 1, "price": None}
                 for i, s in enumerate(FALLBACK_COINS[:max_n])],
                f"statische Fallback-Liste ({type(e).__name__})")


# ==========================================
# NORMALISIERUNG
# ==========================================
def normalize_yf(raw):
    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.drop(columns=[c for c in df.columns if str(c).lower().replace(" ", "") == "adjclose"],
                 errors="ignore")
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns
                            if str(c).strip().lower() in ("open", "high", "low", "close", "volume")})
    df = df.loc[:, ~df.columns.duplicated()]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    if any(c not in df.columns for c in ("open", "high", "low", "close")):
        return pd.DataFrame(columns=OHLCV_COLS)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return _clean(df[OHLCV_COLS])


def _clean(df):
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["high"] > 0) & (df["low"] > 0)]
    return (df.drop_duplicates(subset="timestamp", keep="last")
              .sort_values("timestamp").reset_index(drop=True))


def price_ok(df, expected, tol=0.15):
    """Plausibilitätscheck gegen CoinGecko-Preis (verhindert falsche Ticker-Zuordnung)."""
    if expected is None or df is None or df.empty:
        return df is not None and not df.empty
    last = float(df["close"].iloc[-1])
    return expected > 0 and (1 - tol) <= last / expected <= (1 + tol)


# ==========================================
# YFINANCE
# ==========================================
# Yahoo nutzt bei Namenskollisionen CoinMarketCap-IDs als Suffix. Best effort –
# falsche/fehlende Einträge werden durch den Preis-Check erkannt und fallen auf den nächsten Kandidaten zurück.
YF_OVERRIDES = {
    "UNI": "UNI7083-USD", "SUI": "SUI20947-USD", "APT": "APT21794-USD", "TAO": "TAO22974-USD",
    "PEPE": "PEPE24478-USD", "TON": "TON11419-USD", "STX": "STX4847-USD", "ARB": "ARB11841-USD",
    "IMX": "IMX10603-USD", "MNT": "MNT27075-USD", "GRT": "GRT6719-USD", "TIA": "TIA22861-USD",
    "JUP": "JUP29210-USD", "POL": "POL28321-USD", "S": "S32684-USD", "HYPE": "HYPE32196-USD",
    "TRUMP": "TRUMP35336-USD", "PI": "PI35697-USD", "BEAM": "BEAM28298-USD", "STRK": "STRK22691-USD",
    "ZRO": "ZRO26997-USD", "ENA": "ENA30171-USD", "BONK": "BONK-USD", "WIF": "WIF-USD",
}


def yf_candidates(sym):
    c = [YF_OVERRIDES[sym]] if sym in YF_OVERRIDES else []
    c.append(f"{sym}-USD")
    return list(dict.fromkeys(c))


def yf_days(base, n):
    spec = BASE_SPECS[base]
    days = int(math.ceil(n * spec["dur"] / pd.Timedelta(days=1))) + 2
    if spec["yf_max_days"] is None:
        return None if days > 3000 else days
    return min(days, spec["yf_max_days"])


@st.cache_data(ttl=600, show_spinner=False, max_entries=300)
def yf_batch(tickers: tuple, base: str, days):
    period = "max" if days is None else f"{days}d"
    try:
        raw = yf.download(list(tickers), period=period, interval=BASE_SPECS[base]["yf"], group_by="ticker",
                          progress=False, auto_adjust=False, threads=True)
    except Exception:
        return {}
    out = {}
    if raw is None or raw.empty:
        return out
    for t in tickers:
        sub = None
        if isinstance(raw.columns, pd.MultiIndex):
            for level in range(raw.columns.nlevels):
                if t in raw.columns.get_level_values(level):
                    sub = raw.xs(t, axis=1, level=level)
                    break
        elif len(tickers) == 1:
            sub = raw
        if sub is None:
            continue
        df = normalize_yf(sub)
        if not df.empty:
            out[t] = df
    return out


# ==========================================
# CCXT (Binance / Coinbase / Kraken)
# ==========================================
CCXT_IDS = {"binance": "binance", "coinbase": "coinbaseexchange", "kraken": "kraken"}
CCXT_QUOTES = {"binance": ["USDT", "USDC"], "coinbase": ["USD", "USDC", "USDT"], "kraken": ["USD", "USDT"]}
CCXT_PER_REQ = {"binance": 1000, "coinbase": 300, "kraken": 720}
SOURCE_LABEL = {"binance": "Binance", "coinbase": "Coinbase", "kraken": "Kraken", "yfinance": "yfinance"}


@st.cache_resource(show_spinner=False)
def get_exchange(src):
    if ccxt is None:
        return None
    return getattr(ccxt, CCXT_IDS[src])({"enableRateLimit": True, "timeout": 15000})


@st.cache_data(ttl=1800, show_spinner=False)
def exchange_markets(src):
    """Leere Liste, wenn die Börse nicht erreichbar ist (z. B. Binance von US-Servern: HTTP 451)."""
    ex = get_exchange(src)
    if ex is None:
        return []
    try:
        return sorted(ex.load_markets().keys())
    except Exception:
        return []


def resolve_market(src, sym):
    mk = set(exchange_markets(src))
    for q in CCXT_QUOTES[src]:
        s = f"{sym}/{q}"
        if s in mk:
            return s
    return None


@st.cache_data(ttl=600, show_spinner=False, max_entries=3000)
def fetch_ccxt_base(src, market, base, n_candles, now_key):
    ex = get_exchange(src)
    tf_ms = int(BASE_SPECS[base]["dur"].total_seconds() * 1000)
    end_ms = int(pd.Timestamp(now_key).value // 10 ** 6)
    since = end_ms - n_candles * tf_ms
    per = CCXT_PER_REQ[src]
    rows, guard = [], 0
    while since < end_ms and guard < 250:
        guard += 1
        batch = ex.fetch_ohlcv(market, timeframe=base, since=int(since), limit=per)
        if not batch:
            since += per * tf_ms  # Lücke / vor Listing → weiterspringen
            continue
        rows.extend(batch)
        nxt = batch[-1][0] + tf_ms
        if nxt <= since or src == "kraken":  # Kraken liefert ohnehin nur die letzten 720 Kerzen
            break
        since = nxt
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
    return _clean(df[OHLCV_COLS].astype({c: float for c in OHLCV_COLS[1:]}))


# ==========================================
# HYBRID-ABRUF
# ==========================================
SOURCE_MODES = {
    "Auto (Binance → yfinance → Coinbase → Kraken)": "auto",
    "yfinance": "yfinance",
    "Binance": "binance",
    "Coinbase": "coinbase",
    "Kraken": "kraken",
}


def source_chain(mode):
    if mode == "auto":
        chain = ["binance"] if exchange_markets("binance") else []
        return chain + ["yfinance", "coinbase", "kraken"]
    return [mode]


def _fetch_base_one(sym, base, n, src, expected, now_key):
    if src == "yfinance":
        for t in yf_candidates(sym):
            d = yf_batch((t,), base, yf_days(base, n)).get(t)
            if d is not None and price_ok(d, expected):
                return d.tail(n).reset_index(drop=True), f"yfinance:{t}"
        return None, None
    if not exchange_markets(src):
        return None, None
    m = resolve_market(src, sym)
    if m is None:
        return None, None
    d = fetch_ccxt_base(src, m, base, n, now_key)
    if d is not None and price_ok(d, expected):
        return d, f"{SOURCE_LABEL[src]}:{m}"
    return None, None


def build_tf_frames(base_frames, tfs):
    out = {}
    for tf in tfs:
        d = base_frames.get(TF_SPECS[tf]["base"])
        if d is None or d.empty:
            continue
        r = resample_ohlcv(d, tf)
        if len(r) > 0:
            out[tf] = r
    return out


def get_coin_data(sym, tfs, mode, expected_price, limit, now_key):
    """Alle gewünschten TFs eines Coins. Rückgabe: ({tf: df}, {basis: quelle})."""
    base_frames, used = {}, {}
    for base, n in base_needs(tfs, limit).items():
        for src in source_chain(mode):
            try:
                d, label = _fetch_base_one(sym, base, n, src, expected_price, now_key)
            except Exception:
                d, label = None, None
            if d is not None and not d.empty:
                base_frames[base], used[base] = d, label
                break
    return build_tf_frames(base_frames, tfs), used


def get_many_coins_data(coins, tfs, mode, limit, now_key, progress=None):
    """Batch-Variante für den Screener: yfinance in 25er-Paketen, ccxt pro Coin."""
    needs = base_needs(tfs, limit)
    bases = {c["symbol"]: {} for c in coins}
    used = {c["symbol"]: {} for c in coins}
    chain = source_chain(mode)
    steps, done = max(1, len(chain) * len(needs)), 0

    for src in chain:
        for base, n in needs.items():
            todo = [c for c in coins if base not in bases[c["symbol"]]]
            done += 1
            if progress:
                progress(min(done / steps, 1.0), f"{SOURCE_LABEL[src]}: {base}-Daten für {len(todo)} Coins …")
            if not todo:
                continue
            if src == "yfinance":
                for rnd in range(2):  # Kandidat 1 (Override), Kandidat 2 (SYM-USD)
                    tick_map = {}
                    for c in todo:
                        cand = yf_candidates(c["symbol"])
                        if len(cand) > rnd:
                            tick_map[cand[rnd]] = c
                    ticks = list(tick_map)
                    for i in range(0, len(ticks), 25):
                        chunk = tuple(ticks[i:i + 25])
                        try:
                            data = yf_batch(chunk, base, yf_days(base, n))
                        except Exception:
                            data = {}
                        for t, d in data.items():
                            c = tick_map[t]
                            if base not in bases[c["symbol"]] and price_ok(d, c["price"]):
                                bases[c["symbol"]][base] = d.tail(n).reset_index(drop=True)
                                used[c["symbol"]][base] = f"yfinance:{t}"
                    todo = [c for c in todo if base not in bases[c["symbol"]]]
            else:
                if not exchange_markets(src):
                    continue
                for j, c in enumerate(todo):
                    if progress and j % 5 == 0:
                        progress(min(done / steps, 1.0), f"{SOURCE_LABEL[src]}: {base} {c['symbol']} ({j + 1}/{len(todo)})")
                    try:
                        d, label = _fetch_base_one(c["symbol"], base, n, src, c["price"], now_key)
                    except Exception:
                        d, label = None, None
                    if d is not None and not d.empty:
                        bases[c["symbol"]][base], used[c["symbol"]][base] = d, label

    frames = {sym: build_tf_frames(b, tfs) for sym, b in bases.items()}
    return frames, used


# ==========================================
# KALENDER: Krypto (24/7) vs. Aktien (nur Handelstage)
# ==========================================
# Aktien-TFs aus Tageskerzen: "Nd" = N HANDELStage (Wochenenden zählen nicht), Wochen/Monate kalendarisch
STOCK_NDAYS = {"1d": 1, "2d": 2, "3d": 3, "4d": 4, "6d": 6, "8d": 8, "16d": 16, "32d": 32}
STOCK_ORIGIN = np.datetime64("2000-01-03")        # Montag – fester Anker für N-Handelstage-Kerzen
STOCK_TFS = [tf for tf in TF_ORDER if TF_SPECS[tf]["dur"] >= pd.Timedelta(days=1)]


def tdays_per_bar(tf):
    if tf in STOCK_NDAYS:
        return STOCK_NDAYS[tf]
    if TF_SPECS[tf]["months"]:
        return 21 * TF_SPECS[tf]["months"]
    return int(round(TF_SPECS[tf]["dur"].days * 5 / 7))


def future_ts(last, steps, tf, calendar, delta):
    """Zeitpunkt `steps` Kerzen nach der letzten Kerze `last` (steps darf gebrochen sein)."""
    if calendar == "stock":
        if tf in STOCK_NDAYS:
            b = steps * STOCK_NDAYS[tf]
            whole = int(math.floor(b))
            return pd.Timestamp(last) + pd.offsets.BDay(whole) + pd.Timedelta(days=b - whole)
        return pd.Timestamp(last) + steps * TF_SPECS[tf]["dur"]
    return pd.Timestamp(last) + steps * delta


def prep_stock_daily(df):
    """Tageskerzen auf Kalendertag normieren (yfinance liefert Mitternacht der Börsen-Zeitzone in UTC)."""
    if df is None or df.empty:
        return df
    d = df.copy()
    d["timestamp"] = d["timestamp"].dt.round("1D")
    d = d[d["timestamp"].dt.dayofweek < 5]
    return d.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def resample_stock(daily, tf):
    """Aktien-Timeframes mit expliziter Spalte 'end' (Kerzenende). Nd = N Handelstage ab festem Anker."""
    cols = OHLCV_COLS + ["end"]
    if daily is None or daily.empty:
        return pd.DataFrame(columns=cols)
    agg = {"timestamp": "first", "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if tf in STOCK_NDAYS:
        n = STOCK_NDAYS[tf]
        cnt = np.busday_count(STOCK_ORIGIN, daily["timestamp"].to_numpy().astype("datetime64[D]"))
        key = cnt // n
        g = daily.groupby(key).agg(agg)
        g["end"] = pd.to_datetime(np.busday_offset(STOCK_ORIGIN, (g.index.to_numpy() + 1) * n, roll="forward"))
        first_cnt = int(cnt[0])
        if n > 1 and first_cnt % n != 0 and len(g) > 1:     # erste Kerze unvollständig
            g = g.iloc[1:]
        return g.reset_index(drop=True)[cols]
    g = resample_ohlcv(daily[OHLCV_COLS], tf)
    spec = TF_SPECS[tf]
    if spec["months"]:
        g["end"] = g["timestamp"] + pd.DateOffset(months=spec["months"])
    else:
        g["end"] = g["timestamp"] + pd.Timedelta(spec["rule"])
    return g[cols]


def stock_days_needed(tfs, limit):
    tdays = max(tdays_per_bar(tf) for tf in tfs) * (limit + 10)
    days = int(tdays * 7 / 5 * 1.05) + 10
    return None if days > 3000 else days


def get_stock_data(ticker, tfs, limit):
    """Alle gewünschten TFs einer Aktie aus yfinance-Tageskerzen."""
    d = yf_batch((ticker,), "1d", stock_days_needed(tfs, limit)).get(ticker)
    d = prep_stock_daily(d)
    if d is None or d.empty:
        return {}, {}
    return {tf: resample_stock(d, tf) for tf in tfs}, {"1d": f"yfinance:{ticker}"}


def get_many_stocks_data(tickers, tfs, limit, progress=None, chunk=40):
    days = stock_days_needed(tfs, limit)
    frames = {}
    for i in range(0, len(tickers), chunk):
        part = tuple(tickers[i:i + chunk])
        if progress:
            progress(i / max(1, len(tickers)), f"yfinance: Tageskerzen {i + 1}–{i + len(part)} von {len(tickers)} …")
        try:
            data = yf_batch(part, "1d", days)
        except Exception:
            data = {}
        for t in part:
            d = prep_stock_daily(data.get(t))
            if d is not None and not d.empty:
                frames[t] = {tf: resample_stock(d, tf) for tf in tfs}
    return frames
