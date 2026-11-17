"""
⏱️ Time Predictor v2 – Wendepunkte je Timeframe, Zeit-Projektion und Stichtag-Backtest (Streamlit).
Start:  streamlit run "Time Predictor v2/app.py"
"""
import os
import sys
from dataclasses import replace
from datetime import datetime, time as dtime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Time Predictor v2", page_icon="⏱️")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tp_data import (  # noqa: E402
    SOURCE_MODES, STOCK_TFS, TF_ORDER, TF_SPECS, double_tf, exchange_markets, fetch_top_coins, get_coin_data,
    get_many_coins_data, get_many_stocks_data, get_stock_data, tf_dur, tf_sort,
)
from tp_engine import (  # noqa: E402
    NODE_MODES, POLE_LABEL, QUALITY, SRC_LABEL, TRUTH_MODES, TPParams, analyze_symbol, analyze_tf, build_composite,
    candle_ends,
    evaluate, fmt_ahead, fmt_ts, level_pivots, mark_htf, mark_levels, rolling_backtest, screen_symbol, summarize, truth_for_tf, truth_params,
    utc_now,
)
from tp_charts import (  # noqa: E402
    chart_spec, default_weight, index_figure, lwc_html, matrix_figure, rolling_figure,
)
from tp_stocks import INDEX_SOURCES, stock_universe, tv_stock_symbol  # noqa: E402

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 1rem; padding-left: 1.6rem; padding-right: 1.6rem;
                  max-width: 100%;}
h1, h2, h3 {margin-top: 0.2rem !important;}
[data-testid="stMetricValue"] {font-size: 1.15rem;}
[data-testid="stMetricLabel"] p {font-size: 0.8rem; opacity: 0.8;}
div[data-testid="stHorizontalBlock"] {align-items: flex-end;}
</style>
""", unsafe_allow_html=True)


def _st_version():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


STRETCH = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}
HORIZON_OPTIONS = {
    "3 Tage": pd.Timedelta(days=3), "1 Woche": pd.Timedelta(days=7), "2 Wochen": pd.Timedelta(days=14),
    "1 Monat": pd.Timedelta(days=30), "3 Monate": pd.Timedelta(days=91), "6 Monate": pd.Timedelta(days=182),
    "1 Jahr": pd.Timedelta(days=365),
}
TF_CRYPTO = [tf for tf in TF_ORDER if double_tf(tf)]
TF_STOCK = [tf for tf in STOCK_TFS if double_tf(tf) in STOCK_TFS]
DEFAULT_TFS = {"crypto": ["1w", "3d", "1d", "12h", "4h"], "stock": ["1M", "1w", "3d", "2d", "1d"]}
EXCLUDE_FILE = os.path.join(_HERE, "tr_exclude.txt")
NOW = utc_now().floor("5min")
NOW_KEY = str(NOW)
ss = st.session_state


def _read_exclude():
    try:
        with open(EXCLUDE_FILE, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _set_state(key, value):
    ss[key] = value


def tv_url(sym, tf, market):
    tv_sym = f"BINANCE:{sym}USDT" if market == "crypto" else tv_stock_symbol(sym)
    return f"https://www.tradingview.com/chart/?symbol={tv_sym}&interval={TF_SPECS[tf]['tv']}"


def arrow(pole):
    return "▲ Tief → LONG" if pole == "Low" else "▼ Hoch → SHORT"


def _cell_style(v):
    v = str(v)
    css = ""
    if v.startswith("▲"):
        css = "background-color: rgba(0,230,118,0.22);"
    elif v.startswith("▼"):
        css = "background-color: rgba(255,51,102,0.22);"
    if "jetzt" in v:
        css += "font-weight: bold; border: 1px solid #FFD700;"
    return css


def _bt_style(v):
    v = str(v)
    if v.startswith("✅"):
        return "background-color: rgba(0,230,118,0.18);"
    if v.startswith("🔄"):
        return "background-color: rgba(255,179,0,0.22);"
    if v.startswith("❌"):
        return "background-color: rgba(255,51,102,0.18);"
    return ""


def show_html(html, height):
    """Chart-HTML einbetten: st.iframe (neue Streamlit-Versionen), sonst components.html."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def style_map(df, func, subset):
    sty = df.style
    subset = [c for c in subset if c in df.columns]
    return sty.map(func, subset=subset) if hasattr(sty, "map") else sty.applymap(func, subset=subset)


@st.cache_data(ttl=900, show_spinner=False, max_entries=64)
def cached_truth(sym, tf, Pt, now_key, _df, _df2, _lower):
    """Wahrheit je TF (teuer: ganze Historie) – gecacht je Symbol, TF, Einstellungen und Datenstand."""
    return truth_for_tf(_df, _df2, Pt, pd.Timestamp(now_key), _lower)


# ==========================================
# SIDEBAR
# ==========================================
sb = st.sidebar
sb.header("🪙 Markt & Daten")
market = "crypto" if sb.radio("Markt", ["Krypto", "Aktien"], horizontal=True, key="tp_market") == "Krypto" else "stock"
calendar = market

if market == "crypto":
    src_mode = SOURCE_MODES[sb.selectbox("Datenquelle", list(SOURCE_MODES), index=0)]
    if src_mode in ("auto", "binance"):
        sb.caption("Binance erreichbar ✅" if exchange_markets("binance")
                   else "Binance nicht erreichbar → yfinance zuerst")
    COINS, coin_src = fetch_top_coins(100)
    labels = [f"{c['symbol']} – {c['name']}" + (f" (#{c['rank']})" if c.get("rank") else "") for c in COINS]
    if ss.get("tp_coin_idx", 0) >= len(COINS):
        ss["tp_coin_idx"] = 0
    coin = COINS[sb.selectbox("Coin", list(range(len(COINS))), format_func=lambda i: labels[i], key="tp_coin_idx")]
    symbol, universe_info = coin["symbol"], f"Coin-Liste: {coin_src}"
else:
    src_mode = "yfinance"
    with sb.expander("📋 Aktien-Universum (Trade-Republic-tauglich)", expanded=False):
        st.caption("Large Caps großer Indizes. Ausnahmen unten ausschließen – die Liste wird gespeichert.")
        indices = st.multiselect("Indizes", list(INDEX_SOURCES), default=list(INDEX_SOURCES), key="tp_indices")
        extra = st.text_area("Eigene Ticker (yfinance-Format, z. B. RHM.DE, PLTR)", key="tp_extra", height=68)
        if "tp_exclude" not in ss:
            ss["tp_exclude"] = _read_exclude()
        exclude = st.text_area("Ausschließen", key="tp_exclude", height=68)
        if st.button("💾 Ausschlussliste speichern"):
            try:
                with open(EXCLUDE_FILE, "w", encoding="utf-8") as f:
                    f.write(exclude)
                st.success("Gespeichert.")
            except OSError as e:
                st.error(f"Speichern nicht möglich: {e}")
    STOCKS, idx_sources = stock_universe(tuple(indices), extra, exclude)
    if not STOCKS:
        st.warning("Kein Aktien-Universum – bitte mindestens einen Index oder eigene Ticker wählen.")
        st.stop()
    universe_info = "Universum: " + ", ".join(f"{k}: {v}" for k, v in idx_sources.items())
    sb.caption(f"{len(STOCKS)} Aktien im Universum")
    if ss.get("tp_stock_idx", 0) >= len(STOCKS):
        ss["tp_stock_idx"] = 0
    stock = STOCKS[sb.selectbox("Aktie", list(range(len(STOCKS))), key="tp_stock_idx",
                                format_func=lambda i: f"{STOCKS[i]['symbol']} – {STOCKS[i]['name']}")]
    symbol = stock["symbol"]

limit = sb.slider("Analyse-Fenster (Kerzen je TF)", 150, 1500, 500, 50)
drop_incomplete = sb.checkbox("Laufende (unfertige) Kerze ignorieren", value=True)

tf_opts = TF_CRYPTO if market == "crypto" else TF_STOCK
selected_tfs = tf_sort(sb.multiselect(
    "⏱️ Zeitebenen (empfohlen: 5)", tf_opts, default=DEFAULT_TFS[market], key=f"tp_tfs_{market}",
    help="Nur Timeframes mit 2x-Timeframe (★-Prüfung). Aktien: Tageskerzen und größer."))
doubles = {tf: double_tf(tf) for tf in selected_tfs}

with sb.expander("🎯 Pivot-Erkennung", expanded=False):
    method_lbl = st.radio("Methode", ["Neue Methode (v2)", "Alte Methode (v1)"], index=0, key="tp_method",
                          help="Neu: entgangene Pivots über Gegenkerzen + Messung, nichts wird gelöscht, "
                               "Projektion auf ganze Kerzen aufgerundet. Alt: wie Time Predictor v1.")
    v2 = method_lbl.startswith("Neue")
    use_s2 = st.checkbox("Stufe 2: Messung H→H (⇒ Tief) und L→L (⇒ Hoch)", value=True)
    use_s2a = st.checkbox("Stufe 2a: Messung H→2.H (⇒ Hoch) und L→2.L (⇒ Tief)", value=True)
    max_iter = st.slider("Stufe 2/2a: Durchläufe", 1, 5, 3, help="Eingefügte Pivots ergeben neue Messungen.")
    if v2:
        use_counter = st.checkbox("Gegenkerzen im Trend als entgangene Pivots", value=True,
                                  help="Rote Kerze, deren Tief nicht unter das Vorkerzen-Tief fällt = mögliches Tief; "
                                       "grüne Kerze, deren Hoch nicht über das Vorkerzen-Hoch steigt = mögliches "
                                       "Hoch. Eingefügt nur, wenn eine Messung auf diese Kerze (oder die Kerze davor) zeigt – Toleranz +1.")
        use_lower = st.checkbox("Außenstäbe über kleineren Timeframe auflösen", value=True,
                                help="Hoch und Tief in derselben Kerze: der kleinere TF zeigt, was zuerst kam.")
        use_rsi = st.checkbox("RSI-Markierung (✓/✗, löscht nichts)", value=True)
        rsi_len = st.slider("RSI-Länge", 1, 14, 1, disabled=not use_rsi)
        search_tol, rsi_tol, rsi_remove, rsi_on_insert = 1.0, 1, False, False
        st.caption("Toleranz fest: projizierte Kerze und Kerze +1 (kein −1). Es wird kein Pivot gelöscht.")
    else:
        use_counter, use_lower = False, False
        search_tol = st.slider("Stufe 2/2a: Suchfenster (± Kerzen)", 0.5, 3.0, 1.0, 0.5)
        use_rsi = st.checkbox("Stufe 3: Polaritätsprüfung mit RSI", value=True)
        rsi_len = st.slider("Stufe 3: RSI-Länge", 1, 14, 1, disabled=not use_rsi)
        rsi_tol = st.slider("Stufe 3: Toleranz (± Kerzen)", 0, 2, 1, disabled=not use_rsi)
        rsi_remove = st.radio("Stufe 3: nicht bestätigte Pivots", ["entfernen", "nur markieren"], index=0,
                              disabled=not use_rsi) == "entfernen"
        rsi_on_insert = st.checkbox("Stufe 3 schon beim Einfügen anwenden", value=False, disabled=not use_rsi)
    x2_tol = st.slider("★-Prüfung: Toleranz um die 2x-Kerze (± kleine Kerzen)", 0, 1, 0)

with sb.expander("🔮 Projektion", expanded=False):
    n_lookback = st.slider("Vergangene Tiefs/Hochs je Kette", 1, 30, 10,
                           help="Letztes Tief → 1., 2., 3. … vergangenes Tief: Abstand ÷ 2 (aufgerundet) ⇒ "
                                "Hoch, Tief, Hoch … Hochs analog.")
    proj_tol = float(st.slider("Kreuz-Bestätigung: zweite Kette höchstens … Kerzen später", 0, 1, 1,
                               help="0 = beide Ketten auf derselben Kerze, 1 = auch eine Kerze später."))
    proj_source = "star" if st.radio("Projektion berechnen aus", ["allen Pivots", "nur ★-Pivots (2x-bestätigt)"],
                                     index=0).startswith("nur") else "all"

with sb.expander("🌐 Gesamtkurve", expanded=False):
    horizon = HORIZON_OPTIONS[st.select_slider("Prognose-Horizont", options=list(HORIZON_OPTIONS),
                                               value="1 Monat" if market == "crypto" else "3 Monate")]
    node_mode = NODE_MODES[st.selectbox("Knoten der Gesamtkurve", list(NODE_MODES), index=0)]
    boost_cross = st.slider("Verstärkung kreuzbestätigter Projektionen", 0.0, 2.0, 0.5, 0.1)
    boost_x2 = st.slider("Verstärkung 2x-bestätigter Projektionen", 0.0, 2.0, 0.0, 0.1)
    prominence = st.slider("Mindest-Prominenz Gesamt-Wenden", 0.02, 1.0, 0.15, 0.01)
    weights = {}
    st.caption("Gewichte je Timeframe")
    for tf in selected_tfs:
        weights[tf] = st.slider(f"Gewicht {tf}", 0.0, 1.0, default_weight(tf), 0.05, key=f"w_{tf}")

if not selected_tfs:
    st.warning("Bitte mindestens einen Timeframe wählen.")
    st.stop()

# ==========================================
# DATEN
# ==========================================
# Chart-Zeitebene: jede Zeitebene der Liste. Ist sie nicht unter den gewählten Zeitebenen, wird sie zusätzlich
# geladen und genauso analysiert (Gesamtkurve und Screener nutzen weiterhin nur die gewählten Zeitebenen).
if ss.get("view_tf") not in tf_opts:
    ss["view_tf"] = selected_tfs[min(2, len(selected_tfs) - 1)]
analysis_tfs = tf_sort(list(dict.fromkeys(selected_tfs + [ss["view_tf"]])))
doubles = {tf: double_tf(tf) for tf in analysis_tfs}
load_tfs = list(dict.fromkeys(analysis_tfs + [d for d in doubles.values() if d]))
lower_pool = list(load_tfs)
if market == "crypto" and "1h" not in lower_pool and min(tf_dur(t) for t in analysis_tfs) > pd.Timedelta(hours=1):
    lower_pool.append("1h")
lowers = {}
for tf in analysis_tfs:
    smaller = [t for t in lower_pool if tf_dur(t) < tf_dur(tf)]
    if smaller:
        lowers[tf] = max(smaller, key=tf_dur)
load_tfs = tf_sort(list(dict.fromkeys(lower_pool)))

with st.spinner(f"Lade Marktdaten für {symbol} …"):
    if market == "crypto":
        frames, used_sources = get_coin_data(symbol, load_tfs, src_mode, coin.get("price"), limit, NOW_KEY)
    else:
        frames, used_sources = get_stock_data(symbol, load_tfs, limit)
if not frames:
    st.error(f"Für {symbol} konnten keine Daten geladen werden.")
    st.stop()

# ==========================================
# KOPF + STEUERLEISTE (Chart-TF, Stichtag)
# ==========================================
st.markdown(f"### ⏱️ {symbol} · Time Predictor v2 "
            f"<span style='font-size:0.8rem;opacity:0.6'>· {'neue' if v2 else 'alte'} Methode</span>",
            unsafe_allow_html=True)

earliest = min(df["timestamp"].iloc[min(50, len(df) - 1)] for df in frames.values() if len(df))
ss.setdefault("cut_on", False)
ss.setdefault("cut_date", max(earliest.date(), (NOW - pd.Timedelta(days=14)).date()))
ss.setdefault("cut_time", dtime(0, 0))
if ss["cut_date"] < earliest.date() or ss["cut_date"] > NOW.date():
    ss["cut_date"] = max(earliest.date(), (NOW - pd.Timedelta(days=14)).date())


def _cutoff_from_state():
    return pd.Timestamp(datetime.combine(ss["cut_date"], ss["cut_time"]))


def _step(n, ends_ns):
    """Stichtag auf das Ende der vorigen (n=-1) bzw. nächsten (n=+1) Kerze des Chart-Timeframes setzen."""
    import numpy as np
    c = _cutoff_from_state().value
    ends = np.asarray(ends_ns, dtype=np.int64)
    ends = ends[ends <= NOW.value]
    if not len(ends):
        return
    k = int(np.searchsorted(ends, c, "right")) - 1      # letzte Kerze, die zum Stichtag abgeschlossen ist
    j = min(max(k + n, 0), len(ends) - 2 if len(ends) > 1 else 0)
    new = pd.Timestamp(int(ends[j]))
    ss["cut_date"], ss["cut_time"] = new.date(), new.time()
    ss["cut_on"] = True


def _live():
    ss["cut_on"] = False


bar = st.columns([1.0, 1.25, 1.2, 1.0, 0.45, 0.45, 0.6, 1.3])
view_tf = bar[0].selectbox("Chart-Zeitebene", tf_opts, key="view_tf",
                           help="Alle Zeitebenen wählbar. Nicht unter „Zeitebenen“ gewählte werden zusätzlich "
                                "geladen und genauso analysiert (nicht in der Gesamtkurve).")
bar[1].toggle("⏳ Stichtag-Backtest", key="cut_on",
              help="Prüfzeitpunkt in die Vergangenheit legen: gerechnet wird nur mit Kerzen, die bis dahin "
                   "abgeschlossen waren. Danach siehst du, welche Projektionen eingetroffen sind.")
cut_on = ss["cut_on"]
bar[2].date_input("Stichtag (UTC)", key="cut_date", min_value=earliest.date(), max_value=NOW.date(),
                  disabled=not cut_on, format="DD.MM.YYYY")
bar[3].time_input("Uhrzeit (UTC)", key="cut_time", disabled=not cut_on, step=900)
_vdf = frames.get(view_tf)
_ends = ([int(pd.Timestamp(e).value) for e in candle_ends(_vdf)] if _vdf is not None and len(_vdf) > 1 else [])
bar[4].button("◀", on_click=_step, args=(-1, _ends), help=f"1 Kerze ({view_tf}) zurück", **STRETCH)
bar[5].button("▶", on_click=_step, args=(1, _ends), help=f"1 Kerze ({view_tf}) vor", **STRETCH)
bar[6].button("Jetzt", on_click=_live, help="zurück in den Live-Modus", disabled=not cut_on, **STRETCH)
truth_key = bar[7].selectbox("Wahrheit (Backtest)", list(TRUTH_MODES), format_func=TRUTH_MODES.get, index=0,
                             key="truth_mode", disabled=not cut_on,
                             help="Gegen welche später erkannten Pivots geprüft wird. „Nächsthöherer TF“ = wichtige "
                                  "Wenden: der Pivot fällt mit einem Pivot des nächsthöheren gewählten Timeframes "
                                  "zusammen (für den höchsten TF nicht auswertbar). „bis Timeframe …“ = nur Pivots, "
                                  "die auch auf dem darunter gewählten Timeframe ein Pivot sind.")
use_star_truth = truth_key == "star"
level_pool = tf_sort(list(frames))
lvl_tf = None
if truth_key == "lvl":
    _lvl_opts = [t for t in level_pool if tf_dur(t) > tf_dur(view_tf)]
    if _lvl_opts:
        lvl_tf = bar[7].selectbox("… bestätigt bis Timeframe", list(reversed(_lvl_opts)), key="truth_lvl_tf",
                                  disabled=not cut_on)
    else:
        bar[7].caption("Kein größerer Timeframe geladen.")

cutoff = None
if cut_on:
    cutoff = _cutoff_from_state()
    if cutoff >= NOW:
        st.warning("Stichtag liegt nicht in der Vergangenheit – Live-Modus.")
        cutoff = None

P = TPParams(method="v2" if v2 else "legacy", search_tol=search_tol, max_iter=max_iter, use_stage2=use_s2,
             use_stage2a=use_s2a, use_counter=use_counter, use_lower=use_lower, use_rsi=use_rsi, rsi_len=rsi_len,
             rsi_tol=rsi_tol, rsi_remove=rsi_remove, rsi_on_insert=rsi_on_insert, x2_tol=x2_tol,
             n_lookback=n_lookback, proj_tol=proj_tol, proj_source=proj_source, limit=limit,
             drop_incomplete=drop_incomplete)
a = analyze_symbol(frames, analysis_tfs, doubles, cutoff, P, NOW, calendar=calendar, lowers=lowers)
comp = build_composite(a, selected_tfs, weights, horizon, node_mode, boost_cross, boost_x2, prominence)
tfs_ok = [tf for tf in analysis_tfs if tf in a["tfs"]]
# Stufen-Sterne: jeder Pivot wird mit allen größeren geladenen Timeframes abgeglichen
LVL = level_pivots(frames, level_pool, cutoff, P, NOW, lowers, known=a["tfs"])
mark_levels(a, LVL, tf_dur)
ref = a["ref_now"]
for w in a["warnings"]:
    st.warning(w)
if not tfs_ok:
    st.error("Keine Zeitebene konnte analysiert werden.")
    st.stop()
if view_tf not in tfs_ok:
    st.warning(f"{view_tf}: zu wenige Daten – zeige {tfs_ok[0]}.")
    view_tf = tfs_ok[0]


def higher_tf(tf):
    up = [t for t in tfs_ok if tf_dur(t) > tf_dur(tf)]
    return min(up, key=tf_dur) if up else None


def one_truth(tf, Pt):
    return cached_truth(symbol, tf, Pt, NOW_KEY, frames[tf], frames.get(double_tf(tf)), frames.get(lowers.get(tf)))


def truth_with_htf(tf, Pt):
    tr = dict(one_truth(tf, Pt))
    hi = higher_tf(tf)
    tr["idx_htf"] = mark_htf(tr, one_truth(hi, Pt), frames[hi], NOW) if hi else None
    tr["idx_lvl"] = (mark_htf(tr, one_truth(lvl_tf, Pt), frames[lvl_tf], NOW)
                     if lvl_tf and tf_dur(lvl_tf) > tf_dur(tf) else None)
    return tr


def get_truths(Pv):
    Pt = truth_params(Pv)
    return {tf: truth_with_htf(tf, Pt) for tf in tfs_ok}


bt_rows = None
if cutoff is not None:
    with st.spinner("Prüfe Projektionen gegen die spätere Wahrheit …"):
        truths = get_truths(P)
        bt_rows = evaluate(a, truths, mode=truth_key)

rv = a["tfs"][view_tf]

# ---------- Kennzahlen ----------
fut = sorted([p for p in rv["projs"] if p["date"] > ref], key=lambda p: p["date"])
m = st.columns(5)
if fut:
    t_next = fut[0]["date"]
    poles = {p["pole"] for p in fut if p["date"] == t_next}
    txt = "⚠ Hoch & Tief (Konflikt)" if len(poles) > 1 else ("▼ Hoch" if "High" in poles else "▲ Tief")
    m[0].metric(f"Nächste Projektion {view_tf}", txt, f"{fmt_ts(t_next)} · {fmt_ahead(t_next - ref)}",
                delta_color="off")
else:
    m[0].metric(f"Nächste Projektion {view_tf}", "–")
lp = rv["pivots"][-1] if rv["pivots"] else None
m[1].metric(f"Letzter Pivot {view_tf}", ("▼ Hoch" if lp["type"] == "High" else "▲ Tief") if lp else "–",
            fmt_ts(lp["timestamp"]) if lp else "", delta_color="off")
if bt_rows is not None:
    rows_v = [r for r in bt_rows if r["tf"] == view_tf]
    sv = summarize(rows_v)[0]
    sa = summarize(bt_rows)[0]
    m[2].metric(f"Treffer {view_tf} (+1)", f"{sv.get('Treffer % (+1)', 0)} %" if sv["n"] else "–",
                f"Vorsprung vs. Grundrate {sv.get('Vorsprung (Pp.)', 0):+} Pp. · n = {sv['n']}" if sv["n"] else "",
                delta_color="off")
    m[3].metric(f"Inversion {view_tf} · exakt", (f"{sv.get('Inversion %', 0)} % · {sv.get('exakt %', 0)} %"
                                                 if sv["n"] else "–"),
                f"alle TF: Treffer {sa.get('Treffer % (+1)', 0)} %, Vorsprung {sa.get('Vorsprung (Pp.)', 0):+} Pp."
                if sa["n"] else "", delta_color="off")
    if truth_key == "htf" and not higher_tf(view_tf):
        st.caption(f"ℹ️ {view_tf} ist der höchste gewählte Timeframe – für „wichtige Wenden“ nicht auswertbar.")
    near = [r for r in rows_v if r["rank"] == 1]
    m[4].metric("Nächste Projektion eingetroffen?",
                ("✅ Treffer" if near[0]["hit"] else "🔄 Inversion" if near[0]["inv"] else "❌ verfehlt")
                if near else "–", f"Stichtag {fmt_ts(cutoff)}", delta_color="off")
else:
    n_issue = len(rv["stats"]["issues"])
    m[2].metric("Reihenfolge-Prüfung", "✅ H-L-H-L sauber" if not n_issue else f"⚠ {n_issue} Hinweis(e)",
                f"{len(rv['pivots'])} Pivots", delta_color="off")
    s2 = (rv["stats"].get("s2") or {}).get("inserted", 0) + (rv["stats"].get("s2a") or {}).get("inserted", 0)
    m[3].metric("Entgangene Pivots eingefügt", s2, "inkl. Gegen-Pivots", delta_color="off")
    m[4].metric("Kreuzbestätigt (Zukunft)", sum(1 for p in fut if p["cross"]), f"von {len(fut)} Projektionen",
                delta_color="off")

# ---------- Einblendungen ----------
OVERLAYS = {"Pivots": "pivots", "Eingefügte Pivots": "inserted", "★": "star", "Projektionen": "projs",
            "Toleranz +1": "tol", "Gesamtkurve": "comp", "Pivots 2x-TF": "p2"}
o1, o4, o2, o3 = st.columns([3.0, 1.3, 2.0, 0.9])
if hasattr(st, "pills"):
    sel = o1.pills("Einblenden", list(OVERLAYS), selection_mode="multi", key="ov",
                   default=["Pivots", "Eingefügte Pivots", "★", "Projektionen", "Toleranz +1"])
else:
    sel = o1.multiselect("Einblenden", list(OVERLAYS), key="ov",
                         default=["Pivots", "Eingefügte Pivots", "★", "Projektionen", "Toleranz +1"])
show = {OVERLAYS[s] for s in (sel or [])}
_lv_opts = ["alle"] + [t for t in level_pool if tf_dur(t) > tf_dur(view_tf)]
min_level = o4.selectbox("Pivots zeigen", _lv_opts, key=f"minlvl_{view_tf}",
                         format_func=lambda t: "alle" if t == "alle" else f"nur, die auch auf {t} liegen",
                         help="Blendet alle Pivots aus, die auf dem gewählten größeren Timeframe kein Pivot sind – "
                              "z. B. auf 1h nur die Stunden, in denen eine Tageswende entstanden ist.")
min_level = None if min_level == "alle" else min_level
chain_opts = sorted([p for p in rv["projs"] if p["date"] > ref], key=lambda p: (p["date"], p["chain"], p["k"]))
chain_lbl = ["–"] + [f"{fmt_ts(p['date'])} {POLE_LABEL[p['pole']]} · {p['name']}"
                     + (f" · {p['bt']}" if p.get("bt") else "") for p in chain_opts]
ci = o2.selectbox("Gezählte Kette hervorheben", range(len(chain_lbl)), format_func=lambda i: chain_lbl[i],
                  key=f"chain_{view_tf}", help="Zeigt im Chart, welche Pivots für diese Projektion gezählt wurden "
                                               "(gold: Anker, 1, 2, … k).")
chain = chain_opts[ci - 1] if ci else None
height = o3.select_slider("Höhe", options=[500, 600, 700, 800, 900, 1000, 1200], value=800, key="ch_h")

spec = chart_spec(rv, rv["render"], ref, cutoff, horizon, show, chain, comp, min_level=min_level,
                  title=f"{symbol} · {view_tf}" + (f" · Stichtag {fmt_ts(cutoff)} UTC" if cutoff is not None else ""))
if spec:
    show_html(lwc_html(spec, height, f"{symbol}|{view_tf}"), height + 6)
st.caption("▼/▲ Pivot · orange = rekonstruiert (Alternierung) · ● türkis = Stufe 2 eingefügt · ● violett = Stufe 2a · "
           "G = Gegenkerze · ? = Außenstab unsicher · ★ = auch auf 2x-TF, ★1d = auch Pivot bis 1d (höchste Stufe) · farbige Spalte = projizierte Kerze "
           "(rot Hoch, grün Tief, gelb Konflikt), grau = Kerze +1 (Toleranz) · im Backtest: ✓ Treffer, ⇄ Inversion, ✗ verfehlt. "
           "Mausrad = Zoom, Ziehen = verschieben, Doppelklick auf die Preisachse = automatisch. Zeiten in UTC.")

# ==========================================
# TABS
# ==========================================
tab_proj, tab_det, tab_bt, tab_piv, tab_idx, tab_scr, tab_turns = st.tabs(
    ["🔮 Projektionen", "🔍 Wendefenster im Detail", "🧪 Backtest", "🎯 Pivots & Prüfung",
     "📊 Gesamt-Index & Matrix", f"📡 Screener ({'Krypto' if market == 'crypto' else 'Aktien'})",
     "🌐 Gesamt-Wenden"])

# ------------------------------------------
with tab_proj:
    c1, c2 = st.columns([1, 3])
    only_future = c1.checkbox("Nur ab Stichtag/jetzt", value=True)
    tf_filter = c2.multiselect("Timeframes", tfs_ok, default=tfs_ok, key="proj_tfs")
    rows = []
    for tf in tf_filter:
        for p in a["tfs"][tf]["projs"]:
            if only_future and p["date"] <= ref:
                continue
            row = {"TF": tf, "Kerze (UTC)": fmt_ts(p["date"]), "In": fmt_ahead(p["date"] - ref),
                   "Prognose": ("▼ " if p["pole"] == "High" else "▲ ") + POLE_LABEL[p["pole"]],
                   "Kette": p["name"], "k": p["k"],
                   "Abstand": f"{p['dist_candles']} K. ÷ 2 = {p['proj_candles']} K.",
                   "Gezählt ab": fmt_ts(p["past"]), "Anker": fmt_ts(p["anchor"]),
                   "★ kreuz": "★" if p["cross"] else "", "◆ 2x": "◆" if p["x2"] else "",
                   "Konflikt": "⚠" if p.get("conflict") else ""}
            if bt_rows is not None:
                row["Ergebnis"] = p.get("bt", "⏳ offen")
                row["exakt"] = "✓" if p.get("bt_exact") else ""
                row["Ketten-Diagnose"] = p.get("bt_diag", "")
            row["_d"] = p["date"]
            rows.append(row)
    if rows:
        dfp = pd.DataFrame(rows).sort_values(["_d", "TF"]).drop(columns="_d")
        st.dataframe(style_map(dfp, _bt_style, ["Ergebnis"]).map(_cell_style, subset=["Prognose"])
                     if hasattr(dfp.style, "map") else dfp, hide_index=True, **STRETCH)
    else:
        st.info("Keine Projektionen.")
    st.markdown("**Kreuzbestätigte Zeitpunkte** (Tief-Kette und Hoch-Kette sagen denselben Pol auf derselben oder der nächsten Kerze)")
    cl_rows = [{"TF": tf, "Zeit (UTC)": fmt_ts(c["date"]), "In": fmt_ahead(c["date"] - ref),
                "Wende": POLE_LABEL[c["pole"]], "Projektionen": c["n"], "2x-bestätigt": "◆" if c["x2"] else "",
                "Ketten": c["detail"], "_d": c["date"]}
               for tf in tf_filter for c in a["tfs"][tf]["clusters"] if c["future"] or not only_future]
    if cl_rows:
        st.dataframe(pd.DataFrame(cl_rows).sort_values("_d").drop(columns="_d"), hide_index=True, **STRETCH)
    else:
        st.info("Keine Kreuz-Bestätigungen.")

# ------------------------------------------
with tab_det:
    st.caption("Ein Wendefenster des Chart-Timeframes auf einem kleineren Timeframe ansehen – z. B. eine 1d-Wende "
               "auf 4h oder 1h früh traden.")
    wins = {}
    for p in rv["projs"]:
        if p["date"] > ref:
            wins.setdefault(p["date"], []).append(p)
    if not wins:
        st.info("Keine Wendefenster im Chart-Timeframe.")
    else:
        keys = sorted(wins)

        def _wl(t):
            ps = wins[t]
            poles = {p["pole"] for p in ps}
            pol = "Hoch & Tief ⚠" if len(poles) > 1 else POLE_LABEL[ps[0]["pole"]]
            res = {p.get("bt") for p in ps if p.get("bt")}
            return f"{fmt_ts(t)} · {pol} · {len(ps)} Projektion(en)" + (f" · {', '.join(res)}" if res else "")
        d1, d2 = st.columns([3, 1])
        wsel = d1.selectbox("Wendefenster", keys, format_func=_wl, key=f"win_{view_tf}")
        small = sorted({t for t in list(a["tfs"]) + [lowers.get(view_tf)] + list(frames)
                        if t and tf_dur(t) < tf_dur(view_tf) and t in frames}, key=tf_dur, reverse=True)
        if not small:
            st.info("Kein kleinerer Timeframe geladen – in der Seitenleiste einen kleineren TF hinzufügen.")
        else:
            low_tf = d2.selectbox("Kleinerer Timeframe", small, key=f"low_{view_tf}")
            r_low = a["tfs"].get(low_tf)
            if r_low is None:
                r_low = analyze_tf(frames[low_tf], frames.get(double_tf(low_tf)), low_tf, double_tf(low_tf), cutoff,
                                   P, NOW, calendar=calendar)
                if r_low is not None:
                    r_low["render"] = frames[low_tf][frames[low_tf]["timestamp"] >=
                                                     r_low["visible"]["timestamp"].iloc[0]].reset_index(drop=True)
            if r_low is None:
                st.warning("Zu wenige Kerzen auf dem kleineren Timeframe.")
            else:
                mark_levels({"tfs": {low_tf: r_low}}, LVL, tf_dur)
                if bt_rows is not None:
                    evaluate({"ref_now": ref, "tfs": {low_tf: r_low}},
                             {low_tf: truth_with_htf(low_tf, truth_params(P))}, mode=truth_key)
                dv = rv["delta"]
                w0, w1 = wsel - dv, wsel + 2 * dv
                last_low = r_low["render"]["timestamp"].iloc[-1]
                if w1 <= last_low:
                    spec2 = chart_spec(r_low, r_low["render"], ref, cutoff, horizon, show, None, None,
                                       t_from=w0 - 2 * dv, t_to=w1 + dv,
                                       title=f"{symbol} · {low_tf} · Fenster {fmt_ts(wsel)} ({view_tf})")
                else:
                    spec2 = chart_spec(r_low, r_low["render"], ref, cutoff, max(w1 + dv - ref, pd.Timedelta(0)),
                                       show, None, None, t_from=min(w0, ref) - 3 * dv,
                                       title=f"{symbol} · {low_tf} · Fenster {fmt_ts(wsel)} ({view_tf}) – Zukunft")
                if spec2:
                    show_html(lwc_html(spec2, 560, f"{symbol}|{low_tf}|detail"), 566)
                inside = [p for p in r_low["projs"] if w0 <= p["date"] < w1]
                st.markdown(f"**{low_tf}-Projektionen im Fenster {fmt_ts(w0)} – {fmt_ts(w1)}**")
                if inside:
                    st.dataframe(pd.DataFrame([{
                        "Kerze (UTC)": fmt_ts(p["date"]), "Prognose": POLE_LABEL[p["pole"]], "Kette": p["name"],
                        "★ kreuz": "★" if p["cross"] else "", "Ergebnis": p.get("bt", "")}
                        for p in sorted(inside, key=lambda p: p["date"])]), hide_index=True, **STRETCH)
                else:
                    st.caption("Keine Projektionen des kleineren Timeframes in diesem Fenster.")

# ------------------------------------------
with tab_bt:
    st.markdown(
        "**Wahrheit** = dieselbe Pivot-Erkennung (neue Methode, inkl. entgangener Pivots, ohne Löschen) auf allen "
        "später abgeschlossenen Kerzen. Fenster = **projizierte Kerze und Kerze +1** (kein −1). "
        "**Treffer** = Pivot des vorhergesagten Pols im Fenster; **Inversion** = dort nur ein Pivot des Gegenpols; "
        "**verfehlt** = gar kein Pivot; **exakt** = Pivot des vorhergesagten Pols genau auf der projizierten Kerze. "
        "**Grundrate (beliebige Kerze)** = dieselbe Messung für beliebig herausgegriffene Kerzen nach dem Stichtag – "
        "so oft trifft man, wenn man einfach jede Kerze als Wende bezeichnet. **Vorsprung** = Projektion minus "
        "Grundrate in Prozentpunkten – nur ein positiver Vorsprung zeigt, dass die Projektion mehr weiß. "
        "**Ketten-Diagnose** = Vergleich der am Stichtag gezählten Pivots mit der Wahrheit.")
    st.caption(f"Wahrheit: **{TRUTH_MODES[truth_key].replace('…', lvl_tf or '…')}** (Auswahl oben neben dem "
               "Stichtag). Timeframes ab dieser Stufe sind dabei nicht auswertbar.")
    if cutoff is not None:
        tr_v = get_truths(P).get(view_tf)
        if tr_v and tr_v["n"]:
            idx_v = {"all": tr_v["idx"], "star": tr_v["idx_star"], "htf": tr_v.get("idx_htf"),
                     "lvl": tr_v.get("idx_lvl")}[truth_key]
            if idx_v:
                nh, nl = len(idx_v["High"]), len(idx_v["Low"])
                hit_b = sum(1 for x in range(tr_v["n"] - 1) if x in idx_v["High"] or x + 1 in idx_v["High"])
                st.info(f"**Warum die Grundrate so hoch ist ({view_tf}):** In der Wahrheit ist im Schnitt jede "
                        f"{tr_v['n'] / max(nh, 1):.1f}. Kerze ein Hoch und jede {tr_v['n'] / max(nl, 1):.1f}. Kerze "
                        f"ein Tief. Wer eine beliebige Kerze nimmt und „Hoch auf dieser oder der nächsten Kerze“ "
                        f"sagt, liegt deshalb schon in {100 * hit_b / max(tr_v['n'] - 1, 1):.0f} % der Fälle richtig "
                        "– ganz ohne Berechnung. Eine Projektion ist erst gut, wenn sie deutlich darüber liegt.")
    if bt_rows is None:
        st.info("Oben **⏳ Stichtag-Backtest** einschalten und einen Zeitpunkt in der Vergangenheit wählen.")
    elif not bt_rows:
        st.info("Am Stichtag gibt es noch keine auswertbaren Projektionen (Zukunft noch nicht bekannt).")
    else:
        st.markdown(f"#### Stichtag {fmt_ts(cutoff)} UTC")
        b1, b2 = st.columns(2)
        b1.markdown("**Nach Timeframe**")
        b1.dataframe(pd.DataFrame(summarize(bt_rows, by="tf")), hide_index=True, **STRETCH)
        b2.markdown("**Nach Abstand (1. = zeitlich nächste Projektion)**")
        b2.dataframe(pd.DataFrame(summarize(bt_rows, by="rank")), hide_index=True, **STRETCH)
        b3, b4 = st.columns(2)
        b3.markdown("**Kette am Stichtag vollständig?**")
        b3.dataframe(pd.DataFrame(summarize(bt_rows, by="chain")), hide_index=True, **STRETCH)
        b4.markdown("**Nach Kategorie**")
        b4.dataframe(pd.DataFrame(summarize(bt_rows)), hide_index=True, **STRETCH)
        st.markdown("#### Inversions-Diagnose")
        only_inv = st.radio("Anzeigen", ["nur Inversionen", "alle Projektionen"], horizontal=True, key="inv_only")
        dr = [{"TF": r["tf"], "Kerze (UTC)": fmt_ts(r["date"]), "Rang": r["rank"], "Kerzen voraus": r["ahead"],
               "Prognose": POLE_LABEL[r["pole"]], "k": r["k"],
               "Ergebnis": "✅ Treffer" if r["hit"] else ("🔄 Inversion" if r["inv"] else "❌ verfehlt"),
               "exakt": "✓" if r["hit0"] else "",
               "Ketten-Diagnose": r["diag"], "_d": r["date"]}
              for r in bt_rows if only_inv == "alle Projektionen" or r["inv"]]
        if dr:
            dfd = pd.DataFrame(dr).sort_values(["_d", "TF"]).drop(columns="_d")
            st.dataframe(style_map(dfd, _bt_style, ["Ergebnis"]), hide_index=True, **STRETCH)
            st.caption("Tipp: die Kette einer Projektion oben über „Gezählte Kette hervorheben“ im Chart anzeigen.")
        else:
            st.success("Keine Inversionen an diesem Stichtag.")

    st.markdown("---")
    st.markdown("#### Rollierender Backtest (viele Stichtage)")
    r1, r2 = st.columns(2)
    lookback_days = r1.slider("Zeitraum zurück (Tage)", 14, 1500, 120 if market == "crypto" else 365)
    n_cut = r2.slider("Anzahl Stichtage", 3, 60, 15)
    meta = (symbol, tuple(tfs_ok), P, lookback_days, n_cut, truth_key)
    rb1, rb2 = st.columns(2)
    if rb1.button("▶️ Rollierenden Backtest starten", type="primary"):
        pb = st.progress(0.0, text="Starte …")
        res = rolling_backtest(frames, tfs_ok, doubles, P, NOW, pd.Timedelta(days=lookback_days), n_cut,
                               progress=lambda v, t: pb.progress(min(v, 1.0), text=t),
                               calendar=calendar, lowers=lowers, truths=get_truths(P), mode=truth_key)
        pb.empty()
        ss["tp_roll"] = {"meta": meta, "res": res}
    if rb2.button("⚖️ Alte vs. neue Methode vergleichen"):
        common = dict(n_lookback=n_lookback, proj_tol=proj_tol, proj_source=proj_source, limit=limit,
                      drop_incomplete=drop_incomplete, x2_tol=x2_tol, max_iter=max_iter, use_stage2=use_s2,
                      use_stage2a=use_s2a)
        variants = {"Neue Methode (v2)": TPParams(method="v2", use_counter=True, use_lower=True, **common),
                    "Alte Methode (v1)": TPParams(method="legacy", **common)}
        truths_c = get_truths(variants["Neue Methode (v2)"])
        out, pb = [], st.progress(0.0, text="Starte …")
        for i, (name, Pv) in enumerate(variants.items()):
            pb.progress(i / len(variants), text=f"{name} …")
            res = rolling_backtest(frames, tfs_ok, doubles, Pv, NOW, pd.Timedelta(days=lookback_days), n_cut,
                                   calendar=calendar, lowers=lowers, truths=truths_c, mode=truth_key)
            for by in ("cat", "rank", "chain"):
                for srow in summarize(res["rows"], by):
                    if by == "cat" and srow["Gruppe"] != "Alle":
                        continue
                    out.append({"Methode": name, **srow})
        pb.empty()
        ss["tp_cmp"] = {"meta": meta, "rows": out}
    rs = ss.get("tp_roll")
    if rs is not None:
        if rs["meta"] != meta:
            st.caption("ℹ️ Ergebnis stammt von anderen Einstellungen – ggf. neu starten.")
        c1, c2 = st.columns(2)
        c1.markdown("**Nach Timeframe**")
        c1.dataframe(pd.DataFrame(summarize(rs["res"]["rows"], by="tf")), hide_index=True, **STRETCH)
        c2.markdown("**Nach Abstand**")
        c2.dataframe(pd.DataFrame(summarize(rs["res"]["rows"], by="rank")), hide_index=True, **STRETCH)
        c1.markdown("**Kette vollständig?**")
        c1.dataframe(pd.DataFrame(summarize(rs["res"]["rows"], by="chain")), hide_index=True, **STRETCH)
        c2.markdown("**Nach Kategorie**")
        c2.dataframe(pd.DataFrame(summarize(rs["res"]["rows"])), hide_index=True, **STRETCH)
        if rs["res"]["per_cut"]:
            st.plotly_chart(rolling_figure(rs["res"]["per_cut"]), **STRETCH)
    if ss.get("tp_cmp"):
        st.markdown("**Alte vs. neue Methode (gleiche Wahrheit, gleiche Stichtage)**")
        if ss["tp_cmp"]["meta"] != meta:
            st.caption("ℹ️ Vergleich stammt von anderen Einstellungen.")
        st.dataframe(pd.DataFrame(ss["tp_cmp"]["rows"]), hide_index=True, **STRETCH)

# ------------------------------------------
with tab_piv:
    rows = []
    for tf in tfs_ok:
        r = a["tfs"][tf]
        s = r["stats"]
        s2, s2a, s3 = s.get("s2") or {}, s.get("s2a") or {}, s.get("s3") or {}
        rows.append({
            "TF": tf, "kleinerer TF": lowers.get(tf, "–"), "Stufe 1": s["s1"],
            "Außenstäbe (unsicher)": f"{s['outside']} ({s['outside_unsure']})",
            "S2 geprüft": s2.get("checked"), "S2 bestätigt": s2.get("confirmed"), "S2 eingefügt": s2.get("inserted"),
            "S2 Gegenkerzen": s2.get("counter"), "S2 nicht auflösbar": (s2.get("conflicts") or 0)
            + (s2.get("unresolved") or 0) if s2 else None,
            "S2a geprüft": s2a.get("checked"), "S2a bestätigt": s2a.get("confirmed"),
            "S2a eingefügt": s2a.get("inserted"), "S2a Gegenkerzen": s2a.get("counter"),
            "RSI ✓": s3.get("ok"), "RSI ✗": s3.get("failed"), "entfernt": s3.get("removed"),
            "Final": s["final"], "Reihenfolge-Hinweise": len(s["issues"]), "★": r["n_star"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
    st.caption("„eingefügt“ zählt Pivot + Gegen-Pivot. „nicht auflösbar“ = die Messung zeigt auf eine Kerze, an der "
               "sich kein Pivot nach den Regeln (H-L-H-L, Extrem zwischen den Nachbarn) einfügen lässt."
               + (" „entfernt“ gibt es nur bei der alten Methode." if v2 else ""))
    tf_sel = st.selectbox("Details für", tfs_ok, index=tfs_ok.index(view_tf), key="piv_tf")
    rs_ = a["tfs"][tf_sel]
    p1, p2 = st.columns(2)
    p1.markdown("**Reihenfolge-Prüfung (H-L-H-L, Extrem zwischen den Nachbarn)**")
    if rs_["stats"]["issues"]:
        p1.dataframe(pd.DataFrame([{"Zeit (UTC)": fmt_ts(i["Zeit"]), "Typ": POLE_LABEL[i["Typ"]],
                                    "Hinweis": i["Problem"]} for i in rs_["stats"]["issues"]]),
                     hide_index=True, **STRETCH)
    else:
        p1.success("Alle Pivots wechseln sauber H-L-H-L und sind das Extrem zwischen ihren Nachbarn.")
    p2.markdown("**Einfügungen & Korrekturen**")
    if rs_["stats"]["log"]:
        p2.dataframe(pd.DataFrame([{"Zeit (UTC)": fmt_ts(x["Zeit"]), "Typ": POLE_LABEL[x["Typ"]],
                                    "Aktion": x["Aktion"], "Grund": x["Grund"]}
                                   for x in sorted(rs_["stats"]["log"], key=lambda x: x["Zeit"], reverse=True)]),
                     hide_index=True, **STRETCH)
    else:
        p2.caption("Keine.")
    st.markdown("**Pivot-Liste**")
    st.dataframe(pd.DataFrame([{
        "Zeit (UTC)": fmt_ts(p["timestamp"]), "Typ": POLE_LABEL[p["type"]], "Kurs": round(p["value"], 6),
        "Quelle": SRC_LABEL.get(p["src"], p["src"]) + (" · Gegenkerze" if p.get("counter") else ""),
        "Grund": p.get("reason", ""), "Messungen bestätigt": p["support"],
        "RSI": "–" if p["rsi_ok"] is None else ("✓" if p["rsi_ok"] else "✗"),
        "Außenstab": ("unsicher" if p.get("uncertain") else "ja") if p.get("outside") else "",
        "★ 2x-TF": "★" if p["star"] else "",
        "auch Pivot auf": ", ".join(p.get("levels", [])),
        "höchste Stufe": (p.get("levels") or [""])[-1],
    } for p in reversed(rs_["pivots"])]), hide_index=True, **STRETCH)

# ------------------------------------------
with tab_idx:
    st.plotly_chart(index_figure(comp, ref, cutoff), **STRETCH)
    st.plotly_chart(matrix_figure(a, tfs_ok, show_past=cutoff is not None), **STRETCH)

# ------------------------------------------
with tab_scr:
    st.subheader(f"📡 Bald anstehende Trendwechsel – {'Krypto' if market == 'crypto' else 'Aktien'}")
    st.caption(f"Nutzt die Timeframes ({', '.join(selected_tfs)}) und alle Einstellungen der Seitenleiste.")
    s1, s2, s3, s4 = st.columns(4)
    if market == "crypto":
        n_sym = s1.slider("Anzahl Coins (Top nach Market Cap)", 10, len(COINS), min(50, len(COINS)), 5)
    else:
        n_sym = s1.slider("Anzahl Aktien", 10, len(STOCKS), len(STOCKS), 10) if len(STOCKS) > 10 else len(STOCKS)
    soon_days = s2.slider("„Bald“ = innerhalb von … Tagen", 1, 90, 7 if market == "crypto" else 14)
    quality = QUALITY[s3.selectbox("Mindest-Qualität der Projektion", list(QUALITY), index=3)]
    scr_limit = s4.slider("Kerzen je TF", 150, 1000, 300, 50, help="Weniger = schnellerer Download.")
    f1, f2, f3, f4 = st.columns(4)
    show_active = f4.checkbox("Bereits laufende Fenster zeigen", value=True)
    direction = f1.radio("Richtung", ["beide", "nur ▲ Tief (LONG)", "nur ▼ Hoch (SHORT)"], horizontal=True)
    sort_by = f2.radio("Sortierung", ["nach Zeit", "nach Score"], horizontal=True)
    min_aligned = f3.slider("Mindestens gleichgerichtete weitere TFs", 0, max(0, len(selected_tfs) - 1), 0)
    st.caption("Score = 1 + kreuzbestätigt + 2x-bestätigt + Anzahl gleichgerichteter TFs (+0,5 je weiterer "
               "Projektion im selben Fenster).")

    scr_meta = (market, tuple(selected_tfs), replace(P, limit=scr_limit), n_sym, soon_days, quality, cutoff,
                tuple(sorted(weights.items())), node_mode, boost_cross, boost_x2, prominence)
    if st.button("🚀 Jetzt scannen", type="primary"):
        syms = (COINS if market == "crypto" else STOCKS)[:n_sym]
        pbar = st.progress(0.0, text="Lade Daten …")
        prog = lambda v, t: pbar.progress(min(v, 1.0) * 0.6, text=t)  # noqa: E731
        if market == "crypto":
            all_frames, _used = get_many_coins_data(syms, load_tfs, src_mode, scr_limit, NOW_KEY, progress=prog)
        else:
            all_frames = get_many_stocks_data([s["symbol"] for s in syms], load_tfs, scr_limit, progress=prog)
        Ps = replace(P, limit=scr_limit)
        hz = pd.Timedelta(days=soon_days)
        results, errors = {}, {}
        for j, srow in enumerate(syms):
            sym = srow["symbol"]
            pbar.progress(0.6 + 0.4 * (j + 1) / len(syms), text=f"Analysiere {sym} ({j + 1}/{len(syms)}) …")
            fr = all_frames.get(sym) or {}
            if not fr:
                errors[sym] = "keine Daten"
                continue
            try:
                aa = analyze_symbol(fr, selected_tfs, doubles, cutoff, Ps, NOW, calendar=calendar, lowers=lowers)
                cc = build_composite(aa, selected_tfs, weights, max(hz, pd.Timedelta(days=3)), node_mode,
                                     boost_cross, boost_x2, prominence)
                res = screen_symbol(aa, selected_tfs, hz, quality, cc)
                last_tf = next((t for t in reversed(selected_tfs) if t in aa["tfs"]), None)
                res["price"] = float(aa["tfs"][last_tf]["visible"]["close"].iloc[-1]) if last_tf else None
                results[sym] = res
            except Exception as ex:  # einzelne Symbole dürfen den Scan nicht abbrechen
                errors[sym] = f"{type(ex).__name__}: {ex}"
        pbar.empty()
        ss["tp_scr"] = {"meta": scr_meta, "results": results, "errors": errors, "syms": syms,
                        "ref": ref, "time": NOW, "market": market}

    sr = ss.get("tp_scr")
    if sr is not None and sr["market"] != market:
        st.info("Letzter Scan gehört zum anderen Markt – bitte neu scannen.")
    elif sr is not None:
        if sr["meta"] != scr_meta:
            st.caption("ℹ️ Ergebnis stammt von anderen Einstellungen – ggf. neu scannen.")
        rref = sr["ref"]
        names = {s["symbol"]: s.get("name", s["symbol"]) for s in sr["syms"]}
        order = {s["symbol"]: i for i, s in enumerate(sr["syms"])}
        st.caption(f"Scan {fmt_ts(sr['time'])} UTC | Referenz {fmt_ts(rref)} UTC | "
                   f"{len(sr['results'])}/{len(sr['syms'])} mit Daten")
        if sr["errors"]:
            with st.expander(f"⚠️ Ohne Ergebnis ({len(sr['errors'])})"):
                st.write(", ".join(f"{k} ({v})" for k, v in sr["errors"].items()))

        def ok_event(e):
            if not show_active and e["start"] <= rref:
                return False
            if direction.startswith("nur ▲") and e["pole"] != "Low":
                return False
            if direction.startswith("nur ▼") and e["pole"] != "High":
                return False
            return len(e["aligned"]) >= min_aligned

        def countdown(e):
            return f"jetzt (bis {e['end']:%d.%m.})" if e["start"] <= rref < e["end"] else fmt_ahead(e["start"] - rref)

        tv_tf = selected_tfs[-1]
        top, matrix, longlist = [], [], []
        for sym, res in sr["results"].items():
            evs = [e for tf in selected_tfs for e in res["events"].get(tf, []) if ok_event(e)]
            cn = res["composite_next"]
            comp_txt = (f"{'▲' if cn['pole'] == 'Low' else '▼'} {fmt_ahead(cn['timestamp'] - rref)}" if cn else "–")
            mrow = {"Symbol": sym, "Name": names.get(sym, sym)}
            for tf in selected_tfs:
                lst = [e for e in res["events"].get(tf, []) if ok_event(e)]
                if lst:
                    e = lst[0]
                    mrow[tf] = (f"{'▲' if e['pole'] == 'Low' else '▼'} {countdown(e)}"
                                f"{' ★' if e['cross'] else ''}{' ◆' if e['x2'] else ''}")
                else:
                    mrow[tf] = "–"
            mrow["Gesamtkurve"] = comp_txt
            mrow["TradingView"] = tv_url(sym, tv_tf, market)
            mrow["_o"] = order.get(sym, 0)
            matrix.append(mrow)
            for e in evs:
                longlist.append({
                    "Symbol": sym, "Name": names.get(sym, sym), "TF": e["tf"], "Richtung": arrow(e["pole"]),
                    "Fenster von (UTC)": fmt_ts(e["start"]), "bis": fmt_ts(e["end"]), "In": countdown(e),
                    "★ kreuz": "★" if e["cross"] else "", "◆ 2x": "◆" if e["x2"] else "",
                    "Gleichgerichtet": ", ".join(e["aligned"]) or "–", "Score": round(e["score"], 1),
                    "Ketten": e["name"], "_t": max(0.0, (e["start"] - rref).total_seconds()),
                    "TradingView": tv_url(sym, e["tf"], market),
                })
            if not evs:
                continue
            best = (min(evs, key=lambda e: (e["start"], -e["score"])) if sort_by == "nach Zeit"
                    else max(evs, key=lambda e: (e["score"], -e["start"].value)))
            top.append({
                "Symbol": sym, "Name": names.get(sym, sym), "Kurs": round(res["price"], 6) if res["price"] else None,
                "Richtung": arrow(best["pole"]), "TF": best["tf"], "In": countdown(best),
                "Fenster von (UTC)": fmt_ts(best["start"]), "★ kreuz": "★" if best["cross"] else "",
                "◆ 2x": "◆" if best["x2"] else "", "Gleichgerichtet": ", ".join(best["aligned"]) or "–",
                "Score": round(best["score"], 1), "Signale gesamt": len(evs), "Gesamtkurve nächste Wende": comp_txt,
                "_t": max(0.0, (best["start"] - rref).total_seconds()), "_s": best["score"],
                "TradingView": tv_url(sym, best["tf"], market),
            })

        link_cfg = {"TradingView": st.column_config.LinkColumn("Chart", display_text="öffnen ↗")}
        st.markdown(f"### 🔔 Trendwechsel in den nächsten {soon_days} Tagen ({len(top)} Treffer)")
        if top:
            dft = pd.DataFrame(top)
            dft = (dft.sort_values(["_t", "_s"], ascending=[True, False]) if sort_by == "nach Zeit"
                   else dft.sort_values(["_s", "_t"], ascending=[False, True])).drop(columns=["_t", "_s"])
            st.dataframe(style_map(dft, _cell_style, ["Richtung", "In"]), hide_index=True, column_config=link_cfg,
                         **STRETCH)
            q1, q2 = st.columns([3, 1])
            pick = q1.selectbox("Im Chart öffnen", list(dft["Symbol"]), key="tp_scr_pick")
            all_syms = [c["symbol"] for c in (COINS if market == "crypto" else STOCKS)]
            if pick in all_syms:
                q2.button("📈 Öffnen", on_click=_set_state,
                          args=("tp_coin_idx" if market == "crypto" else "tp_stock_idx", all_syms.index(pick)))
        else:
            st.info("Keine Treffer mit diesen Filtern – Qualität senken oder Zeitraum vergrößern.")

        st.markdown("### 🧮 Matrix: nächste Wende je Symbol × Timeframe")
        if matrix:
            dfm = pd.DataFrame(matrix).sort_values("_o").drop(columns="_o")
            st.dataframe(style_map(dfm, _cell_style, selected_tfs + ["Gesamtkurve"]), hide_index=True,
                         column_config=link_cfg, **STRETCH)

        st.markdown("### 📋 Alle Wendefenster")
        if longlist:
            dfl = pd.DataFrame(longlist).sort_values("_t").drop(columns="_t")
            st.dataframe(dfl, hide_index=True, column_config=link_cfg, **STRETCH)
            st.download_button("📥 Liste als CSV", dfl.to_csv(index=False).encode("utf-8"),
                               file_name=f"screener_{market}_{NOW:%Y%m%d_%H%M}.csv", mime="text/csv")

# ------------------------------------------
with tab_turns:
    if comp and comp["turns"]:
        st.dataframe(pd.DataFrame([{
            "Zeit (UTC)": fmt_ts(t["timestamp"]), "In": fmt_ahead(t["timestamp"] - ref),
            "Wende": POLE_LABEL[t["pole"]], "Index": round(t["index"], 2), "Prominenz": round(t["prominence"], 2),
            "Preis (heuristisch)": round(t["price"], 6), "Dominanter TF": t["dominant"],
            "Gleichgerichtete TFs": ", ".join(t["aligned"]) or "–",
        } for t in comp["turns"]]), hide_index=True, **STRETCH)
    else:
        st.info("Keine Gesamt-Wendepunkte im Horizont (Prominenz senken oder Horizont vergrößern).")

st.caption(f"Daten {symbol}: " + " | ".join(f"{b} → {s}" for b, s in used_sources.items()) + f" | {universe_info}"
           + f" | Stand {fmt_ts(NOW)} UTC")
