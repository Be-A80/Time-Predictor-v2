"""Time Predictor v2 – Charts.

Hauptchart: TradingView Lightweight Charts (Mausrad-Zoom, Ziehen, automatische Preisachse, Vollbild).
Zusatz-Charts (Gesamt-Index, Projektions-Matrix): Plotly.
"""
import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tp_data import TF_ORDER, tf_dur
from tp_engine import POLE_COLOR, POLE_LABEL, POLE_SYMBOL, SRC_LABEL, fmt_ts, make_idx_to_ts

_PALETTE = ["#AB47BC", "#7E57C2", "#5C6BC0", "#42A5F5", "#29B6F6", "#26C6DA", "#26A69A", "#66BB6A",
            "#9CCC65", "#D4E157", "#FFEE58", "#FFCA28", "#FFA726", "#FF7043", "#EC407A", "#EF5350",
            "#8D6E63", "#BDBDBD", "#78909C"]
TF_COLORS = {tf: _PALETTE[i % len(_PALETTE)] for i, tf in enumerate(TF_ORDER)}
GOLD = "#FFD700"
BG = "#0e1117"
SRC_COLOR = {"S1": None, "S1r": "#FFA726", "S2": "#00E5FF", "S2a": "#E040FB"}
BT_SYMBOL = {"hit": "✓", "inv": "⇄", "miss": "✗"}
LWC_URL = "https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"


def default_weight(tf):
    """Log-skaliert nach Kerzendauer: 4h ≈ 0,25 … 1M ≈ 1,0 (begrenzt auf 0,1–1,0)."""
    lo, hi = math.log(pd.Timedelta(hours=4).total_seconds()), math.log(pd.Timedelta(days=30).total_seconds())
    x = (math.log(tf_dur(tf).total_seconds()) - lo) / (hi - lo)
    return round(min(1.0, max(0.1, 0.25 + 0.75 * x)), 2)


def _sec(ts):
    return int(pd.Timestamp(ts).value // 10 ** 9)


def _rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# ==========================================
# HAUPTCHART – Daten aufbereiten
# ==========================================
def chart_spec(r, render, ref_now, cutoff, horizon, show=None, chain=None, comp=None, t_from=None, t_to=None,
               title="", min_level=None):
    """Baut die Daten für den Lightweight-Chart eines Timeframes.

    r       = Analyse-Ergebnis des TF (analyze_tf)
    render  = alle Kerzen ab Analyse-Beginn (auch nach dem Stichtag – werden abgedunkelt)
    show    = Menge der Einblendungen: pivots, inserted, star, projs, tol, comp, p2
    chain   = Projektion, deren gezählte Kette hervorgehoben wird
    min_level = nur Pivots zeigen, die auch auf diesem größeren TF ein Pivot sind (Stufen-Sterne)
    """
    show = show or set()
    vis = r["visible"]
    if t_from is not None:
        render = render[render["timestamp"] >= t_from]
    if t_to is not None:
        render = render[render["timestamp"] <= t_to]
    render = render.reset_index(drop=True)
    if render.empty:
        return None
    last_vis = vis["timestamp"].iloc[-1]
    candles = []
    for row in render.itertuples(index=False):
        dim = cutoff is not None and row.timestamp > last_vis
        c = {"time": _sec(row.timestamp), "open": float(row.open), "high": float(row.high),
             "low": float(row.low), "close": float(row.close)}
        if dim:
            col = "rgba(140,140,140,0.45)"
            c.update(color=col, borderColor=col, wickColor=col)
        candles.append(c)
    times = [c["time"] for c in candles]

    # Zukunfts-Kerzen (leere Zeitpunkte), damit Projektionen rechts vom Kursende Platz haben
    idx_to_ts = make_idx_to_ts(vis, r["tf"], r.get("calendar", "crypto"))
    future = []
    if t_to is None:
        end = ref_now + horizon
        n_vis = len(vis)
        j = 1
        while j < 2000:
            t = idx_to_ts(n_vis - 1 + j)
            if t > end and j > 20:
                break
            s = _sec(t)
            if s > times[-1]:
                future.append(s)
            j += 1
    grid = np.array(times + future, dtype=np.int64)

    def snap(ts):
        """Zeitpunkt → Kerzen-Zeit (Kerze, die den Zeitpunkt enthält)."""
        s = _sec(ts)
        step = int(grid[-1] - grid[-2]) if len(grid) > 1 else 0
        if s < grid[0] or s >= grid[-1] + max(step, 1):
            return None
        i = int(np.searchsorted(grid, s, "right")) - 1
        return int(grid[i])

    info = {}

    def add_info(t, txt):
        info.setdefault(str(t), []).append(txt)

    # --- Pivots ---
    markers = []
    t0, t1 = grid[0], grid[-1]
    chain_set = set()
    if chain is not None:
        chain_set = {_sec(t) for t in chain["chain_ts"]}
    if "pivots" in show:
        for p in r["pivots"]:
            s = _sec(p["timestamp"])
            if s < t0 or s > t1:
                continue
            levels = p.get("levels") or []
            if min_level and min_level not in levels:
                continue
            base = p["src"].rstrip("g")
            inserted = base in ("S2", "S2a")
            if inserted and "inserted" not in show:
                color = POLE_COLOR[p["type"]]
            else:
                color = SRC_COLOR.get(base) or POLE_COLOR[p["type"]]
            hi = p["type"] == "High"
            txt = ""
            if p.get("counter"):
                txt += "G"
            if p.get("outside") and p.get("uncertain"):
                txt += "?"
            if "star" in show and levels:
                txt += "★" + levels[-1]          # höchste Stufe, auf der der Pivot ebenfalls ein Pivot ist
            elif "star" in show and p["star"]:
                txt += "★"
            shape = ("circle" if inserted and "inserted" in show else ("arrowDown" if hi else "arrowUp"))
            if s in chain_set and chain is not None and p["type"] == chain["chain"]:
                k = len(chain["chain_ts"]) - 1 - sorted(chain_set).index(s)
                color, shape, txt = GOLD, "square", f"{'Anker' if k == 0 else k}"
            markers.append({"time": s, "position": "aboveBar" if hi else "belowBar", "color": color,
                            "shape": shape, "text": txt, "size": 1})
            label = SRC_LABEL.get(p["src"], p["src"])
            extra = []
            if p.get("reason"):
                extra.append(p["reason"])
            if p.get("rsi_ok") is not None:
                extra.append("RSI ✓" if p["rsi_ok"] else "RSI ✗")
            if levels:
                extra.append("★ auch Pivot auf: " + ", ".join(levels))
            elif p["star"]:
                extra.append("★ auch auf 2x-TF")
            if p.get("outside"):
                extra.append("Außenstab" + (" – Reihenfolge unsicher" if p.get("uncertain") else
                                            " – Reihenfolge aus kleinerem TF"))
            add_info(s, f"<b style='color:{POLE_COLOR[p['type']]}'>Pivot {POLE_LABEL[p['type']]}</b> "
                        f"{p['value']:,.6g} · {label}" + (" · " + " · ".join(extra) if extra else ""))
    if "p2" in show and r.get("pivots2"):
        for p in r["pivots2"]:
            s = snap(p["timestamp"])
            if s is None or s < t0:
                continue
            markers.append({"time": s, "position": "aboveBar" if p["type"] == "High" else "belowBar",
                            "color": "#B388FF", "shape": "square", "text": r["tf2"], "size": 0.6})

    # --- Projektionen: Fenster (Histogramm über volle Höhe) + Marker ---
    windows, wmarkers = {}, []
    if "projs" in show:
        groups = {}
        for p in r["projs"]:
            if p["date"] <= ref_now and "bt" not in p:
                continue
            s = snap(p["date"])
            if s is None or s < t0:
                continue
            groups.setdefault(s, []).append(p)
        for s, ps in sorted(groups.items()):
            poles = {p["pole"] for p in ps}
            col = "#FFB300" if len(poles) > 1 else POLE_COLOR[ps[0]["pole"]]
            windows[s] = _rgba(col, 0.28)
            for pole in ("High", "Low"):
                pp = [p for p in ps if p["pole"] == pole]
                if not pp:
                    continue
                txt = "★" if any(p["cross"] for p in pp) else ""
                res = [p.get("bt_key") for p in pp if p.get("bt_key")]
                if res:
                    txt += " " + " ".join(f"{BT_SYMBOL[k]}{res.count(k) if res.count(k) > 1 else ''}"
                                          for k in ("hit", "inv", "miss") if k in res)
                wmarkers.append({"time": s, "position": "aboveBar" if pole == "High" else "belowBar",
                                 "color": POLE_COLOR[pole], "shape": "arrowDown" if pole == "High" else "arrowUp",
                                 "text": txt.strip(), "size": 1.3})
            for p in ps:
                line = (f"<b style='color:{POLE_COLOR[p['pole']]}'>Projektion {POLE_LABEL[p['pole']]}</b> "
                        f"{p['name']}: {p['dist_candles']} K. → {p['proj_candles']} K."
                        + (" · ★ kreuzbestätigt" if p["cross"] else "") + (" · ◆ 2x" if p["x2"] else "")
                        + (" · ⚠ Konflikt" if p.get("conflict") else ""))
                if p.get("bt"):
                    line += (f" · <b>{p['bt']}</b>" + (" (exakt)" if p.get("bt_exact") else "")
                             + f" · {p.get('bt_diag', '')}")
                add_info(s, line)
        if "tol" in show:
            gi = {int(t): i for i, t in enumerate(grid)}
            for s in list(windows):
                i = gi.get(s)
                for o in (1,):          # Toleranz nur +1 Kerze
                    if i is not None and 0 <= i + o < len(grid):
                        t = int(grid[i + o])
                        if t not in windows:
                            windows[t] = "rgba(200,200,200,0.07)"
    win_data = [{"time": t, "value": 1, "color": c} for t, c in sorted(windows.items())]

    # --- Gesamtkurve (auf Kerzen-Zeiten abgetastet) ---
    comp_line = []
    if comp is not None and "comp" in show:
        g_ns = comp["grid"].asi8 // 10 ** 9 if hasattr(comp["grid"], "asi8") else np.asarray(comp["grid"])
        ts_c = [int(t) for t in grid if g_ns[0] <= t <= g_ns[-1]]
        if ts_c:
            vals = np.interp(ts_c, g_ns.astype(float), comp["composite_price"])
            comp_line = [{"time": t, "value": float(v)} for t, v in zip(ts_c, vals)]

    cut_idx = len(times) - 1
    if cutoff is not None:
        cut_idx = int(np.searchsorted(np.array(times), _sec(last_vis), "right")) - 1
    return {
        "candles": candles, "future": future, "markers": sorted(markers, key=lambda m: m["time"]),
        "windows": win_data, "wmarkers": sorted(wmarkers, key=lambda m: m["time"]), "comp": comp_line,
        "cut": _sec(ref_now) if cutoff is not None else None, "cut_label": "Stichtag" if cutoff is not None else "",
        "cut_idx": cut_idx, "info": info, "title": title,
        "precision": _precision(float(render["close"].iloc[-1])),
    }


def _precision(price):
    if price >= 1000:
        return 1
    if price >= 10:
        return 2
    if price >= 1:
        return 4
    return 6 if price >= 0.01 else 8


# ==========================================
# HAUPTCHART – HTML / JavaScript
# ==========================================
def lwc_html(spec, height, chart_key):
    data = json.dumps(spec, separators=(",", ":"))
    key = json.dumps(chart_key)
    return f"""
<div id="wrap" style="position:relative;width:100%;height:{height}px;background:{BG};
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div id="chart" style="position:absolute;inset:0;"></div>
  <div id="vl" style="position:absolute;inset:0;pointer-events:none;z-index:3;"></div>
  <div id="legend" style="position:absolute;left:10px;top:8px;z-index:4;color:#d1d4dc;font-size:12px;
       line-height:1.45;max-width:62%;pointer-events:none;text-shadow:0 0 3px #000;"></div>
  <div id="btns" style="position:absolute;right:70px;top:8px;z-index:5;display:flex;gap:4px;"></div>
  <div id="err" style="color:#ff8a80;padding:20px;display:none;">Chart-Bibliothek konnte nicht geladen werden
       (Internetverbindung zu unpkg.com nötig).</div>
</div>
<script src="{LWC_URL}"></script>
<script>
(function() {{
const S = {data};
const KEY = "tpv2:" + {key};
const wrap = document.getElementById('wrap');
if (typeof LightweightCharts === 'undefined') {{ document.getElementById('err').style.display='block'; return; }}
const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
  autoSize: true,
  layout: {{ background: {{ type: 'solid', color: '{BG}' }}, textColor: '#d1d4dc', fontSize: 12 }},
  grid: {{ vertLines: {{ color: 'rgba(255,255,255,0.04)' }}, horzLines: {{ color: 'rgba(255,255,255,0.05)' }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor: 'rgba(255,255,255,0.15)', scaleMargins: {{ top: 0.08, bottom: 0.08 }} }},
  timeScale: {{ borderColor: 'rgba(255,255,255,0.15)', timeVisible: true, secondsVisible: false, rightOffset: 4,
               barSpacing: 7, minBarSpacing: 0.5 }},
  localization: {{ locale: 'de-DE', dateFormat: 'dd.MM.yyyy' }},
  handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }},
  handleScale: {{ mouseWheel: true, pinch: true, axisPressedMouseMove: true, axisDoubleClickReset: true }},
}});
const win = chart.addHistogramSeries({{ priceScaleId: 'win', lastValueVisible: false, priceLineVisible: false,
  base: 0 }});
chart.priceScale('win').applyOptions({{ scaleMargins: {{ top: 0, bottom: 0 }}, visible: false }});
const fut = S.future.map(t => ({{ time: t }}));
win.setData(S.windows.length ? S.windows : [{{ time: S.candles[0].time }}]);
win.setMarkers(S.wmarkers);
const cs = chart.addCandlestickSeries({{
  upColor: '#26a69a', downColor: '#ef5350', borderUpColor: '#26a69a', borderDownColor: '#ef5350',
  wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  priceFormat: {{ type: 'price', precision: S.precision, minMove: Math.pow(10, -S.precision) }},
}});
cs.setData(S.candles.concat(fut));
cs.setMarkers(S.markers);
if (S.comp.length) {{
  const ln = chart.addLineSeries({{ color: '{GOLD}', lineWidth: 2, lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false }});
  ln.setData(S.comp);
}}
// --- senkrechte Linie (Stichtag) ---
const vl = document.getElementById('vl');
function drawLines() {{
  vl.innerHTML = '';
  if (S.cut === null) return;
  const ts = chart.timeScale();
  let x = ts.timeToCoordinate(S.cut);
  if (x === null) {{
    const i = S.cut_idx; x = ts.logicalToCoordinate(i + 0.5);
  }}
  if (x === null) return;
  const d = document.createElement('div');
  d.style.cssText = 'position:absolute;top:0;bottom:28px;width:0;border-left:2px dashed #FFA726;left:' + x + 'px';
  const lb = document.createElement('div');
  lb.textContent = '⏳ ' + S.cut_label;
  lb.style.cssText = 'position:absolute;bottom:32px;left:' + (x + 4) + 'px;color:#FFA726;font-size:11px;';
  vl.appendChild(d); vl.appendChild(lb);
}}
// --- Legende (Fadenkreuz) ---
const legend = document.getElementById('legend');
const fmt = v => v.toLocaleString('de-DE', {{ maximumFractionDigits: S.precision }});
function showLegend(t) {{
  let html = '<b>' + S.title + '</b>';
  const c = t ? S.candles.find(x => x.time === t) : S.candles[S.candles.length - 1];
  if (t) {{
    const d = new Date(t * 1000);
    html += ' · ' + d.toLocaleString('de-DE', {{ timeZone: 'UTC', day: '2-digit', month: '2-digit',
      year: 'numeric', hour: '2-digit', minute: '2-digit' }}) + ' UTC';
  }}
  if (c) html += '<br>O ' + fmt(c.open) + ' · H ' + fmt(c.high) + ' · T ' + fmt(c.low) + ' · S ' + fmt(c.close);
  const inf = t ? S.info[String(t)] : null;
  if (inf) html += '<br>' + inf.join('<br>');
  legend.innerHTML = html;
}}
chart.subscribeCrosshairMove(p => showLegend(p && p.time ? p.time : null));
showLegend(null);
// --- Ausschnitt: gespeicherten Zoom relativ zum Stichtag wiederherstellen ---
const N = S.candles.length, F = S.future.length;
function setLast(n) {{
  const after = (N - 1 - S.cut_idx) + F;   // Kerzen nach dem Stichtag + Zukunft
  const to = S.cut_idx + Math.max(8, Math.min(after, Math.round(n * 0.3)));
  chart.timeScale().setVisibleLogicalRange({{ from: Math.max(0, to - n), to: to }});
}}
let ready = false;
function initRange() {{
  let sv = null;
  try {{ sv = JSON.parse(localStorage.getItem(KEY) || 'null'); }} catch (e) {{}}
  if (sv && sv.t > sv.f) chart.timeScale().setVisibleLogicalRange({{ from: S.cut_idx + sv.f, to: S.cut_idx + sv.t }});
  else setLast(150);
  ready = true;
  drawLines();
}}
// erst setzen, wenn der Chart seine endgültige Breite hat (autoSize)
requestAnimationFrame(() => setTimeout(initRange, 60));
chart.timeScale().subscribeVisibleLogicalRangeChange(r => {{
  drawLines();
  if (!r || !ready) return;
  try {{ localStorage.setItem(KEY, JSON.stringify({{ f: r.from - S.cut_idx, t: r.to - S.cut_idx }})); }} catch (e) {{}}
}});
new ResizeObserver(() => setTimeout(drawLines, 30)).observe(wrap);
setTimeout(drawLines, 50);
// --- Knöpfe ---
const btns = document.getElementById('btns');
function btn(label, title, fn) {{
  const b = document.createElement('button');
  b.textContent = label; b.title = title; b.onclick = fn;
  b.style.cssText = 'background:rgba(40,44,52,0.85);color:#d1d4dc;border:1px solid rgba(255,255,255,0.15);' +
                    'border-radius:4px;padding:2px 8px;font-size:12px;cursor:pointer;';
  btns.appendChild(b);
}}
[50, 150, 500].forEach(n => btn(String(n), 'letzte ' + n + ' Kerzen', () => setLast(n)));
btn('Alle', 'alle Kerzen', () => chart.timeScale().fitContent());
btn('⛶', 'Vollbild', () => {{
  if (document.fullscreenElement) document.exitFullscreen();
  else if (wrap.requestFullscreen) wrap.requestFullscreen().catch(() => {{}});
}});
document.addEventListener('fullscreenchange', () => {{
  wrap.style.height = document.fullscreenElement ? '100vh' : '{height}px';
}});
}})();
</script>
"""


# ==========================================
# ZUSATZ-CHARTS (Plotly)
# ==========================================
def _base_layout(fig, height, title):
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=36, b=10),
                      title=dict(text=title, font=dict(size=13)), paper_bgcolor=BG, plot_bgcolor=BG,
                      hovermode="closest", legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    return fig


def index_figure(comp, ref_now, cutoff):
    fig = go.Figure()
    if comp:
        for tf in comp["usable"]:
            fig.add_trace(go.Scatter(x=comp["grid"], y=comp["waves"][tf], mode="lines", name=f"{tf}",
                                     line=dict(color=TF_COLORS[tf], width=1), opacity=0.6))
        fig.add_trace(go.Scatter(x=comp["grid"], y=comp["composite"], mode="lines", name="Gesamt",
                                 line=dict(color=GOLD, width=3)))
        for pole in ("High", "Low"):
            ts_ = [t for t in comp["turns"] if t["pole"] == pole]
            if ts_:
                fig.add_trace(go.Scatter(
                    x=[t["timestamp"] for t in ts_], y=[t["index"] for t in ts_], mode="markers",
                    marker=dict(size=12, symbol=POLE_SYMBOL[pole], color=POLE_COLOR[pole],
                                line=dict(width=1, color=GOLD)), name=f"Gesamt-Wende {POLE_LABEL[pole]}",
                    customdata=[[t["dominant"], ", ".join(t["aligned"]) or "–"] for t in ts_],
                    hovertemplate=("%{x|%d.%m.%Y %H:%M}<br>dominant: %{customdata[0]}<br>"
                                   "gleichgerichtet: %{customdata[1]}<extra></extra>")))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_yaxes(range=[-1.15, 1.15], title_text="Index")
    return _base_layout(fig, 300, "Gesamt-Index (gewichtete Summe der TF-Wellen, −1 = Tief … +1 = Hoch)")


def matrix_figure(a, tfs, show_past=False):
    ref_now = a["ref_now"]
    fig = go.Figure()
    for kind, name, sel, mk in (
        ("plain", "Projektion", lambda p: not p["cross"] and not p["x2"], lambda pole: dict(
            size=9, symbol=POLE_SYMBOL[pole] + "-open", color=POLE_COLOR[pole], line=dict(width=1.5))),
        ("cross", "kreuzbestätigt", lambda p: p["cross"] and not p["x2"], lambda pole: dict(
            size=13, symbol="star", color=POLE_COLOR[pole], line=dict(width=1, color="white"))),
        ("x2", "2x-bestätigt", lambda p: p["x2"], lambda pole: dict(
            size=12, symbol="diamond", color=POLE_COLOR[pole], line=dict(width=2, color=GOLD))),
    ):
        for pole in ("High", "Low"):
            ps = [p for tf in tfs if tf in a["tfs"] for p in a["tfs"][tf]["projs"]
                  if p["pole"] == pole and sel(p) and (show_past or p["date"] > ref_now)]
            if not ps:
                continue
            fig.add_trace(go.Scatter(
                x=[p["date"] for p in ps], y=[p["timeframe"] for p in ps], mode="markers", marker=mk(pole),
                name=f"{name} {POLE_LABEL[pole]}",
                customdata=[[p["name"], p.get("bt", "")] for p in ps],
                hovertemplate=(f"<b>%{{y}} {POLE_LABEL[pole]}</b><br>%{{customdata[0]}}<br>"
                               "%{x|%d.%m.%Y %H:%M} UTC<br>%{customdata[1]}<extra></extra>")))
    fig.add_vline(x=pd.Timestamp(ref_now).value / 1e6, line_width=1.5, line_dash="dashdot",
                  line_color="#FFA726" if a.get("cutoff") is not None else "#888")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed([t for t in tfs if t in a["tfs"]])))
    return _base_layout(fig, 90 + 34 * len(tfs), "Projektionen aller Timeframes (▲ Tief | ▼ Hoch | ★ kreuz | ◆ 2x)")


def rolling_figure(per_cut):
    dfc = pd.DataFrame(per_cut)
    fig = go.Figure()
    for col, color, dash in (("Treffer %", "#00E5FF", None), ("Grundrate %", "#00E5FF", "dot"),
                             ("exakt %", "#FFD700", None), ("Inversion %", "#FF3366", None)):
        if col in dfc:
            fig.add_trace(go.Scatter(x=dfc["Stichtag"], y=dfc[col], name=col, mode="lines+markers",
                                     line=dict(color=color, dash=dash)))
    fig.update_yaxes(title_text="%")
    return _base_layout(fig, 320, "Rollierender Backtest je Stichtag")


def fmt_window(p):
    return f"{fmt_ts(p['date'])} UTC"
