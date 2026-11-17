"""
Time Predictor v2 – Analyse-Engine (ohne UI, ohne scipy).

Neue Methode (method="v2", Standard):
  Stufe 1   Pivots mit 1 Kerze links / 1 Kerze rechts, Folge H-L-H-L erzwingen. Bei Außenstäben (Hoch UND Tief
            in derselben Kerze) entscheidet der kleinere Timeframe, was zuerst kam.
  Stufe 2   Messung aus BENACHBARTEN Paaren:  H→H ergibt ein Tief, L→L ergibt ein Hoch
  Stufe 2a  Messung aus dem ZWEITNÄCHSTEN Paar: H→2.H ergibt ein Hoch, L→2.L ergibt ein Tief
            Ziel = jüngerer Pivot + aufgerundet(Abstand / 2) Kerzen. Liegt dort (Kerze +1) kein Pivot des
            erwarteten Pols, wird ein entgangener Pivot gesucht: zuerst eine Gegenkerze im Trend (rote Kerze,
            deren Tief nicht unter das Vorkerzen-Tief fällt = mögliches Tief; grüne Kerze, deren Hoch nicht über
            das Vorkerzen-Hoch steigt = mögliches Hoch), sonst – wenn im Fenster gar kein Pivot liegt – das
            Extrem im Fenster. Der Gegen-Pivot wird mit eingefügt, damit H-L-H-L erhalten bleibt.
  Stufe 3   RSI(1) markiert nur (✓/✗), es wird KEIN Pivot gelöscht.
  Prüfung   Reihenfolge H-L-H-L und Extrem-Regel (jedes Hoch = höchstes Hoch zwischen seinen Nachbar-Tiefs).
  Projektion  ab letztem Tief: zum 1., 2., 3. … vergangenen Tief, aufgerundet(Abstand / 2) → Hoch, Tief, Hoch …
              ab letztem Hoch analog → Tief, Hoch, Tief …

Alte Methode (method="legacy"): unverändert wie in Time Predictor v1 (0,5× mit halben Kerzen, Pol-Konflikt
blockiert das Einfügen, RSI-Prüfung kann Pivots entfernen).
"""
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from tp_data import future_ts

POLE_LABEL = {"High": "Hoch", "Low": "Tief"}
POLE_COLOR = {"High": "#FF3366", "Low": "#00E676"}
POLE_SYMBOL = {"High": "triangle-down", "Low": "triangle-up"}
OPP = {"High": "Low", "Low": "High"}
SRC_LABEL = {
    "S1": "Stufe 1 (1L/1R)",
    "S1r": "Stufe 1 (rekonstruiert, Alternierung)",
    "S2": "Stufe 2 (H→H / L→L)",
    "S2g": "Stufe 2 (Gegen-Pivot)",
    "S2a": "Stufe 2a (H→2.H / L→2.L)",
    "S2ag": "Stufe 2a (Gegen-Pivot)",
}
# Toleranz – fest nach Vorgabe: nur die projizierte Kerze und die EINE Kerze danach ("+1", kein "−1")
TOL_BEFORE, TOL_AFTER = 0, 1
TOL_LABEL = "+1 Kerze"


@dataclass(frozen=True)
class TPParams:
    method: str = "v2"           # "v2" = neue Methode, "legacy" = alte Methode
    search_tol: float = 1.0      # Stufe 2/2a: Suchfenster ± Kerzen (v2: fest 1)
    max_iter: int = 3            # Stufe 2/2a: Durchläufe (neue Pivots erzeugen neue Messungen)
    use_stage2: bool = True
    use_stage2a: bool = True
    use_counter: bool = True     # v2: Gegenkerzen im Trend als entgangene Pivots
    use_lower: bool = True       # v2: Außenstäbe über den kleineren Timeframe auflösen
    use_rsi: bool = True         # Stufe 3
    rsi_len: int = 1
    rsi_tol: int = 1             # Stufe 3: Toleranz ± Kerzen
    rsi_remove: bool = True      # nur alte Methode: nicht bestätigte Pivots entfernen
    rsi_on_insert: bool = False  # nur alte Methode
    x2_tol: int = 0              # Stufe 3a: Toleranz ± Kerzen (des kleinen TF) um die 2x-Kerze
    n_lookback: int = 10         # Projektion: Anzahl vergangener Tiefs/Hochs (bis 30)
    proj_tol: float = 1.0        # Kreuz-Bestätigung Tief-Kette ↔ Hoch-Kette (± Kerzen)
    proj_source: str = "all"     # "all" = alle Pivots, "star" = nur ★-Pivots (2x-bestätigt)
    limit: int = 500
    drop_incomplete: bool = True

    @property
    def v2(self):
        return self.method == "v2"


# ==========================================
# HILFSFUNKTIONEN
# ==========================================
def utc_now():
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def fmt_ts(ts):
    return pd.Timestamp(ts).strftime("%d.%m.%Y %H:%M")


def fmt_ahead(td):
    h = td.total_seconds() / 3600.0
    sign = "+" if h >= 0 else "−"
    h = abs(h)
    return f"{sign}{h:.1f} h" if h < 48 else f"{sign}{h / 24:.1f} Tage"


def _rate(xs):
    return round(100.0 * float(np.mean(xs)), 1) if len(xs) else 0.0


def bar_delta(df):
    d = df["timestamp"].diff().dropna()
    return d.median() if len(d) else pd.Timedelta(days=1)


def candle_end_times(df, delta):
    ts = df["timestamp"]
    nxt = ts.shift(-1)
    return nxt.where(nxt.notna() & (nxt <= ts + 1.5 * delta), ts + delta)


def slice_visible(df_full, cutoff, limit, drop_incomplete, now):
    """Nur Kerzen, die vor dem Stichtag (bzw. jetzt) vollständig abgeschlossen sind (kein Lookahead)."""
    if df_full is None or df_full.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    delta = bar_delta(df_full)
    end = candle_end_times(df_full, delta)
    if "end" in df_full.columns:               # Aktien: explizites Kerzenende (Handelstage-Kalender)
        end = df_full["end"]
    if cutoff is not None:
        mask = end <= cutoff
    elif drop_incomplete:
        mask = end <= now
    else:
        mask = pd.Series(True, index=df_full.index)
    return df_full[mask].tail(limit).reset_index(drop=True)


def _arrays(df):
    return (df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float),
            df["open"].to_numpy(dtype=float), df["close"].to_numpy(dtype=float), df["timestamp"])


def _mk(i, typ, ts, highs, lows, src):
    return {"index": int(i), "timestamp": ts.iloc[int(i)], "type": typ,
            "value": float(highs[i] if typ == "High" else lows[i]),
            "src": src, "support": 0, "rsi_ok": None, "star": False, "star_ts": None,
            "reason": "", "counter": False, "outside": False, "uncertain": False}


def _ext_idx(highs, lows, s, e, typ):
    """Index des Extrems vom Typ `typ` im Bereich [s, e] (inklusive) oder None."""
    if e < s:
        return None
    return s + (int(np.argmax(highs[s:e + 1])) if typ == "High" else int(np.argmin(lows[s:e + 1])))


def lower_tf_order(df, lower):
    """Funktion k → 'High'/'Low'/None: welches Extrem der Kerze k kam zuerst (laut kleinerem Timeframe)?"""
    if lower is None or lower.empty or df is None or df.empty:
        return None
    lts = lower["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    lh = lower["high"].to_numpy(dtype=float)
    ll = lower["low"].to_numpy(dtype=float)
    starts = df["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    ends = (df["end"] if "end" in df.columns else candle_end_times(df, bar_delta(df)))
    ends = pd.to_datetime(ends).to_numpy(dtype="datetime64[ns]").astype(np.int64)

    def order(k):
        a, b = np.searchsorted(lts, starts[k], "left"), np.searchsorted(lts, ends[k], "left")
        if b - a < 2:
            return None
        ih, il = a + int(np.argmax(lh[a:b])), a + int(np.argmin(ll[a:b]))
        if ih == il:
            return None
        return "High" if ih < il else "Low"
    return order


# ==========================================
# STUFE 1 – 1 Kerze links / 1 Kerze rechts
# ==========================================
def stage1(df, order=None, log=None):
    """Hoch: Hoch > Hoch der Kerze links und ≥ Hoch der Kerze rechts (Tief analog).
    Danach wird die Folge H-L-H-L erzwungen: zwischen zwei gleichartigen Pivots wird das
    Gegen-Extrem rekonstruiert (Quelle 'S1r'), bei direkt benachbarten Kerzen bleibt das extremere.
    order(k) (optional) liefert für Außenstäbe, welches Extrem zuerst kam."""
    highs, lows, opens, closes, ts = _arrays(df)
    n = len(highs)
    if n < 3:
        return []
    i = np.arange(1, n - 1)
    hs = set(i[(highs[i] > highs[i - 1]) & (highs[i] >= highs[i + 1])].tolist())
    ls = set(i[(lows[i] < lows[i - 1]) & (lows[i] <= lows[i + 1])].tolist())
    raw = []
    for k in sorted(hs | ls):
        if k in hs and k in ls:      # Außenstab: Hoch und Tief in derselben Kerze
            first = order(k) if order else None
            unsure = first is None
            if first is None:
                last = raw[-1]["type"] if raw else None
                first = ("Low" if last == "High" else "High") if last else ("Low" if closes[k] >= opens[k] else "High")
            a, b = _mk(k, first, ts, highs, lows, "S1"), _mk(k, OPP[first], ts, highs, lows, "S1")
            for q in (a, b):
                q["outside"], q["uncertain"] = True, unsure
            raw += [a, b]
        else:
            raw.append(_mk(k, "High" if k in hs else "Low", ts, highs, lows, "S1"))
    if not raw:
        return []
    out = [raw[0]]
    for p in raw[1:]:
        last = out[-1]
        if p["type"] != last["type"]:
            out.append(p)
            continue
        s, e = last["index"] + 1, p["index"] - 1
        j = _ext_idx(highs, lows, s, e, OPP[p["type"]])
        if j is not None:
            r = _mk(j, OPP[p["type"]], ts, highs, lows, "S1r")
            r["reason"] = "Alternierung: Gegen-Extrem zwischen zwei gleichen Pivots"
            out += [r, p]
            if log is not None:
                log.append({"Zeit": r["timestamp"], "Typ": r["type"], "Aktion": "rekonstruiert",
                            "Grund": f"zwischen {POLE_LABEL[p['type']]} {fmt_ts(last['timestamp'])} und "
                                     f"{fmt_ts(p['timestamp'])}"})
        else:
            better = (p["value"] > last["value"]) if p["type"] == "High" else (p["value"] < last["value"])
            keep, drop = (p, last) if better else (last, p)
            out[-1] = keep
            if log is not None:
                log.append({"Zeit": drop["timestamp"], "Typ": drop["type"], "Aktion": "zusammengelegt",
                            "Grund": f"direkt benachbart mit gleichem Pol – extremeres {fmt_ts(keep['timestamp'])}"
                                     " bleibt"})
    return out


# ==========================================
# STUFE 2 / 2a – ALTE METHODE (unverändert)
# ==========================================
def _targets(pivs, gap):
    out = []
    for typ in ("High", "Low"):
        same = [p for p in pivs if p["type"] == typ]
        for j in range(gap, len(same)):
            a, b = same[j - gap], same[j]
            d = b["index"] - a["index"]
            if d < 2:
                continue
            out.append((b["index"] + 0.5 * d, OPP[typ] if gap == 1 else typ, a, b))
    return sorted(out, key=lambda x: x[0])


def _check_or_insert(pivs, t, X, P, highs, lows, ts, last_ok, src, st):
    w = P.search_tol
    near = [p for p in pivs if abs(p["index"] - t) <= w]
    hit = [p for p in near if p["type"] == X]
    if hit:
        min(hit, key=lambda p: abs(p["index"] - t))["support"] += 1
        st["confirmed"] += 1
        return
    if near:
        st["conflicts"] += 1
        return
    s, e = max(1, int(math.ceil(t - w))), min(last_ok, int(math.floor(t + w)))
    c = _ext_idx(highs, lows, s, e, X)
    if c is None:
        st["unresolved"] += 1
        return
    idxs = [p["index"] for p in pivs]
    k = bisect_right(idxs, c)
    if k == 0:
        st["unresolved"] += 1
        return
    A = pivs[k - 1]
    B = pivs[k] if k < len(pivs) else None
    Y = OPP[X]
    if A["type"] == X:
        d = _ext_idx(highs, lows, A["index"] + 1, c - 1, Y)
        new = None if d is None else [_mk(d, Y, ts, highs, lows, src + "g"), _mk(c, X, ts, highs, lows, src)]
    elif B is not None:
        d = _ext_idx(highs, lows, c + 1, B["index"] - 1, Y)
        new = None if d is None else [_mk(c, X, ts, highs, lows, src), _mk(d, Y, ts, highs, lows, src + "g")]
    else:
        new = [_mk(c, X, ts, highs, lows, src)]
    if new is None:
        st["unresolved"] += 1
        return
    if len(new) == 2:
        lo_p, hi_p = (new[0], new[1]) if new[0]["type"] == "Low" else (new[1], new[0])
        if lo_p["value"] >= hi_p["value"]:
            st["unresolved"] += 1
            return
    if P.use_rsi and P.rsi_on_insert and st.get("_turns") is not None:
        top, bot = st["_turns"]
        wr = int(P.rsi_tol)
        for q in new:
            arr = top if q["type"] == "High" else bot
            if not arr[max(0, q["index"] - wr):q["index"] + wr + 1].any():
                st["rsi_rejected"] += 1
                return
    new[0 if new[0]["type"] == X else 1]["support"] = 1
    pivs[k:k] = new
    st["inserted"] += len(new)


def stage_halfcycle(pivs, df, P, gap, src):
    highs, lows, _, _, ts = _arrays(df)
    last_ok = len(highs) - 2
    st = {"checked": 0, "confirmed": 0, "conflicts": 0, "inserted": 0, "unresolved": 0, "rsi_rejected": 0,
          "_turns": rsi_turns(rsi(df["close"].to_numpy(dtype=float), P.rsi_len)) if P.rsi_on_insert else None}
    for _ in range(max(1, int(P.max_iter))):
        before = st["inserted"]
        for key in ("checked", "confirmed", "conflicts", "unresolved", "rsi_rejected"):
            st[key] = 0
        for t, X, _a, _b in _targets(pivs, gap):
            if t > last_ok:
                continue
            st["checked"] += 1
            _check_or_insert(pivs, t, X, P, highs, lows, ts, last_ok, src, st)
        if st["inserted"] == before:
            break
    st.pop("_turns", None)
    return st


# ==========================================
# STUFE 2 / 2a – NEUE METHODE (entgangene Pivots)
# ==========================================
def _targets_v2(pivs, gap):
    """Ziel = jüngerer Pivot + aufgerundet(Abstand / 2) Kerzen (ganze Kerze).
    gap=1: H→H ⇒ Tief, L→L ⇒ Hoch; gap=2: H→2.H ⇒ Hoch, L→2.L ⇒ Tief."""
    out = []
    for typ in ("High", "Low"):
        same = [p for p in pivs if p["type"] == typ]
        for j in range(gap, len(same)):
            a, b = same[j - gap], same[j]
            d = b["index"] - a["index"]
            if d < 2:
                continue
            out.append((b["index"] + math.ceil(d / 2), OPP[typ] if gap == 1 else typ, a, b, d))
    return sorted(out, key=lambda x: x[0])


def _is_counter(c, X, opens, closes, highs, lows):
    """Gegenkerze im Trend: mögliches Tief = rote Kerze, deren Tief nicht unter das Vorkerzen-Tief fällt;
    mögliches Hoch = grüne Kerze, deren Hoch nicht über das Vorkerzen-Hoch steigt."""
    if c < 1:
        return False
    if X == "Low":
        return closes[c] < opens[c] and lows[c] >= lows[c - 1]
    return closes[c] > opens[c] and highs[c] <= highs[c - 1]


def _val(highs, lows, i, typ):
    return highs[i] if typ == "High" else lows[i]


def _insert_pair(pivs, c, X, highs, lows, ts, last_ok, src):
    """Pivot X an Kerze c einfügen, plus Gegen-Pivot Y, so dass H-L-H-L und die Extrem-Regel gelten.
    Rückgabe (Position, [neue Pivots]) oder None."""
    idxs = [p["index"] for p in pivs]
    k = bisect_right(idxs, c)
    if k == 0:
        return None
    A = pivs[k - 1]
    if A["index"] == c:
        return None
    B = pivs[k] if k < len(pivs) else None
    Y = OPP[X]
    vc = _val(highs, lows, c, X)
    if A["type"] == X:                              # A(X) … d(Y) … c(X) … B(Y)
        d = _ext_idx(highs, lows, A["index"] + 1, c - 1, Y)
        if d is None:
            return None
        end = B["index"] - 1 if B is not None else last_ok
        e = _ext_idx(highs, lows, d + 1, end, X)
        if e is None or _val(highs, lows, e, X) != vc:
            return None
        new = [_mk(d, Y, ts, highs, lows, src + "g"), _mk(c, X, ts, highs, lows, src)]
    elif B is not None:                             # A(Y) … c(X) … d(Y) … B(X)
        d = _ext_idx(highs, lows, c + 1, B["index"] - 1, Y)
        if d is None:
            return None
        e = _ext_idx(highs, lows, A["index"] + 1, d - 1, X)
        if e is None or _val(highs, lows, e, X) != vc:
            return None
        new = [_mk(c, X, ts, highs, lows, src), _mk(d, Y, ts, highs, lows, src + "g")]
    else:                                           # A(Y) … c(X) … Datenende
        e = _ext_idx(highs, lows, A["index"] + 1, last_ok, X)
        if e is None or _val(highs, lows, e, X) != vc:
            return None
        new = [_mk(c, X, ts, highs, lows, src)]
    if len(new) == 2:
        lo_p, hi_p = (new[0], new[1]) if new[0]["type"] == "Low" else (new[1], new[0])
        if lo_p["value"] >= hi_p["value"]:
            return None
    return k, new


def stage_halfcycle_v2(pivs, df, P, gap, src, log=None):
    highs, lows, opens, closes, ts = _arrays(df)
    last_ok = len(highs) - 2                # Pivot braucht 1 abgeschlossene Kerze rechts
    st = {"checked": 0, "confirmed": 0, "inserted": 0, "counter": 0, "conflicts": 0, "unresolved": 0}
    for _ in range(max(1, int(P.max_iter))):
        before = st["inserted"]
        for key in ("checked", "confirmed", "conflicts", "unresolved"):
            st[key] = 0
        for t, X, a, b, d in _targets_v2(pivs, gap):
            if t > last_ok:
                continue                     # liegt in der Zukunft → das ist Projektion, nicht Historie
            st["checked"] += 1
            idxs = [p["index"] for p in pivs]
            near = pivs[bisect_left(idxs, t - TOL_BEFORE):bisect_right(idxs, t + TOL_AFTER)]
            hit = [p for p in near if p["type"] == X]
            if hit:
                min(hit, key=lambda p: abs(p["index"] - t))["support"] += 1
                st["confirmed"] += 1
                continue
            wlo, whi = max(1, t - TOL_BEFORE), min(last_ok, t + TOL_AFTER)
            taken = {p["index"] for p in near}
            cands = []
            if P.use_counter:
                cands += [(c, True) for c in range(wlo, whi + 1)
                          if c not in taken and _is_counter(c, X, opens, closes, highs, lows)]
            if not near:
                e = _ext_idx(highs, lows, wlo, whi, X)
                if e is not None and all(e != c for c, _ in cands):
                    cands.append((e, False))
            done = False
            for c, is_ctr in cands:
                res = _insert_pair(pivs, c, X, highs, lows, ts, last_ok, src)
                if res is None:
                    continue
                k, new = res
                rule = (f"{a['type'][0]}→{'2.' if gap == 2 else ''}{b['type'][0]} "
                        f"{fmt_ts(a['timestamp'])} → {fmt_ts(b['timestamp'])}: {d} K. → "
                        f"{POLE_LABEL[X]} nach {math.ceil(d / 2)} K.")
                for q in new:
                    if q["type"] == X:
                        q["support"], q["counter"] = 1, is_ctr
                        q["reason"] = rule + (" (Gegenkerze)" if is_ctr else " (Extrem im Fenster)")
                    else:
                        q["reason"] = "Gegen-Pivot für H-L-H-L zu " + fmt_ts(ts.iloc[c])
                pivs[k:k] = new
                st["inserted"] += len(new)
                st["counter"] += int(is_ctr)
                if log is not None:
                    for q in new:
                        log.append({"Zeit": q["timestamp"], "Typ": q["type"],
                                    "Aktion": "eingefügt (" + ("Gegenkerze" if q["counter"] else
                                                               "Gegen-Pivot" if q["src"].endswith("g") else
                                                               "Extrem") + ")",
                                    "Grund": q["reason"]})
                done = True
                break
            if not done:
                st["conflicts" if near else "unresolved"] += 1
        if st["inserted"] == before:
            break
    return st


# ==========================================
# STUFE 3 – RSI(1)-Polaritätsprüfung
# ==========================================
def rsi(close, length=1):
    close = np.asarray(close, dtype=float)
    diff = np.diff(close, prepend=close[0])
    gain = pd.Series(np.clip(diff, 0, None)).ewm(alpha=1.0 / max(1, length), adjust=False).mean().to_numpy()
    loss = pd.Series(np.clip(-diff, 0, None)).ewm(alpha=1.0 / max(1, length), adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + gain / loss)
    out = np.where(loss == 0, np.where(gain == 0, 50.0, 100.0), out)
    return out


def rsi_turns(r):
    n = len(r)
    top = np.zeros(n, bool)
    bot = np.zeros(n, bool)
    if n >= 3:
        top[1:-1] = (r[1:-1] >= r[:-2]) & (r[1:-1] > r[2:])
        bot[1:-1] = (r[1:-1] <= r[:-2]) & (r[1:-1] < r[2:])
    return top, bot


def stage3(pivs, df, P):
    """RSI-Wende innerhalb ± rsi_tol Kerzen. Neue Methode: nur markieren. Alte Methode: optional entfernen."""
    st = {"checked": len(pivs), "ok": 0, "failed": 0, "removed": 0}
    if not P.use_rsi or not pivs:
        return pivs, st
    r = rsi(df["close"].to_numpy(dtype=float), P.rsi_len)
    top, bot = rsi_turns(r)
    n, w = len(r), int(P.rsi_tol)
    for p in pivs:
        lo, hi = max(0, p["index"] - w), min(n, p["index"] + w + 1)
        p["rsi_ok"] = bool((top if p["type"] == "High" else bot)[lo:hi].any())
        st["ok" if p["rsi_ok"] else "failed"] += 1
    if P.v2 or not P.rsi_remove:
        return pivs, st
    kept = [p for p in pivs if p["rsi_ok"]]
    st["removed"] = len(pivs) - len(kept)
    out = []
    for p in kept:
        if out and out[-1]["type"] == p["type"]:
            if (p["value"] > out[-1]["value"]) if p["type"] == "High" else (p["value"] < out[-1]["value"]):
                out[-1] = p
            st["removed"] += 1
            continue
        out.append(p)
    return out, st


# ==========================================
# REIHENFOLGE-PRÜFUNG
# ==========================================
def validate_sequence(pivs, df):
    """H-L-H-L und Extrem-Regel: jedes Hoch ist das höchste Hoch zwischen seinen Nachbar-Tiefs
    (Tief analog; die Kerzen der Nachbar-Pivots selbst zählen nicht mit). Es wird nichts verändert – nur gemeldet."""
    highs, lows = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    issues = []
    for i, p in enumerate(pivs):
        if i > 0:
            q = pivs[i - 1]
            if q["type"] == p["type"]:
                issues.append({"Zeit": p["timestamp"], "Typ": p["type"],
                               "Problem": f"zwei {POLE_LABEL[p['type']]}s in Folge (vorher {fmt_ts(q['timestamp'])})"})
            if q["index"] > p["index"]:
                issues.append({"Zeit": p["timestamp"], "Typ": p["type"], "Problem": "zeitliche Reihenfolge verletzt"})
        if 0 < i < len(pivs) - 1:
            # Bereich ZWISCHEN den Nachbar-Pivots (deren eigene Kerzen ausgenommen – wie bei der Rekonstruktion)
            s, e = pivs[i - 1]["index"] + 1, pivs[i + 1]["index"] - 1
            if s <= p["index"] <= e:
                if p["type"] == "High" and highs[s:e + 1].max() > p["value"]:
                    j = s + int(np.argmax(highs[s:e + 1]))
                    issues.append({"Zeit": p["timestamp"], "Typ": p["type"],
                                   "Problem": f"nicht das höchste Hoch zwischen den Nachbar-Tiefs "
                                              f"(höher: {fmt_ts(df['timestamp'].iloc[j])})"})
                if p["type"] == "Low" and lows[s:e + 1].min() < p["value"]:
                    j = s + int(np.argmin(lows[s:e + 1]))
                    issues.append({"Zeit": p["timestamp"], "Typ": p["type"],
                                   "Problem": f"nicht das tiefste Tief zwischen den Nachbar-Hochs "
                                              f"(tiefer: {fmt_ts(df['timestamp'].iloc[j])})"})
    return issues


# ==========================================
# PIPELINE
# ==========================================
def detect_pivots(df, P: TPParams, halfcycle=True, order=None):
    """halfcycle=False → nur Stufe 1 + 3."""
    stats = {"s1": 0, "s2": None, "s2a": None, "s3": None, "final": 0, "log": [], "issues": [],
             "outside": 0, "outside_unsure": 0}
    if df is None or len(df) < 5:
        return [], stats
    log = stats["log"]
    pivs = stage1(df, order if P.v2 and P.use_lower else None, log)
    stats["s1"] = len(pivs)
    stats["outside"] = sum(p["outside"] for p in pivs) // 2
    stats["outside_unsure"] = sum(p["outside"] and p["uncertain"] for p in pivs) // 2
    if P.v2:
        if halfcycle and P.use_stage2 and len(pivs) >= 3:
            stats["s2"] = stage_halfcycle_v2(pivs, df, P, 1, "S2", log)
        if halfcycle and P.use_stage2a and len(pivs) >= 5:
            stats["s2a"] = stage_halfcycle_v2(pivs, df, P, 2, "S2a", log)
    else:
        if halfcycle and P.use_stage2 and len(pivs) >= 3:
            stats["s2"] = stage_halfcycle(pivs, df, P, 1, "S2")
        if halfcycle and P.use_stage2a and len(pivs) >= 5:
            stats["s2a"] = stage_halfcycle(pivs, df, P, 2, "S2a")
    pivs, stats["s3"] = stage3(pivs, df, P)
    stats["final"] = len(pivs)
    stats["issues"] = validate_sequence(pivs, df)
    return pivs, stats


# ==========================================
# STUFE 3a – Abgleich mit dem 2x-Timeframe
# ==========================================
def candle_ends(vis):
    if "end" in vis.columns:
        return list(vis["end"])
    return list(candle_end_times(vis, bar_delta(vis)))


def mark_2x(pivs, pivs2, delta, ends2, tol_bars=0):
    for p in pivs:
        p["star"], p["star_ts"] = False, None
    tol = tol_bars * delta
    n = 0
    for q in pivs2:
        lo, hi = q["timestamp"] - tol, ends2[q["index"]] + tol
        cand = [p for p in pivs if p["type"] == q["type"] and lo <= p["timestamp"] < hi and not p["star"]]
        if not cand:
            continue
        best = min(cand, key=lambda p: (abs(p["value"] - q["value"]), p["timestamp"]))
        best["star"], best["star_ts"] = True, q["timestamp"]
        n += 1
    return n


# ==========================================
# STUFEN-STERNE – Abgleich mit ALLEN größeren Timeframes
# ==========================================
def match_higher(pivs, pivs_hi, ends_hi):
    """Pivots des kleinen TF, die einen Pivot des größeren TF bilden: je Pivot q des großen TF wird unter den
    gleichpoligen kleinen Pivots in der Zeitspanne der großen Kerze [Beginn, Ende) genau EINER gewählt – der,
    dessen Kurs dem Extrem der großen Kerze am nächsten liegt (= die Kerze, die das Hoch/Tief gemacht hat).
    Rückgabe: Liste der getroffenen kleinen Pivots (Objekte aus `pivs`)."""
    by_type = {"High": [], "Low": []}
    for p in pivs:
        by_type[p["type"]].append(p)
    keys = {t: [p["timestamp"].value for p in ps] for t, ps in by_type.items()}
    out = []
    for q in pivs_hi:
        if q["index"] >= len(ends_hi):
            continue
        lo, hi = q["timestamp"].value, pd.Timestamp(ends_hi[q["index"]]).value
        ps, ks = by_type[q["type"]], keys[q["type"]]
        cand = ps[bisect_left(ks, lo):bisect_left(ks, hi)]
        if cand:
            out.append(min(cand, key=lambda p: (abs(p["value"] - q["value"]), p["timestamp"])))
    return out


def level_pivots(frames, pool, cutoff, P, now, lowers=None, known=None):
    """Pivots je Timeframe des Pools zum Stichtag (gleiche Erkennung). known = bereits analysierte TFs."""
    out = {}
    for t in pool:
        if known and t in known:
            out[t] = (known[t]["pivots"], candle_ends(known[t]["visible"]))
            continue
        df = frames.get(t)
        if df is None or df.empty:
            continue
        vis = slice_visible(df, cutoff, P.limit, P.drop_incomplete, now)
        if len(vis) < 10:
            continue
        pivs, _ = detect_pivots(vis, P, order=lower_tf_order(vis, frames.get((lowers or {}).get(t))))
        out[t] = (pivs, candle_ends(vis))
    return out


def mark_levels(a, lvl, dur):
    """Setzt für jeden Pivot jedes analysierten TF p['levels'] = größere TFs, auf denen er ebenfalls ein Pivot ist
    (sortiert klein → groß). dur(tf) liefert die Kerzendauer."""
    for tf, r in a["tfs"].items():
        for p in r["pivots"]:
            p["levels"] = []
        for H in sorted((t for t in lvl if dur(t) > dur(tf)), key=dur):
            pivs_hi, ends_hi = lvl[H]
            for p in match_higher(r["pivots"], pivs_hi, ends_hi):
                p["levels"].append(H)


# ==========================================
# PROJEKTION
# ==========================================
def make_idx_to_ts(vis, tf, calendar):
    ns = vis["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    n = len(ns)
    last = pd.Timestamp(int(ns[-1]))
    delta = bar_delta(vis)

    def f(x):
        if x <= n - 1:
            x = max(0.0, float(x))
            i = int(math.floor(x))
            if i >= n - 1:
                return last
            return pd.Timestamp(int(ns[i] + (x - i) * (ns[i + 1] - ns[i])))
        return future_ts(last, x - (n - 1), tf, calendar, delta)
    return f


def project(pivs, tf, n_lookback, ref_now, idx_to_ts, v2=True):
    """Ab dem letzten Tief: Abstand zum 1., 2., 3. … vergangenen Tief, halbiert → Hoch, Tief, Hoch …
    Ab dem letzten Hoch analog → Tief, Hoch, Tief … Neue Methode: aufgerundet auf ganze Kerzen."""
    out = []
    for typ in ("Low", "High"):
        same = [p for p in pivs if p["type"] == typ]
        if len(same) < 2:
            continue
        anchor = same[-1]
        for k in range(1, min(int(n_lookback), len(same) - 1) + 1):
            past = same[-(k + 1)]
            dist = anchor["index"] - past["index"]
            half = math.ceil(dist / 2) if v2 else 0.5 * dist
            pidx = anchor["index"] + half
            date = idx_to_ts(pidx)
            pole = OPP[typ] if k % 2 == 1 else typ
            out.append({
                "id": f"{tf}|{typ}|{k}", "timeframe": tf, "chain": typ, "k": k,
                "pole": pole, "pidx": pidx, "date": date, "date_end": idx_to_ts(pidx + 1),
                "anchor": anchor["timestamp"], "past": past["timestamp"],
                "chain_ts": [q["timestamp"] for q in same[-(k + 1):]],
                "dist_candles": int(dist), "proj_candles": half,
                "name": f"{typ[0]}_letztes ➔ {typ[0]}[-{k}]",
                "future": date > ref_now,
                "cross": False, "partners": [], "x2": False, "conflict": False,
            })
    return out


def mark_conflicts(projs):
    """Gleiche Kerze, aber unterschiedlicher Pol → Konflikt (beide Projektionen werden markiert)."""
    by = {}
    for p in projs:
        by.setdefault(round(float(p["pidx"]), 3), set()).add(p["pole"])
    for p in projs:
        p["conflict"] = len(by[round(float(p["pidx"]), 3)]) > 1


def cross_confirm(projs, tol):
    lows = [p for p in projs if p["chain"] == "Low"]
    highs = [p for p in projs if p["chain"] == "High"]
    for a in lows:
        for b in highs:
            if a["pole"] == b["pole"] and abs(a["pidx"] - b["pidx"]) <= tol:
                a["cross"] = b["cross"] = True
                a["partners"].append(b["id"])
                b["partners"].append(a["id"])
    clusters = []
    for pole in ("High", "Low"):
        cs = sorted([p for p in projs if p["cross"] and p["pole"] == pole], key=lambda p: p["pidx"])
        cur = []
        for p in cs:
            if cur and p["pidx"] - cur[-1]["pidx"] > tol:
                clusters.append(cur)
                cur = []
            cur.append(p)
        if cur:
            clusters.append(cur)
    out = []
    for cl in clusters:
        ns = [p["date"].value for p in cl]
        out.append({
            "timeframe": cl[0]["timeframe"], "pole": cl[0]["pole"],
            "date": pd.Timestamp(int(np.mean(ns))), "start": cl[0]["date"], "end": cl[-1]["date"],
            "n": len(cl), "members": [p["id"] for p in cl], "x2": any(p["x2"] for p in cl),
            "detail": ", ".join(p["name"] for p in cl),
        })
    return sorted(out, key=lambda c: c["date"])


def mark_proj_2x(projs, projs2, delta, tol_bars=0):
    tol = tol_bars * delta
    for p in projs:
        p["x2"] = any(q["pole"] == p["pole"] and q["date"] - tol <= p["date"] < q["date_end"] + tol
                      for q in projs2)


# ==========================================
# ANALYSE JE TF UND GESAMT
# ==========================================
def analyze_tf(df_full, df2_full, tf, tf2, cutoff, P: TPParams, now, calendar="crypto", lower_full=None):
    ref_now = cutoff if cutoff is not None else now
    vis = slice_visible(df_full, cutoff, P.limit, P.drop_incomplete, now)
    if len(vis) < 10:
        return None
    delta = bar_delta(vis)
    pivs, stats = detect_pivots(vis, P, order=lower_tf_order(vis, lower_full))
    res = {"tf": tf, "tf2": tf2, "visible": vis, "delta": delta, "pivots": pivs, "stats": stats,
           "pivots2": [], "delta2": None, "visible2": None, "stats2": None, "n_star": 0, "projs2": [],
           "calendar": calendar, "lower_full": lower_full}

    if df2_full is not None and not df2_full.empty:
        vis2 = slice_visible(df2_full, cutoff, P.limit, P.drop_incomplete, now)
        vis2 = vis2[vis2["timestamp"] >= vis["timestamp"].iloc[0] - 4 * delta].reset_index(drop=True)
        if len(vis2) >= 10:
            pivs2, stats2 = detect_pivots(vis2, P, order=lower_tf_order(vis2, df_full))
            res.update(visible2=vis2, delta2=bar_delta(vis2), pivots2=pivs2, stats2=stats2,
                       n_star=mark_2x(pivs, pivs2, delta, candle_ends(vis2), P.x2_tol))

    src = [p for p in pivs if p["star"]] if P.proj_source == "star" else pivs
    projs = project(src, tf, P.n_lookback, ref_now, make_idx_to_ts(vis, tf, calendar), P.v2)
    if res["pivots2"]:
        res["projs2"] = project(res["pivots2"], tf2, P.n_lookback, ref_now,
                                make_idx_to_ts(res["visible2"], tf2, calendar), P.v2)
        mark_proj_2x(projs, res["projs2"], delta, P.x2_tol)
    mark_conflicts(projs)
    res["projs"] = projs
    res["clusters"] = cross_confirm(projs, P.proj_tol)
    for c in res["clusters"]:
        c["future"] = c["date"] > ref_now
    return res


def analyze_symbol(frames, tfs, doubles, cutoff, P: TPParams, now, calendar="crypto", lowers=None):
    """lowers = {tf: kleinerer TF} für die Auflösung von Außenstäben (nur neue Methode)."""
    ref_now = cutoff if cutoff is not None else now
    out = {"ref_now": ref_now, "tfs": {}, "warnings": [], "calendar": calendar, "cutoff": cutoff}
    lowers = lowers or {}
    for tf in tfs:
        tf2 = doubles.get(tf)
        df = frames.get(tf)
        if df is None or df.empty:
            out["warnings"].append(f"{tf}: keine Daten.")
            continue
        df2 = frames.get(tf2) if tf2 else None
        if tf2 and (df2 is None or df2.empty):
            out["warnings"].append(f"{tf}: keine Daten für den 2x-TF {tf2} – keine ★-Markierung.")
        r = analyze_tf(df, df2, tf, tf2, cutoff, P, now, calendar=calendar, lower_full=frames.get(lowers.get(tf)))
        if r is None:
            out["warnings"].append(f"{tf}: zu wenige abgeschlossene Kerzen – übersprungen.")
            continue
        r["render"] = df[df["timestamp"] >= r["visible"]["timestamp"].iloc[0]].reset_index(drop=True)
        out["tfs"][tf] = r
    return out


# ==========================================
# GESAMTKURVE
# ==========================================
NODE_MODES = {
    "Alle Projektionen": "all",
    "Nur kreuzbestätigte (Tief- & Hoch-Kette)": "cross",
    "Nur 2x-bestätigte": "x2",
    "Kreuz- ODER 2x-bestätigt": "any",
    "Kreuz- UND 2x-bestätigt": "both",
}


def _keep(p, mode):
    return {"all": True, "cross": p["cross"], "x2": p["x2"], "any": p["cross"] or p["x2"],
            "both": p["cross"] and p["x2"]}[mode]


def build_wave_nodes(projs, ref_now, mode, boost_cross, boost_x2):
    fut = sorted((p for p in projs if p["date"] > ref_now and _keep(p, mode)), key=lambda p: p["date"])
    groups = []
    for p in fut:
        if groups and groups[-1]["pole"] == p["pole"]:
            groups[-1]["items"].append(p)
        else:
            groups.append({"pole": p["pole"], "items": [p]})
    nodes = []
    for g in groups:
        ns = int(np.mean([p["date"].value for p in g["items"]]))
        if nodes and ns <= nodes[-1]["ns"]:
            continue
        amp = max(1.0 + boost_cross * p["cross"] + boost_x2 * p["x2"] for p in g["items"])
        nodes.append({"ns": ns, "value": amp if g["pole"] == "High" else -amp, "pole": g["pole"],
                      "cross": any(p["cross"] for p in g["items"]), "x2": any(p["x2"] for p in g["items"])})
    return nodes


def current_wave_position(df_vis, pivots):
    last_h = next((p for p in reversed(pivots) if p["type"] == "High"), None)
    last_l = next((p for p in reversed(pivots) if p["type"] == "Low"), None)
    if df_vis.empty or last_h is None or last_l is None or last_h["value"] <= last_l["value"]:
        return 0.0
    close = float(df_vis["close"].iloc[-1])
    return float(np.clip((close - last_l["value"]) / (last_h["value"] - last_l["value"]) * 2 - 1, -1, 1))


def relative_swing(pivots, df_vis, prefer_star=True):
    ps = [p for p in pivots if p["star"]] if prefer_star else pivots
    if len(ps) < 3:
        ps = pivots
    sw = [abs(b["value"] - a["value"]) / a["value"] for a, b in zip(ps[:-1], ps[1:]) if a["value"] > 0]
    if sw:
        return float(np.median(sw[-6:]))
    r = float(((df_vis["high"] - df_vis["low"]) / df_vis["close"]).median() * 3)
    return r if np.isfinite(r) else 0.0


def wave_on_grid(grid_ns, ref_ns, start_val, nodes, fallback_fade_ns):
    xs, ys = [float(ref_ns)], [start_val]
    for nd in nodes:
        if nd["ns"] > xs[-1]:
            xs.append(float(nd["ns"]))
            ys.append(nd["value"])
    xs, ys = np.array(xs), np.array(ys)
    g = grid_ns.astype(np.float64)
    out = np.empty_like(g)
    seg = np.clip(np.searchsorted(xs, g, side="right") - 1, 0, None)
    inside = seg < len(xs) - 1
    if inside.any():
        s = seg[inside]
        u = (g[inside] - xs[s]) / (xs[s + 1] - xs[s])
        out[inside] = ys[s] + (ys[s + 1] - ys[s]) * (1 - np.cos(np.pi * u)) / 2
    after = ~inside
    if after.any():
        fade = max(float(np.median(np.diff(xs))) if len(xs) > 1 else float(fallback_fade_ns), 1.0)
        out[after] = ys[-1] * np.exp(-(g[after] - xs[-1]) / fade)
    return out


def find_turns(y, prominence, distance):
    n = len(y)
    if n < 3:
        return [], []
    cand = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    res = []
    for i in cand:
        left, right = y[:i][::-1], y[i + 1:]
        hl = np.where(left > y[i])[0]
        hr = np.where(right > y[i])[0]
        lmin = left[:hl[0]].min() if len(hl) else left.min()
        rmin = right[:hr[0]].min() if len(hr) else right.min()
        prom = y[i] - max(lmin, rmin)
        if prom >= prominence:
            res.append((int(i), float(prom)))
    kept = []
    for i, pr in sorted(res, key=lambda x: -x[1]):
        if all(abs(i - j) >= distance for j, _ in kept):
            kept.append((i, pr))
    kept.sort()
    return [i for i, _ in kept], [p for _, p in kept]


def build_composite(a, tfs, weights, horizon, mode="all", boost_cross=0.5, boost_x2=0.0, prominence=0.15):
    ref_now = a["ref_now"]
    usable = [tf for tf in tfs if tf in a["tfs"] and weights.get(tf, 0) > 0]
    if not usable:
        return None
    finest = min(a["tfs"][tf]["delta"] for tf in usable)
    n = int(np.clip(horizon / (finest / 4), 400, 4000))
    grid_ns = np.linspace(ref_now.value, (ref_now + horizon).value, n).astype(np.int64)
    norm = 1.0 + max(boost_cross, 0) + max(boost_x2, 0)

    raw, rel_sw, nodes_by_tf = {}, {}, {}
    for tf in usable:
        r = a["tfs"][tf]
        nodes = build_wave_nodes(r["projs"], ref_now, mode, boost_cross, boost_x2)
        nodes_by_tf[tf] = nodes
        raw[tf] = wave_on_grid(grid_ns, ref_now.value, current_wave_position(r["visible"], r["pivots"]),
                               nodes, 10 * r["delta"].value)
        rel_sw[tf] = relative_swing(r["pivots"], r["visible"])

    W = np.array([weights[tf] for tf in usable], dtype=float)
    stack = np.vstack([raw[tf] for tf in usable])
    composite = (W[:, None] * stack).sum(axis=0) / W.sum() / norm

    p0 = float(a["tfs"][usable[-1]]["visible"]["close"].iloc[-1])
    log_amp = {tf: math.log1p(rel_sw[tf]) / 2 for tf in usable}
    tf_price = {tf: p0 * np.exp((raw[tf] - raw[tf][0]) * log_amp[tf]) for tf in usable}
    log_change = sum(weights[tf] * (raw[tf] - raw[tf][0]) * log_amp[tf] for tf in usable)
    composite_price = p0 * np.exp(log_change)

    grid = pd.to_datetime(grid_ns)
    turns = []
    for pole, sign in (("High", 1.0), ("Low", -1.0)):
        idx, proms = find_turns(sign * composite, prominence, max(1, n // 100))
        for i, prom in zip(idx, proms):
            contrib = W * stack[:, i] * sign
            turns.append({
                "timestamp": grid[i], "pole": pole, "index": float(composite[i]), "prominence": prom,
                "price": float(composite_price[i]), "dominant": usable[int(np.argmax(contrib))],
                "aligned": [tf for tf in usable if raw[tf][i] * sign > 0.3],
            })
    turns.sort(key=lambda t: t["timestamp"])
    return {"usable": usable, "grid": grid, "composite": composite,
            "waves": {tf: raw[tf] / norm for tf in usable}, "tf_price": tf_price,
            "composite_price": composite_price, "turns": turns, "p0": p0, "nodes": nodes_by_tf}


# ==========================================
# BACKTEST (Stichtag & rollierend)
# ==========================================
def truth_params(P: TPParams):
    """Die Wahrheit nutzt immer die NEUE Erkennung (inkl. entgangener Pivots, ohne Löschen) – mit den
    gewählten Einstellungen der neuen Methode. So laufen alte und neue Methode gegen dieselbe Wahrheit."""
    return replace(P, method="v2", limit=10 ** 9, drop_incomplete=True)


def truth_for_tf(df_full, df2_full, P: TPParams, now, lower_full=None):
    """„Wahrheit“ = dieselbe Pivot-Erkennung auf ALLEN später abgeschlossenen Kerzen."""
    Pt = truth_params(P)
    full = slice_visible(df_full, None, len(df_full), True, now)
    pivs, _ = detect_pivots(full, Pt, order=lower_tf_order(full, lower_full))
    if df2_full is not None and not df2_full.empty:
        full2 = slice_visible(df2_full, None, len(df2_full), True, now)
        p2, _ = detect_pivots(full2, Pt, order=lower_tf_order(full2, df_full))
        mark_2x(pivs, p2, bar_delta(full), candle_ends(full2), P.x2_tol)
    ts_ns = full["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    by_idx = {"High": set(), "Low": set()}
    by_idx_star = {"High": set(), "Low": set()}
    for p in pivs:
        i = int(np.searchsorted(ts_ns, p["timestamp"].value))
        by_idx[p["type"]].add(i)
        if p["star"]:
            by_idx_star[p["type"]].add(i)
    return {"all": pivs, "star": [p for p in pivs if p["star"]], "idx": by_idx, "idx_star": by_idx_star,
            "last": full["timestamp"].iloc[-1] if len(full) else None, "ts_ns": ts_ns, "n": len(full)}


def chain_diag(p, cut_pivs, truth_pivs):
    """Vergleicht die am Stichtag gezählte Kette (vergangener Pivot … Anker) mit der späteren Wahrheit."""
    t0, t1 = p["past"], p["anchor"]
    cut = {(q["timestamp"], q["type"]) for q in cut_pivs if t0 <= q["timestamp"] <= t1}
    tru = {(q["timestamp"], q["type"]) for q in truth_pivs if t0 <= q["timestamp"] <= t1}
    missing, extra = sorted(tru - cut), sorted(cut - tru)
    typ = p["chain"]
    k_truth = sum(1 for q in truth_pivs if q["type"] == typ and t0 < q["timestamp"] <= t1)
    anchors_ok = (t0, typ) in tru and (t1, typ) in tru
    ok = not missing and not extra
    parts = []
    if missing:
        parts.append("fehlte am Stichtag: " + ", ".join(f"{POLE_LABEL[t]} {fmt_ts(ts)}" for ts, t in missing[:4])
                     + (" …" if len(missing) > 4 else ""))
    if extra:
        parts.append("später kein Pivot: " + ", ".join(f"{POLE_LABEL[t]} {fmt_ts(ts)}" for ts, t in extra[:4])
                     + (" …" if len(extra) > 4 else ""))
    if not anchors_ok:
        parts.append("Anker/Startpivot ist in der Wahrheit kein Pivot")
    if k_truth != p["k"] and anchors_ok:
        parts.append(f"Zählung k = {p['k']} am Stichtag, k = {k_truth} in der Wahrheit")
    return ok, ("Kette vollständig" if ok else "; ".join(parts)), k_truth


CATEGORIES = {"Alle": lambda p: True, "Kreuzbestätigt": lambda p: p["cross"], "2x-bestätigt": lambda p: p["x2"],
              "Konflikt (Hoch & Tief auf derselben Kerze)": lambda p: p["conflict"]}
RANK_LABEL = lambda r: f"{r}. Projektion" if r <= 3 else ("4.–5. Projektion" if r <= 5 else "6.+ Projektion")  # noqa: E731


TRUTH_MODES = {"all": "alle Pivots", "star": "nur ★-Pivots (auch auf 2x-TF)",
               "htf": "nur Pivots, die auch auf dem nächsthöheren TF liegen",
               "lvl": "nur Pivots, die bis Timeframe … bestätigt sind"}


def mark_htf(tr, tr_hi, df_hi_full, now):
    """Wichtige Wenden: Pivots des TF, die mit einem gleichpoligen Pivot des nächsthöheren TF zusammenfallen
    (der Pivot liegt in der Kerze des höheren Pivots; je höherem Pivot wird nur einer markiert)."""
    full_hi = slice_visible(df_hi_full, None, len(df_hi_full), True, now)
    ends = candle_ends(full_hi)
    ts_ns = tr["ts_ns"]
    out = {"High": set(), "Low": set()}
    for q in tr_hi["all"]:
        lo, hi = q["timestamp"], ends[min(q["index"], len(ends) - 1)] if q["index"] < len(ends) else q["timestamp"]
        cand = [p for p in tr["all"] if p["type"] == q["type"] and lo <= p["timestamp"] < hi]
        if not cand:
            continue
        best = min(cand, key=lambda p: (abs(p["value"] - q["value"]), p["timestamp"]))
        out[q["type"]].add(int(np.searchsorted(ts_ns, best["timestamp"].value)))
    return out


def evaluate(a, truths, use_star=False, n_random=300, seed=0, mode=None):
    """Für jede Projektion nach dem Stichtag (Zukunft bekannt), Fenster = projizierte Kerze und Kerze +1:
       Treffer   = Pivot des vorhergesagten Pols im Fenster
       Inversion = kein solcher, aber ein Pivot des Gegenpols im Fenster
       verfehlt  = gar kein Pivot im Fenster
       exakt     = Pivot des vorhergesagten Pols genau auf der projizierten Kerze
    Grundrate = dieselbe Messung für beliebige Kerzen nach dem Stichtag (gleicher Pol) → Vorsprung.
    mode: "all" | "star" | "htf" (wichtige Wenden). Zusätzlich: Rang, Kerzen voraus, Ketten-Diagnose."""
    mode = mode or ("star" if use_star else "all")
    rng = np.random.default_rng(seed)
    ref = a["ref_now"]
    rows = []
    for tf, r in a["tfs"].items():
        tr = truths.get(tf)
        if not tr or tr["last"] is None or tr["last"] <= ref:
            continue
        idx = {"all": tr["idx"], "star": tr["idx_star"], "htf": tr.get("idx_htf"), "lvl": tr.get("idx_lvl")}[mode]
        if idx is None:
            continue
        ts_ns, n = tr["ts_ns"], tr["n"]
        x_last = int(np.searchsorted(ts_ns, r["visible"]["timestamp"].iloc[-1].value))

        def cidx(t):
            return int(np.searchsorted(ts_ns, pd.Timestamp(t).value, "right")) - 1

        def judge(x, pole):
            win = range(x - TOL_BEFORE, x + TOL_AFTER + 1)
            exact = x in idx[pole]
            if any(i in idx[pole] for i in win):
                return "hit", exact
            if any(i in idx[OPP[pole]] for i in win):
                return "inv", exact
            return "miss", exact
        cands = []
        for p in r["projs"]:
            for key in ("bt", "bt_key", "bt_diag", "bt_exact"):
                p.pop(key, None)
            x = cidx(p["date"])
            if p["date"] <= ref or x <= x_last or x + TOL_AFTER > n - 1:
                continue
            cands.append((x, p))
        ranks = {x: i + 1 for i, x in enumerate(sorted({x for x, _ in cands}))}
        hi_rand = n - 1 - TOL_AFTER
        rand = rng.integers(x_last + 1, hi_rand + 1, size=n_random) if hi_rand > x_last else np.array([], int)
        base = {}
        for pole in ("High", "Low"):
            js = [judge(int(x), pole) for x in rand]
            base[pole] = tuple(float(np.mean(v)) if js else 0.0 for v in (
                [j[0] == "hit" for j in js], [j[0] == "inv" for j in js], [j[1] for j in js]))
        for x, p in cands:
            res, exact = judge(x, p["pole"])
            ok, diag, k_truth = chain_diag(p, r["pivots"], tr["all"])
            p["bt"] = {"hit": "✅ Treffer", "inv": "🔄 Inversion", "miss": "❌ verfehlt"}[res]
            p["bt_key"] = res
            p["bt_exact"] = exact
            p["bt_diag"] = diag
            p["bt_rank"] = ranks[x]
            p["bt_ahead"] = x - x_last
            b = base[p["pole"]]
            rows.append({"tf": tf, "hit": float(res == "hit"), "inv": float(res == "inv"), "hit0": float(exact),
                         "b_hit": b[0], "b_inv": b[1], "b_hit0": b[2],
                         "rank": ranks[x], "ahead": x - x_last, "chain_ok": ok, "k": p["k"],
                         "id": p["id"], "date": p["date"], "pole": p["pole"], "diag": diag,
                         **{k: f(p) for k, f in CATEGORIES.items()}})
    return rows


def summarize(rows, by="cat"):
    out = []
    if by == "cat":
        groups = [(k, [r for r in rows if r[k]]) for k in CATEGORIES]
    elif by == "rank":
        labels = list(dict.fromkeys(RANK_LABEL(r) for r in sorted(range(1, 7))))
        groups = [(lb, [r for r in rows if RANK_LABEL(r["rank"]) == lb]) for lb in labels]
    elif by == "chain":
        groups = [("Kette vollständig (wie später)", [r for r in rows if r["chain_ok"]]),
                  ("Kette abweichend (Pivot fehlte / zu viel)", [r for r in rows if not r["chain_ok"]])]
    else:
        groups = [(tf, [r for r in rows if r["tf"] == tf]) for tf in dict.fromkeys(r["tf"] for r in rows)]
    for name, rs in groups:
        if not rs:
            out.append({"Gruppe": name, "n": 0})
            continue
        h, i, h0 = (_rate([r[k] for r in rs]) for k in ("hit", "inv", "hit0"))
        bh, bi, bh0 = (_rate([r[k] for r in rs]) for k in ("b_hit", "b_inv", "b_hit0"))
        out.append({"Gruppe": name, "n": len(rs),
                    "Treffer % (+1)": h, "Grundrate Treffer %": bh, "Vorsprung (Pp.)": round(h - bh, 1),
                    "Inversion %": i, "Grundrate Inversion %": bi, "verfehlt %": round(100 - h - i, 1),
                    "exakt %": h0, "Grundrate exakt %": bh0, "Vorsprung exakt (Pp.)": round(h0 - bh0, 1)})
    return out


def rolling_backtest(frames, tfs, doubles, P, now, lookback, n_cut, use_star=False, progress=None,
                     calendar="crypto", lowers=None, truths=None, mode=None):
    lowers = lowers or {}
    if truths is None:
        truths = {tf: truth_for_tf(frames[tf], frames.get(doubles.get(tf)), P, now, frames.get(lowers.get(tf)))
                  for tf in tfs if tf in frames}
    start = now - lookback
    end = now - max(pd.Timedelta(days=1), lookback * 0.15)
    cuts = pd.to_datetime(np.linspace(start.value, end.value, n_cut).astype(np.int64))
    rows, per_cut = [], []
    for i, co in enumerate(cuts):
        a = analyze_symbol(frames, tfs, doubles, co, P, now, calendar=calendar, lowers=lowers)
        rs = evaluate(a, truths, use_star, seed=i, mode=mode)
        rows += rs
        per_cut.append({"Stichtag": co, "n": len(rs), "Treffer %": _rate([r["hit"] for r in rs]),
                        "Grundrate %": _rate([r["b_hit"] for r in rs]), "Inversion %": _rate([r["inv"] for r in rs]),
                        "exakt %": _rate([r["hit0"] for r in rs])})
        if progress:
            progress((i + 1) / len(cuts), f"Stichtag {i + 1}/{len(cuts)}: {fmt_ts(co)}")
    return {"rows": rows, "per_cut": per_cut}


# ==========================================
# SCREENER
# ==========================================
QUALITY = {
    "Alle Projektionen": "all",
    "Kreuzbestätigt (Tief- & Hoch-Kette)": "cross",
    "2x-bestätigt": "x2",
    "Kreuz- ODER 2x-bestätigt": "any",
    "Kreuz- UND 2x-bestätigt": "both",
}


def screen_symbol(a, tfs, horizon, quality="all", comp=None):
    ref = a["ref_now"]
    end_h = ref + horizon
    events = {}
    for tf in tfs:
        r = a["tfs"].get(tf)
        if r is None:
            continue
        evs = []
        for p in r["projs"]:
            if p["date_end"] <= ref or p["date"] > end_h or not _keep(p, quality):
                continue
            evs.append({"tf": tf, "pole": p["pole"], "start": p["date"], "end": p["date_end"],
                        "cross": p["cross"], "x2": p["x2"], "name": p["name"], "delta": r["delta"]})
        evs.sort(key=lambda e: e["start"])
        merged = []
        for e in evs:
            m = merged[-1] if merged else None
            if m and m["pole"] == e["pole"] and e["start"] <= m["end"]:
                m["end"] = max(m["end"], e["end"])
                m["cross"] |= e["cross"]
                m["x2"] |= e["x2"]
                m["n"] += 1
                m["name"] += ", " + e["name"]
            else:
                merged.append(dict(e, n=1))
        events[tf] = merged
    for tf, evs in events.items():
        for e in evs:
            e["aligned"] = []
            for tf2, evs2 in events.items():
                if tf2 == tf:
                    continue
                pad = max(e["delta"], evs2[0]["delta"]) if evs2 else e["delta"]
                if any(x["pole"] == e["pole"] and x["start"] - pad < e["end"] and x["end"] + pad > e["start"]
                       for x in evs2):
                    e["aligned"].append(tf2)
            e["score"] = 1 + e["cross"] + e["x2"] + len(e["aligned"]) + 0.5 * (e["n"] - 1)
    nxt_turn = None
    if comp and comp["turns"]:
        nxt_turn = next((t for t in comp["turns"] if t["timestamp"] > ref), None)
    return {"events": events, "composite_next": nxt_turn}
