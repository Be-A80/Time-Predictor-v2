"""
Time Predictor – Aktien-Universum.

Trade Republic bietet keine öffentliche Instrumenten-Liste/API. Deshalb besteht das Universum aus den
Mitgliedern großer Indizes (S&P 500, DAX 40, Euro Stoxx 50) – diese Large Caps sind bei Trade Republic
praktisch vollständig handelbar. Eine automatische Prüfung je Aktie ist nicht möglich; über die
Ausschlussliste lassen sich einzelne Ausnahmen dauerhaft entfernen, über „Eigene Ticker“ weitere ergänzen.
Die Listen werden aus Wikipedia gelesen (täglich gecacht), mit statischem Fallback.
"""
import re
from html.parser import HTMLParser

import requests
import streamlit as st

UA = {"User-Agent": "Mozilla/5.0 (TimePredictor; Streamlit app)"}

INDEX_SOURCES = {
    "DAX 40": ("https://en.wikipedia.org/wiki/DAX", ("Ticker",), ("Company",)),
    "Euro Stoxx 50": ("https://en.wikipedia.org/wiki/Euro_Stoxx_50", ("Ticker",), ("Name",)),
    "S&P 500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ("Symbol",), ("Security",)),
}

FALLBACK = {
    "DAX 40": ["ADS.DE", "AIR.PA", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
               "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "FME.DE", "G1A.DE", "HNR1.DE",
               "HEI.DE", "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
               "RWE.DE", "SAP.DE", "G24.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"],
    "Euro Stoxx 50": ["ADYEN.AS", "AD.AS", "AI.PA", "ABI.BR", "ARGX.BR", "ASML.AS", "CS.PA", "BBVA.MC", "SAN.MC",
                      "BNP.PA", "BN.PA", "ENEL.MI", "ENGI.PA", "ENI.MI", "EL.PA", "RACE.MI", "RMS.PA", "IBE.MC",
                      "ITX.MC", "INGA.AS", "ISP.MI", "OR.PA", "MC.PA", "NOKIA.HE", "NDA-FI.HE", "PRX.AS", "SAF.PA",
                      "SGO.PA", "SAN.PA", "SU.PA", "TTE.PA", "DG.PA", "UCG.MI"],
    "S&P 500": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "JPM", "LLY", "V", "MA",
                "UNH", "XOM", "COST", "WMT", "HD", "PG", "JNJ", "NFLX", "ABBV", "BAC", "CRM", "ORCL", "KO", "CVX",
                "MRK", "AMD", "PEP", "ADBE", "TMO", "LIN", "CSCO", "MCD", "ACN", "ABT", "WFC", "IBM", "GE", "QCOM",
                "TXN", "INTU", "DIS", "CAT", "AMGN", "PM", "ISRG", "NOW", "GS", "UBER", "PLTR", "INTC", "MU"],
}


class _Tables(HTMLParser):
    """Minimaler HTML-Tabellen-Parser (kein lxml nötig)."""

    def __init__(self):
        super().__init__()
        self.tables, self.depth, self.row, self.cell = [], 0, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.tables.append({"id": dict(attrs).get("id", ""), "rows": []})
        if self.depth != 1:
            return
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag == "table":
            self.depth -= 1
        if self.depth != 1:
            return
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(re.sub(r"\s+", " ", "".join(self.cell)).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.tables[-1]["rows"].append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None and self.depth == 1:
            self.cell.append(data)


def _parse_constituents(html, sym_cols, name_cols):
    p = _Tables()
    p.feed(html)
    tables = sorted(p.tables, key=lambda t: (t["id"] != "constituents", -len(t["rows"])))
    for t in tables:
        if len(t["rows"]) < 10:
            continue
        head = [re.sub(r"\[.*?\]", "", h).strip() for h in t["rows"][0]]
        si = next((head.index(c) for c in sym_cols if c in head), None)
        ni = next((head.index(c) for c in name_cols if c in head), None)
        if si is None:
            continue
        out = []
        for r in t["rows"][1:]:
            if len(r) <= si:
                continue
            sym = r[si].strip()
            if sym and re.fullmatch(r"[A-Z0-9.\-]{1,12}", sym):
                out.append((sym, r[ni] if ni is not None and len(r) > ni else sym))
        if len(out) >= 10:
            return out
    return []


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_index(name):
    """Liste [(yfinance-Ticker, Name)] eines Index + Quelle."""
    url, sym_cols, name_cols = INDEX_SOURCES[name]
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        rows = _parse_constituents(r.text, sym_cols, name_cols)
        if name == "S&P 500":
            rows = [(s.replace(".", "-"), n) for s, n in rows]       # BRK.B → BRK-B (yfinance)
        if len(rows) >= 10:
            return rows, "Wikipedia"
        raise ValueError("Tabelle nicht gefunden")
    except Exception as e:
        return [(s, s) for s in FALLBACK[name]], f"statischer Fallback ({type(e).__name__})"


def stock_universe(indices, extra="", exclude=""):
    """Vereinigung der gewählten Indizes + eigene Ticker − Ausschlüsse. Rückgabe: Liste von Dicts, Quellen."""
    seen, out, sources = set(), [], {}
    excl = {t.strip().upper() for t in re.split(r"[,;\s]+", exclude or "") if t.strip()}
    for idx in indices:
        rows, src = fetch_index(idx)
        sources[idx] = f"{src} ({len(rows)})"
        for sym, name in rows:
            if sym.upper() in excl or sym in seen:
                continue
            seen.add(sym)
            out.append({"symbol": sym, "name": name, "index": idx})
    for sym in re.split(r"[,;\s]+", extra or ""):
        sym = sym.strip().upper()
        if sym and sym not in seen and sym not in excl:
            seen.add(sym)
            out.append({"symbol": sym, "name": sym, "index": "eigene"})
    return out, sources


_TV_EXCHANGE = {".DE": "XETR", ".PA": "EURONEXT", ".AS": "EURONEXT", ".BR": "EURONEXT", ".MI": "MIL",
                ".MC": "BME", ".HE": "OMXHEX", ".L": "LSE", ".SW": "SIX", ".VI": "VIE"}


def tv_stock_symbol(sym):
    for suf, ex in _TV_EXCHANGE.items():
        if sym.endswith(suf):
            base = sym[: -len(suf)]
            if suf == ".HE":
                base = base.replace("-FI", "")
            return f"{ex}:{base}"
    return sym.replace("-", ".")
