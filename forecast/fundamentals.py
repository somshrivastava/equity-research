
import requests
from datetime import datetime, date
USER_AGENT_EMAIL = "som.shrivastava@gmail.com"
CIK = "0000320193"
FACTS_URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"

def fetch_companyfacts():
    headers = {
        "User-Agent": f"AAPL-fundamentals/1.0 ({USER_AGENT_EMAIL})",
        "Accept": "application/json",
    }
    r = requests.get(FACTS_URL, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_dt(s: str):
    try: return datetime.fromisoformat(s)
    except: return None

def approx_days(start: str, end: str) -> int:
    ds, de = parse_dt(start), parse_dt(end)
    if not ds or not de: return 0
    return (de - ds).days

def is_true_quarter(entry: dict) -> bool:
    q = entry.get("qtrs")
    if isinstance(q, int): return q == 1
    return approx_days(entry.get("start",""), entry.get("end","")) <= 100

def subtract_years(d: date, years: int) -> date:
    try: return d.replace(year=d.year - years)
    except ValueError: return d.replace(month=2, day=28, year=d.year - years)

def collect_quarterly(facts: dict, concept: str):
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    obj = usgaap.get(concept)
    if not obj: return []
    out = []
    for unit, series in obj.get("units", {}).items():
        if unit != "USD": continue
        for e in series:
            if e.get("fp") not in {"Q1","Q2","Q3","Q4"}: continue
            if not isinstance(e.get("val"), (int,float)): continue
            if not is_true_quarter(e): continue
            out.append({
                "val": e.get("val"),
                "fy": e.get("fy"),
                "fp": e.get("fp"),
                "form": e.get("form"),
                "start": e.get("start"),
                "end": e.get("end"),
                "filed": e.get("filed"),
            })
    return out

def dedupe(entries):
    def dt(x): 
        d = parse_dt(x or "")
        return d if d else datetime.min
    by_period = {}
    for e in entries:
        key = (e.get("fy"), e.get("fp"))
        if not key[0] or key[1] not in {"Q1","Q2","Q3","Q4"}:
            continue
        cur = by_period.get(key)
        better = False
        if not cur:
            better = True
        else:
            if dt(e.get("end")) > dt(cur.get("end")):
                better = True
            elif dt(e.get("end")) == dt(cur.get("end")):
                e10, c10 = (e.get("form")=="10-Q"), (cur.get("form")=="10-Q")
                if e10 and not c10: better = True
                elif e10 == c10 and dt(e.get("filed")) > dt(cur.get("filed")):
                    better = True
        if better: by_period[key] = e
    rows = list(by_period.values())
    rows.sort(key=lambda r:(r.get("fy",0), {"Q1":1,"Q2":2,"Q3":3,"Q4":4}.get(r.get("fp",""),0)))
    return rows

def get_flow_metric(concepts: list[str], label: str, years_back: int = 5):
    facts = fetch_companyfacts()
    cutoff = subtract_years(date.today(), years_back)
    quarterly = {}
    annuals = {}
    
    for concept in concepts:
        entries = collect_quarterly(facts, concept)
        best_rows = dedupe(entries)
        for r in best_rows:
            end_dt = parse_dt(r.get("end", ""))
            if end_dt and end_dt.date() >= cutoff:
                quarterly[f"FY{r['fy']} {r['fp']}"] = r["val"]

        usgaap = facts.get("facts", {}).get("us-gaap", {})
        obj = usgaap.get(concept)
        if obj:
            for unit, series in obj.get("units", {}).items():
                if unit != "USD": continue
                for e in series:
                    if e.get("fp") != "FY": continue
                    if not isinstance(e.get("val"), (int,float)): continue
                    if e.get("form") != "10-K": continue
                    fy = e.get("fy")
                    end_dt = parse_dt(e.get("end", ""))
                    if not fy or not end_dt: continue
                    if end_dt.date() < cutoff: continue
                    if str(fy) not in str(end_dt.year): continue
                    annuals[fy] = max(annuals.get(fy,0), e.get("val"))

    # Compute Q4
    for fy,total in annuals.items():
        q1 = quarterly.get(f"FY{fy} Q1",0)
        q2 = quarterly.get(f"FY{fy} Q2",0)
        q3 = quarterly.get(f"FY{fy} Q3",0)
        quarterly[f"FY{fy} Q4"] = total - (q1+q2+q3)

    def qsort(period):
        fy,q = period.split()
        return (int(fy.replace("FY","")), {"Q1":1,"Q2":2,"Q3":3,"Q4":4}[q])

    return [{ "Period": p, label: v }
            for p,v in sorted(quarterly.items(), key=lambda x: qsort(x[0]))]

def get_balance_metric(concepts: list[str], label: str, years_back: int = 5):
    facts = fetch_companyfacts()
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    cutoff = subtract_years(date.today(), years_back)
    quarterly = {}

    for concept in concepts:
        obj = usgaap.get(concept)
        if not obj: continue
        for unit, series in obj.get("units", {}).items():
            if unit != "USD": continue
            for e in series:
                if not isinstance(e.get("val"), (int,float)): continue
                form = e.get("form")
                fp   = e.get("fp")
                fy   = e.get("fy")
                end_dt = parse_dt(e.get("end",""))
                if not fy or not end_dt: continue
                if end_dt.date() < cutoff: continue

                if fp in {"Q1","Q2","Q3"} and form == "10-Q":
                    quarterly[f"FY{fy} {fp}"] = e["val"]

                if fp == "FY" and form == "10-K":
                    quarterly[f"FY{fy} Q4"] = e["val"]

    def qsort(period):
        fy,q = period.split()
        return (int(fy.replace("FY","")), {"Q1":1,"Q2":2,"Q3":3,"Q4":4}[q])

    return [{ "Period": p, label: v }
            for p,v in sorted(quarterly.items(), key=lambda x: qsort(x[0]))]
