"""Compact, read-only statistics while the full raw data are being retrieved."""
import csv
import json
from pathlib import Path
from statistics import fmean
ROOT = Path(__file__).resolve().parent
A = ROOT/"analysis_other"
def rows(name):
    with (A/f"{name}.csv").open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
events = [r for r in rows("events") if r["rule"]=="primary" and r["generation"]=="300"]
local = [r for r in rows("local") if r["rule"]=="primary" and r["generation"]=="300"]
result = []
for t,eid in sorted({(r["target"],r["event_id"]) for r in events}):
    rr = [r for r in events if r["target"]==t and r["event_id"]==eid]
    if rr[0]["width_evaluable"]!="True":
        continue
    meta = json.loads((ROOT/"inputs_core"/t/"target.json").read_text())
    target = next(e for e in meta["events"] if e["event_id"]==eid)
    ll = [float(r["amp_fraction"]) for r in local if r["target"]==t and r["event_id"]==eid and r["amp_fraction"]!=""]
    match = [r for r in rr if r["matched"]=="True"]
    n = len(rr)
    result.append(dict(target=t,event=eid,frequency=target["frequency_hz"],width_hz=target["width_3db_hz"],width_decades=target["width_3db_decades"],height_db=target["height_db"],seeds=n,
                       missing=sum(r["matched"]=="False" for r in rr),complete=sum(r["complete"]=="True" for r in rr),
                       position_bad=sum(abs(float(r["shift_decades"]))>.02 for r in match),height_bad=sum(abs(float(r["height_delta_db"]))>3 for r in match),width_bad=sum(r["width_ratio"]=="" or not .5<=float(r["width_ratio"])<=2 for r in match),
                       height_delta_mean=fmean(float(r["height_delta_db"]) for r in match) if match else None,amp_fraction_mean=fmean(ll) if ll else None))
out = dict(events=result,population=rows("population"),pairs=[r for r in rows("similar_scores_different_shapes") if r["threshold"]=="0.05"])
(ROOT/"PREVIEW_STATISTICS.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(sorted([r for r in result if r["complete"]<r["seeds"]],key=lambda r:(r["complete"],r["target"],r["event"]))[:12]))
