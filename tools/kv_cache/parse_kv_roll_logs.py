#!/usr/bin/env python3
"""Summarize rolling-lookahead device logs per docs/kv-fsu-rolling-device-test.ko.md."""
import hashlib, json, re, sys, glob, os

LOGDIR = os.path.expanduser(
    "~/NNTrainer_commit/general/nntrainer/build-fsu/logs/roll74")

KV_KEYS = ["lookahead", "read_bytes", "retained_bytes", "kv_payload_peak_bytes",
           "retained_owners", "layers", "flash_owners", "owner_lookahead",
           "capped", "total_budget_bytes", "rolling", "borrowed_owners",
           "discarded_prefetch", "budget_blocks"]

def grab(pattern, txt, cast=float, last=False):
    vals = [cast(m) for m in re.findall(pattern, txt)]
    return vals if not last else (vals[-1] if vals else None)

def parse(path):
    txt = open(path, errors="replace").read()
    if "FATAL" in txt:
        return {"file": os.path.basename(path), "fatal": True}
    # request results are [REQUEST] {...} single-line JSON
    reqs = []
    for m in re.finditer(r'^\[REQUEST\] (\{.*\})\s*$', txt, re.M):
        try:
            obj = json.loads(m.group(1))
            ids = obj.get("output_ids")
            h = hashlib.md5(json.dumps(ids).encode()).hexdigest()[:8] if ids else "?"
            reqs.append({"idx": obj.get("index"), "before": obj.get("cached_before"),
                         "after": obj.get("cached_after"), "md5": h,
                         "gen": len(ids) if ids else 0})
        except Exception:
            pass
    steady = grab(r'steady_ms=([0-9.]+)', txt)
    p95 = grab(r'p95_ms=([0-9.]+)', txt)
    # [kv-fsu] lines: split into key=value tokens
    kvsu = []
    for line in re.findall(r'\[kv-fsu\] [^\n]+', txt):
        kv = dict(re.findall(r'(\w+)=([\d.]+)', line))
        kvsu.append(kv)
    reads = re.findall(r'\[kv-fsu-reads\] total owner_reads=(\d+) owner_read_us=([0-9.]+) demand_read_us=([0-9.]+) consumer_wait_us=([0-9.]+)', txt)
    out = {"file": os.path.basename(path), "fatal": False, "reqs": reqs,
           "steady": steady, "p95": p95}
    if kvsu:
        out["capped"] = int(kvsu[-1].get("capped", 0))
        out["total_budget"] = int(kvsu[-1].get("total_budget_bytes", 0))
        out["rolling"] = int(kvsu[-1].get("rolling", 0))
        out["payload_peak"] = max(int(k.get("kv_payload_peak_bytes", 0)) for k in kvsu)
        out["peak_ok"] = out["payload_peak"] <= out["total_budget"]
        out["borrowed"] = int(kvsu[-1].get("borrowed_owners", 0))
        out["budget_blocks"] = int(kvsu[-1].get("budget_blocks", 0))
        out["discarded"] = int(kvsu[-1].get("discarded_prefetch", 0))
        out["retained_owners"] = int(kvsu[-1].get("retained_owners", 0))
        rb = [int(k.get("read_bytes", 0)) for k in kvsu]
        out["read_bytes_delta"] = rb[-1] - rb[0] if len(rb) > 1 else rb[0]
    if reads:
        out["owner_reads"] = sum(int(r[0]) for r in reads)
        out["demand_s"] = sum(float(r[2]) for r in reads) / 1e6
        out["cwait_s"] = sum(float(r[3]) for r in reads) / 1e6
    return out

def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*/*"
    files = sorted(glob.glob(os.path.join(LOGDIR, pattern + ".log")))
    rows = [parse(f) for f in files]
    ref = {}
    for r in rows:
        if r["file"].startswith("resident") and not r.get("fatal"):
            for q in r["reqs"]:
                ref.setdefault(q["idx"], q["md5"])
    for r in rows:
        if r.get("fatal"):
            print(f'{r["file"]}: FATAL'); continue
        match = all(ref.get(q["idx"]) == q["md5"] for q in r["reqs"]) if ref else None
        st = " ".join(f'{s:.1f}' for s in r["steady"])
        p95 = " ".join(f'{p:.1f}' for p in r["p95"])
        extra = ""
        if "capped" in r:
            tokens = sum(q["gen"] for q in r["reqs"]) or 1
            rbt = f' read/tok={r.get("read_bytes_delta",0)/tokens/1e6:.2f}MB'
            extra = (f' la? rol={r["rolling"]} C={r["total_budget"]/2**20:.0f}M peak={r["payload_peak"]/2**20:.1f}M '
                     f'capOK={r["peak_ok"]} borr={r["borrowed"]} blk={r["budget_blocks"]} '
                     f'dis={r["discarded"]} ret={r["retained_owners"]} reads={r.get("owner_reads")} '
                     f'demand_s={r.get("demand_s",0):.2f}{rbt}')
        print(f'{r["file"]}: outmatch={match} mean=[{st}] p95=[{p95}]{extra}')
        print('   reqs: ' + " | ".join(f'#{q["idx"]} {q["before"]}->{q["after"]} n={q["gen"]} {q["md5"]}' for q in r["reqs"]))

main()
