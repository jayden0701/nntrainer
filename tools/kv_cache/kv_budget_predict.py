#!/usr/bin/env python3
"""Predict FSU decode speed from the resident-retention budget.

Calibrated on the SM-S947N budget sweep (2026-11, see
kv-cache-fsu-status.ko.md section 9.4). The measured decode-vs-budget
curve is U-shaped:

  * full budget (every owner retained) reaches the compute floor,
  * low budgets stream densely and keep the storage awake
    (cheap per-read service),
  * MID budgets are the WORST: few, sparsely spaced reads let the
    device enter low-power states, so every read pays a wake-up
    latency. `decode = max(compute, read_bytes/BW)` does NOT hold
    there; a per-read service-time term does.

Model
-----
    k(budget)      = min(owners, budget // (rows * row_bytes))
    streamers      = owners - k
    gap            = owners / streamers          (layers between reads)
    t_eff(gap)     = t_service + W * clamp((gap - g0) / ramp, 0, 1)
    decode(budget) = T_c + O + streamers * max(0, t_eff - T_c / owners)

    T_c        compute floor (resident decode, ms/token)
    O          fixed FSU overhead (binding/accounting, ms/token)
    t_service  dense-read service time (ms per owner read)
    W, g0, ramp  storage wake-latency growth with read spacing

Usage
-----
    # fit a profile from a sweep log directory (decode_ms/[kv-fsu] lines)
    kv_budget_predict.py calibrate --logs LOGDIR --rows 630 --out profile.json

    # predict decode for budgets (bytes or MiB suffix)
    kv_budget_predict.py predict --profile profile.json --rows 630 \
        --budget 24MiB 52MiB 72253440

    # decode target -> smallest budget that reaches it (when possible)
    kv_budget_predict.py solve --profile profile.json --rows 630 --target 34

    # draw the curve
    kv_budget_predict.py curve --profile profile.json --rows 630 --out curve.png
"""

import argparse
import glob
import json
import re
import statistics
import sys

ROW_BYTES_DEFAULT = 4096          # K+V bytes per row per owner (FP16 kv_dim 1024)
OWNERS_DEFAULT = 28               # qwen3-1.7b: one owner per layer, no aliases

DEFAULT_PROFILE = {
    "name": "SM-S947N qwen3-1.7b 2026-11 sweep",
    "T_c": 31.87,        # resident decode ms/token (today's thermal state)
    "O": 1.56,           # full-budget decode minus resident
    "t_service": 1.28,   # main-visible per-read cost at gap <= g0 (fitted)
    "t_service_worker": 1.45,  # worker-side per-read service (io_read accounting);
                               # drives only the worker-saturation check
    "W": 0.78,           # max wake-latency adder, ms per read
    "g0": 1.4,           # gap (layers) below which no wake penalty
    "ramp": 1.2,         # gap width over which the penalty ramps to W
    "rows_cal": 630,     # rows at calibration
}


def parse_budget(text):
    text = text.strip().lower()
    if text.endswith("mib"):
        return int(float(text[:-3]) * 1024 * 1024)
    if text.endswith("mb"):
        return int(float(text[:-2]) * 1000 * 1000)
    return int(text)


def owners_for_budget(budget, rows, row_bytes, owners):
    if budget <= 0:
        return 0
    return min(owners, budget // (rows * row_bytes))


def t_eff(gap, p):
    ramp = max(1e-9, (gap - p["g0"]) / p["ramp"])
    return p["t_service"] + p["W"] * max(0.0, min(1.0, ramp))


def predict_decode(budget, rows, p, row_bytes=ROW_BYTES_DEFAULT, owners=OWNERS_DEFAULT):
    """Predicted decode ms/token for a retention budget in bytes.

    Returns (decode, k, stall_total, saturated). `saturated` marks the regime
    the model was NOT calibrated on: when streamers * t_service exceeds the
    compute floor the read worker cannot keep up, decode becomes read-bound
    (queueing) and the additive stall term underestimates it.
    """
    k = owners_for_budget(budget, rows, row_bytes, owners)
    streamers = owners - k
    if streamers == 0:
        return p["T_c"] + p["O"], k, 0.0, False
    gap = owners / streamers
    t_service_gap = t_eff(gap, p)
    stall = max(0.0, t_service_gap - p["T_c"] / owners)
    saturated = streamers * p.get("t_service_worker", p["t_service"]) > p["T_c"]
    return p["T_c"] + p["O"] + streamers * stall, k, streamers * stall, saturated


def solve_budget(target, rows, p, row_bytes=ROW_BYTES_DEFAULT, owners=OWNERS_DEFAULT):
    """Budgets reaching the target decode, split by arm of the U curve.

    Returns (right_arm, left_arm, best_unreachable). The right arm is the
    smallest retention-heavy budget (robust, calibrated); the left arm is a
    dense-streaming budget (cheap per-read service but close to worker
    saturation — the risky option).
    """
    right, left = None, None
    for k in range(0, owners + 1):
        budget = k * rows * row_bytes
        d, _, _, sat = predict_decode(budget, rows, p, row_bytes, owners)
        if d > target or sat:
            continue
        entry = (budget, d, k)
        if k * 2 >= owners:
            if right is None:
                right = entry
        elif left is None:
            left = entry
    if right is None:
        best = min(((k * rows * row_bytes,) + predict_decode(
            k * rows * row_bytes, rows, p, row_bytes, owners)[:2])
            for k in range(owners + 1))
        return None, left, best
    return right, left, None


# ---------------------------------------------------------------- calibration
def calibrate_from_logs(logdir, rows, row_bytes, owners):
    runs = {}
    for path in sorted(glob.glob(f"{logdir}/**/*.log", recursive=True)):
        txt = open(path, errors="ignore").read()
        m = re.search(r'"decode_ms":([0-9.]+)', txt)
        if not m:
            continue
        tag_m = re.search(r"(\S+)\.log$", path)
        tag = re.sub(r"_r\d+$", "", tag_m.group(1).split("/")[-1]) if tag_m else path
        d = {"decode": float(m.group(1))}
        for pat, key, cast in [
            (r"retained_bytes=(\d+)", "retained", int),
            (r"io_read_us=([0-9.]+)", "io_read_us", float),
            (r"parallel_reads=(\d+)", "reads", int),
        ]:
            mm = re.search(pat, txt)
            if mm:
                d[key] = cast(mm.group(1))
        runs.setdefault(tag, []).append(d)

    med = {t: statistics.median(r["decode"] for r in rs) for t, rs in runs.items()}
    resident = next((v for t, v in med.items() if t.startswith("resident")), None)
    if resident is None:
        sys.exit("calibrate: need a resident_*.log in the sweep for the floor")
    full = max(
        ((t, v, runs[t]) for t, v in med.items()
         if not t.startswith("resident") and all(r.get("retained", 0) for r in runs[t])),
        key=lambda tv: statistics.median(r["retained"] for r in tv[2]),
        default=None,
    )
    if full is None:
        sys.exit("calibrate: need a full-coverage FSU run for the fixed overhead")
    o = full[1] - resident

    # per-config k from retained accounting
    points = []  # (k, decode)
    for t, v in sorted(med.items()):
        if t.startswith("resident"):
            continue
        ret = statistics.median(r["retained"] for r in runs[t] if "retained" in r)
        k = round(ret / (rows * row_bytes)) if ret else owners
        points.append((k, v, ret))

    # fit t_eff(gap) = t_service + W * clamp((gap-g0)/ramp,0,1).
    # t_service_worker is PINNED to the measured dense per-read worker service
    # time (io_read accounting of the config with the most reads); it drives
    # only the worker-saturation flag so the low-k extrapolation cannot be
    # silently optimistic. The main-visible t_eff is fitted freely.
    T_c = resident
    c_layer = T_c / owners
    dense = max(
        ((t, runs[t]) for t in runs
         if not t.startswith("resident") and any(r.get("reads") for r in runs[t])),
        key=lambda tv: statistics.median(r["reads"] for r in tv[1] if r.get("reads")),
        default=None,
    )
    if dense is None:
        sys.exit("calibrate: need at least one streaming FSU run")
    t_service_worker = statistics.median(
        r["io_read_us"] / r["reads"] / 1000 for r in dense[1] if r.get("reads"))

    best, best_err = None, 1e9
    for t_service in [x / 100 for x in range(90, 190, 4)]:
        for W in [x / 100 for x in range(30, 160, 5)]:
            for g0 in [x / 10 for x in range(10, 30, 2)]:
                for ramp in [x / 10 for x in range(6, 30, 2)]:
                    p = {"T_c": T_c, "O": o, "t_service": t_service,
                         "t_service_worker": t_service_worker, "W": W,
                         "g0": g0, "ramp": ramp}
                    err = 0.0
                    for k, dec, _ in points:
                        s = owners - k
                        if s == 0:
                            continue
                        pred = T_c + o + s * max(0.0, t_eff(owners / s, p) - c_layer)
                        err += (pred - dec) ** 2
                    if err < best_err:
                        best_err, best = err, dict(p)

    best.update({"rows_cal": rows})
    return best, best_err, points


# -------------------------------------------------------------------- output
def ascii_curve(rows, p, budgets):
    print(f"{'budget':>12} {'k':>3} {'decode':>8}")
    for b in budgets:
        d, k, _ = predict_decode(b, rows, p)
        bar = "#" * int((d - p["T_c"] - p["O"]) * 3 + 1)
        print(f"{b:>12} {k:>3} {d:7.2f}  {bar}")


def png_curve(rows, p, out_path, row_bytes, owners):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = owners * rows * row_bytes
    xs = list(range(0, full + 1, max(1, full // 400)))
    ys = [predict_decode(x, rows, p, row_bytes, owners)[0] for x in xs]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([x / 1024 / 1024 for x in xs], ys, lw=2)
    ax.axhline(p["T_c"], ls="--", lw=1, color="green", label=f"resident floor {p['T_c']:.1f}")
    ax.axhline(p["T_c"] + p["O"], ls=":", lw=1, color="olive",
               label=f"full-budget floor {p['T_c'] + p['O']:.1f}")
    for b, label, color in [(24 * 1024**2, "24 MiB", "tab:orange"),
                            (52 * 1024**2, "52 MiB (valley)", "red"),
                            (74 * 1024**2, "74 MiB (full)", "tab:blue")]:
        d, _, _, _ = predict_decode(min(b, full), rows, p, row_bytes, owners)
        ax.plot(b / 1024 / 1024, d, "o", color=color)
        ax.annotate(f"{label}: {d:.1f} ms", (b / 1024 / 1024, d),
                    textcoords="offset points", xytext=(6, 8), fontsize=8)
    ax.set_xlabel("resident budget (MiB)")
    ax.set_ylabel("decode ms/token (predicted)")
    ax.set_title(f"FSU budget vs decode ({p.get('name', 'calibrated')}, rows={rows})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_rows(sp):
        sp.add_argument("--rows", type=int, required=True,
                        help="cache rows at decode end (prefill+decode, clamped to plan)")
        sp.add_argument("--row-bytes", type=int, default=ROW_BYTES_DEFAULT)
        sp.add_argument("--owners", type=int, default=OWNERS_DEFAULT)

    c = sub.add_parser("calibrate", help="fit a profile from sweep logs")
    c.add_argument("--logs", required=True)
    common_rows(c)
    c.add_argument("--out", default="kv_budget_profile.json")

    pr = sub.add_parser("predict", help="decode prediction for budgets")
    pr.add_argument("--profile", default=None, help="profile JSON (default: built-in)")
    common_rows(pr)
    pr.add_argument("--budget", nargs="+", required=True)

    sv = sub.add_parser("solve", help="smallest budget reaching a decode target")
    sv.add_argument("--profile", default=None)
    common_rows(sv)
    sv.add_argument("--target", type=float, required=True, help="ms/token")

    cv = sub.add_parser("curve", help="draw the decode-vs-budget curve")
    cv.add_argument("--profile", default=None)
    common_rows(cv)
    cv.add_argument("--out", default="kv_budget_curve.png")

    args = ap.parse_args()
    p = dict(DEFAULT_PROFILE)
    if getattr(args, "profile", None):
        p.update(json.load(open(args.profile)))

    if args.cmd == "calibrate":
        prof, err, points = calibrate_from_logs(args.logs, args.rows, args.row_bytes, args.owners)
        json.dump(prof, open(args.out, "w"), indent=1)
        print(f"profile -> {args.out}  (fit RMSE {err ** 0.5:.2f} ms over {len(points)} points)")
        for k, dec, ret in points:
            pred = predict_decode(k * args.rows * args.row_bytes, args.rows,
                                  prof, args.row_bytes, args.owners)[0]
            print(f"  k={k:2d} measured={dec:6.2f}  predicted={pred:6.2f}")
        return

    if args.cmd == "predict":
        print(f"{'budget':>12} {'k':>3} {'streamers':>9} {'decode(ms)':>10}  note")
        for btext in args.budget:
            b = parse_budget(btext)
            d, k, _, sat = predict_decode(b, args.rows, p, args.row_bytes, args.owners)
            note = "read worker saturated (unreliable, read-bound)" if sat else ""
            print(f"{btext:>12} {k:>3} {args.owners - k:>9} {d:>10.2f}  {note}")
        return

    if args.cmd == "solve":
        right, left, best = solve_budget(args.target, args.rows, p, args.row_bytes, args.owners)
        if right:
            b, d, k = right
            print(f"safe (right arm): {b} bytes ({b / 1024 / 1024:.1f} MiB, k={k}) "
                  f"-> predicted {d:.2f} ms")
        if left:
            b, d, k = left
            print(f"risky (dense-streaming arm, worker near saturation): "
                  f"{b} bytes ({b / 1024 / 1024:.1f} MiB, k={k}) -> predicted {d:.2f} ms")
        if not right and not left:
            b, d = best[0], best[1]
            print(f"target unreachable; best is {b} bytes at {d:.2f} ms "
                  f"(floor = T_c + O = {p['T_c'] + p['O']:.2f})")
        return

    if args.cmd == "curve":
        png_curve(args.rows, p, args.out, args.row_bytes, args.owners)


if __name__ == "__main__":
    main()
