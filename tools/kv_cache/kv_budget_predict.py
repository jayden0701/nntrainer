#!/usr/bin/env python3
"""Find the trade-off between CPU FSU retention and steady decode latency.

calibrate --logs DIR --rows 630 --out profile.json
solve --profile profile.json --rows 630 --memory kv-peak --cap 80MiB
solve --profile profile.json --rows 630 --metric p95 --slowdown-pct 10
predict --profile profile.json --rows 630 --budget 24MiB 52MiB
curve --profile profile.json --rows 630 --out curve.png

Compare one device, pack, workload and I/O policy per sweep. Use --requests and
NNTR_KV_TIMING=1. For older logs supply --budget-map budgets.json, mapping log
stems (without _rN) to the configured byte count or a size such as "24MiB".
Resident logs are named resident*.log. Legacy parameter-only profiles still
support exploratory predictions. A measured SLA needs actual token timings.
"""

import argparse
from bisect import bisect_left
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import statistics
import sys

MIB = 1024 ** 2
ROW_BYTES_DEFAULT = 4096
OWNERS_DEFAULT = 28
# Published medians and nominal settings from kv-cache-fsu-status.ko.md §9.4.
# There are no published per-token p95 or total KV peaks for this sweep.
DEFAULT_PROFILE = {
    "name": "SM-S947N Qwen3-1.7B (published sweep)",
    "device": "SM-S947N", "model": "Qwen3-1.7B",
    "rows_cal": 630, "row_bytes": ROW_BYTES_DEFAULT, "owners": OWNERS_DEFAULT,
    "resident": {"p50": 31.9},
    "points": [{"budget": mib * MIB, "p50": ms}
               for mib, ms in [(24, 36.2), (40, 41.8), (48, 41.5),
                               (52, 40.7), (60, 37.7), (74, 33.5)]],
    # Original fit, retained for explicitly labelled extrapolations.
    "T_c": 31.87, "O": 1.56, "t_service": 1.28, "W": 0.78,
    "g0": 1.4, "ramp": 1.2, "page_bytes": 4096,
}


def parse_budget(text):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(mib|mb|b)?", str(text).strip(), re.I)
    if not match:
        raise ValueError(f"invalid byte budget {text!r}")
    units = {"mib": MIB, "mb": 10**6, "b": 1, "": 1}
    value = Fraction(match[1]) * units[(match[2] or "").lower()]
    if value.denominator != 1:
        raise ValueError(f"fractional byte budget {text!r}")
    return int(value)


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def owner_size(rows, row_bytes, page_bytes):
    if min(rows, row_bytes, page_bytes) <= 0 or row_bytes % 2:
        raise ValueError("expected positive geometry and equally sized K/V planes")
    plane = rows * (row_bytes // 2)
    return 2 * ((plane + page_bytes - 1) // page_bytes) * page_bytes


def model_decode(k, owners, profile):
    missing = owners - k
    floor = profile["T_c"] + profile["O"]
    if not missing:
        return floor
    gap = owners / missing
    fraction = min(1., max(0., (gap - profile["g0"]) / profile["ramp"]))
    cost = profile["t_service"] + profile["W"] * fraction
    return floor + missing * max(0., cost - profile["T_c"] / owners)


def log_records(path):
    """The FSU report and token percentile precede their [REQUEST] record."""
    fsu = tail = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[kv-fsu] lookahead=" in line:
            fsu = {name: int(value) for name, value in re.findall(r"(\w+)=(\d+)", line)}
        if "[kv-decode]" in line:
            m = re.search(r"steady_tokens=(\d+) p95_ms=([\d.]+)", line)
            if not m:
                raise ValueError(f"{path}: malformed steady-token timing")
            tail = positive(float(m[2]), "p95") if int(m[1]) else None
        if "[REQUEST]" in line:
            request = json.loads(line.split("[REQUEST]", 1)[1].strip())
            yield request, fsu, tail
            fsu = tail = None


def summarize(runs):
    result = {"p50": statistics.median(r["p50"] for r in runs), "runs": len(runs)}
    # Conservative across repetitions: these are per-request token p95s and
    # process-lifetime KV high-water marks, not percentiles of request medians.
    for metric in ("p95", "kv_peak"):
        if all(r.get(metric) is not None for r in runs):
            result[metric] = max(r[metric] for r in runs)
    return result


def fit_model(points, resident, owners):
    """Fit the original spacing model as a diagnostic, separate from solve.

    With a fixed spacing ramp, the stall is a nonnegative base plus a
    nonnegative ramp coefficient per missing owner. Fit both by least squares.
    """
    full = [p["p50"] for p in points if p.get("k") == owners and p.get("zero_reads")]
    streaming = [p for p in points if p.get("k") is not None and p["k"] < owners]
    if not full or len({p["k"] for p in streaming}) < 4:
        return {}
    floor = statistics.median(full)
    gaps = sorted({owners / (owners - p["k"]) for p in streaming})
    best = {}
    best_error = math.inf
    for i, start in enumerate(gaps):
        for end in gaps[i + 1:]:
            xx = xy = yy = xz = yz = 0.
            samples = []
            for p in streaming:
                x = owners - p["k"]
                y = x * max(0., min(1., (owners / x - start) / (end - start)))
                z = p["p50"] - floor
                xx += x*x
                xy += x*y
                yy += y*y
                xz += x*z
                yz += y*z
                samples.append((x, y, z))
            candidates = [(max(0., xz/xx), 0.), (0., max(0., yz/yy))]
            determinant = xx*yy - xy*xy
            if determinant > 0:
                candidates.append(((xz*yy - yz*xy)/determinant, (yz*xx - xz*xy)/determinant))
            for base, extra in candidates:
                if min(base, extra) < 0:
                    continue
                error = sum((x*base + y*extra - z)**2 for x, y, z in samples)
                if error < best_error:
                    best_error = error
                    best = dict(T_c=resident, O=floor-resident, t_service=resident/owners+base,
                                W=extra, g0=start, ramp=end-start)
    best["streaming_fit_rmse_ms"] = math.sqrt(best_error / len(streaming))
    return best


def calibrate(logdir, rows, row_bytes, owners, mapping, page_bytes=None):
    resident = []
    groups = defaultdict(list)
    workload = None
    io_mode = None
    geometry = None
    write_modes = set()
    for path in sorted(Path(logdir).rglob("*.log")):
        tag = re.sub(r"_r\d+$", "", path.stem)
        last_bytes = 0
        for request, fsu, p95 in log_records(path):
            if fsu is not None:
                read_bytes = fsu["read_bytes"]
                read_delta = read_bytes - last_bytes
                if read_delta < 0:
                    raise ValueError(f"{path}: write each process run to a separate log")
                last_bytes = read_bytes
            if request["cached_after"] != rows:
                continue
            sig = (request["cached_before"], request["prefill_tokens"],
                   request["output_ids"], request.get("pending_token"))
            if workload is not None and sig != workload:
                raise ValueError(f"{path}: workload or generated IDs differ at {rows} rows")
            workload = sig
            run = {"p50": positive(request["decode_ms"], f"{path} decode_ms"), "p95": p95}
            if tag.startswith("resident") and fsu is None:
                resident.append(run)
                continue
            if fsu is None or tag.startswith("resident"):
                raise ValueError(f"{path}: inconsistent FSU/resident run")
            mode = (fsu["direct_io"], fsu["lookahead"], fsu.get("page_bytes"),
                    fsu.get("parallel_reader"), fsu.get("async_writer"))
            shape = (fsu.get("flash_owners", fsu.get("layers")), fsu.get("uniform_row_bytes"),
                     fsu.get("layers"))
            if geometry is not None and geometry != shape:
                raise ValueError(f"{path}: mixed model geometries")
            geometry = shape
            if io_mode is not None and mode != io_mode:
                raise ValueError(f"{path}: separate sweeps by I/O mode, workers, lookahead and page size")
            io_mode = mode
            budget = fsu.get("budget_bytes")
            if tag in mapping:
                declared = parse_budget(mapping[tag])
                if budget is not None and declared != budget:
                    raise ValueError(f"{path}: inconsistent configured budget")
                budget = declared
            if budget is None:
                raise ValueError(f"{path}: supply configured bytes for {tag!r} in --budget-map")
            if fsu["retained_bytes"] > budget:
                raise ValueError(f"{path}: retained bytes exceed configured budget")
            if budget:
                write_modes.add((fsu.get("write_back"), fsu.get("flush_sync")))
                if len(write_modes) > 1:
                    raise ValueError(f"{path}: separate sweeps by write policy")
            run["kv_peak"] = fsu.get("kv_payload_peak_bytes")
            run["k"] = fsu.get("retained_owners")
            # Older logs have only bytes. Divide only for an explicitly
            # specified uniform geometry and its actual per-plane alignment.
            if run["k"] is None and row_bytes and owners and page_bytes:
                count, rem = divmod(fsu["retained_bytes"], owner_size(rows, row_bytes, page_bytes))
                if rem or count > owners:
                    raise ValueError(f"{path}: retained bytes do not match uniform owner geometry")
                run["k"] = count
            run["zero_reads"] = read_delta == 0
            groups[budget].append(run)
    if not resident or not groups:
        raise ValueError(f"need resident and FSU [REQUEST] records at exactly {rows} rows")
    log_owners, log_width, layers = geometry
    if owners is not None and log_owners is not None and owners != log_owners:
        raise ValueError("--owners disagrees with the recorded geometry")
    if row_bytes is not None and log_width is not None and row_bytes != log_width:
        raise ValueError("--row-bytes disagrees with the recorded geometry")
    owners = log_owners if log_owners is not None else owners
    row_bytes = log_width if log_width is not None else row_bytes
    profile = {"name": Path(logdir).resolve().name, "rows_cal": rows,
               "row_bytes": row_bytes, "owners": owners, "layers": layers,
               "resident": summarize(resident),
               "io": dict(zip(("direct_io", "lookahead", "page_bytes", "parallel_reader",
                               "async_writer"), io_mode)),
               "points": []}
    if write_modes:
        profile["io"]["write_back"], profile["io"]["flush_sync"] = write_modes.pop()
    for b, rs in sorted(groups.items()):
        point = dict(budget=b, **summarize(rs))
        counts = {r["k"] for r in rs}
        if len(counts) == 1:
            point["k"] = counts.pop()
        point["zero_reads"] = all(r["zero_reads"] for r in rs)
        profile["points"].append(point)
    if page_bytes and io_mode[2] and page_bytes != io_mode[2]:
        raise ValueError("--page-bytes disagrees with the recorded page size")
    if io_mode[2] or page_bytes:
        profile["page_bytes"] = io_mode[2] or page_bytes
    if owners and row_bytes:
        profile.update(fit_model(profile["points"], profile["resident"]["p50"], owners))
    return profile


def measured_points(profile, metric, memory):
    points = profile.get("points", [])
    if not points:
        raise ValueError("recommendations need a measured sweep; use calibrate (or predict for legacy fits)")
    for point in points:
        if metric not in point or memory not in point:
            raise ValueError(f"sweep lacks {metric}/{memory} data; collect matched logs with NNTR_KV_TIMING=1")
        positive(point[metric], metric)
        for field in ("budget", memory):
            if type(point[field]) is not int or point[field] < 0:
                raise ValueError(f"invalid measured {field}")
    return points


def solve(points, metric, memory, cap, target):
    eligible = [p for p in points if cap is None or p[memory] <= cap]
    if target is not None:
        eligible = [p for p in eligible if p[metric] <= target]
        return min(eligible, key=lambda p: (p[memory], p[metric], p["budget"]), default=None)
    return min(eligible, key=lambda p: (p[metric], p[memory], p["budget"]), default=None)


def predict(budget, rows, profile, metric, extrapolate=False):
    points = sorted(profile.get("points", []), key=lambda p: p["budget"])
    if points and rows == profile["rows_cal"]:
        measured_points(profile, metric, "budget")
        xs = [p["budget"] for p in points]
        i = bisect_left(xs, budget)
        if i < len(xs) and xs[i] == budget:
            return points[i][metric], "measured"
        if 0 < i < len(xs):
            fraction = (budget - xs[i-1]) / (xs[i] - xs[i-1])
            return points[i-1][metric] + fraction * (points[i][metric] - points[i-1][metric]), "interpolated"
    if metric != "p50" or "t_service" not in profile or (points and not extrapolate):
        raise ValueError("outside measured coverage; calibrate for this budget range/context")
    size = owner_size(rows, profile["row_bytes"], profile["page_bytes"])
    owners = profile["owners"]
    k = min(owners, budget // size)
    return model_decode(k, owners, profile), "legacy model estimate; unverified"


def draw_curve(rows, profile, metric, output, extrapolate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = profile.get("points", [])
    if points and rows == profile["rows_cal"]:
        points = sorted(measured_points(profile, metric, "budget"), key=lambda p: p["budget"])
        xs, ys = [p["budget"] for p in points], [p[metric] for p in points]
        label = "measured settings (lines interpolate)"
    else:
        if "t_service" not in profile or (points and not extrapolate):
            raise ValueError("no measured curve for this context; calibrate this workload")
        size = owner_size(rows, profile["row_bytes"], profile["page_bytes"])
        xs = [k * size for k in range(profile["owners"] + 1)]
        ys = [predict(b, rows, profile, metric, extrapolate)[0] for b in xs]
        label = "legacy model estimate"
    fig, ax = plt.subplots()
    ax.plot([b / MIB for b in xs], ys, ".-", label=label)
    ax.set(xlabel="Configured retention (MiB)", ylabel=f"Steady decode {metric} (ms/token)",
           title=f"{profile.get('name', 'FSU')} at {rows} rows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "predict", "solve", "curve"):
        sp = sub.add_parser(name)
        sp.add_argument("--rows", type=int, required=True)
        sp.add_argument("--owners", type=int)
        sp.add_argument("--row-bytes", type=int)
        sp.add_argument("--page-bytes", type=int, help="system page size for legacy uniform-geometry logs/profiles")
        sp.add_argument("--device", help="device ID for this sweep/profile")
        sp.add_argument("--model", help="pack ID for this sweep/profile")
        if name == "calibrate":
            sp.add_argument("--logs", required=True)
            sp.add_argument("--budget-map", help='JSON file, e.g. {"b24":"24MiB"} for legacy logs')
            sp.add_argument("--out", default="kv_budget_profile.json")
            continue
        if name == "solve":
            source = sp.add_mutually_exclusive_group(required=True)
            source.add_argument("--profile", help="profile collected for the target workload")
            source.add_argument("--example", action="store_true", help="use the published SM-S947N example")
        else:
            sp.add_argument("--profile", help="default: published 630-row Qwen example")
        sp.add_argument("--metric", choices=("p50", "p95"), default="p50")
        if name in ("curve", "predict"):
            sp.add_argument("--extrapolate", action="store_true",
                            help="allow unvalidated p50 estimates outside a measured sweep")
        if name == "predict":
            sp.add_argument("--budget", nargs="+", required=True)
        elif name == "curve":
            sp.add_argument("--out", default="kv_budget_curve.png")
        else:
            sp.add_argument("--memory", choices=("retention", "kv-peak"), default="retention",
                            help="configured retention or measured KV payload high-water in bytes")
            sp.add_argument("--cap", help="maximum selected memory (bytes, MB or MiB)")
            group = sp.add_mutually_exclusive_group()
            group.add_argument("--target", type=float, help="maximum steady ms/token")
            group.add_argument("--slowdown-pct", type=float, help="allowed %% above same-sweep resident")
    args = ap.parse_args(argv)
    try:
        positive(args.rows, "rows")
        for n in (args.owners, args.row_bytes, args.page_bytes):
            if n is not None:
                positive(n, "geometry")
        if args.cmd == "calibrate":
            mapping = json.loads(Path(args.budget_map).read_text()) if args.budget_map else {}
            p = calibrate(args.logs, args.rows, args.row_bytes, args.owners, mapping, args.page_bytes)
            p.update(device=args.device, model=args.model)
            Path(args.out).write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
            print(f"{len(p['points'])} measured budget settings -> {args.out}")
            if "streaming_fit_rmse_ms" in p:
                print(f"spacing model in-sample streaming RMSE: {p['streaming_fit_rmse_ms']:.2f} ms")
            return 0

        p = json.loads(Path(args.profile).read_text()) if args.profile else dict(DEFAULT_PROFILE)
        if not args.profile:
            print("Example only: published SM-S947N / Qwen3-1.7B sweep.", file=sys.stderr)
        for key in ("device", "model"):
            requested = getattr(args, key)
            if requested and requested != p.get(key):
                raise ValueError(f"{key} differs from the selected profile; calibrate this target")
        for key, arg in (("owners", args.owners), ("row_bytes", args.row_bytes),
                         ("page_bytes", args.page_bytes)):
            if key in p and arg is not None and p[key] != arg:
                raise ValueError(f"{key} differs from supplied profile")
            if not p.get("points"):
                if key not in p and arg is None:
                    raise ValueError(f"legacy profile needs explicit --{key.replace('_', '-')}")
                p.setdefault(key, arg)
                positive(p[key], key)
        if args.cmd == "curve":
            draw_curve(args.rows, p, args.metric, args.out, args.extrapolate)
            return 0
        if args.cmd == "predict":
            for value in args.budget:
                ms, origin = predict(parse_budget(value), args.rows, p, args.metric, args.extrapolate)
                print(f"{value} -> {ms:.2f} ms/token ({args.metric}, {origin})")
            return 0

        if p.get("points") and args.rows != p["rows_cal"]:
            raise ValueError("solve requires measurements at the requested context length")
        cap = parse_budget(args.cap) if args.cap is not None else None
        target = args.target
        if args.slowdown_pct is not None:
            if not math.isfinite(args.slowdown_pct) or args.slowdown_pct < 0:
                raise ValueError("slowdown-pct must be finite and nonnegative")
            baseline = p.get("resident", {"p50": p.get("T_c")}).get(args.metric)
            if baseline is None:
                raise ValueError(f"profile lacks matched resident {args.metric} measurements")
            target = baseline * (1 + args.slowdown_pct / 100)
        if target is not None:
            positive(target, "target")
        if cap is None and target is None:
            raise ValueError("specify --cap, --target or --slowdown-pct")
        memory = "budget" if args.memory == "retention" else "kv_peak"
        source = "measured sweep"
        if not p.get("points") and "t_service" in p:
            if args.metric != "p50" or memory != "budget":
                raise ValueError("legacy fits provide only estimated p50 and retention bytes")
            size = owner_size(args.rows, p["row_bytes"], p["page_bytes"])
            points = [{"budget": k*size, "p50": model_decode(k, p["owners"], p)}
                      for k in range(p["owners"] + 1)
                      if not p.get("t_service_worker") or
                      (p["owners"] - k)*p["t_service_worker"] <= p["T_c"]]
            source = "legacy model estimate; verify on device"
        else:
            points = measured_points(p, args.metric, memory)
        result = solve(points, args.metric, memory, cap, target)
        if result is None:
            print(f"No setting satisfies the constraints ({source}).", file=sys.stderr)
            fastest = solve(points, args.metric, memory, cap, None)
            if fastest is not None:
                print(f"Fastest within the memory limit: {fastest[args.metric]:.2f} ms/token "
                      f"at {fastest['budget']} retention bytes.", file=sys.stderr)
            return 1
        print(f"{p.get('name', 'profile')} ({args.rows} rows): "
              f"NNTR_KV_RESIDENT_BYTES={result['budget']}  "
              f"{args.memory}={result[memory]} bytes  {args.metric}={result[args.metric]:.2f} ms/token "
              f"({source})")
        return 0
    except (ValueError, KeyError, OSError, ImportError, TypeError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
