#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Recompute recorded KV16 latency units; does not execute model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.repo.resolve()
folder = root / 'docs/validation/kv16-npu-device-20261111'
inputs = {}

def read(path):
    data = path.read_bytes()
    inputs[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8-sig')

record = json.loads(read(folder.with_suffix('.json')))
report = {'hardware_run_by_this_audit': False, 'units': {
    'decode_steady_ms': 'median milliseconds per token, already normalized',
    'decode_steady_tokens': 'sample count, not a divisor',
    'median_rate': '1000 / decode_steady_ms; not measured aggregate throughput'},
    'historical_pairs': [], 'optimized_pairs': [], 'profiles': {}}

def pairs(left_file, right_file, claimed=None):
    left = json.loads(read(folder / left_file))['switched']
    right = json.loads(read(folder / right_file))['switched']
    assert len(left) == len(right)
    result = []
    for i, (a, b) in enumerate(zip(left, right)):
        assert (a['session'], a['turn']) == (b['session'], b['turn'])
        row = {'session': a['session'], 'turn': a['turn'],
               'prefill_tokens': a['prefill_tokens'],
               'ttft_ms': [a['ttft_ms'], b['ttft_ms']],
               'correct_median_ms_per_token': [a['decode_steady_ms'], b['decode_steady_ms']],
               'samples': [a['decode_steady_tokens'], b['decode_steady_tokens']],
               'median_rate_tokens_per_second': [1000 / a['decode_steady_ms'], 1000 / b['decode_steady_ms']],
               'ids_and_position_equal': a['ids'] == b['ids'] and a['position'] == b['position']}
        if claimed is not None:
            wrong = [round(x['decode_steady_ms'] / x['decode_steady_tokens'], 2) for x in (a, b)]
            row['recorded_tpot_ms'] = claimed[i]['tpot_ms']
            row['double_division_reproduces_record'] = wrong == row['recorded_tpot_ms']
        result.append(row)
    return result

report['historical_pairs'] = pairs('perf-baseline-sessions.json', 'perf-codec-sessions.json',
                                   record['performance_no_trace']['rows'])
report['optimized_pairs'] = pairs('encode-opt/perf-off-sessions.json', 'encode-opt/perf-on-sessions.json')
for mode in ('baseline', 'codec'):
    log = read(folder / 'evidence-full' / mode / 'run.log')
    patterns = {'dsp_wall_us_per_token': r'dsp_wall=([\d.]+)',
                'round_trip_us_per_token': r' rt=([\d.]+)',
                'arm_forward_us_per_token': r'arm_fwd=([\d.]+)',
                'arm_between_calls_us_aggregate': r'arm_us=([\d.]+)',
                'miss_wait_us_per_token': r'miss_wait_us/token=([\d.]+)',
                'misses_per_token': r'misses/token=([\d.]+)',
                'predictor_drain_us_per_token': r'predictor=([\d.]+)',
                'prefetch_drain_us_per_token': r'prefetch=([\d.]+)'}
    row = {key: float(re.search(pattern, log).group(1)) for key, pattern in patterns.items()}
    row['seed_rpc_ms_by_prompt'] = [float(x) for x in re.findall(r'kv_seed_rpc_ms=([\d.]+)', log)]
    row['seed_calls_by_prompt'] = [int(x) for x in re.findall(r' seeds=(\d+)', log)]
    row['miss_fraction_of_dsp_wall'] = row['miss_wait_us_per_token'] / row['dsp_wall_us_per_token']
    report['profiles'][mode] = row
report['limitations'] = [
    'Profiles are stored diagnostic runs, not fresh HEAD device benchmarks.',
    'arm_us aggregate includes request/prefill gaps and is not steady host overhead.',
    'Do not add miss_wait to DSP wall or DSP wall to round-trip: they overlap.',
    'Optimized timing pair is a single pair with uncontrolled thermal variance.',
]
report['input_sha256'] = inputs
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('Historical TPOT double division reproduced:',
      sum(r['double_division_reproduces_record'] for r in report['historical_pairs']),
      '/', len(report['historical_pairs']))
for row in report['optimized_pairs']:
    print('session', row['session'], 'turn', row['turn'], 'median ms/token OFF/ON',
          [round(v, 3) for v in row['correct_median_ms_per_token']])
print(json.dumps(report['profiles'], indent=2))
