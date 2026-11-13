#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description="KV16 review probes; race exits with the sanitizer process status.")
parser.add_argument('mode', choices=('ab', 'race', 'race-control'))
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--engine-build', type=Path)
args = parser.parse_args()
ROOT = args.repo.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
if args.mode != 'ab' and args.engine_build is None:
    parser.error('--engine-build is required for race probes')

def run(cmd, name, cwd=ROOT, env=None):
    with (OUT / (name + '.log')).open('w') as log:
        p = subprocess.run([str(x) for x in cmd], cwd=cwd, stdout=log,
                           stderr=subprocess.STDOUT, env=env, timeout=300)
    print(name, p.returncode, flush=True)
    return p.returncode

if args.mode == 'ab':
    source = subprocess.check_output(['git', 'show', '5ededcb00:Applications/CausalLM/kv16_lossless.cpp'], cwd=ROOT, text=True)
    source = source.replace('namespace causallm {', 'namespace causalleg {\nusing causallm::kv16::Geometry;\nusing causallm::kv16::Decoded;\nusing causallm::kv16::max_decoded_bytes;')
    old = OUT / 'baseline.cpp'
    old.write_text(source)
    exe = OUT / 'ab'
    assert run(['g++', '-O2', '-std=c++17', '-I.', '-IApplications/CausalLM',
                'tools/kv16/kv16_encode_ab_harness.cpp', old, '-o', exe], 'ab-build') == 0
    for arg, name in [([], 'ab'), (['predict'], 'ab-predict'), (['bench'], 'ab-bench')]:
        assert run([exe, *arg], name) == 0
elif args.mode in ('race', 'race-control'):
    spec = importlib.util.spec_from_file_location('integration', ROOT / 'test/htp/host/run_kv16_integration_checks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Args:
        repo = ROOT
        engine_build = args.engine_build.resolve()
    base, library = module.prepare(Args(), {})
    source = OUT / 'race.cpp'
    source.write_text(r'''
#define main original_prefix_check_main
#include "test/htp/host/kv16_prefix_check.cpp"
#undef main
int main() {
  std::map<void *, std::shared_ptr<void>> slabs;
  Manager cache;
  cache.setSharedAllocator([&](size_t n) -> void * {
    auto *p = std::malloc(n);
    if (p) slabs.emplace(p, std::shared_ptr<void>(p, std::free));
    return p;
  }, [&](void *p) { slabs.erase(p); });
  cache.setSharedRetain([&](void *p) { return slabs.at(p); });
  cache.allocate(1, 1, 4096, std::vector<unsigned>{64}, DataType::UINT16, Format::NCHW);
  fill(cache);
  cache.configurePrefixCodec(0, geometry());
  for (unsigned i = 0; i < 30; ++i) {
    cache.setPosition(4096);
    auto snapshot = cache.capturePrefix(4096);
    cache.reset();
    cache.drainPrefixCodec();
    auto raw = snapshot->layers[0].pages[0]->decodeCopy();
    require(!raw->key.empty(), "empty snapshot");
  }
}
''')
    extra = ['-O1', '-g', '-fsanitize=thread', '-fno-omit-frame-pointer', '-I' + str(ROOT)]
    tag = args.mode
    manager = ROOT / 'Applications/CausalLM/kv_cache_manager.cpp'
    if tag == 'race-control':
        contents = manager.read_text()
        assert contents.count('          prefix_views_[l].push_back(page);') == 1
        manager = OUT / 'manager-control.cpp'
        manager.write_text(contents.replace('          prefix_views_[l].push_back(page);',
                                           '          if (!codec)\n            prefix_views_[l].push_back(page);'))
    objects = []
    for index, item in enumerate([manager, ROOT / 'Applications/CausalLM/kv16_lossless.cpp', source]):
        obj = OUT / f'{tag}-{index}.o'
        assert run([*base, *extra, '-c', item, '-o', obj], f'{tag}-build-{index}', Args.engine_build) == 0
        objects.append(obj)
    exe = OUT / tag
    assert run([base[0], *objects, *extra, '-L' + str(library.parent), '-lnntrainer', '-Wl,-rpath,' + str(library.parent), '-pthread', '-ldl', '-lm', '-o', exe], tag + '-link', Args.engine_build) == 0
    env = dict(os.environ, TSAN_OPTIONS='halt_on_error=1')
    raise SystemExit(run([exe], tag, env=env))
