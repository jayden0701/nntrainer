"""Import the public vLLM Ridge safetensors artifact without refitting.

Original format: https://github.com/alesha-pro/vllm-cross-model-kv
Only unscaled split-half RoPE, all-head FP32 Ridge artifacts are accepted.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from safetensors import safe_open

from mapper import Geometry, Mapping
from snapshot import ModelSpec, new_directory, read_json, write_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def convert(manifest_path, tensor_path, source_dir, target_dir, output, revision):
    manifest = read_json(manifest_path)
    if (manifest['schema_version'] != 1 or manifest['kind'] != 'ridge'
            or manifest['rope_mode'] != 'stripped'):
        raise ValueError('requires version-1 content-space Ridge artifact')
    specs = []
    for side, directory in [('source', source_dir), ('target', target_dir)]:
        identity = manifest[side + '_model']
        config = read_json(directory / 'config.json')
        if config.get('rope_scaling'):
            raise ValueError('scaled RoPE is unsupported')
        for filename, key in [('config.json', 'config_sha256'),
                              ('tokenizer.json', 'tokenizer_json_sha256')]:
            if digest(directory / filename) != identity[key]:
                raise ValueError(f'{side} {filename} hash differs from upstream')
        model_id = 'Qwen/' + identity['model_id']
        spec = ModelSpec(model_id, config['num_hidden_layers'],
                         config['num_key_value_heads'], config['head_dim'],
                         config['rope_theta'], config['head_dim'])
        expected = (manifest['num_' + side + '_layers'],
                    manifest[side + '_kv_heads'], manifest[side + '_head_dim'])
        if (spec.layers, spec.kv_heads, spec.head_dim) != expected:
            raise ValueError(f'{side} geometry mismatch')
        if spec.theta != identity['rope_theta']:
            raise ValueError(f'{side} RoPE mismatch')
        specs.append(spec)
    with safe_open(tensor_path, framework='numpy') as tensors:
        if set(tensors.keys()) != {'source_layers', 'W_K', 'W_V', 'b_K', 'b_V'}:
            raise ValueError('unexpected tensor names')
        picks = tensors.get_tensor('source_layers')
        if picks.dtype.kind not in 'iu':
            raise ValueError('layer IDs must be integers')
        geometry = Geometry(*specs, 'all-head', picks.tolist())
        if picks.shape != (specs[1].layers, manifest['k']):
            raise ValueError('layer count differs from manifest')
        with new_directory(output) as staging:
            weights, bias = geometry.arrays(staging, 'w+')
            for plane, suffix in enumerate(('K', 'V')):
                for key, dest in [('W_' + suffix, weights[plane, :, 0]),
                                  ('b_' + suffix, bias[plane, :, 0])]:
                    value = tensors.get_tensor(key)
                    if value.dtype != np.dtype('float32') or value.shape != dest.shape:
                        raise ValueError(f'{key}: FP32 shape mismatch')
                    if not np.isfinite(value).all():
                        raise ValueError(f'{key}: nonfinite coefficients')
                    dest[:] = value
            weights.flush()
            bias.flush()
            write_json(staging / 'metadata.json', {
                'version': 1, 'fit_space': 'pre_rope', **geometry.metadata(),
                'alpha': manifest['lambda'],
                'import': {'repo': 'alesha-pro/vllm-cross-model-kv',
                           'revision': revision, 'mapper_id': manifest['mapper_id'],
                           'manifest_sha256': digest(manifest_path),
                           'safetensors_sha256': digest(tensor_path),
                           'refitted': False, 'precision': 'original FP32'},
                'upstream_manifest': manifest,
            })
    # Verify all coefficients after reopening the native memory-mapped format.
    mapped = Mapping(output)
    with safe_open(tensor_path, framework='numpy') as tensors:
        for plane, suffix in enumerate(('K', 'V')):
            np.testing.assert_array_equal(mapped.weights[plane, :, 0], tensors.get_tensor('W_' + suffix))
            np.testing.assert_array_equal(mapped.bias[plane, :, 0], tensors.get_tensor('b_' + suffix))
    return {'passed': True, 'all_coefficients_bitwise_equal': True,
            'source_layers_preserved': True, 'weights_sha256': digest(output / 'weights.f32'),
            'bias_sha256': digest(output / 'bias.f32')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'tensors', 'source_dir', 'target_dir', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    result = convert(args.manifest, args.tensors, args.source_dir, args.target_dir,
                     args.output, args.revision)
    write_json(args.output / 'conversion-check.json', result)
    print(result)


if __name__ == '__main__':
    main()
