"""Chunked affine ridge regression between aligned model KV caches."""

import math
from pathlib import Path

import numpy as np

from dataset import describe_split
from snapshot import (
    PLANES, ModelSpec, Snapshot, create_snapshot, new_directory, open_plane,
    positive_int, read_json, rotate_keys, write_json,
)


def select_layers(source_count, target_count, k):
    if not 1 <= k <= source_count:
        raise ValueError("source layer count k must be between one and source depth")
    centers = np.linspace(0, source_count - 1, target_count)
    if target_count == 1:
        centers[0] = (source_count - 1) / 2
    return [
        sorted(sorted(range(source_count), key=lambda i: (abs(i - center), i))[:k])
        for center in centers
    ]


def validate_layers(layers, source, target):
    if not isinstance(layers, list) or len(layers) != target.layers:
        raise ValueError("layer map must have one list per target layer")
    if not isinstance(layers[0], list) or not layers[0]:
        raise ValueError("each target layer needs at least one source layer")
    k = len(layers[0])
    for row in layers:
        if (not isinstance(row, list) or len(row) != k
                or any(type(i) is not int or not 0 <= i < source.layers for i in row)
                or len(set(row)) != k):
            raise ValueError("each layer map row needs k distinct valid source indices")
    return k


class Geometry:
    def __init__(self, source, target, layout, layers):
        self.source = source
        self.target = target
        self.layout = layout
        self.layers = layers
        k = validate_layers(layers, source, target)
        if layout == "headwise":
            if source.kv_heads != target.kv_heads:
                raise ValueError("headwise mapping requires equal source and target KV heads")
            self.units = target.kv_heads
            self.features = k * source.head_dim
            self.outputs = target.head_dim
        elif layout == "all-head":
            self.units = 1
            self.features = k * source.kv_heads * source.head_dim
            self.outputs = target.kv_heads * target.head_dim
        else:
            raise ValueError(f"unknown mapping layout: {layout}")

    def metadata(self):
        return {
            "source": self.source.metadata(), "target": self.target.metadata(),
            "layout": self.layout, "source_layers": self.layers,
        }

    def arrays(self, path, mode="r"):
        units = len(PLANES), self.target.layers, self.units
        return (
            open_plane(path / "weights.f32", units + (self.features, self.outputs), mode),
            open_plane(path / "bias.f32", units + (self.outputs,), mode),
        )

    def head(self, unit):
        return unit if self.layout == "headwise" else slice(None)


class Mapping:
    def __init__(self, path):
        path = Path(path)
        meta = read_json(path / "metadata.json")
        if (type(meta.get("version")) is not int or meta["version"] != 1
                or meta["fit_space"] != "pre_rope"):
            raise ValueError(f"{path}: unsupported mapping format")
        self.geometry = Geometry(
            ModelSpec.parse(meta["source"]), ModelSpec.parse(meta["target"]),
            meta["layout"], meta["source_layers"],
        )
        self.weights, self.bias = self.geometry.arrays(path)


def center_batch(x, y, count, x_mean, y_mean):
    """Center rows and encode the covariance merge as one weighted row."""
    n = len(x)
    batch_x_mean = x.mean(axis=0)
    batch_y_mean = y.mean(axis=0)
    x -= batch_x_mean
    y -= batch_y_mean
    dx = batch_x_mean - x_mean
    dy = batch_y_mean - y_mean
    total = count + n
    if count:
        scale = math.sqrt(count * n / total)
        # The outer product of this row adds the between-batch term to
        # the same matrix product that accumulates the within-batch term.
        x = np.concatenate((x, (scale * dx)[None, :]))
        y = np.concatenate((y, (scale * dy)[None, :]))
    x_mean += dx * (n / total)
    y_mean += dy * (n / total)
    return x, y


class Ridge:
    """Accumulate centered moments in bounded token batches in float64."""

    def __init__(self, features, outputs):
        self.count = 0
        self.x_mean = np.zeros(features)
        self.y_mean = np.zeros(outputs)
        self.xx = np.zeros((features, features))
        self.xy = np.zeros((features, outputs))

    def update(self, x, y):
        n = len(x)
        x, y = center_batch(x, y, self.count, self.x_mean, self.y_mean)
        self.xx += x.T @ x
        self.xy += x.T @ y
        self.count += n

    def solve(self, alpha):
        self.xx.flat[::len(self.xx) + 1] += alpha
        weights = np.linalg.solve(self.xx, self.xy)
        return weights, self.y_mean - self.x_mean @ weights


class ErrorStats:
    def __init__(self):
        self.count = 0
        self.error = 0.0
        self.reference = 0.0
        self.predicted = 0.0
        self.dot = 0.0

    def update(self, predicted, reference):
        predicted = rows64(predicted)
        reference = rows64(reference)
        residual = predicted - reference
        self.count += reference.size
        self.error += float(np.vdot(residual, residual))
        self.reference += float(np.vdot(reference, reference))
        self.predicted += float(np.vdot(predicted, predicted))
        self.dot += float(np.vdot(predicted, reference))

    def metrics(self):
        return {
            "elements": self.count,
            "rmse": math.sqrt(self.error / self.count),
            "target_rms": math.sqrt(self.reference / self.count),
            "relative_l2": math.sqrt(self.error / self.reference) if self.reference else None,
            "cosine": min(1.0, max(-1.0, self.dot / math.sqrt(self.reference * self.predicted)))
            if self.reference and self.predicted else None,
        }


def blocks(tokens, size):
    for start in range(0, tokens, size):
        yield slice(start, min(start + size, tokens))


def sampled_blocks(tokens, size, stride):
    """Yield at most size rows per block, starting at row zero in each prompt."""
    for start in range(0, tokens, size * stride):
        yield slice(start, min(start + size * stride, tokens), stride)


def rows64(block):
    result = np.array(block, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError("non-finite KV or mapping data")
    return result


def features(source, plane, channel, layer_ids, span, head):
    # Move tokens to the leading dimension before applying model RoPE.
    block = np.swapaxes(plane[layer_ids, span, head, :], 0, 1)
    if channel == "keys":
        block = rotate_keys(block, source.positions[span], source.spec, inverse=True)
    return block.reshape(len(block), -1)


def calibration_batches(pairs, channel, input_layers, output_layers, head, block_size, stride):
    """Pack sampled rows across prompts while preserving each prompt's stride."""
    source_spec, target_spec = pairs[0].source.spec, pairs[0].target.spec
    source_heads = source_spec.kv_heads if isinstance(head, slice) else 1
    target_heads = target_spec.kv_heads if isinstance(head, slice) else 1
    x = np.empty((block_size, len(input_layers) * source_heads * source_spec.head_dim))
    y = np.empty((block_size, len(output_layers) * target_heads * target_spec.head_dim))
    count = 0
    for pair in pairs:
        source = pair.source.plane(channel)
        target = pair.target.plane(channel)
        for span in sampled_blocks(pair.source.tokens, block_size, stride):
            input_rows = features(pair.source, source, channel, input_layers, span, head)
            output_rows = features(pair.target, target, channel, output_layers, span, head)
            if not np.isfinite(input_rows).all() or not np.isfinite(output_rows).all():
                raise ValueError("non-finite KV calibration data")
            offset = 0
            while offset < len(input_rows):
                take = min(block_size - count, len(input_rows) - offset)
                x[count:count + take] = input_rows[offset:offset + take]
                y[count:count + take] = output_rows[offset:offset + take]
                count += take
                offset += take
                if count == block_size:
                    yield x, y
                    count = 0
    if count:
        yield x[:count], y[:count]


def check_block_budget(geometry, block_size, memory_mib, *, fitting):
    positive_int(block_size, "block size")
    positive_int(memory_mib, "memory budget")
    p, q = geometry.features, geometry.outputs
    # Include ridge moments, LAPACK copies, multiplication temporaries and a
    # predicted/reference block. File-backed arrays can use OS page cache.
    row_width = geometry.target.kv_heads * geometry.target.head_dim
    estimate = 8 * block_size * (6 * (p + q) + 8 * row_width)
    if fitting:
        estimate += 8 * (4 * p * p + 5 * p * q)
    if estimate > memory_mib * (1 << 20):
        raise ValueError(
            f"estimated working set {math.ceil(estimate / (1 << 20))} MiB "
            f"exceeds {memory_mib} MiB; reduce k or block size or use headwise"
        )


def fit(path, train, validation, geometry, *, alpha, block_size, memory_mib,
        token_stride=1, selection=None):
    positive_int(token_stride, "token stride")
    if not train:
        raise ValueError("fit needs at least one calibration prompt")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge alpha must be finite and positive")
    check_block_budget(geometry, block_size, memory_mib, fitting=True)
    with new_directory(path) as staging:
        weights, bias = geometry.arrays(staging, "w+")
        try:
            for channel_id, channel in enumerate(PLANES):
                for layer, source_layers in enumerate(geometry.layers):
                    for unit in range(geometry.units):
                        ridge = Ridge(geometry.features, geometry.outputs)
                        head = geometry.head(unit)
                        for x, y in calibration_batches(
                                train, channel, source_layers, [layer], head, block_size,
                                token_stride):
                            ridge.update(x, y)
                        index = channel_id, layer, unit
                        weights[index], bias[index] = ridge.solve(alpha)
                        if (not np.isfinite(weights[index]).all()
                                or not np.isfinite(bias[index]).all()):
                            raise ValueError("ridge produced non-finite coefficients")
            report = (held_out(validation, geometry, weights, bias, block_size)
                      if validation else None)
            weights.flush()
            bias.flush()
        finally:
            weights._mmap.close()
            bias._mmap.close()
        metadata = {
            "version": 1, "fit_space": "pre_rope", **geometry.metadata(),
            "alpha": alpha, "token_stride": token_stride,
            "layer_selection": selection if selection is not None else {"method": "explicit"},
            "train": describe_split(train, token_stride),
            "validation": describe_split(validation), "held_out": report,
        }
        write_json(staging / "metadata.json", metadata)
    return metadata


def predict(source, plane, geometry, channel, span, layer, weights, bias):
    shape = geometry.target.kv_heads, geometry.target.head_dim
    block = np.empty((span.stop - span.start, *shape), dtype=np.float32)
    for unit in range(geometry.units):
        head = geometry.head(unit)
        x = features(source, plane, channel, geometry.layers[layer], span, head)
        y = x @ weights[unit] + bias[unit]
        view = block[:, head, :]
        view[...] = y.reshape(view.shape)
    if channel == "keys":
        block = rotate_keys(block, source.positions[span], geometry.target)
    if not np.isfinite(block).all():
        raise ValueError("mapping produced non-finite cache data")
    return block


def held_out(pairs, geometry, weights, bias, block_size):
    stats = {channel: ErrorStats() for channel in PLANES}
    for pair in pairs:
        for channel_id, channel in enumerate(PLANES):
            source = pair.source.plane(channel)
            target = pair.target.plane(channel)
            for layer in range(geometry.target.layers):
                index = channel_id, layer
                for span in blocks(pair.source.tokens, block_size):
                    predicted = predict(pair.source, source, geometry, channel, span,
                                        layer, weights[index], bias[index])
                    stats[channel].update(predicted, target[layer, span])
    return {channel: stat.metrics() for channel, stat in stats.items()}


def apply(mapping, source, output, *, block_size, memory_mib):
    geometry = mapping.geometry
    if source.spec != geometry.source:
        raise ValueError("source snapshot model or shape does not match mapping")
    check_block_budget(geometry, block_size, memory_mib, fitting=False)
    with create_snapshot(output, geometry.target, source.token_ids, source.positions) as planes:
        for channel_id, channel in enumerate(PLANES):
            source_plane = source.plane(channel)
            for layer in range(geometry.target.layers):
                index = channel_id, layer
                for span in blocks(source.tokens, block_size):
                    planes[channel][layer, span] = predict(
                        source, source_plane, geometry, channel, span, layer,
                        mapping.weights[index], mapping.bias[index],
                    )


def compare(predicted, reference, block_size):
    positive_int(block_size, "block size")
    if predicted.spec != reference.spec:
        raise ValueError("prediction and reference model or shape do not match")
    if predicted.token_ids != reference.token_ids or predicted.positions != reference.positions:
        raise ValueError("prediction and reference are not token-aligned")
    report = {}
    for channel in PLANES:
        stat = ErrorStats()
        predicted_plane = predicted.plane(channel)
        reference_plane = reference.plane(channel)
        for layer in range(reference.spec.layers):
            for span in blocks(reference.tokens, block_size):
                stat.update(predicted_plane[layer, span], reference_plane[layer, span])
        report[channel] = stat.metrics()
    return report
