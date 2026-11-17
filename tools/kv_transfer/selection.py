"""Rank source layers by calibration-only same-head OLS probes."""

import math

import numpy as np

from mapper import calibration_batches, center_batch
from snapshot import PLANES, positive_int


class ProbeMoments:
    """Centered batch moments for every source/target layer pair of one head."""

    def __init__(self, source, target):
        self.source_layers = list(range(source.layers))
        self.target_layers = list(range(target.layers))
        self.source_dim = source.head_dim
        self.target_dim = target.head_dim
        p = source.layers * source.head_dim
        q = target.layers * target.head_dim
        self.count = 0
        self.x_mean = np.zeros(p)
        self.y_mean = np.zeros(q)
        self.xx = np.zeros((source.layers, source.head_dim, source.head_dim))
        self.xy = np.zeros((p, q))
        self.yy = np.zeros(q)

    def update(self, x, y):
        n = len(x)
        x, y = center_batch(x, y, self.count, self.x_mean, self.y_mean)
        # OLS probes each source layer separately, so only diagonal source
        # covariance blocks are needed. All cross-covariances share one GEMM.
        x_by_layer = x.reshape(len(x), len(self.source_layers), self.source_dim).transpose(1, 0, 2)
        self.xx += x_by_layer.swapaxes(1, 2) @ x_by_layer
        self.xy += x.T @ y
        self.yy += np.einsum("ij,ij->j", y, y)
        self.count += n

    def r2(self):
        total_variance = self.yy.reshape(-1, self.target_dim).sum(axis=1)
        valid = total_variance > 0
        scores = np.zeros((len(self.target_layers), len(self.source_layers)))
        for source_layer, xx in enumerate(self.xx):
            eigenvalues, eigenvectors = np.linalg.eigh(xx)
            # Discard the numerical null space of the Gram matrix, including
            # duplicate and constant source coordinates.
            cutoff = np.finfo(np.float64).eps * self.source_dim * eigenvalues[-1]
            active = eigenvalues > max(0, cutoff)
            start = source_layer * self.source_dim
            projected = eigenvectors[:, active].T @ self.xy[start:start + self.source_dim]
            projected /= np.sqrt(eigenvalues[active])[:, None]
            explained = np.einsum("ij,ij->j", projected, projected)
            explained = explained.reshape(-1, self.target_dim).sum(axis=1)
            # In-sample OLS with an intercept has R² in [0, 1].
            scores[valid, source_layer] = np.clip(explained[valid] / total_variance[valid], 0, 1)
        return scores, valid


def select_layers_by_r2(train, k, *, block_size, token_stride, memory_mib):
    """Return each target layer's top k calibration-scored source layers."""
    positive_int(block_size, "block size")
    positive_int(memory_mib, "memory budget")
    positive_int(token_stride, "token stride")
    if not train:
        raise ValueError("OLS selection needs at least one calibration prompt")
    source, target = train[0].source.spec, train[0].target.spec
    if not 1 <= k <= source.layers:
        raise ValueError("source layer count k must be between one and source depth")
    if source.kv_heads != target.kv_heads:
        raise ValueError("OLS layer selection requires equal source and target KV heads")

    p = source.layers * source.head_dim
    q = target.layers * target.head_dim
    # Covariance, its largest multiplication temporary and the row blocks.
    estimate = 8 * (2 * p * q + 4 * p * source.head_dim + 6 * block_size * (p + q))
    if estimate > memory_mib * (1 << 20):
        raise ValueError(
            f"estimated OLS selection working set {math.ceil(estimate / (1 << 20))} MiB "
            f"exceeds {memory_mib} MiB; reduce block size or increase memory budget"
        )

    shape = target.layers, source.layers
    plane_scores = np.zeros(shape)
    plane_counts = np.zeros((target.layers, 1), dtype=int)
    head_counts = {}
    for channel in PLANES:
        channel_scores = np.zeros(shape)
        channel_counts = np.zeros((target.layers, 1), dtype=int)
        for head in range(target.kv_heads):
            probe = ProbeMoments(source, target)
            for x, y in calibration_batches(
                    train, channel, probe.source_layers, probe.target_layers, head,
                    block_size, token_stride):
                probe.update(x, y)
            scores, valid = probe.r2()
            channel_scores += scores
            channel_counts[:, 0] += valid
            # Release this head's matrices before allocating the next head.
            del probe
        head_counts[channel] = channel_counts[:, 0].tolist()
        plane_scores += channel_scores / np.maximum(channel_counts, 1)
        plane_counts += channel_counts > 0

    scores = plane_scores / np.maximum(plane_counts, 1)
    layers = [
        sorted(range(source.layers), key=lambda i: (-row[i], i))[:k]
        for row in scores
    ]
    return layers, {
        "method": "ols-r2",
        "scores_by_target_layer": scores.tolist(),
        "nonconstant_heads_by_target_layer": head_counts,
    }
