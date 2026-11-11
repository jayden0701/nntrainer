#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline, lossless FP16 K/V codec reference; never used by inference.

Inputs are little-endian binary16 arrays in contiguous [token, head, dim] order.
Predictor 1 multiplies binary16 V by a binary32 gamma, rounds the product to
binary32 (nearest, ties to even), then to binary16 by integer arithmetic. It
preserves subnormals and signed zero; predictor NaNs are canonical 0x7e00.
Original NaN payloads, including signaling NaNs, survive in the XOR residual.

The little-endian v1 file has a fixed header, gamma words, rotary indices, a
16-byte index per block, then contiguous block payloads. RAW16 stores K then V.
EXACT16 stores V, rotary K, then LSB-first fixed-width XOR residuals.
EXACT16_SPARSE stores V, rotary K, then an LSB-first nonzero bitmap and nonzero
uint16 residuals in lane order. Each block chooses the smallest payload; ties
with RAW16 choose RAW16, and ties between exact modes choose fixed-width.
A whole-file CRC32 covers every byte except its own header field. Shape and
index lengths are validated independently.
This experimental format is not a checkpoint or HTP wire format.
"""

from array import array
import struct
import sys
import zlib


MAGIC = b"KV16"
VERSION = 1
PREDICTOR = 1
HEADER = struct.Struct("<4sHHQIIIIIIQI")
INDEX = struct.Struct("<QIBBH")
RAW16 = 0
EXACT16 = 1
EXACT16_SPARSE = 2
# Bound allocation even for an otherwise structurally valid untrusted file.
MAX_DECODED_BYTES = 1 << 30


def _round_shift(value, shift):
    if shift <= 0:
        return value << -shift
    quotient, remainder = divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    return quotient + (remainder > halfway or (remainder == halfway and quotient & 1))


def _round_binary(sign, significand, exponent, exponent_bits, fraction_bits):
    sign_word = sign << (exponent_bits + fraction_bits)
    if not significand:
        return sign_word
    bias = (1 << (exponent_bits - 1)) - 1
    minimum, maximum = 1 - bias, bias
    actual_exponent = significand.bit_length() - 1 + exponent
    if actual_exponent < minimum:
        rounded = _round_shift(significand, minimum - fraction_bits - exponent)
        return sign_word | rounded
    rounded = _round_shift(significand, actual_exponent - fraction_bits - exponent)
    if rounded == 1 << (fraction_bits + 1):
        rounded >>= 1
        actual_exponent += 1
    if actual_exponent > maximum:
        return sign_word | (((1 << exponent_bits) - 1) << fraction_bits)
    return sign_word | ((actual_exponent + bias) << fraction_bits) | (rounded - (1 << fraction_bits))


def _parts(word, exponent_bits, fraction_bits):
    sign = word >> (exponent_bits + fraction_bits)
    exponent = (word >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = word & ((1 << fraction_bits) - 1)
    bias = (1 << (exponent_bits - 1)) - 1
    if exponent == (1 << exponent_bits) - 1:
        return sign, "nan" if fraction else "inf", 0, 0
    if not exponent:
        return sign, "finite", fraction, 1 - bias - fraction_bits
    return sign, "finite", (1 << fraction_bits) | fraction, exponent - bias - fraction_bits


def predict_fp16(v_bits, gamma_bits):
    """Return predictor-1 bits, without depending on host FP modes or casts."""
    if not isinstance(v_bits, int) or not 0 <= v_bits <= 0xffff:
        raise ValueError("V must be a binary16 word")
    if not isinstance(gamma_bits, int) or not 0 <= gamma_bits <= 0xffffffff:
        raise ValueError("gamma must be a binary32 word")
    vs, vk, vm, ve = _parts(v_bits, 5, 10)
    gs, gk, gm, ge = _parts(gamma_bits, 8, 23)
    sign = vs ^ gs
    if vk == "nan" or gk == "nan":
        return 0x7e00
    if vk == "inf" or gk == "inf":
        if (vk == "finite" and vm == 0) or (gk == "finite" and gm == 0):
            return 0x7e00
        return (sign << 15) | 0x7c00
    product32 = _round_binary(sign, vm * gm, ve + ge, 8, 23)
    ps, pk, pm, pe = _parts(product32, 8, 23)
    if pk == "inf":
        return (ps << 15) | 0x7c00
    return _round_binary(ps, pm, pe, 5, 10)


def _words(data):
    words = array("H")
    words.frombytes(data)
    if sys.byteorder != "little":
        words.byteswap()
    return words


def _bytes(words):
    if sys.byteorder == "little":
        return words.tobytes()
    result = array("H", words)
    result.byteswap()
    return result.tobytes()


def _uint(value, limit, name):
    if type(value) is not int or not 0 <= value <= limit:
        raise ValueError(f"invalid {name}")
    return value


def _geometry(shape, gamma_bits, rotary_dims, block_rows):
    if len(shape) != 3:
        raise ValueError("shape must be (tokens, heads, dimensions)")
    tokens = _uint(shape[0], (1 << 64) - 1, "token count")
    heads = _uint(shape[1], (1 << 32) - 1, "head count")
    dimensions = _uint(shape[2], (1 << 32) - 1, "head dimension")
    block_rows = _uint(block_rows, (1 << 32) - 1, "block rows")
    if not heads or not dimensions or not block_rows:
        raise ValueError("heads, dimensions, and block rows must be positive")
    if tokens * heads * dimensions * 4 > MAX_DECODED_BYTES:
        raise ValueError("decoded K/V exceeds the reference allocation limit")
    gamma = tuple(_uint(word, 0xffffffff, "gamma word") for word in gamma_bits)
    if len(gamma) not in (dimensions, heads * dimensions):
        raise ValueError("gamma must have D or H*D binary32 words")
    rotary = tuple(_uint(dim, dimensions - 1, "rotary dimension") for dim in rotary_dims)
    if tuple(sorted(set(rotary))) != rotary:
        raise ValueError("rotary dimensions must be sorted and unique")
    return (tokens, heads, dimensions), gamma, rotary, block_rows


def _pack_residuals(residuals, width):
    packed = bytearray()
    accumulator = available = 0
    for residual in residuals:
        accumulator |= residual << available
        available += width
        while available >= 8:
            packed.append(accumulator & 255)
            accumulator >>= 8
            available -= 8
    if available:
        packed.append(accumulator)
    return bytes(packed)


def _unpack_residuals(packed, count, width):
    mask = (1 << width) - 1
    accumulator = available = offset = 0
    for _ in range(count):
        while available < width:
            accumulator |= packed[offset] << available
            offset += 1
            available += 8
        yield accumulator & mask
        accumulator >>= width
        available -= width


def _pack_sparse(residuals):
    bitmap = bytearray((len(residuals) + 7) // 8)
    nonzero = array("H")
    for index, residual in enumerate(residuals):
        if residual:
            bitmap[index // 8] |= 1 << (index % 8)
            nonzero.append(residual)
    return bytes(bitmap) + _bytes(nonzero)


def _unpack_sparse(packed, count):
    value_offset = (count + 7) // 8
    for index in range(count):
        if packed[index // 8] & (1 << (index % 8)):
            yield struct.unpack_from("<H", packed, value_offset)[0]
            value_offset += 2
        else:
            yield 0


def encode(k, v, shape, gamma_bits, rotary_dims, block_rows=16):
    """Encode original bytes; the complete container may exceed raw K+V."""
    return encode_with_stats(k, v, shape, gamma_bits, rotary_dims, block_rows)[0]


def encode_with_stats(k, v, shape, gamma_bits, rotary_dims, block_rows=16):
    shape, gamma, rotary, block_rows = _geometry(shape, gamma_bits, rotary_dims, block_rows)
    tokens, heads, dimensions = shape
    rows = tokens * heads
    expected = rows * dimensions * 2
    if len(k) != expected or len(v) != expected:
        raise ValueError("K/V byte length does not match shape")
    kw, vw = _words(k), _words(v)
    rotary_set = set(rotary)
    nonrotary = tuple(dim for dim in range(dimensions) if dim not in rotary_set)
    blocks = []
    exact_count = predicted_count = 0
    residual_width_histogram = [0] * 17
    for row_start in range(0, rows, block_rows):
        row_end = min(rows, row_start + block_rows)
        start, end = row_start * dimensions, row_end * dimensions
        residuals, rotated = [], array("H")
        for row in range(row_start, row_end):
            row_offset = row * dimensions
            gamma_offset = (row % heads) * dimensions if len(gamma) != dimensions else 0
            rotated.extend(kw[row_offset + dim] for dim in rotary)
            for dim in nonrotary:
                residual = kw[row_offset + dim] ^ predict_fp16(vw[row_offset + dim], gamma[gamma_offset + dim])
                residuals.append(residual)
                predicted_count += 1
                exact_count += residual == 0
                residual_width_histogram[residual.bit_length()] += 1
        width = max(residuals, default=0).bit_length()
        raw = _bytes(kw[start:end]) + _bytes(vw[start:end])
        base = _bytes(vw[start:end]) + _bytes(rotated)
        bitpacked = base + _pack_residuals(residuals, width)
        sparse = base + _pack_sparse(residuals)
        chosen = (RAW16, 0, raw)
        if len(bitpacked) < len(chosen[2]):
            chosen = (EXACT16, width, bitpacked)
        if len(sparse) < len(chosen[2]):
            chosen = (EXACT16_SPARSE, 0, sparse)
        blocks.append(chosen)
    metadata = struct.pack(f"<{len(gamma)}I", *gamma) + struct.pack(f"<{len(rotary)}I", *rotary)
    offset = HEADER.size + len(metadata) + INDEX.size * len(blocks)
    index, payload = bytearray(), bytearray()
    for mode, width, data in blocks:
        index.extend(INDEX.pack(offset, len(data), mode, width, 0))
        payload.extend(data)
        offset += len(data)
    body = metadata + index + payload
    header = HEADER.pack(MAGIC, VERSION, PREDICTOR, tokens, heads, dimensions, block_rows,
                         len(gamma), len(rotary), len(blocks), offset, 0)
    checksum = zlib.crc32(body, zlib.crc32(header[:-4]))
    header = header[:-4] + struct.pack("<I", checksum)
    blob = header + body
    stats = inspect_blob(blob)
    stats.update(exact_prediction_count=exact_count, predicted_count=predicted_count,
                 exact_prediction_fraction=exact_count / predicted_count if predicted_count else None,
                 residual_width_histogram=residual_width_histogram)
    return blob, stats


def _parse(blob):
    if len(blob) < HEADER.size:
        raise ValueError("truncated KV16 header")
    (magic, version, predictor, tokens, heads, dimensions, block_rows,
     gamma_count, rotary_count, block_count, file_size, checksum) = HEADER.unpack_from(blob)
    if magic != MAGIC or version != VERSION or predictor != PREDICTOR:
        raise ValueError("unsupported KV16 magic/version/predictor")
    if file_size != len(blob):
        raise ValueError("KV16 file size mismatch")
    # Validate all untrusted dimensions before metadata slicing or allocation.
    if (not heads or not dimensions or not block_rows or
            tokens * heads * dimensions * 4 > MAX_DECODED_BYTES or
            gamma_count not in (dimensions, heads * dimensions) or
            rotary_count > dimensions):
        raise ValueError("invalid KV16 geometry")
    rows = tokens * heads
    if block_count != (rows + block_rows - 1) // block_rows:
        raise ValueError("invalid KV16 block count")
    gamma_end = HEADER.size + gamma_count * 4
    rotary_end = gamma_end + rotary_count * 4
    payload_start = rotary_end + block_count * INDEX.size
    if payload_start > len(blob):
        raise ValueError("truncated KV16 metadata/index")
    if zlib.crc32(memoryview(blob)[HEADER.size:], zlib.crc32(memoryview(blob)[:HEADER.size - 4])) != checksum:
        raise ValueError("KV16 checksum mismatch")
    gamma = struct.unpack_from(f"<{gamma_count}I", blob, HEADER.size)
    rotary = struct.unpack_from(f"<{rotary_count}I", blob, gamma_end)
    shape, gamma, rotary, block_rows = _geometry((tokens, heads, dimensions), gamma, rotary, block_rows)
    expected_offset = payload_start
    entries = []
    for block in range(block_count):
        offset, length, mode, width, reserved = INDEX.unpack_from(blob, rotary_end + block * INDEX.size)
        count_rows = min(block_rows, rows - block * block_rows)
        if reserved or offset != expected_offset or width > 16:
            raise ValueError("invalid KV16 index")
        if offset + length > len(blob):
            raise ValueError("KV16 payload exceeds file")
        if mode == RAW16:
            expected_length = count_rows * dimensions * 4
            if width:
                raise ValueError("RAW16 residual width must be zero")
        elif mode in (EXACT16, EXACT16_SPARSE):
            residual_count = count_rows * (dimensions - rotary_count)
            base_length = count_rows * (dimensions + rotary_count) * 2
            if mode == EXACT16:
                expected_length = base_length + (residual_count * width + 7) // 8
                # Require canonical zero padding in the final residual byte.
                meaningful_bits = residual_count * width
                if meaningful_bits % 8 and length:
                    if blob[offset + length - 1] >> (meaningful_bits % 8):
                        raise ValueError("nonzero KV16 residual padding")
            else:
                if width:
                    raise ValueError("EXACT16_SPARSE residual width must be zero")
                bitmap_size = (residual_count + 7) // 8
                if length < base_length + bitmap_size:
                    raise ValueError("truncated KV16 sparse bitmap")
                bitmap = memoryview(blob)[offset + base_length:offset + base_length + bitmap_size]
                if residual_count % 8 and bitmap[-1] >> (residual_count % 8):
                    raise ValueError("nonzero KV16 sparse bitmap padding")
                nonzero_count = sum(word.bit_count() for word in bitmap)
                expected_length = base_length + bitmap_size + nonzero_count * 2
                if length != expected_length:
                    raise ValueError("KV16 sparse bitmap/popcount length mismatch")
                residual_data = memoryview(blob)[offset + base_length + bitmap_size:offset + length]
                if any(word == 0 for (word,) in struct.iter_unpack("<H", residual_data)):
                    raise ValueError("noncanonical zero KV16 sparse residual")
            if expected_length >= count_rows * dimensions * 4:
                raise ValueError("EXACT16 payload is not smaller than RAW16")
        else:
            raise ValueError("unknown KV16 block mode")
        if length != expected_length or offset + length > len(blob):
            raise ValueError("invalid KV16 payload length")
        entries.append((offset, length, mode, width, count_rows))
        expected_offset += length
    if expected_offset != len(blob):
        raise ValueError("trailing KV16 payload")
    return shape, gamma, rotary, block_rows, entries, payload_start


def inspect_blob(blob):
    """Validate the complete container and account for every stored byte."""
    shape, gamma, rotary, block_rows, entries, payload_start = _parse(blob)
    dimensions = shape[2]
    raw_bytes = v_bytes = rotary_bytes = residual_bytes = 0
    raw_blocks = encoded_blocks = sparse_blocks = bitpacked_blocks = 0
    bitmap_bytes = nonzero_bytes = bitpacked_bytes = 0
    widths = [0] * 17
    for _, length, mode, width, count_rows in entries:
        if mode == RAW16:
            raw_bytes += length
            raw_blocks += 1
        else:
            encoded_blocks += 1
            v_bytes += count_rows * dimensions * 2
            rotary_bytes += count_rows * len(rotary) * 2
            block_residual_bytes = length - count_rows * (dimensions + len(rotary)) * 2
            residual_bytes += block_residual_bytes
            if mode == EXACT16:
                bitpacked_blocks += 1
                widths[width] += 1
                bitpacked_bytes += block_residual_bytes
            else:
                sparse_blocks += 1
                block_bitmap_bytes = (count_rows * (dimensions - len(rotary)) + 7) // 8
                bitmap_bytes += block_bitmap_bytes
                nonzero_bytes += block_residual_bytes - block_bitmap_bytes
    baseline = shape[0] * shape[1] * shape[2] * 4
    return {
        "format_version": VERSION, "predictor_id": PREDICTOR, "dtype": "fp16",
        "shape": list(shape), "block_rows": block_rows, "gamma_bits": list(gamma),
        "rotary_dims": list(rotary), "original_bytes": baseline, "encoded_bytes": len(blob),
        "global_payload_ratio": len(blob) / baseline if baseline else None,
        "header_bytes": HEADER.size, "gamma_bytes": len(gamma) * 4,
        "rotary_index_bytes": len(rotary) * 4, "block_index_bytes": len(entries) * INDEX.size,
        "padding_bytes": 0, "metadata_bytes": payload_start,
        "raw_payload_bytes": raw_bytes, "v_payload_bytes": v_bytes,
        "rotary_k_payload_bytes": rotary_bytes, "residual_payload_bytes": residual_bytes,
        "raw_blocks": raw_blocks, "encoded_blocks": encoded_blocks,
        "bitpacked_blocks": bitpacked_blocks, "sparse_blocks": sparse_blocks,
        "bitpacked_residual_bytes": bitpacked_bytes,
        "residual_bitmap_bytes": bitmap_bytes, "residual_nonzero_bytes": nonzero_bytes,
        "residual_accounting_note": "bitpacked_residual_bytes + residual_bitmap_bytes + residual_nonzero_bytes equals residual_payload_bytes; subcounts are not additional storage",
        "encoded_block_width_histogram": widths,
        "width_histogram_note": "Only fixed-width EXACT16 blocks; sparse blocks use a bitmap instead",
        "raw_block_fraction": raw_blocks / len(entries) if entries else None,
        "predictor_uses_hardware_fp": False,
        "memory_note": "Python reference materializes original words and complete output; allocator peak is not measured",
    }


def decode(blob):
    """Validate fully, then return restored K/V bytes and container metadata.

    Caller receives no partial output on a validation/decode error. Restored
    arrays are newly owned; blob and any original cache are never mutated.
    """
    shape, gamma, rotary, block_rows, entries, _ = _parse(blob)
    _, heads, dimensions = shape
    rotary_set = set(rotary)
    nonrotary = tuple(dim for dim in range(dimensions) if dim not in rotary_set)
    restored_k, restored_v = array("H"), array("H")
    for block, (offset, length, mode, width, count_rows) in enumerate(entries):
        data = memoryview(blob)[offset:offset + length]
        word_count = count_rows * dimensions
        if mode == RAW16:
            restored_k.extend(_words(data[:word_count * 2]))
            restored_v.extend(_words(data[word_count * 2:]))
            continue
        vw = _words(data[:word_count * 2])
        rotated_end = (word_count + count_rows * len(rotary)) * 2
        rotated = iter(_words(data[word_count * 2:rotated_end]))
        if mode == EXACT16:
            residuals = iter(_unpack_residuals(data[rotated_end:], count_rows * len(nonrotary), width))
        else:
            residuals = iter(_unpack_sparse(data[rotated_end:], count_rows * len(nonrotary)))
        kw = array("H", [0]) * word_count
        for local_row in range(count_rows):
            row = block * block_rows + local_row
            row_offset = local_row * dimensions
            gamma_offset = (row % heads) * dimensions if len(gamma) != dimensions else 0
            for dim in rotary:
                kw[row_offset + dim] = next(rotated)
            for dim in nonrotary:
                kw[row_offset + dim] = predict_fp16(vw[row_offset + dim], gamma[gamma_offset + dim]) ^ next(residuals)
        restored_k.extend(kw)
        restored_v.extend(vw)
    return _bytes(restored_k), _bytes(restored_v), inspect_blob(blob)
