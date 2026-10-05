"""Row-exact multi-row affine QMV for native-MTP verification.

Why: a native-MTP verify forward feeds 2..8 token rows through every
projection at once. MLX ``quantized_matmul`` switches from its single-row
``qmv_fast`` kernel to a different multi-row kernel for M >= 2, whose
reduction order differs, so every verify row's logits differ from the logits
the same token gets in an ordinary single-token decode step (measured on
Qwen3.8 Flash-Next JANGH4, MLX 0.32.3: tens of elements per projection row,
~1.3k of 248k lm_head logits at M=4). Greedy native MTP therefore did not
reproduce AR output byte for byte. TensorFold, mlx-serve and oMLX all keep
each verify row bit-identical to its solo decode call.

How: this kernel is MLX's ``qmv_fast_impl`` with every x row handled inside
the same threadgroup. Each row runs exactly the per-row operation sequence of
``qmv_fast`` (same ``load_vector``/``qdot`` calls, same k-block order, same
``simd_sum``), so its result equals the M=1 call bit for bit, while the
weight block is fetched once for all rows. ``load_vector`` and ``qdot`` are
taken verbatim from the installed MLX headers at runtime, so the arithmetic
follows the exact MLX build. Every (shape, bits, group, rows, dtype) is
admitted only after a bit-equality self-check against stock M=1 calls on the
real weights; a failed check keeps the stock path for that shape and logs it.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

MAX_ROWS = 8
_ADMITTED: dict[tuple, str | None] = {}
STATS = {"calls": 0, "admitted": 0, "rejected": 0}


def row_exact_qmv_requested() -> bool:
    """Default on; ``VMLX_ROW_EXACT_VERIFY_QMV=0`` restores stock MLX rows."""
    return os.environ.get("VMLX_ROW_EXACT_VERIFY_QMV", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _extract_template_function(source: str, signature: str) -> str | None:
    at = source.find(signature)
    if at < 0:
        return None
    start = source.rfind("template", 0, at)
    open_brace = source.find("{", at)
    if start < 0 or open_brace < 0:
        return None
    depth = 0
    for index in range(open_brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    return None


@lru_cache(maxsize=1)
def _mlx_qmv_header() -> str | None:
    """MLX's own load_vector/qdot/pack helpers from the installed headers."""
    try:
        package = Path(mx.__file__).resolve().parent
        path = package / "include/mlx/backend/metal/kernels/quantized.h"
        source = path.read_text()
    except Exception as exc:  # pragma: no cover - packaging dependent
        logger.warning("row-exact verify QMV unavailable: MLX header unreadable (%s)", exc)
        return None
    pieces = []
    for signature in (
        "inline constexpr short get_pack_factor()",
        "inline constexpr short get_bytes_per_pack()",
        "inline U load_vector(const device T* x, thread U* x_thread)",
        "inline U qdot(\n    const device uint8_t* w,",
        "inline U load_vector_safe(const device T* x, thread U* x_thread, int N)",
        "inline U qdot_safe(",
    ):
        piece = _extract_template_function(source, signature)
        if piece is None:
            logger.warning("row-exact verify QMV unavailable: %r not found in %s", signature, path)
            return None
        pieces.append(piece)
    # qmv_fast_impl itself must still have the traversal this kernel copies.
    impl = _extract_template_function(source, "METAL_FUNC void qmv_fast_impl(")
    generic = _extract_template_function(source, "METAL_FUNC void qmv_impl(")
    if impl is None or not _traversal_matches(impl) or generic is None or not _generic_traversal_matches(generic):
        logger.warning("row-exact verify QMV unavailable: MLX qmv traversal changed")
        return None
    # Register copies: identical arithmetic, weights read from a thread-local
    # copy so each weight byte is fetched once for all verify rows.
    for piece in list(pieces[2:]):
        if "inline U qdot" in piece:
            pieces.append(
                piece.replace("inline U qdot_safe(", "inline U qdot_safe_reg(")
                .replace("inline U qdot(", "inline U qdot_reg(")
                .replace("device", "thread")
            )
    return "#ifndef SIMD_SIZE\n#define SIMD_SIZE 32\n#endif\n" + "\n\n".join(pieces)


def _traversal_matches(impl: str) -> bool:
    compact = re.sub(r"\s+", " ", impl)
    needles = (
        "constexpr int packs_per_thread = bits == 2 ? 1 : 2;",
        "constexpr int num_simdgroups = 2;",
        "constexpr int results_per_simdgroup = 4;",
        "constexpr int pack_factor = get_pack_factor<bits, 32>();",
        "constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();",
        "U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);",
        "result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);",
        "result[row] = simd_sum(result[row]);",
        "y[row] = static_cast<T>(result[row]);",
    )
    return all(needle in compact for needle in needles)


def _generic_traversal_matches(impl: str) -> bool:
    compact = re.sub(r"\s+", " ", impl)
    needles = (
        "constexpr int packs_per_thread = 1;",
        "const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);",
        "for (; k < in_vec_size - block_size; k += block_size) {",
        "U sum = load_vector_safe<T, U, values_per_thread, bits>( x, x_thread, remaining);",
        "result[row] += qdot_safe<U, values_per_thread, bits>( wl, x_thread, s, b, sum, remaining);",
    )
    return all(needle in compact for needle in needles)


# MLX qmv_impl (the M=1 kernel when K is not a qmv_fast multiple). Both of its
# branches: N >= 8 moves the last tile back and recomputes identical outputs;
# N < 8 keeps out_row and guards every output row. Every x row r runs the
# identical per-row sequence.
_SOURCE_GENERIC = r"""
  constexpr int bits = BITS;
  constexpr int group_size = GS;
  constexpr int in_vec_size = K;
  constexpr int out_vec_size = N;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  typedef float U;

  uint3 tid = threadgroup_position_in_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w;
  const device T* sc = scales;
  const device T* bi = biases;
  thread U x_thread[ROWS][values_per_thread];
  thread U result[ROWS][results_per_simdgroup];
  for (int r = 0; r < ROWS; r++)
    for (int row = 0; row < results_per_simdgroup; row++) result[r][row] = 0;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  constexpr bool small_n = out_vec_size < (num_simdgroups * results_per_simdgroup);
  const int used_out_row = small_n ? out_row : min(out_vec_size - results_per_simdgroup, out_row);
  if (out_row >= out_vec_size) {
    return;
  }
  ws += used_out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  sc += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  bi += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xs = x + simd_lid * values_per_thread;

  int k = 0;
  for (; k < in_vec_size - block_size; k += block_size) {
    U sum[ROWS];
    for (int r = 0; r < ROWS; r++) {
      sum[r] = load_vector<T, U, values_per_thread, bits>(xs + r * in_vec_size, x_thread[r]);
    }
    for (int row = 0; row < results_per_simdgroup && (!small_n || out_row + row < out_vec_size); row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = sc + row * in_vec_size_g;
      const device T* bl = bi + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
#if REG
      thread uint32_t wbw[(packs_per_thread * bytes_per_pack + 3) / 4];
      thread uint8_t* wb = (thread uint8_t*)wbw;
      for (int j = 0; j < packs_per_thread * bytes_per_pack; j++) wb[j] = wl[j];
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot_reg<U, values_per_thread, bits>(wb, x_thread[r], s, b, sum[r]);
      }
#else
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread[r], s, b, sum[r]);
      }
#endif
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bi += block_size / group_size;
    xs += block_size;
  }
  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    U sum[ROWS];
    for (int r = 0; r < ROWS; r++) {
      sum[r] = load_vector_safe<T, U, values_per_thread, bits>(xs + r * in_vec_size, x_thread[r], remaining);
    }
    for (int row = 0; row < results_per_simdgroup && (!small_n || out_row + row < out_vec_size); row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = sc + row * in_vec_size_g;
      const device T* bl = bi + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
#if REG
      thread uint32_t wbw[(packs_per_thread * bytes_per_pack + 3) / 4];
      thread uint8_t* wb = (thread uint8_t*)wbw;
      for (int j = 0; j < packs_per_thread * bytes_per_pack; j++) wb[j] = wl[j];
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot_safe_reg<U, values_per_thread, bits>(wb, x_thread[r], s, b, sum[r], remaining);
      }
#else
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot_safe<U, values_per_thread, bits>(wl, x_thread[r], s, b, sum[r], remaining);
      }
#endif
    }
  }
  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < results_per_simdgroup && (!small_n || out_row + row < out_vec_size); row++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        y[r * out_vec_size + used_out_row + row] = static_cast<T>(v);
      }
    }
  }
"""


# MLX qmv_quad_impl (the M=1 kernel for K == 64 or 128): one simdgroup per
# threadgroup, eight 4-lane quads, eight output rows per quad. Quad indices are
# derived from the simd lane exactly as Metal defines them for 32-thread groups.
_SOURCE_QUAD = r"""
  constexpr int bits = BITS;
  constexpr int group_size = GS;
  constexpr int in_vec_size = K;
  constexpr int out_vec_size = N;
  constexpr int QUAD = 4;
  constexpr int quads_per_simd = SIMD_SIZE / QUAD;
  constexpr int pack_factor = 32 / bits;
  constexpr int values_per_thread = K / QUAD;
  constexpr int packs_per_thread = values_per_thread / pack_factor;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int results_per_quadgroup = 8;
  typedef float U;

  uint3 tid = threadgroup_position_in_grid;
  uint lane = thread_index_in_simdgroup;
  uint quad_gid = lane / QUAD;
  uint quad_lid = lane % QUAD;

  thread U x_thread[ROWS][values_per_thread];
  thread U result[ROWS][results_per_quadgroup];
  for (int r = 0; r < ROWS; r++)
    for (int row = 0; row < results_per_quadgroup; row++) result[r][row] = 0;

  const int in_vec_size_w = in_vec_size / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * quads_per_simd * results_per_quadgroup + quad_gid;

  const device uint32_t* wp = w + out_row * in_vec_size_w + quad_lid * packs_per_thread;
  const device T* sc = scales + out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  const device T* bi = biases + out_row * in_vec_size_g + quad_lid / scale_step_per_thread;

  U sum[ROWS];
  for (int r = 0; r < ROWS; r++) {
    sum[r] = load_vector<T, U, values_per_thread, bits>(
        x + r * in_vec_size + quad_lid * values_per_thread, x_thread[r]);
  }
  for (int row = 0; row < results_per_quadgroup; row++) {
    auto wl = (const device uint8_t*)(wp + row * in_vec_size_w * quads_per_simd);
    const device T* sl = sc + row * in_vec_size_g * quads_per_simd;
    const device T* bl = bi + row * in_vec_size_g * quads_per_simd;
    U s = sl[0];
    U b = bl[0];
    if (row * quads_per_simd + out_row < out_vec_size) {
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread[r], s, b, sum[r]);
      }
    }
  }
  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < results_per_quadgroup; row++) {
      U v = quad_sum(result[r][row]);
      if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {
        y[r * out_vec_size + out_row + row * quads_per_simd] = static_cast<T>(v);
      }
    }
  }
"""


_SOURCE = r"""
  // MLX qmv_fast_impl, transcribed; every x row r runs the identical per-row
  // sequence, all rows share one threadgroup and one weight walk.
  constexpr int bits = BITS;
  constexpr int group_size = GS;
  constexpr int in_vec_size = K;
  constexpr int out_vec_size = N;
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  typedef float U;

  uint3 tid = threadgroup_position_in_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w;
  const device T* sc = scales;
  const device T* bi = biases;

  thread U x_thread[ROWS][values_per_thread];
  thread U result[ROWS][results_per_simdgroup];
  for (int r = 0; r < ROWS; r++)
    for (int row = 0; row < results_per_simdgroup; row++) result[r][row] = 0;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  bi += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xs = x + simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum[ROWS];
    for (int r = 0; r < ROWS; r++) {
      sum[r] = load_vector<T, U, values_per_thread, bits>(xs + r * in_vec_size, x_thread[r]);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = sc + row * in_vec_size_g;
      const device T* bl = bi + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
#if REG
      thread uint32_t wbw[(packs_per_thread * bytes_per_pack + 3) / 4];
      thread uint8_t* wb = (thread uint8_t*)wbw;
      for (int j = 0; j < packs_per_thread * bytes_per_pack; j++) wb[j] = wl[j];
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot_reg<U, values_per_thread, bits>(wb, x_thread[r], s, b, sum[r]);
      }
#else
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread[r], s, b, sum[r]);
      }
#endif
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bi += block_size / group_size;
    xs += block_size;
  }

  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        y[r * out_vec_size + out_row + row] = static_cast<T>(v);
      }
    }
  }
"""


def _register_weights(rows: int) -> bool:
    """Read each weight block into registers once for all rows.

    Measured on Flash-Next JANGH4 (in-process verify ladder, 2 interleaved
    process pairs): registers win from three rows (S3 30.24 -> 29.42 ms,
    S4 34.0 -> 33.3 ms) and lose slightly at two (25.46 -> 25.74 ms).
    ``VMLX_ROW_EXACT_QMV_REG=0/1`` pins either choice.
    """
    pinned = os.environ.get("VMLX_ROW_EXACT_QMV_REG", "").strip()
    if pinned in {"0", "1"}:
        return pinned == "1"
    return rows >= 3


@lru_cache(maxsize=8)
def _kernel(variant: str = "fast", reg: bool | None = None):
    header = _mlx_qmv_header()
    if header is None:
        return None
    reg = bool(reg)
    source = {"fast": _SOURCE, "generic": _SOURCE_GENERIC, "quad": _SOURCE_QUAD}[variant]
    return mx.fast.metal_kernel(
        name=f"vmlx_row_exact_verify_qmv_{variant}{'_reg' if reg else ''}",
        input_names=["w", "scales", "biases", "x"],
        output_names=["y"],
        header=header,
        source=f"#define REG {1 if reg else 0}\n" + source,
    )


def _values_per_thread(bits: int) -> int:
    pack_factor = 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)
    return pack_factor * (1 if bits == 2 else 2)


def _max_rows(bits: int) -> int:
    # Rows live in registers; packs wider than 8 values per thread spill past
    # four rows (measured 10x slower at 8 rows for 2/3/4/5-bit).
    return MAX_ROWS if bits in (6, 8) else 4


def _variants(rows, in_dim, out_dim, bits, group_size, dtype) -> tuple[str, ...]:
    """Candidate transcriptions in admission order (the self-check decides)."""
    if bits not in (2, 3, 4, 5, 6, 8) or group_size not in (32, 64, 128):
        return ()
    if dtype not in (mx.float16, mx.bfloat16) or not 2 <= rows <= _max_rows(bits):
        return ()
    if in_dim % group_size:
        return ()
    fast_ok = (in_dim % (_values_per_thread(bits) * 32) == 0 and out_dim % 8 == 0
               and group_size % _values_per_thread(bits) == 0)
    quad_ok = (in_dim in (64, 128) and bits in (2, 4, 8)
               and group_size % (in_dim // 4) == 0 and rows * in_dim // 4 <= 128)
    variants = ("quad",) if quad_ok else ()
    return variants + (("fast", "generic") if fast_ok else ("generic",))


def _device_sized(a):
    # metal_kernel binds inputs smaller than 8 elements as `constant` buffers;
    # the transcribed MLX code addresses `device` memory. Zero padding at the
    # end keeps every flat index and therefore every result unchanged.
    if a.size >= 8:
        return a
    return mx.concatenate([a.reshape(-1), mx.zeros((8,), dtype=a.dtype)])


def _launch(x2d, weight, scales, biases, bits, group_size, variant="fast"):
    rows, in_dim = x2d.shape
    out_dim = weight.shape[0]
    weight, scales, biases = (_device_sized(a) for a in (weight, scales, biases))
    (y,) = _kernel(variant, _register_weights(rows))(
        inputs=[weight, scales, biases, x2d],
        template=[("T", x2d.dtype), ("BITS", bits), ("GS", group_size),
                  ("K", in_dim), ("N", out_dim), ("ROWS", rows)],
        grid=(32, (out_dim + 63) // 64, 1) if variant == "quad" else (32, 2 * ((out_dim + 7) // 8), 1),
        threadgroup=(32, 1, 1) if variant == "quad" else (32, 2, 1),
        output_shapes=[(rows, out_dim)],
        output_dtypes=[x2d.dtype],
    )
    return y


def _admit(key, weight, scales, biases, bits, group_size, rows, in_dim, dtype, variants):
    if key in _ADMITTED:
        return _ADMITTED[key]
    probe = (mx.random.normal((rows, in_dim), key=mx.random.key(1234)) * 0.5).astype(dtype)
    want = mx.concatenate([
        mx.quantized_matmul(probe[r:r + 1], weight, scales, biases, transpose=True,
                            group_size=group_size, bits=bits, mode="affine")
        for r in range(rows)
    ], axis=0)
    chosen = None
    for variant in variants:
        got = _launch(probe, weight, scales, biases, bits, group_size, variant)
        if bool(mx.array_equal(got, want).item()):
            chosen = variant
            break
    _ADMITTED[key] = chosen
    STATS["admitted" if chosen else "rejected"] += 1
    if chosen:
        logger.info("row-exact verify QMV admitted variant=%s rows=%d K=%d N=%d bits=%d gs=%d dtype=%s",
                    chosen, rows, in_dim, weight.shape[0], bits, group_size, dtype)
    else:
        logger.warning("row-exact verify QMV REJECTED (no variant bit-equal to MLX M=1) rows=%d K=%d "
                       "N=%d bits=%d gs=%d dtype=%s; stock MLX keeps this shape", rows, in_dim,
                       weight.shape[0], bits, group_size, dtype)
    return chosen


def row_exact_qmv(x, weight, scales, biases, *, group_size: int, bits: int, mode: str = "affine"):
    """``x @ dequant(weight).T`` with each row bit-equal to its M=1 call, or None.

    None means "use stock MLX" (unsupported shape, kernel unavailable, or the
    shape failed its bit-equality admission check).
    """
    if mode != "affine" or biases is None or x.ndim < 2 or _mlx_qmv_header() is None:
        return None
    in_dim = x.shape[-1]
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    out_dim = weight.shape[0]
    variants = _variants(rows, in_dim, out_dim, bits, group_size, x.dtype)
    if not variants:
        return None
    if weight.dtype != mx.uint32 or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return None
    key = (rows, in_dim, out_dim, bits, group_size, str(x.dtype))
    variant = _admit(key, weight, scales, biases, bits, group_size, rows, in_dim, x.dtype, variants)
    if variant is None:
        return None
    STATS["calls"] += 1
    y = _launch(mx.contiguous(x.reshape(rows, in_dim)), weight, scales, biases, bits, group_size, variant)
    return y.reshape(*x.shape[:-1], out_dim)


# --------------------------------------------------------------------------- #
# Verify scope: only the native-MTP target verify forward uses the kernel.
# --------------------------------------------------------------------------- #
import contextlib
import contextvars

import mlx.nn as nn

_SCOPE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "vmlx_row_exact_verify_qmv", default=False
)
_INSTALLED = {"done": False}


def row_exact_scope_active() -> bool:
    return _SCOPE.get()


_DENSE_ADMITTED: dict[tuple, bool] = {}


def row_exact_linear(x, weight, bias=None):
    """Dense ``x @ weight.T (+ bias)`` with each row bit-equal to its M=1 call, or None.

    MLX's single-row ``x @ W.T`` is a GEMV; at M >= 2 it switches to a GEMM
    whose reduction order differs (measured on fp16/bf16 router, hyper-connection
    and square shapes). A batched matrix-vector product over a broadcast view of
    the weight keeps every row on the GEMV path without copying the weight.
    """
    if x.ndim < 2 or weight.ndim != 2 or x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        return None
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    if not 2 <= rows <= MAX_ROWS or weight.dtype != x.dtype:
        return None
    out_dim, in_dim = weight.shape

    def run(x2d):
        y = mx.matmul(mx.broadcast_to(weight, (rows, out_dim, in_dim)), x2d.reshape(rows, in_dim, 1))
        return y.reshape(rows, out_dim)

    key = (rows, in_dim, out_dim, str(x.dtype))
    admitted = _DENSE_ADMITTED.get(key)
    if admitted is None:
        probe = mx.random.normal((rows, in_dim), key=mx.random.key(4321)).astype(x.dtype)
        want = mx.concatenate([probe[r:r + 1] @ weight.T for r in range(rows)], axis=0)
        admitted = bool(mx.array_equal(run(probe), want).item())
        _DENSE_ADMITTED[key] = admitted
        STATS["dense_admitted" if admitted else "dense_rejected"] = STATS.get(
            "dense_admitted" if admitted else "dense_rejected", 0) + 1
        if not admitted:
            logger.warning("row-exact verify dense linear REJECTED rows=%d K=%d N=%d dtype=%s",
                           rows, in_dim, out_dim, x.dtype)
    if not admitted:
        return None
    STATS["dense_calls"] = STATS.get("dense_calls", 0) + 1
    y = run(x.reshape(rows, in_dim)).reshape(*x.shape[:-1], out_dim)
    return y + bias if bias is not None else y


def _install() -> None:
    if _INSTALLED["done"]:
        return
    original_dense = nn.Linear.__call__

    def patched_dense(self, x):  # type: ignore[no-untyped-def]
        if _SCOPE.get():
            y = row_exact_linear(x, self["weight"], self["bias"] if "bias" in self else None)
            if y is not None:
                return y
        return original_dense(self, x)

    nn.Linear.__call__ = patched_dense
    original = nn.QuantizedLinear.__call__

    def patched(self, x):  # type: ignore[no-untyped-def]
        if _SCOPE.get() and "biases" in self:
            y = row_exact_qmv(
                x, self["weight"], self["scales"], self["biases"],
                group_size=int(self.group_size), bits=int(self.bits),
                mode=str(getattr(self, "mode", "affine") or "affine"),
            )
            if y is not None:
                return y + self["bias"] if "bias" in self else y
        return original(self, x)

    nn.QuantizedLinear.__call__ = patched
    _INSTALLED["done"] = True


@contextlib.contextmanager
def row_exact_verify_scope():
    """Route 2..MAX_ROWS-row affine projections through the row-exact kernel."""
    if not row_exact_qmv_requested():
        yield False
        return
    _install()
    token = _SCOPE.set(True)
    try:
        yield True
    finally:
        _SCOPE.reset(token)
