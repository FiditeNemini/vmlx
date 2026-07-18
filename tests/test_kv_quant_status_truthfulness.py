"""HEALTH-KV-QUANT-FLAG-FALSE-WHILE-TQ-ACTIVE regression.

The /health and /v1/cache/stats kv_cache_quantization summary must report an
active TurboQuant STORAGE codec as enabled instead of contradicting the
turboquant_kv_cache section on the same response (live-caught on
Bonsai-27b-1bit-JANG: storage q8 active with tq_native_writes/hits while the
summary said enabled=false).
"""

from types import SimpleNamespace

from vmlx_engine.server import _kv_cache_quantization_status


def _sched(bits: int = 0) -> SimpleNamespace:
    return SimpleNamespace(_kv_cache_bits=bits, _kv_cache_group_size=64)


def test_disabled_without_tq_status() -> None:
    assert _kv_cache_quantization_status(_sched(0)) == {"enabled": False}


def test_none_scheduler_returns_none() -> None:
    assert _kv_cache_quantization_status(None) is None


def test_active_storage_codec_reports_enabled_truthfully() -> None:
    tq = {
        "storage_encode_enabled": True,
        "storage_key_bits": 8,
        "storage_value_bits": 8,
        "stored_prefix_quantization": "turboquant-q8",
        "auto_policy": "bonsai_hybrid_attention_kv_storage_tq8",
        "live_encode_enabled": False,
    }
    status = _kv_cache_quantization_status(_sched(0), tq)
    assert status is not None
    assert status["enabled"] is True
    assert status["mode"] == "turboquant-storage"
    # bits mirrors key_bits so the panel's "N-bit" InfoCard renders truthfully.
    assert status["bits"] == 8
    assert status["key_bits"] == 8
    assert status["value_bits"] == 8
    assert status["stored_prefix_quantization"] == "turboquant-q8"
    assert status["auto_policy"] == "bonsai_hybrid_attention_kv_storage_tq8"
    assert status["live_encode_enabled"] is False


def test_tq_objects_without_storage_encode_stay_disabled() -> None:
    # TQ objects can be active with the storage codec off; the summary must
    # not overclaim in that state.
    tq = {"enabled": True, "objects_active": True, "storage_encode_enabled": False}
    assert _kv_cache_quantization_status(_sched(0), tq) == {"enabled": False}


def test_legacy_explicit_live_bits_keep_reporting() -> None:
    status = _kv_cache_quantization_status(_sched(4))
    assert status == {
        "enabled": True,
        "mode": "live",
        "bits": 4,
        "group_size": 64,
    }
