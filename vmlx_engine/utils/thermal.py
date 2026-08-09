"""macOS thermal-pressure readout without a pyobjc dependency.

Sustained decode/prefill slowdowns on Apple Silicon are frequently OS thermal
throttling (idle-heals, no code defect), so perf log lines carry the
NSProcessInfo thermalState to separate throttle from real regressions.
"""

import ctypes
import ctypes.util
import sys

_STATE_NAMES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
_get_state = None


def _unavailable():
    return None


def _init():
    global _get_state
    if sys.platform != "darwin":
        _get_state = _unavailable
        return
    try:
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        # Foundation must be loaded so the NSProcessInfo class is registered.
        ctypes.CDLL(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send_ptr = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
        )
        send_long = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p),
        )
        cls = objc.objc_getClass(b"NSProcessInfo")
        sel_proc = objc.sel_registerName(b"processInfo")
        sel_thermal = objc.sel_registerName(b"thermalState")
        if not (cls and sel_proc and sel_thermal):
            _get_state = _unavailable
            return

        def get_state():
            try:
                # Re-fetch the singleton each call: holding the raw pointer
                # across calls would bypass ARC lifetime guarantees.
                proc = send_ptr(cls, sel_proc)
                if not proc:
                    return None
                return int(send_long(proc, sel_thermal))
            except Exception:
                return None

        _get_state = get_state
    except Exception:
        _get_state = _unavailable


def thermal_state_name() -> str:
    """Return 'nominal'/'fair'/'serious'/'critical', or 'unknown'."""
    global _get_state
    if _get_state is None:
        _init()
    return _STATE_NAMES.get(_get_state(), "unknown")
