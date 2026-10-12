"""Guard: --disable-prefix-cache must NOT reach the DFlash2 lanes' native stores.

The app launches the GLM-5.3 and Qwen DFlash2 lanes WITH --disable-prefix-cache (it switches off the batch scheduler
cache those lanes do not use) while their native prefix/session stores serve reuse. Mapping the flag onto those
stores (d2de6f4a, reverted 7b42af15) made every GLM follow-up re-prefill the whole prompt (56k: 0.66 s -> 136-146 s).
"""
import inspect

import vmlx_engine.cli as cli


def test_cli_never_turns_off_the_native_dflash2_stores():
    src = inspect.getsource(cli)
    for name in ("VMLX_GLM_DFLASH2_PREFIX_CACHE", "VMLX_DFLASH2_PREFIX_REUSE"):
        assert f'os.environ["{name}"]' not in src and f"os.environ['{name}']" not in src
        assert f'setdefault("{name}"' not in src
