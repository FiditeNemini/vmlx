"""Exercise the production lifespan without importing or loading MLX models."""
import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace


def lifespan_for(engine):
    source = Path(__file__).parents[1] / 'vmlx_engine' / 'server.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'lifespan')
    noop = lambda *args, **kwargs: None
    ns = dict(asynccontextmanager=asynccontextmanager, FastAPI=object,
              platform=SimpleNamespace(system=lambda: 'Test'), asyncio=asyncio,
              _engine=engine, _mcp_manager=None, _max_prompt_tokens=None,
              _flash_moe_enabled=False, _enable_jit=False, _image_gen=None,
              _lifecycle_progress=SimpleNamespace(report=noop, PHASE_RESTORING_ACCELERATION='restore'),
              _record_metal_ws_model_baseline=noop, _refresh_loaded_max_prompt_tokens=noop,
              _discover_mcp_config_for_startup=lambda: None,
              _shutdown_image_gen_executor=noop, logger=SimpleNamespace(info=noop, warning=noop, error=noop))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    return ns['lifespan']


def test_graceful_server_shutdown_keeps_engine_drain_alive_past_old_deadline():
    async def run():
        completed = asyncio.Event()
        cancelled = []
        class Engine:
            _loaded = True
            async def stop(self):
                try:
                    # Longer than the previous real ten-second lifespan timeout.
                    await asyncio.sleep(10.1)
                    completed.set()
                except asyncio.CancelledError:
                    cancelled.append(True)
                    raise
        async with lifespan_for(Engine())(None):
            pass
        assert completed.is_set(), 'lifespan returned before queued persistence completed'
        assert not cancelled, 'lifespan cancelled its persistence owner'
    asyncio.run(run())
