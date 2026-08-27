import asyncio

from vmlx_engine import server


def test_direct_admin_wakes_share_one_model_reload(monkeypatch):
    async def scenario():
        original_state = server._standby_state
        original_lock = server._wake_lock
        original_progress = server._wake_in_progress
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def fake_wake_impl():
            nonlocal calls
            calls += 1
            assert server._wake_in_progress is True
            started.set()
            await release.wait()
            server._standby_state = None
            return {"status": "active"}

        monkeypatch.setattr(server, "_admin_wake_impl", fake_wake_impl)
        server._standby_state = "deep"
        server._wake_lock = asyncio.Lock()
        server._wake_in_progress = False
        try:
            first = asyncio.create_task(server.admin_wake())
            await started.wait()
            second = asyncio.create_task(server.admin_wake())
            await asyncio.sleep(0)

            assert calls == 1
            assert second.done() is False

            release.set()
            first_result, second_result = await asyncio.gather(first, second)
            assert first_result == {"status": "active"}
            assert second_result == {"status": "already_active"}
            assert calls == 1
            assert server._wake_in_progress is False
        finally:
            server._standby_state = original_state
            server._wake_lock = original_lock
            server._wake_in_progress = original_progress

    asyncio.run(scenario())


def test_jit_middleware_uses_lock_held_wake_path():
    source = open(server.__file__, encoding="utf-8").read()
    middleware_start = source.index("async def track_request_time")
    middleware_end = source.index("# Gate: image servers", middleware_start)
    middleware = source[middleware_start:middleware_end]

    assert "async with _wake_lock" in middleware
    assert "wake_result = await _wake_with_lock_held()" in middleware
    assert "wake_result = await admin_wake()" not in middleware


def test_public_admin_wake_owns_shared_lock():
    source = open(server.__file__, encoding="utf-8").read()
    endpoint_start = source.index("async def admin_wake()")
    endpoint_end = source.index('@app.get("/v1/cache/stats"', endpoint_start)
    endpoint = source[endpoint_start:endpoint_end]

    assert "async with _wake_lock" in endpoint
    assert "return await _wake_with_lock_held()" in endpoint
