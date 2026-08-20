"""vmlx#256: the image-prefill guard rejected without actionable diagnostics.

The guard itself is correctly placed — a Metal OOM inside the media forward
raises in a completion handler and kills the engine uncatchably, so it must
predict rather than try-and-recover. The defect was that every number needed
to act (prompt composition, the cost model, and the budget that WOULD fit)
was available at the raise site and none of it reached the user.
"""

from vmlx_engine.mllm_batch_generator import (
    _max_tokens_under_attention_bytes,
    _vlm_image_prefill_budget,
)

GB = 1024**3


def _reject(**over):
    kw = dict(
        has_images=True,
        seq_len=19729,
        num_attention_heads=32,
        active_memory_bytes=int(60 * GB),
        max_working_set_bytes=int(107.5 * GB),
        reject_pct=98.0,
        single_buffer_limit_bytes=int(17.2 * GB),
        guard_enabled=True,
    )
    kw.update(over)
    return _vlm_image_prefill_budget(**kw)


class TestInverseBudget:
    def test_quadratic_inverse(self):
        # heads * t^2 * 2 <= budget  ->  t = isqrt(budget / (2*heads))
        assert _max_tokens_under_attention_bytes(2 * 32 * 1000 * 1000, 32) == 1000

    def test_degenerate_inputs_are_zero_not_crash(self):
        assert _max_tokens_under_attention_bytes(0, 32) == 0
        assert _max_tokens_under_attention_bytes(-5, 32) == 0
        assert _max_tokens_under_attention_bytes(GB, 0) == 0


class TestMessageIsActionable:
    def test_names_the_cost_model(self):
        d = _reject()
        assert d.should_reject
        assert "32 heads" in d.detail and "19,729 tokens" in d.detail

    def test_states_a_fitting_budget(self):
        d = _reject()
        assert "fits about" in d.detail
        fits = _max_tokens_under_attention_bytes(int(17.2 * GB), 32)
        assert f"{fits:,}" in d.detail
        assert fits < 19729

    def test_splits_image_and_text_when_known(self):
        d = _reject(image_token_count=14308)
        assert "14,308 image" in d.detail
        assert "5,421 text" in d.detail
        # remaining text budget = fits - image tokens
        fits = _max_tokens_under_attention_bytes(int(17.2 * GB), 32)
        assert f"{fits - 14308:,} tokens" in d.detail

    def test_says_so_when_images_alone_blow_the_budget(self):
        d = _reject(image_token_count=19000)
        assert "images alone exceed the budget" in d.detail

    def test_working_set_reason_names_resident_and_limit(self):
        d = _reject(single_buffer_limit_bytes=0, active_memory_bytes=int(100 * GB))
        assert d.should_reject
        assert "already resident" in d.detail and "107.5GB" in d.detail

    def test_still_explains_why_chunking_is_impossible(self):
        assert "cannot be chunked safely" in _reject().detail

    def test_passing_prompt_is_not_rejected(self):
        d = _reject(seq_len=2000)
        assert not d.should_reject
