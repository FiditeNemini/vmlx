from types import SimpleNamespace

import pytest

from vmlx_engine.mllm_batch_generator import (
    _call_processor_direct,
    _mllm_media_cache_extra_keys,
)


class Gemma4ProcessorProbe:
    model_type = "gemma4_unified"

    def __init__(self):
        self.image_processor = SimpleNamespace(max_soft_tokens=280)
        self.seen_budgets = []

    def process(self, images=None, text=None, **kwargs):
        self.seen_budgets.append(self.image_processor.max_soft_tokens)
        return {"input_ids": [[1, 2, 3]]}


def test_request_local_gemma4_image_budget_is_applied_then_restored():
    processor = Gemma4ProcessorProbe()

    result = _call_processor_direct(
        processor,
        prompts="<|image|>read this",
        images=["image.png"],
        add_special_tokens=False,
        image_token_budget=1120,
    )

    assert result["input_ids"] == [[1, 2, 3]]
    assert processor.seen_budgets == [1120]
    assert processor.image_processor.max_soft_tokens == 280


def test_request_local_gemma4_image_budget_rejects_unsupported_value():
    with pytest.raises(ValueError, match="image_token_budget must be one of"):
        _call_processor_direct(
            Gemma4ProcessorProbe(),
            prompts="<|image|>read this",
            images=["image.png"],
            add_special_tokens=False,
            image_token_budget=1000,
        )


def test_media_prefix_key_separates_visual_token_budgets(tmp_path):
    image = tmp_path / "same.png"
    image.write_bytes(b"same-image")

    def request(budget):
        return SimpleNamespace(images=[str(image)], image_token_budget=budget)

    assert _mllm_media_cache_extra_keys(request(280)) != _mllm_media_cache_extra_keys(
        request(1120)
    )
