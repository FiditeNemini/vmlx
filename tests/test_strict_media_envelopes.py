"""Strict media requests cannot silently lose foreign processor controls."""
import pytest
from pydantic import ValidationError

from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest


@pytest.fixture(params=[ChatCompletionRequest, ResponsesRequest])
def schema_and_input(request):
    cls = request.param
    body = {"model": "test"}
    body.update({"messages": [{"role": "user", "content": "test"}]}
                if cls is ChatCompletionRequest else {"input": "test"})
    return cls, body


@pytest.mark.parametrize("field,value", [
    ("media_io_kwargs", {"video": {"fps": 2}}),
    ("mm_processor_kwargs", {"max_image_tokens": 1024}),
])
@pytest.mark.parametrize("strict", [True, "true"])
def test_strict_rejects_foreign_envelope(schema_and_input, field, value, strict):
    cls, body = schema_and_input
    with pytest.raises(ValidationError, match=field):
        cls.model_validate({**body, "media_controls_strict": strict, field: value})


def test_native_controls_and_permissive_compatibility(schema_and_input):
    cls, body = schema_and_input
    native = cls.model_validate({**body, "media_controls_strict": True,
                                 "video_fps": 2, "video_max_frames": 8})
    assert native.video_fps == 2 and native.video_max_frames == 8
    permissive = cls.model_validate({**body, "media_io_kwargs": {"video": {"fps": 2}}})
    assert permissive.video_fps is None
