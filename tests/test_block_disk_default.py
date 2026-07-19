from types import SimpleNamespace

from vmlx_engine.cli import _apply_paged_block_disk_default


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)


def _args(**overrides):
    values = {
        "enable_block_disk_cache": None,
        "continuous_batching": True,
        "use_paged_cache": True,
        "enable_prefix_cache": True,
        "disable_prefix_cache": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_paged_cache_defaults_block_disk_l2_on():
    args = _args()
    logger = _Logger()

    assert _apply_paged_block_disk_default(args, logger) is True
    assert args.enable_block_disk_cache is True
    assert any("enabling block disk cache" in message for message in logger.messages)


def test_explicit_block_disk_opt_out_is_preserved():
    args = _args(enable_block_disk_cache=False)

    assert _apply_paged_block_disk_default(args, _Logger()) is False
    assert args.enable_block_disk_cache is False


def test_block_disk_default_stays_on_without_paged_cache():
    args = _args(use_paged_cache=False)

    assert _apply_paged_block_disk_default(args, _Logger()) is True
    assert args.enable_block_disk_cache is True


def test_explicit_block_disk_opt_in_is_preserved_without_paged_cache():
    args = _args(use_paged_cache=False, enable_block_disk_cache=True)

    assert _apply_paged_block_disk_default(args, _Logger()) is True
    assert args.enable_block_disk_cache is True


def test_block_disk_default_stays_off_without_prefix_cache():
    args = _args(enable_prefix_cache=False, disable_prefix_cache=True)

    assert _apply_paged_block_disk_default(args, _Logger()) is False
    assert args.enable_block_disk_cache is False


def test_block_disk_default_stays_off_without_continuous_batching():
    args = _args(continuous_batching=False)

    assert _apply_paged_block_disk_default(args, _Logger()) is False
    assert args.enable_block_disk_cache is False
