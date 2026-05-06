import logging
from unittest.mock import MagicMock

from wandb.sdk.lib.logger_capture import WandbLoggerHandler


def _make_logger(name: str) -> logging.Logger:
    """Create a uniquely-named logger for test isolation."""
    logger = logging.getLogger(f"{__name__}.{name}")
    logger.setLevel(logging.DEBUG)
    return logger


def _make_mock_run() -> MagicMock:
    """Create a mock run with a write_logs method."""
    run = MagicMock()
    run.write_logs = MagicMock()
    return run


def test_handler_calls_write_logs():
    """The handler calls run.write_logs with the formatted message."""
    run = _make_mock_run()
    handler = WandbLoggerHandler(run, level=logging.INFO)
    logger = _make_logger("test_basic")
    logger.addHandler(handler)
    try:
        logger.info("hello world")
    finally:
        logger.removeHandler(handler)

    run.write_logs.assert_called_once()
    text = run.write_logs.call_args[0][0]
    assert "hello world" in text


def test_handler_does_not_propagate_errors():
    """A broken run.write_logs must not crash the user's logger call."""
    run = _make_mock_run()
    run.write_logs.side_effect = RuntimeError("write_logs broke")
    handler = WandbLoggerHandler(run, level=logging.INFO)
    logger = _make_logger("test_error_handling")
    logger.addHandler(handler)
    try:
        # This should NOT raise
        logger.error("this should not crash")
    finally:
        logger.removeHandler(handler)

    run.write_logs.assert_called_once()


def test_handler_default_level_is_notset():
    """Default level is NOTSET, capturing everything."""
    run = _make_mock_run()
    handler = WandbLoggerHandler(run)
    assert handler.level == logging.NOTSET
