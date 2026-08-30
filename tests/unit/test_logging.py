from __future__ import annotations

import logging

from voice_service.core.logging import NarrationPollingAccessFilter


def _record(method: str, path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1", method, path, "1.1", status),
        None,
    )


def test_successful_narration_status_poll_is_suppressed() -> None:
    access_filter = NarrationPollingAccessFilter()
    assert access_filter.filter(
        _record("GET", "/v1/narrations/33f40306ad8b4c318a07fa23b46348b9", 200)
    ) is False


def test_failures_and_non_polling_requests_remain_visible() -> None:
    access_filter = NarrationPollingAccessFilter()
    assert access_filter.filter(
        _record("GET", "/v1/narrations/33f40306ad8b4c318a07fa23b46348b9", 500)
    ) is True
    assert access_filter.filter(_record("GET", "/health/ready", 200)) is True
    assert access_filter.filter(_record("GET", "/v1/narrations/job/audio", 200)) is True
