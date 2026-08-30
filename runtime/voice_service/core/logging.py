import logging
import re

_NARRATION_STATUS_PATH = re.compile(r"^/v1/narrations/[0-9a-f]{32}(?:\?.*)?$")


class NarrationPollingAccessFilter(logging.Filter):
    """Keep successful status polling from drowning lifecycle/quality events."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        method = args[1]
        path = args[2]
        status_code = args[4]
        return not (
            method == "GET"
            and isinstance(path, str)
            and _NARRATION_STATUS_PATH.fullmatch(path)
            and isinstance(status_code, int)
            and status_code < 400
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, NarrationPollingAccessFilter) for item in access_logger.filters):
        access_logger.addFilter(NarrationPollingAccessFilter())
