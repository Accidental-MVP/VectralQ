import logging
import sys
import uuid
from typing import Optional
import contextvars


# Context variables used for structured logs
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
tenant_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("tenant_id", default=None)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        record.tenant_id = tenant_id_var.get() or "-"
        return True


def configure_logging() -> None:
    logger = logging.getLogger()
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s tenant_id=%(tenant_id)s - %(message)s",
        )
        handler.setFormatter(formatter)
        handler.addFilter(ContextFilter())
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)


