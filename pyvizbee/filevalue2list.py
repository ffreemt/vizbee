"""Convert file1.value to list."""
from typing import List
from logzero import logger
import cchardet as chardet


def filevalue2list(value: bytes) -> List[str]:
    """Convert file1.value to list."""
    if value is None:
        return []

    if not isinstance(value, bytes):
        raise Exception("not bytes fed to me, cant handle it.")

    encoding = chardet.detect(value).get("encoding") or "utf8"
    try:
        _ = value.decode(encoding=encoding)
    except Exception as _:
        logger.error(_)
        raise
    return [elm.strip() for elm in _.splitlines() if elm.strip()]
