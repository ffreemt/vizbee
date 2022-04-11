"""Set loglevel."""
# pylint: disable=duplicate-code
import environs


def loglevel(
    level: int = 20,
    force: bool = False,
) -> int:
    """Return an integer based on env LOGLEVEL.

    Args:
        level: set loglevel if env LOGLEVEL is not set.
        force: bool, use level if set.
        if force is not set, env LOGLEVEL takes precedence.
        set env LOGLEVEL to 10/debug/DEBUG to turn on debug
    Returns:
        an integer for using in logzero.loglevel()
        if env LOGLEVEL is not set, use level.

    >>> loglevel(force=True)
    20
    >>> import os; os.environ['logleve'] = 'debug'
    >>> loglevel()
    10
    """
    try:
        level = int(level)
    except Exception:
        level = 20

    if force:
        return level

    try:
        _ = environs.Env().log_level("LOGLEVEL")
    except (environs.EnvError, environs.EnvValidationError):
        _ = None
    except Exception:
        _ = None

    _ = _ or level
    return _
