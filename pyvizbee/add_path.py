"""Add path(s) to sys.path."""
from typing import List, Union
import sys
from pathlib import Path
from icecream import ic


def add_path(
    path_: Union[str, List[str]] = "",
    remove: bool = False,
    resolve: bool = True,
):
    """Add path to sys.path.

    Args:
        path_: paths to add to sys.path
        remove: if True, remove paths insteading of add
        resolve: if True, resolve to absolute paths first before adding to sys.path.
    """
    if isinstance(path_, str):
        paths = [path_]
    else:
        paths = path_[:]
    for path in paths:
        if resolve:
            path = Path(path).expanduser().resolve().as_posix()

        if remove:
            if path in sys.path:
                sys.path.remove(path)
                ic(f"[{path}] removed")
            else:
                ic(f"[{path}] not in sys.path.")
        else:
            if path not in sys.path:
                sys.path.insert(0, path)
                ic(f"[{path}] added to sys.path.")
            else:
                ic(f"[{path}] already in sys.path")
