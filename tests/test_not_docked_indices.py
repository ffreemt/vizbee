"""Test docked_or_not."""
from tests.ns_df import df_

from pyvizbee.not_docked_indices import not_docked_indices


def test_docked_or_not():
    """Test docked_or_not."""
    _ = not_docked_indices(df_)
    assert len(_) == 8
