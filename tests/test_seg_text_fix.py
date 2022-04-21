"""Test seg_text fix."""
from pyvizbee.seg_text import seg_text


def test_seg_text_fix():
    """Test seg_text fix."""
    text = "I commenced again. `Do you intend parting with the little ones, madam?'"
    assert seg_text(text).__len__() > 1  # 2
