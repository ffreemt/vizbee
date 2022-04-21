"""Test seg_text."""
import pytest
from seg_text import seg_text


def test_seg_text1():
    """Test seg_text 1."""
    text = " text 1\n\n test 2. test 3"
    _ = seg_text(text)
    assert len(_) == 2

    text = " text 1\n\n test 2. Test 3"
    _ = seg_text(text)
    assert len(_) == 3


@pytest.mark.parametrize(
    "test_input,expected", [
        ("", []),
        (" ", []),
        (" \n ", []),
    ]
)
def test_seg_text_blanks(test_input, expected):
    """Test blanks."""
    assert seg_text(test_input) == expected


def test_seg_text_semicolon():
    """Test semicolon."""
    text = """ “元宇宙”，英文為“Metaverse”。該詞出自1992年；的科幻小說《雪崩》。 """
    assert len(seg_text(text)) == 2
    assert len(seg_text(text, 'zh')) == 2
    assert len(seg_text(text, 'ja')) == 2
    assert len(seg_text(text, 'ko')) == 2
    assert len(seg_text(text, 'en')) == 1


def test_seg_text_semicolon_extra():
    """Test semicolon."""
    extra = "[;；]"
    text = """ “元宇宙”，英文為“Metaverse”。該詞出自1992年；的科幻小說《雪崩》。 """
    assert len(seg_text(text, extra=extra)) == 2 + 1
    assert len(seg_text(text, 'zh', extra=extra)) == 2 + 1
    assert len(seg_text(text, 'ja', extra=extra)) == 2 + 1
    assert len(seg_text(text, 'ko', extra=extra)) == 2 + 1
    assert len(seg_text(text, 'en', extra=extra)) == 1 + 1
