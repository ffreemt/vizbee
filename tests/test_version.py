"""Test versioin."""
from vizbee import __version__

def test_version():
    """Test version."""
    assert __version__[:3] == "0.1"
