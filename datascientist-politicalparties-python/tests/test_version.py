import sys
import os
# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from src.political_party_analysis.loader import DataLoader
from src.political_party_analysis import __version__


def test_version():
    assert __version__ == "0.1.0"
