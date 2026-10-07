import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a GPU, HF_TOKEN and gated model access")
