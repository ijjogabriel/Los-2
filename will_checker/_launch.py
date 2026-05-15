"""Entry point for the `will-checker-ui` command."""
import subprocess
import sys
from pathlib import Path


def run_ui():
    app = Path(__file__).parent / "app.py"
    sys.exit(subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.headless", "true",
    ]))
