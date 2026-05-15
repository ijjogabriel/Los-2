"""Convenience launcher: python run_app.py"""
import subprocess, sys, pathlib
app = pathlib.Path(__file__).parent / "will_checker" / "app.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true"])
