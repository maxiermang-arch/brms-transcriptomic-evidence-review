"""Compatibility entry point for revised evidence figures."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('bmc_figures.py')),run_name='__main__')
