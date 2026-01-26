"""
Entry point for running benchmark as a module.

Usage:
    python -m benchmark run --debug
    python -m benchmark run --full -e all
"""

from .cli import main

if __name__ == "__main__":
    main()
