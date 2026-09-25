# src/menu.py
"""Launcher entry — opens the decorative GUI (not a terminal menu).

    python -m src.menu
    python -m src.menu --cam 2
"""
from .gui_app import main

if __name__ == "__main__":
    main()
