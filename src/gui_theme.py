# src/gui_theme.py
"""Black / white / gold theme + Poppins font loading (Windows)."""
from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk
import tkinter.font as tkfont

BLACK = "#0a0a0a"
BLACK_2 = "#141414"
WHITE = "#ffffff"
WHITE_DIM = "#c8c8c8"
GOLD = "#d4af37"
GOLD_HOT = "#f0d78c"
GOLD_DIM = "#8a7020"
DANGER = "#e74c3c"
OK = "#2ecc71"

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"


def _register_poppins() -> str:
    """Add Poppins TTF files to the process font table; return family name."""
    family = "Poppins"
    if not sys.platform.startswith("win"):
        return family if (FONT_DIR / "Poppins-Regular.ttf").exists() else "Segoe UI"
    try:
        from ctypes import windll

        FR_PRIVATE = 0x10
        for name in (
            "Poppins-Regular.ttf",
            "Poppins-Medium.ttf",
            "Poppins-SemiBold.ttf",
            "Poppins-Bold.ttf",
        ):
            path = FONT_DIR / name
            if path.exists():
                windll.gdi32.AddFontResourceExW(str(path), FR_PRIVATE, 0)
        return family
    except Exception:
        return "Segoe UI"


def load_poppins(root: tk.Misc) -> dict[str, tkfont.Font]:
    fam = _register_poppins()
    # Force Tk to notice newly registered fonts
    root.update_idletasks()
    return {
        "hero": tkfont.Font(family=fam, size=36, weight="bold"),
        "title": tkfont.Font(family=fam, size=18, weight="bold"),
        "subtitle": tkfont.Font(family=fam, size=12),
        "body": tkfont.Font(family=fam, size=11),
        "button": tkfont.Font(family=fam, size=11, weight="bold"),
        "small": tkfont.Font(family=fam, size=9),
        "status": tkfont.Font(family=fam, size=10),
    }
