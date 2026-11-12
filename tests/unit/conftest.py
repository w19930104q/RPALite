"""Pytest configuration for the offline unit tests.

The image-matching regressions are fully synthetic (PIL/numpy only) and must run
without a display, OCR backend or any network access. Importing the RPALite
class on a headless Linux box pulls in pyautogui (which needs tkinter via
mouseinfo) together with the X11 mouse/keyboard bindings. The lightweight stub
modules below satisfy those imports purely in-process so the RPALite-level
coordinate conversion tests stay hermetic and offline.
"""
import sys
import types
from pathlib import Path

# Make ``src`` importable regardless of the directory pytest is launched from.
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_headless_stubs():
    if "pyautogui" not in sys.modules:
        pyautogui = types.ModuleType("pyautogui")

        class _Size:
            width = 800
            height = 600

        pyautogui.size = lambda: _Size()
        pyautogui.screenshot = lambda *args, **kwargs: None
        pyautogui.position = lambda: (0, 0)
        pyautogui.moveTo = lambda *args, **kwargs: None
        pyautogui.click = lambda *args, **kwargs: None
        pyautogui.doubleClick = lambda *args, **kwargs: None
        pyautogui.hotkey = lambda *args, **kwargs: None
        pyautogui.press = lambda *args, **kwargs: None
        pyautogui.write = lambda *args, **kwargs: None
        pyautogui.mouseDown = lambda *args, **kwargs: None
        pyautogui.mouseUp = lambda *args, **kwargs: None
        pyautogui.scroll = lambda *args, **kwargs: None
        sys.modules["pyautogui"] = pyautogui

    if "mouse" not in sys.modules:
        mouse = types.ModuleType("mouse")
        mouse.move = lambda *args, **kwargs: None
        mouse.click = lambda *args, **kwargs: None
        mouse.double_click = lambda *args, **kwargs: None
        mouse.wheel = lambda *args, **kwargs: None
        sys.modules["mouse"] = mouse

    if "keyboard" not in sys.modules:
        keyboard = types.ModuleType("keyboard")
        keyboard.write = lambda *args, **kwargs: None
        sys.modules["keyboard"] = keyboard


_install_headless_stubs()
