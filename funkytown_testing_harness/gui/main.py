"""Entry point: python -m funkytown_testing_harness.gui.main

Applies a custom comfy-prompt-tools folder from settings (if set) to sys.path
*before* anything imports run_test.py/live_workflow.py, so it takes
precedence over their default sibling-directory guess.
"""

import sys


def main():
    from funkytown_testing_harness.gui.app_settings import load_settings

    settings = load_settings()
    custom_dir = settings.get("comfy_prompt_tools_dir")
    if custom_dir and custom_dir not in sys.path:
        sys.path.insert(0, custom_dir)

    from PySide6.QtWidgets import QApplication

    from funkytown_testing_harness.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
