#!/usr/bin/env python3
"""
Run the Black Widow Optimization Algorithm GUI.

This script launches the graphical user interface for the Black Widow Optimization Algorithm,
allowing users to visualize the optimization process and interact with the algorithm.
"""

from black_widow_gui import BlackWidowGUI

if __name__ == "__main__":
    app = BlackWidowGUI()
    app.mainloop()
