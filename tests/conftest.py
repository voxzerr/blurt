"""Pytest bootstrap for the blurt test suite.

Puts the project root on ``sys.path`` so ``import blurt`` resolves whether the
suite is run as ``python3 -m pytest`` from the project root, as
``python3 -m pytest tests/`` from anywhere, or from an IDE with a different
working directory. blurt is not installed as a package on the floor machine
(no Homebrew, no virtualenv in the repo), so the import has to work straight
from the checkout.

What can go wrong on macOS: nothing here touches the OS beyond reading the
path of this file. The whole suite is deliberately hardware-free -- no
microphone, no model download, no network, no Accessibility or Microphone
permission prompts -- so that it runs unattended on the Intel floor machine
under the Apple system Python (3.9.6).

Python 3.9 floor: lazy annotations, no PEP 585/604 syntax.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
