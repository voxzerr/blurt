"""Pytest bootstrap, and the fixtures more than one test module needs.

Puts the project root on ``sys.path`` so ``import blurt`` resolves whether the
suite is run as ``python3 -m pytest`` from the project root, as
``python3 -m pytest tests/`` from anywhere, or from an IDE with a different
working directory. blurt is not installed as a package on the floor machine
(no Homebrew, no virtualenv in the repo), so the import has to work straight
from the checkout.

This directory goes on ``sys.path`` too, so that ``from conftest import
config_on_disk`` is a supported import rather than a lucky one. pytest happens
to insert the directory of every module it collects under its default import
mode, but relying on that would make the suite depend on an import mode nobody
chose deliberately and which ``--import-mode=importlib`` changes. Two lines
here cost nothing and make the shared helpers importable however the suite is
invoked.

WHY ANYTHING SHARED LIVES HERE. ``home`` was written out verbatim in both
tests/test_learn_cli.py and tests/test_config_cli.py, and ``config_on_disk``
in both again with cosmetic differences. Two copies of an isolation fixture is
one copy too many for something whose entire job is keeping a test off the
developer's real config file: the day someone adds ``XDG_STATE_HOME`` or a
fourth variable to one copy, the other module keeps running against whatever
the developer's shell happens to say, and it keeps passing while it does it.
Fixtures used by exactly one module stay in that module -- ``seeded`` (a
journal with findings in it) and ``_snapshot`` (the exact bytes of the config
file) are each meaningful in one place only, and moving them here would just
put them further from the tests that explain them.

What can go wrong on macOS: nothing here touches the OS beyond reading the
path of this file and, inside the ``home`` fixture, redirecting three
environment variables at ``tmp_path`` through monkeypatch (which undoes them
at teardown). The whole suite is deliberately hardware-free -- no microphone,
no model download, no network, no Accessibility or Microphone permission
prompts -- so that it runs unattended on the Intel floor machine under the
Apple system Python (3.9.6).

Python 3.9 floor: lazy annotations, typing generics, no PEP 585/604 syntax.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)

for _entry in (_PROJECT_ROOT, _TESTS_DIR):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated XDG home, so no test can touch the developer's real config.

    ``HOME`` is redirected as well as the two XDG variables, and that third one
    is not redundant: ``default_config_path`` falls back to ``~/.config`` when
    ``XDG_CONFIG_HOME`` is unset or relative, so a test that unsets the variable
    to exercise that fallback would otherwise land on the developer's own
    ``~/.config/blurt/config.json`` -- which by now holds a replacement
    dictionary built from months of their speech and exists nowhere else.

    Returns ``tmp_path`` so a test can reach the tree directly (to plant a
    corrupt file, or to check that nothing was created).
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def config_on_disk() -> Dict[str, Any]:
    """The config file parsed as JSON. Fails the test if it is not valid JSON.

    Reads the file rather than calling ``load_config``, always, and that is the
    point of it rather than an implementation detail. ``load_config`` drops keys
    it does not recognise and substitutes defaults for values it rejects, so a
    test asserting through it cannot tell "the setting is on disk" from "the
    setting is absent and the default happens to match". Both test modules that
    use this are checking what a *write* actually put in the file, and only the
    file can answer that.

    Imported by the test modules rather than requested as a fixture because it
    is a plain question about the filesystem with nothing to set up or tear
    down, and because tests call it twice (before and after a command) inside a
    single test body.
    """
    from blurt.config import default_config_path

    return json.loads(default_config_path().read_text(encoding="utf-8"))
