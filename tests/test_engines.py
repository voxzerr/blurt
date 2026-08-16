"""Tests for blurt's ASR engine registry (blurt/engines/__init__.py).

The registry makes exactly one promise, and it is the reason this file exists:
**blurt never silently substitutes a backend the user did not ask for.** Someone
who wrote ``"engine": "apple-speech"`` chose on-device Apple recognition over a
model download; someone who wrote ``"engine": "faster-whisper"`` chose accuracy
over the permission dialog. Handing either of them the other backend reverses a
privacy and latency decision without a word. It would also be invisible -- the
dictation still works, so nothing ever prompts the user to look. That is the
asymmetric failure this suite is built around, so the substitution tests are
written from both directions (alternatives exist / no alternatives exist) and
assert on the CONTENT of the refusal, not merely that something was raised.

The second property defended here is that the diagnostic survives. When nothing
works, the exception message is the only artefact a user ever sees -- there is no
menu bar yet at selection time. A message that says "no engine available" and
drops the per-engine reasons has technically raised correctly and has still
failed the user, so every rejection-reason test asserts the reason text is
actually present in ``str(exc)``.

The third is that selection is cheap and total. ``select_engine`` must return an
UNLOADED engine (loading is where the one-time model download lives; doing it
during selection would freeze startup before the UI can say "preparing"), and
probing must swallow everything -- including the ``OSError`` that a wheel built
for a newer macOS raises out of the dynamic loader rather than as ``ImportError``,
which is precisely how pywhispercpp failed on the floor machine.

How the fakes work: the registry resolves backends lazily through ``importlib``
against the private ``_ENGINE_MODULES`` map, so :func:`install` swaps that map for
one pointing at fake module suffixes and swaps the module-global ``importlib``
for a stub that hands back a namespace containing a fake class. No
faster-whisper, no pyobjc, no dynamic loader, no macOS speech-permission dialog
is ever reached -- which is also the point of
:func:`test_auto_does_not_probe_apple_speech_when_whisper_works`, since probing
apple-speech is itself a side effect in a bundled build.

``ENGINE_NAMES`` is deliberately left unpatched: the preference order is part of
the behaviour under test, so the fakes are installed into the real slots.

THE LAST SECTION USES NO FAKE FOR apple-speech, on purpose. Everything above is
about the registry's logic and wants both backends replaced; the packaging
decision in ``pyproject.toml`` is about the real stub and cannot be tested
against a fake at all. faster-whisper moved out of the ``[whisper]`` extra and
into the base dependencies on the strength of one claim -- that
``AppleSpeechEngine.is_available()`` is False and always will be, so an install
without faster-whisper has no second candidate and dies at startup. If that
claim ever stopped holding, the fakes above would keep passing while the
justification written into ``pyproject.toml`` had quietly become false, so
:func:`install_with_real_apple_speech` leaves that slot pointing at the shipped
module.

Python 3.9 floor: lazy annotations, typing.Optional, no PEP 585/604 syntax.
"""

from __future__ import annotations

import builtins
import importlib as real_importlib
import types

import pytest

from blurt import engines
from blurt.config import Config
from blurt.engines import NoEngineAvailable, available_engines, select_engine
from blurt.engines.apple_speech_engine import REJECTION_REASON, AppleSpeechEngine
from blurt.types import ASREngine, Hardware

WHISPER = "faster-whisper"
APPLE = "apple-speech"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def _hw() -> Hardware:
    """A synthetic Hardware. Nothing in the registry reads it; it only has to be
    the real type so a fake that inspects it is inspecting the real shape."""
    return Hardware(
        arch="x86_64",
        is_apple_silicon=False,
        under_rosetta=False,
        cpu_brand="Intel(R) Core(TM) i5",
        physical_cores=2,
        ram_gb=8.0,
        macos_version=(12, 7, 6),
        tier="slow",
    )


def fake_engine_class(
    name,
    available=True,
    reason=None,
    raise_on_construct=None,
    raise_on_probe=None,
):
    """Build a one-off engine class with the failure mode the test needs.

    Subclasses the real :class:`ASREngine` on purpose: if the interface grows an
    abstract method, these fakes stop instantiating and the suite says so,
    instead of quietly testing a shape the registry no longer receives.

    ``reason`` is only assigned as an attribute when it is given, so the default
    fake reproduces the awkward real case -- an engine that says "no" and offers
    no explanation -- rather than the tidy one.

    Each class carries its own ``attempts`` / ``instances`` / ``loads`` lists so a
    test can assert on work that did NOT happen: an engine the registry should
    never have reached has an empty ``attempts``, and an engine the registry
    should never have loaded has an empty ``loads``.
    """

    class Fake(ASREngine):
        attempts = []  # (cfg, hw) per constructor call, recorded before any raise
        instances = []  # successfully constructed instances
        loads = []  # instances whose load() was called

        def __init__(self, cfg, hw):
            Fake.attempts.append((cfg, hw))
            if raise_on_construct is not None:
                raise raise_on_construct
            self.name = name
            self.cfg = cfg
            self.hw = hw
            self.load_count = 0
            if reason is not None:
                self.unavailable_reason = reason
            Fake.instances.append(self)

        def is_available(self):
            if raise_on_probe is not None:
                raise raise_on_probe
            return available

        def load(self):
            self.load_count += 1
            Fake.loads.append(self)

        def transcribe(self, pcm, sample_rate):  # pragma: no cover - never called
            return ""

    Fake.__name__ = "Fake_" + name.replace("-", "_")
    return Fake


def install(monkeypatch, whisper, apple):
    """Point the registry at fakes for both real backends.

    ``whisper`` and ``apple`` are each either an engine class or a BaseException
    instance -- the latter is raised out of the import, which is how a broken
    wheel behaves. Returns the list of module suffixes the registry actually
    imported, so a test can prove a backend was never even reached.
    """
    specs = {WHISPER: whisper, APPLE: apple}
    modules = {}
    by_suffix = {}
    for engine_name in engines.ENGINE_NAMES:
        suffix = "._fake_" + engine_name.replace("-", "_")
        modules[engine_name] = (suffix, "FakeEngine")
        by_suffix[suffix] = specs[engine_name]

    imported = []

    def fake_import_module(suffix, package=None):
        imported.append(suffix)
        spec = by_suffix[suffix]
        if isinstance(spec, BaseException):
            raise spec
        return types.SimpleNamespace(FakeEngine=spec)

    monkeypatch.setattr(engines, "_ENGINE_MODULES", modules)
    monkeypatch.setattr(
        engines, "importlib", types.SimpleNamespace(import_module=fake_import_module)
    )
    return imported


def install_with_real_apple_speech(monkeypatch, whisper):
    """Fake out faster-whisper only, and leave apple-speech resolving for real.

    The machine this simulates is the one the packaging change is about: blurt
    installed without faster-whisper, which until this changeset was the default
    outcome of ``pip install blurt``. faster-whisper has to be faked because a
    test cannot uninstall it from the developer's machine (and must not depend
    on whether it happens to be installed); apple-speech must NOT be faked,
    because whether that backend is usable is the entire question.

    The apple-speech entry is copied out of the real ``_ENGINE_MODULES`` rather
    than written out again here, so a rename of the module or the class fails
    this section loudly instead of leaving it testing a path the registry no
    longer takes.
    """
    modules = dict(engines._ENGINE_MODULES)
    suffix = "._fake_faster_whisper"
    modules[WHISPER] = (suffix, "FakeEngine")

    def import_module(name, package=None):
        if name != suffix:
            return real_importlib.import_module(name, package)
        if isinstance(whisper, BaseException):
            raise whisper
        return types.SimpleNamespace(FakeEngine=whisper)

    monkeypatch.setattr(engines, "_ENGINE_MODULES", modules)
    monkeypatch.setattr(
        engines, "importlib", types.SimpleNamespace(import_module=import_module)
    )


#: What a machine without faster-whisper actually raises out of the import.
def _whisper_missing():
    return ImportError("No module named 'faster_whisper'")


def _select(engine_name):
    """Run selection for a configured engine value, with real Config/Hardware."""
    return select_engine(Config(engine=engine_name), _hw())


# --------------------------------------------------------------------------- #
# engine="auto": preference order
# --------------------------------------------------------------------------- #
def test_engine_names_pins_faster_whisper_as_the_preferred_backend():
    """The order is a decision, not an accident: faster-whisper is the proven
    engine and apple-speech is experimental. Flipping this silently downgrades
    every ``auto`` user, so it is pinned rather than inferred."""
    assert engines.ENGINE_NAMES == (WHISPER, APPLE)


def test_auto_picks_the_first_available_engine_in_preference_order(monkeypatch):
    whisper = fake_engine_class(WHISPER, available=True)
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, whisper, apple)

    engine = _select("auto")

    assert isinstance(engine, whisper)
    assert engine.name == WHISPER


def test_auto_does_not_probe_apple_speech_when_whisper_works(monkeypatch):
    """Probing apple-speech is not free: from a bundled .app it can raise the
    macOS speech-permission dialog. A user whose preferred engine works must
    never be shown a prompt for the one blurt did not need."""
    whisper = fake_engine_class(WHISPER, available=True)
    apple = fake_engine_class(APPLE, available=True)
    imported = install(monkeypatch, whisper, apple)

    _select("auto")

    assert apple.attempts == []
    assert imported == ["._fake_faster_whisper"]


def test_auto_falls_through_to_the_second_engine_when_the_first_is_unavailable(
    monkeypatch,
):
    whisper = fake_engine_class(WHISPER, available=False, reason="no model on disk")
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, whisper, apple)

    engine = _select("auto")

    assert isinstance(engine, apple)
    assert engine.name == APPLE


def test_auto_falls_through_when_the_first_backend_will_not_import(monkeypatch):
    """One broken wheel must cost the user a fallback, not the whole app."""
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, ImportError("No module named 'faster_whisper'"), apple)

    engine = _select("auto")

    assert isinstance(engine, apple)


def test_auto_falls_through_when_the_first_backends_constructor_raises(monkeypatch):
    whisper = fake_engine_class(WHISPER, raise_on_construct=ValueError("bad ctor"))
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, whisper, apple)

    engine = _select("auto")

    assert isinstance(engine, apple)
    assert whisper.attempts, "the first engine should still have been tried"


# --------------------------------------------------------------------------- #
# engine="auto": the diagnostic when nothing works
#
# At selection time there is no menu bar and no UI. This message is the entire
# user-facing artefact, so losing a reason from it IS the bug.
# --------------------------------------------------------------------------- #
def test_auto_with_nothing_available_raises_no_engine_available(monkeypatch):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=False, reason="a"),
        fake_engine_class(APPLE, available=False, reason="b"),
    )

    with pytest.raises(NoEngineAvailable):
        _select("auto")


def test_the_failure_message_names_every_engine_that_was_tried(monkeypatch):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=False, reason="no model on disk"),
        fake_engine_class(APPLE, available=False, reason="macOS too old"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("auto")

    message = str(excinfo.value)
    for engine_name in engines.ENGINE_NAMES:
        assert engine_name in message


def test_the_failure_message_keeps_the_specific_reason_for_each_rejection(monkeypatch):
    """A diagnostic that survives the raise but drops the reasons is the failure
    mode this test exists to catch: the user is told "nothing worked" and given
    nothing to act on."""
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=False, reason="model file is missing"),
        fake_engine_class(APPLE, available=False, reason="needs macOS 13 or newer"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("auto")

    message = str(excinfo.value)
    assert "model file is missing" in message
    assert "needs macOS 13 or newer" in message


def test_the_failure_message_keeps_reasons_from_engines_that_would_not_import(
    monkeypatch,
):
    """The loader-level failure is the one most likely to baffle a user, so its
    text has to reach them too, not just the polite is_available() rejections."""
    install(
        monkeypatch,
        OSError("dlopen(_pywhispercpp.so): symbol not found"),
        fake_engine_class(APPLE, available=False, reason="no speech framework"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("auto")

    message = str(excinfo.value)
    assert "OSError" in message
    assert "symbol not found" in message
    assert "no speech framework" in message


def test_the_failure_message_offers_the_most_likely_fix(monkeypatch):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=False, reason="not installed"),
        fake_engine_class(APPLE, available=False, reason="not on this OS"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("auto")

    assert "pip install faster-whisper" in str(excinfo.value)


def test_no_engine_available_is_a_runtime_error():
    """Documented as a RuntimeError subclass so callers that already guard
    startup against runtime failures need no extra handler. Changing the base
    class would silently uncover startup crashes in those callers."""
    assert issubclass(NoEngineAvailable, RuntimeError)


# --------------------------------------------------------------------------- #
# An explicit engine is honoured exactly, or refused. Never substituted.
#
# This is the headline promise of the module and the reason the file exists.
# --------------------------------------------------------------------------- #
def test_an_explicitly_configured_available_engine_is_returned(monkeypatch):
    whisper = fake_engine_class(WHISPER, available=True)
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, whisper, apple)

    engine = _select(APPLE)

    assert isinstance(engine, apple)
    assert engine.name == APPLE


def test_an_explicit_choice_that_works_does_not_probe_the_other_engine(monkeypatch):
    """Honouring a choice means going straight to it. Probing the alternative
    anyway would import a backend the user declined -- and for apple-speech that
    import can itself prompt for permission."""
    whisper = fake_engine_class(WHISPER, available=True)
    apple = fake_engine_class(APPLE, available=True)
    imported = install(monkeypatch, whisper, apple)

    _select(APPLE)

    assert whisper.attempts == []
    assert imported == ["._fake_apple_speech"]


def test_an_unavailable_explicit_engine_is_refused_rather_than_substituted(monkeypatch):
    """The whole point. faster-whisper is sitting right there and working; the
    user asked for apple-speech; blurt must raise instead of handing over the
    other one."""
    whisper = fake_engine_class(WHISPER, available=True)
    apple = fake_engine_class(APPLE, available=False, reason="no speech permission")
    install(monkeypatch, whisper, apple)

    with pytest.raises(NoEngineAvailable):
        _select(APPLE)


def test_the_refusal_says_blurt_will_not_switch(monkeypatch):
    """Saying "not available" while a working alternative exists invites the user
    to assume blurt quietly coped. It has to state that it did not."""
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=True),
        fake_engine_class(APPLE, available=False, reason="no speech permission"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select(APPLE)

    message = str(excinfo.value)
    assert "will NOT switch" in message
    assert repr(APPLE) in message


def test_the_refusal_names_the_alternatives_that_were_available(monkeypatch):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=True),
        fake_engine_class(APPLE, available=False, reason="no speech permission"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select(APPLE)

    message = str(excinfo.value)
    assert "Available instead: " + WHISPER in message
    # Naming the alternative is an offer, not an act: the user still has to edit
    # the config, and the message has to say so.
    assert "config" in message


def test_the_refusal_carries_the_engines_own_reason(monkeypatch):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=True),
        fake_engine_class(APPLE, available=False, reason="Info.plist has no usage text"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select(APPLE)

    assert "Info.plist has no usage text" in str(excinfo.value)


def test_an_unavailable_explicit_engine_is_refused_when_nothing_else_works_either(
    monkeypatch,
):
    """The other direction: with no alternative there is nothing to substitute,
    but the refusal must still be explicit and must say the machine has nothing
    else -- otherwise the user goes hunting for a config change that cannot
    help."""
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=False, reason="not installed"),
        fake_engine_class(APPLE, available=False, reason="no speech permission"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select(APPLE)

    message = str(excinfo.value)
    assert "No other engine is usable" in message
    assert "no speech permission" in message
    # No alternative exists, so nothing must be dangled as one.
    assert "Available instead" not in message


def test_an_explicit_engine_that_will_not_import_is_refused_not_substituted(
    monkeypatch,
):
    """A broken wheel for the requested backend is still not permission to run a
    different one."""
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=True),
        OSError("dlopen(Speech.framework): image not found"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select(APPLE)

    message = str(excinfo.value)
    assert "OSError" in message
    assert "will NOT switch" in message


def test_an_unknown_engine_name_raises_and_lists_the_valid_values(monkeypatch):
    """A typo in the config must not fall back to auto. Silently "fixing" it is
    the same substitution bug wearing a different hat, and it also hides the
    typo forever."""
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=True),
        fake_engine_class(APPLE, available=True),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("wisper")

    message = str(excinfo.value)
    assert "wisper" in message
    assert "auto" in message
    for engine_name in engines.ENGINE_NAMES:
        assert engine_name in message


def test_an_unknown_engine_name_never_reaches_a_backend(monkeypatch):
    """Rejection happens on the name alone, so a typo cannot import anything or
    trigger a permission prompt on the way to failing."""
    whisper = fake_engine_class(WHISPER, available=True)
    apple = fake_engine_class(APPLE, available=True)
    imported = install(monkeypatch, whisper, apple)

    with pytest.raises(NoEngineAvailable):
        _select("wisper")

    assert imported == []
    assert whisper.attempts == []
    assert apple.attempts == []


# --------------------------------------------------------------------------- #
# Name normalisation
#
# The value arrives from a hand-edited JSON file. Casing and stray whitespace
# are user typos, not engine choices, and must not become "unknown engine".
# --------------------------------------------------------------------------- #
def test_auto_is_matched_case_insensitively(monkeypatch):
    whisper = fake_engine_class(WHISPER, available=True)
    install(monkeypatch, whisper, fake_engine_class(APPLE, available=True))

    assert isinstance(_select("AUTO"), whisper)


def test_auto_tolerates_surrounding_whitespace(monkeypatch):
    whisper = fake_engine_class(WHISPER, available=True)
    install(monkeypatch, whisper, fake_engine_class(APPLE, available=True))

    assert isinstance(_select(" Auto "), whisper)


def test_an_explicit_name_is_normalised_the_same_way(monkeypatch):
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, fake_engine_class(WHISPER, available=True), apple)

    assert isinstance(_select("  Apple-Speech  "), apple)


def test_an_empty_engine_setting_means_auto_rather_than_unknown(monkeypatch):
    """An empty string is a user who deleted the value, not a user who named a
    backend called "". Treating it as a typo would refuse to start."""
    whisper = fake_engine_class(WHISPER, available=True)
    install(monkeypatch, whisper, fake_engine_class(APPLE, available=True))

    assert isinstance(_select(""), whisper)


# --------------------------------------------------------------------------- #
# Probing is total: an engine that explodes is an engine that does not work
#
# Every one of these would, if it escaped, take down startup before the menu bar
# exists -- a total failure caused by a backend the user may not even be using.
# --------------------------------------------------------------------------- #
def test_a_loader_level_import_failure_is_treated_as_unavailable(monkeypatch):
    """The real pywhispercpp failure on the floor machine: a wheel built for a
    newer macOS fails inside dyld, so Python sees OSError and an ``except
    ImportError`` would not catch it."""
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, OSError("dlopen: incompatible library version"), apple)

    engine = _select("auto")

    assert isinstance(engine, apple)


def test_a_loader_level_import_failure_still_reports_its_reason(monkeypatch):
    install(
        monkeypatch,
        OSError("dlopen: incompatible library version"),
        fake_engine_class(APPLE, available=False, reason="unsupported"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("auto")

    message = str(excinfo.value)
    assert "incompatible library version" in message
    assert "could not be loaded" in message


def test_a_constructor_that_raises_is_treated_as_unavailable_not_propagated(
    monkeypatch,
):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, raise_on_construct=RuntimeError("ctor exploded")),
        fake_engine_class(APPLE, available=False, reason="unsupported"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("auto")

    assert "ctor exploded" in str(excinfo.value)


def test_an_is_available_that_raises_is_treated_as_unavailable_not_propagated(
    monkeypatch,
):
    """A backend that cannot answer "can you run?" cannot be trusted to
    transcribe, but it also must not be allowed to abort selection: the
    exception has to be converted into a reason, not re-raised."""
    apple = fake_engine_class(APPLE, available=True)
    install(
        monkeypatch,
        fake_engine_class(WHISPER, raise_on_probe=RuntimeError("probe exploded")),
        apple,
    )

    engine = _select("auto")

    assert isinstance(engine, apple)


def test_an_is_available_that_raises_reports_why(monkeypatch):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, raise_on_probe=RuntimeError("probe exploded")),
        fake_engine_class(APPLE, available=False, reason="unsupported"),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("auto")

    message = str(excinfo.value)
    assert "availability check raised" in message
    assert "probe exploded" in message


def test_an_engine_that_declines_without_explaining_still_yields_a_reason(monkeypatch):
    """``unavailable_reason`` is optional on the interface, so the registry has to
    invent text rather than emit a bare "faster-whisper: " with nothing after the
    colon -- which reads like the message was truncated."""
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=False),
        fake_engine_class(APPLE, available=False),
    )

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("auto")

    message = str(excinfo.value)
    assert message.count("no reason given") == 2
    assert WHISPER + ": no reason given" in message


# --------------------------------------------------------------------------- #
# available_engines()
#
# Runs during startup, before the menu bar exists. It reports; it never raises.
# --------------------------------------------------------------------------- #
def test_available_engines_returns_both_names_in_preference_order(monkeypatch):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=True),
        fake_engine_class(APPLE, available=True),
    )

    assert available_engines() == [WHISPER, APPLE]


def test_available_engines_reports_only_the_usable_one(monkeypatch):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=False, reason="not installed"),
        fake_engine_class(APPLE, available=True),
    )

    assert available_engines() == [APPLE]


def test_available_engines_returns_an_empty_list_when_nothing_works(monkeypatch):
    """An empty list is information, not an error: it is what the "no engine"
    screen is built from."""
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=False, reason="not installed"),
        fake_engine_class(APPLE, available=False, reason="unsupported"),
    )

    assert available_engines() == []


def test_available_engines_never_raises_even_when_every_probe_explodes(monkeypatch):
    """Import failure, constructor failure and a throwing is_available() all at
    once -- the worst machine we can imagine -- must still return, because the
    caller has no handler for this and the app has no UI yet."""
    install(
        monkeypatch,
        OSError("dlopen: image not found"),
        fake_engine_class(APPLE, raise_on_probe=KeyError("kaboom")),
    )

    assert available_engines() == []


def test_available_engines_never_raises_when_a_constructor_explodes(monkeypatch):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, raise_on_construct=SystemError("ctor")),
        fake_engine_class(APPLE, raise_on_construct=MemoryError("ctor")),
    )

    assert available_engines() == []


def test_available_engines_probes_with_default_configuration(monkeypatch):
    """Documented contract: availability does not depend on model size or thread
    count, so it probes with no config. If that ever stopped being true, an
    engine could be listed as available and then be rejected by select_engine."""
    whisper = fake_engine_class(WHISPER, available=True)
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, whisper, apple)

    available_engines()

    assert whisper.attempts == [(None, None)]
    assert apple.attempts == [(None, None)]


# --------------------------------------------------------------------------- #
# Selection never loads
#
# load() is where the one-time model download and the native context allocation
# live. Doing it during selection would block startup before the UI could say
# "preparing", so a user on a cold cache would stare at nothing.
# --------------------------------------------------------------------------- #
def test_select_engine_returns_an_unloaded_engine(monkeypatch):
    whisper = fake_engine_class(WHISPER, available=True)
    install(monkeypatch, whisper, fake_engine_class(APPLE, available=True))

    engine = _select("auto")

    assert engine.load_count == 0
    assert whisper.loads == []


def test_an_explicitly_selected_engine_is_also_returned_unloaded(monkeypatch):
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, fake_engine_class(WHISPER, available=True), apple)

    engine = _select(APPLE)

    assert engine.load_count == 0
    assert apple.loads == []


def test_nothing_is_loaded_while_auto_falls_through_a_rejected_engine(monkeypatch):
    """The rejected engine was constructed and asked a question. It must not have
    been loaded on the way past -- that would download a model for a backend
    blurt is about to discard."""
    whisper = fake_engine_class(WHISPER, available=False, reason="not installed")
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, whisper, apple)

    engine = _select("auto")

    assert whisper.attempts, "the rejected engine should still have been probed"
    assert whisper.loads == []
    assert apple.loads == []
    assert engine.load_count == 0


def test_available_engines_never_loads_anything(monkeypatch):
    whisper = fake_engine_class(WHISPER, available=True)
    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, whisper, apple)

    available_engines()

    assert whisper.loads == []
    assert apple.loads == []


def test_select_engine_hands_the_real_config_and_hardware_to_the_backend(monkeypatch):
    """Unlike available_engines(), selection builds the engine the app will use,
    so it has to pass the user's settings through -- a backend constructed with
    None would silently run at defaults."""
    whisper = fake_engine_class(WHISPER, available=True)
    install(monkeypatch, whisper, fake_engine_class(APPLE, available=True))

    cfg = Config(engine="auto", model="tiny.en", cpu_threads=2)
    hw = _hw()
    engine = select_engine(cfg, hw)

    assert engine.cfg is cfg
    assert engine.hw is hw


def test_select_engine_returns_the_instance_it_probed(monkeypatch):
    """Selection must hand back the object it just verified, not construct a
    fresh one afterwards: a second construction is a second chance to fail, and
    it would fail outside the probe's guard."""
    whisper = fake_engine_class(WHISPER, available=True)
    install(monkeypatch, whisper, fake_engine_class(APPLE, available=True))

    engine = _select("auto")

    assert whisper.instances == [engine]


# --------------------------------------------------------------------------- #
# Typing sanity: what comes back is the interface the rest of blurt codes to
# --------------------------------------------------------------------------- #
def test_the_selected_engine_satisfies_the_asr_engine_interface(monkeypatch):
    install(
        monkeypatch,
        fake_engine_class(WHISPER, available=True),
        fake_engine_class(APPLE, available=True),
    )

    engine = _select("auto")

    assert isinstance(engine, ASREngine)


def test_unavailable_reason_is_read_from_the_instance_not_the_class(monkeypatch):
    """The registry reads ``unavailable_reason`` off the constructed object, so an
    engine that computes its reason during is_available() -- the natural
    implementation -- is reported correctly rather than as "no reason given"."""

    class LateReason(ASREngine):
        def __init__(self, cfg, hw):
            self.name = WHISPER

        def is_available(self):
            self.unavailable_reason = "model directory is empty"
            return False

        def load(self):  # pragma: no cover - never reached
            raise AssertionError("load() must not be called during selection")

        def transcribe(self, pcm, sample_rate):  # pragma: no cover - never called
            return ""

    install(monkeypatch, LateReason, fake_engine_class(APPLE, available=False))

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("auto")

    assert "model directory is empty" in str(excinfo.value)


def test_a_falsy_non_bool_from_is_available_is_treated_as_unavailable(monkeypatch):
    """``is_available`` is documented to return bool, but the registry coerces
    with bool() so a backend returning 0 or None -- easy to do when forwarding a
    C API's result -- is not mistaken for a yes."""

    class ReturnsNone(ASREngine):
        def __init__(self, cfg, hw):
            self.name = WHISPER
            self.unavailable_reason = "returned None"

        def is_available(self):
            return None

        def load(self):  # pragma: no cover - never reached
            raise AssertionError("load() must not be called during selection")

        def transcribe(self, pcm, sample_rate):  # pragma: no cover - never called
            return ""

    apple = fake_engine_class(APPLE, available=True)
    install(monkeypatch, ReturnsNone, apple)

    assert isinstance(_select("auto"), apple)
    assert available_engines() == [APPLE]


# --------------------------------------------------------------------------- #
# apple-speech is a rejection stub, and pyproject.toml depends on it being one
#
# `faster-whisper>=1.2` sits in the base dependencies instead of behind the
# `[whisper]` extra for exactly one reason, written out in the comment next to
# it: there is no second engine. If apple-speech were a working fallback, an
# install without faster-whisper would be degraded but alive and the extra would
# have been defensible; because apple-speech always declines, that install is a
# program that starts, probes the hardware, and dies.
#
# Nothing tested that. The stub could have been made to return True by someone
# reviving the idea -- the module docstring exists precisely because the idea
# keeps looking attractive -- and every test in this file would still have
# passed, because every one of them replaces apple-speech with a fake. The
# packaging comment would then be describing a program that no longer existed.
#
# So this section asserts the stub's own behaviour, and then the consequence the
# packaging decision actually rests on.
# --------------------------------------------------------------------------- #
def test_the_registry_still_resolves_apple_speech_to_the_rejection_stub():
    """The premise of everything below: this is the class `auto` really probes.

    Asserted rather than assumed, because the tests below are only about the
    shipped backend while the registry still points at it. A rewrite that
    swapped in a new module would leave them testing a class blurt no longer
    loads, quietly and while staying green.
    """
    assert engines._ENGINE_MODULES[APPLE] == (".apple_speech_engine", "AppleSpeechEngine")
    assert APPLE in engines.ENGINE_NAMES
    assert AppleSpeechEngine.name == APPLE


def test_apple_speech_reports_itself_unavailable():
    """The single fact the packaging change is built on."""
    assert AppleSpeechEngine(None, None).is_available() is False


@pytest.mark.parametrize(
    "args",
    [(), (None, None), (Config(), None), (None, "hw"), (Config(engine=APPLE), None)],
    ids=["no-args", "none-none", "cfg-only", "hw-only", "explicitly-chosen"],
)
def test_apple_speech_is_unavailable_however_it_was_constructed(args):
    """No argument makes it available, including the user asking for it by name.

    The registry constructs engines as ``engine_class(cfg, hw)`` and
    ``available_engines`` constructs them as ``engine_class(None, None)``, so
    both shapes have to answer the same way. Passing a config that names
    apple-speech is in the list because "the user asked for it" is the one input
    that could plausibly have been wired to a different answer -- and must not
    be: wanting this backend has never been what makes it safe.
    """
    engine = AppleSpeechEngine(*args)
    assert engine.is_available() is False


def test_apple_speech_probing_never_raises_and_stays_stable():
    """A backend that throws when asked "can you run?" takes startup down with
    it -- before the menu bar exists, so there is nowhere to show the error.

    The registry does catch that (``test_an_is_available_that_raises_...``
    above), but a stub relying on the caller's guard is one refactor away from
    being an outage, and ``ASREngine`` documents ``is_available`` as total.
    Probed repeatedly because a stub has no state that could make the second
    answer differ from the first, and if that ever stops being true this is
    where it should show up.
    """
    engine = AppleSpeechEngine(Config(), _hw())
    assert [engine.is_available() for _ in range(3)] == [False, False, False]


def test_apple_speech_says_why_rather_than_just_declining():
    """A silent "no" reads as a bug on the user's machine. This one is a verdict.

    ``unavailable_reason`` is optional on the interface, and an engine that
    omits it is reported by the registry as "no reason given" -- fine for a
    backend that failed for a boring local reason, useless for one that will
    never work anywhere, for reasons a user cannot discover or fix. The text has
    to carry the disqualifying finding (audio can be uploaded to Apple), say
    that the rejection is deliberate, and point at where the rest of it is
    written down.
    """
    reason = AppleSpeechEngine(None, None).unavailable_reason

    assert reason.strip()
    assert "not supported" in reason
    # The reason that makes this a hard rejection rather than a "not yet".
    assert "servers" in reason
    assert "never leaves your machine" in reason
    # Where the full findings live, so the next person to have this idea reads
    # them instead of spending a day rediscovering them.
    assert "apple_speech_engine.py" in reason


def test_apple_speech_imports_nothing_at_all_while_being_probed(monkeypatch):
    """Probing has to stay inert, which is most of the value of a stub.

    From a signed .app bundle, importing and querying ``SFSpeechRecognizer`` is
    what raises the macOS speech-permission dialog -- the registry's own
    docstring warns that ``available_engines`` is therefore not guaranteed
    side-effect-free. Since this backend has already decided the answer is no,
    prompting a user for permission on the way to saying so would be a dialog
    spent on nothing. It also keeps the answer independent of whether the
    ``[speech]`` extra is installed, which is what the README promises: the stub
    cannot consult pyobjc, so it cannot answer differently when pyobjc is there.

    Written as a spy on ``__import__`` rather than as a before/after diff of
    ``sys.modules``, and that is not a stylistic choice -- the diff version was
    written first and was already blind. An earlier test in this file probes the
    same engine, so by the time this one took its "before" snapshot the module
    would already be loaded and nothing would ever look new. Mutation-testing it
    is what surfaced that; the spy has no such history because it observes the
    call rather than its lingering effect. Watching every import rather than a
    list of framework names also means a revival that reaches the framework
    through some other package still trips it.
    """
    imported = []
    real_import = builtins.__import__

    def spy(name, *args, **kwargs):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    engine = AppleSpeechEngine(Config(), _hw())
    monkeypatch.setattr(builtins, "__import__", spy)
    try:
        usable = engine.is_available()
    finally:
        monkeypatch.undo()

    assert usable is False
    assert imported == []


def test_apple_speech_refuses_to_load_rather_than_pretending_to_work():
    """A stub that declines and then loads happily would be worse than no stub.

    ``is_available()`` returning False is a claim about the whole object.
    Anything that reaches ``load()`` or ``transcribe()`` -- a caller that skips
    the probe, a future refactor that forgets it -- must hit the same verdict
    and the same text, not silence and an empty transcript that looks like the
    user simply was not heard.
    """
    engine = AppleSpeechEngine(Config(), _hw())

    with pytest.raises(RuntimeError) as load_error:
        engine.load()
    assert REJECTION_REASON in str(load_error.value)

    with pytest.raises(RuntimeError) as transcribe_error:
        engine.transcribe(None, 16000)
    assert REJECTION_REASON in str(transcribe_error.value)


# --------------------------------------------------------------------------- #
# The consequence: without faster-whisper there is no engine at all
# --------------------------------------------------------------------------- #
def test_without_faster_whisper_auto_finds_nothing_usable(monkeypatch):
    """The property `faster-whisper>=1.2` was promoted to a base dependency for.

    Not "auto degrades to apple-speech" and not "auto is slower" -- auto raises.
    An extra that is easy to miss cannot gate the only engine there is.
    """
    install_with_real_apple_speech(monkeypatch, _whisper_missing())

    with pytest.raises(NoEngineAvailable):
        _select("auto")


def test_without_faster_whisper_available_engines_is_empty(monkeypatch):
    """The same fact as the startup probe sees it. ``doctor`` and the no-engine
    screen are both built from this list, so it has to agree with selection."""
    install_with_real_apple_speech(monkeypatch, _whisper_missing())

    assert available_engines() == []


def test_the_no_engine_message_names_both_engines_and_both_reasons(monkeypatch):
    """The only artefact the user gets, on the install this changeset prevents.

    Both halves have to be in there. The missing import alone reads like blurt
    forgot to look for its other engine; apple-speech's rejection alone reads
    like the machine is at fault. Together they say: one engine is not installed,
    the other never works, install the first -- which is the whole content of the
    packaging decision, delivered at the only moment the user needs it.
    """
    install_with_real_apple_speech(monkeypatch, _whisper_missing())

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select("auto")

    message = str(excinfo.value)
    assert WHISPER in message
    assert APPLE in message
    assert "No module named 'faster_whisper'" in message
    assert REJECTION_REASON in message
    assert "pip install faster-whisper" in message


def test_choosing_apple_speech_explicitly_is_refused_with_its_own_reason(monkeypatch):
    """Installing the ``[speech]`` extra does not buy a second engine.

    A user who reads about ``pyobjc-framework-Speech``, installs it and writes
    ``"engine": "apple-speech"`` has done everything the old docs implied and
    still has nothing. The refusal must therefore carry the real reason rather
    than a bare "not available", and must not dangle an alternative that is not
    there either.
    """
    install_with_real_apple_speech(monkeypatch, _whisper_missing())

    with pytest.raises(NoEngineAvailable) as excinfo:
        _select(APPLE)

    message = str(excinfo.value)
    assert REJECTION_REASON in message
    assert "No other engine is usable" in message
    assert "Available instead" not in message


def test_faster_whisper_is_still_chosen_over_the_stub_when_it_is_there(monkeypatch):
    """The control case, and it is not decoration.

    Everything above would also pass if the registry had stopped being able to
    select anything at all. With faster-whisper importable, ``auto`` picks it and
    never reaches the stub -- which is what the ordinary install this changeset
    creates now looks like.
    """
    whisper = fake_engine_class(WHISPER, available=True)
    install_with_real_apple_speech(monkeypatch, whisper)

    assert isinstance(_select("auto"), whisper)
    assert available_engines() == [WHISPER]
