"""Insert transcribed text at the user's cursor, via the clipboard and a synthetic Cmd+V.

blurt's job ends with the words appearing where the user was already typing. This
module does that and nothing else: it does not know what a Transcript is, does not
clean text, and never decides *whether* to insert -- the caller decides, this file
carries it out.

WHY CLIPBOARD + Cmd+V AND NOT THE "PROPER" ACCESSIBILITY API
-------------------------------------------------------------
The tidy-looking approach is to find the focused element through the Accessibility
API and set its value directly::

    AXUIElementSetAttributeValue(focused, kAXValueAttribute, text)

Do not build on that. It works in a handful of native Cocoa text fields and fails
everywhere the user actually types. Electron applications (Slack, VS Code, Discord,
Notion) expose a single opaque AXTextArea whose value cannot be set from outside, or
expose nothing writable at all. Web views hand back a container element rather than
the caret's field. Editors that draw their own text (the Vim/Emacs family, terminal
emulators, anything canvas-backed) have no AX value to set. Even where the write
succeeds it frequently *replaces the whole field* instead of inserting at the caret,
silently destroying whatever the user had already typed, and it bypasses the app's
own undo stack so Cmd+Z cannot save them.

Synthetic Cmd+V has the opposite profile. It is crude, it costs us the clipboard
dance below, and it goes through the application's real paste handler -- which means
correct caret insertion, correct undo, correct behaviour in every app that supports
pasting, which is every app. Boring and proven beats elegant and broken.

The alternative of *typing* the text as synthetic keystrokes (CGEventKeyboardSetUnicodeString,
one event per character) is slower by orders of magnitude for a paragraph of dictation,
drops characters under load, and mangles anything with an input method active. Rejected.

THE CLIPBOARD IS THE USER'S, NOT OURS
--------------------------------------
Silently eating whatever someone had copied is an unacceptable side effect -- they
copied it for a reason and will not find out it is gone until they paste and get
their own dictation back. So :func:`insert_text` snapshots the pasteboard before
overwriting and puts it back afterwards.

The restore is guarded by ``changeCount``. macOS increments that counter on every
pasteboard write from any process, so if it moved between our write and our restore,
somebody else -- the user hitting Cmd+C, a clipboard manager, another app -- wrote
after us. In that case we deliberately do NOT restore: clobbering the user's *fresh*
copy with a stale snapshot is a worse failure than leaving our dictated text sitting
on the clipboard. Last writer should win, and after the user copies something, that
is not us.

Restoration is best effort by nature. We copy the raw bytes of every type on every
pasteboard item, which handles text, RTF, HTML, images and most everything else --
but *promised* content (the "I'll generate that data when someone asks" pattern used
by some design tools for large exports) cannot be captured, because reading it forces
the promise and the promising app may already be gone. Those types are dropped from
the snapshot. A snapshot that fails entirely never blocks the paste; the user gets
their text and loses their clipboard, which is the lesser of the two harms.

WHAT CAN GO WRONG ON macOS
---------------------------
  * **Secure Event Input -- the hard stop.** When a password field has focus, or a
    terminal has Secure Keyboard Entry switched on (Terminal.app and iTerm2 both
    offer it, iTerm2 enables it automatically at some password prompts), macOS puts
    the window server into secure input mode and *discards synthetic keyboard events
    entirely*. Our Cmd+V never arrives. This is the OS refusing on purpose, it is not
    a bug in this file, and there is no entitlement, permission, or API that lets a
    normal application opt out of it -- that is the whole point of the feature. We
    detect it up front with ``IsSecureEventInputEnabled()`` and return ``False``
    *without touching the clipboard*, so the caller can fall back to
    :func:`copy_to_clipboard` and tell the user to paste it themselves.
    Note the check is process-global, not per-window: it reports that *some*
    application on the system has secure input enabled, which is the condition that
    matters to us, but it means a stray app holding secure input on can block pasting
    even while the user's focused field looks perfectly ordinary.

  * **Accessibility permission is required to post synthetic events.** ``CGEventPost``
    is gated behind System Settings > Privacy & Security > Accessibility. Untrusted,
    it does not raise and does not return an error -- the event is simply dropped and
    the paste silently does not happen. Since the failure is invisible we check
    ``AXIsProcessTrusted()`` beforehand and refuse rather than eating the clipboard
    for a paste that was never going to land. As with the hotkey, when blurt is run
    from a shell the trust belongs to the *terminal application*, not to Python and
    not to blurt, so it is Terminal/iTerm/VS Code the user must tick in that list.

  * **We cannot confirm the paste actually happened.** Nothing reports back. A
    ``True`` return means "we held the necessary permissions and posted the events",
    not "the characters are on screen". An app that ignores Cmd+V, or a field that
    was not editable, looks identical to success from here. Hence "apparent success"
    in the docstring -- do not build retry logic on this return value.

  * **Keyboard layout.** We post virtual keycode 9, which is the physical key labelled
    V on ANSI hardware. macOS resolves Cmd-key equivalents through the active layout,
    so this is right for QWERTY and for the "Dvorak - QWERTY Cmd" layout that exists
    precisely to keep the shortcut keys in place. On a plain Dvorak or Colemak layout
    keycode 9 is not the paste key and the shortcut may land elsewhere. Rare, and
    fixing it properly needs UCKeyTranslate over the current layout; documented rather
    than half-solved.

  * **Held modifiers bleed into the event.** If the user is still physically holding
    a modifier when we fire -- easily done, the dictation hotkey *is* a held modifier
    and this runs right after they let go -- the receiving app can see Cmd+Opt+V
    instead of Cmd+V and do something else entirely. We set the flags explicitly on
    each event rather than inheriting global state, and :data:`_SETTLE_S` gives the
    key they were holding a moment to come up first. That shrinks the window; it does
    not slam it shut.

  * **Universal Clipboard.** A plain pasteboard write is broadcast to the user's other
    Apple devices. Dictated text has no business landing on someone's iPhone, so we
    write with ``NSPasteboardContentsCurrentHostOnly``. This is also why the transient
    markers below matter: dictation can contain anything the user says out loud, and
    it should not be archived forever by a clipboard-history app.

  * **No run loop needed.** NSPasteboard and CGEventPost work from any thread and do
    not require a running NSApplication, so this is safe to call from blurt's worker
    thread. Do not add AppKit UI calls to this module or that stops being true.

  * **We never synthesize Return.** Deliberate, and load-bearing. Dictated text is
    frequently misheard and must stay visible and editable so the user can fix it
    before it goes anywhere. Auto-submitting someone's half-right dictation into a
    chat box, a terminal, or a search field is unrecoverable in a way that a wrong
    word simply is not. There is no option for this and there should not be one.

Python 3.9 floor: ``from __future__ import annotations``, and typing.List / Optional
rather than PEP 585/604 forms.
"""

from __future__ import annotations

import ctypes
import logging
import time
from typing import Any, Dict, List, Optional

__all__ = [
    "insert_text",
    "copy_to_clipboard",
    "secure_input_active",
    "accessibility_trusted",
]

_log = logging.getLogger(__name__)

# Virtual keycode for the physical "V" key on ANSI layouts. See the layout note
# in the module docstring before changing this.
_KEYCODE_V = 9

# Grace period before posting, so a modifier the user is still holding from the
# push-to-talk gesture has a moment to come up and stop contaminating our flags.
_SETTLE_S = 0.02

# Upper bound on how much pasteboard content we will copy aside before overwriting.
# Someone with a 400 MB image on the clipboard should not have blurt quietly
# duplicate it in RAM on every dictation -- the floor machine has 16 GB and two
# cores and would feel that. Past the cap we keep plain text only, which is the
# part users actually miss.
_SNAPSHOT_BYTE_CAP = 8 * 1024 * 1024

# Community conventions from nspasteboard.org, honoured by the mainstream
# clipboard-history apps (Maccy, Alfred, Pastebot, Copied and friends): content
# tagged with these is meant to be skipped rather than archived. There is no
# official Apple API for "do not record this", so these de-facto marker types are
# the whole of what we can do. An app that ignores them will still archive the
# text -- we cannot enforce this, only ask.
_MARKER_TYPES = (
    "org.nspasteboard.TransientType",
    "org.nspasteboard.ConcealedType",
    "org.nspasteboard.AutoGeneratedType",
)


def _appkit() -> Optional[Any]:
    """Import AppKit, or return None if pyobjc is unavailable."""
    try:
        import AppKit
    except Exception:  # pragma: no cover - non-macOS or broken pyobjc install
        _log.debug("AppKit unavailable; clipboard operations disabled", exc_info=True)
        return None
    return AppKit


def _quartz() -> Optional[Any]:
    """Import Quartz, or return None if pyobjc is unavailable."""
    try:
        import Quartz
    except Exception:  # pragma: no cover - non-macOS or broken pyobjc install
        _log.debug("Quartz unavailable; cannot post key events", exc_info=True)
        return None
    return Quartz


def secure_input_active() -> Optional[bool]:
    """Report whether macOS is currently discarding synthetic keyboard events.

    Returns ``True`` when some application has Secure Event Input enabled (a
    password field has focus, or a terminal has Secure Keyboard Entry on),
    ``False`` when it does not, and ``None`` when the answer cannot be determined.

    ``True`` means a synthetic Cmd+V cannot possibly work, no matter what
    permissions we hold. It is a system-wide condition, not a property of the
    focused window -- see the module docstring.
    """
    # IsSecureEventInputEnabled lives in Carbon's HIToolbox and pyobjc does not
    # wrap it, so we reach it through ctypes. Verified present on macOS 13.7.
    try:
        carbon = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/Carbon.framework/Carbon"
        )
        probe = carbon.IsSecureEventInputEnabled
    except Exception:  # pragma: no cover - non-macOS, or framework moved
        return None
    probe.restype = ctypes.c_bool
    probe.argtypes = []
    try:
        return bool(probe())
    except Exception:  # pragma: no cover - ABI changed underneath us
        return None


def accessibility_trusted() -> Optional[bool]:
    """Report whether this process may post synthetic events.

    ``True`` when macOS trusts the host process for Accessibility, ``False`` when
    it does not, ``None`` when undeterminable. ``False`` is the one that matters:
    ``CGEventPost`` will accept our events and quietly drop them.

    Asking does not prompt the user. The trust belongs to the host application --
    the terminal, when blurt runs from a shell -- not to blurt itself.
    """
    # pyobjc exposes AXIsProcessTrusted from more than one framework module and
    # which are installed varies with the pyobjc split, so try each in turn.
    for module_name in ("HIServices", "ApplicationServices", "Quartz"):
        try:
            module = __import__(module_name)
        except Exception:  # pragma: no cover - non-macOS or pyobjc missing
            continue
        probe = getattr(module, "AXIsProcessTrusted", None)
        if probe is None:
            continue
        try:
            return bool(probe())
        except Exception:  # pragma: no cover - API shape changed
            return None
    return None


def _snapshot(pasteboard: Any) -> Optional[List[Dict[str, bytes]]]:
    """Copy the pasteboard's current contents out as plain bytes.

    Returns one dict of ``{type: data}`` per pasteboard item, preserving item and
    type order (order encodes preference -- the first type is what a pasting app
    reaches for first). Returns None if nothing could be captured.

    The bytes must be copied *now*: NSPasteboardItem objects belonging to the
    pasteboard are invalidated the moment we clear it, so holding references to
    them and reading later gives nothing back.
    """
    try:
        items = pasteboard.pasteboardItems()
    except Exception:  # pragma: no cover - defensive
        _log.debug("could not enumerate pasteboard items", exc_info=True)
        return None
    if not items:
        # An empty pasteboard is a real state worth restoring to, not a failure.
        return []

    captured: List[Dict[str, bytes]] = []
    total = 0
    truncated = False
    for item in items:
        try:
            types = list(item.types())
        except Exception:  # pragma: no cover - defensive
            continue
        entry: Dict[str, bytes] = {}
        for type_name in types:
            try:
                data = item.dataForType_(type_name)
            except Exception:
                # Promised/lazy content that refuses to materialise. Skip the
                # type rather than abandoning the whole snapshot.
                continue
            if data is None:
                continue
            try:
                raw = bytes(data)
            except Exception:  # pragma: no cover - exotic NSData subclass
                continue
            total += len(raw)
            if total > _SNAPSHOT_BYTE_CAP:
                truncated = True
                break
            entry[str(type_name)] = raw
        if entry:
            captured.append(entry)
        if truncated:
            break

    if truncated:
        # Over budget: fall back to preserving just the plain-text flavour, which
        # is small, and is the loss users actually notice.
        appkit = _appkit()
        text_type = str(appkit.NSPasteboardTypeString) if appkit else "public.utf8-plain-text"
        for entry in captured:
            if text_type in entry:
                _log.debug("clipboard snapshot over cap; preserving text only")
                return [{text_type: entry[text_type]}]
        _log.debug("clipboard snapshot over cap and no text flavour; not restoring")
        return None

    return captured


def _restore(pasteboard: Any, snapshot: List[Dict[str, bytes]]) -> bool:
    """Put a snapshot from :func:`_snapshot` back onto the pasteboard."""
    appkit = _appkit()
    if appkit is None:  # pragma: no cover - checked by callers already
        return False
    try:
        pasteboard.clearContents()
        if not snapshot:
            # Restoring to genuinely empty: clearing is the whole job.
            return True
        items = []
        for entry in snapshot:
            item = appkit.NSPasteboardItem.alloc().init()
            for type_name, raw in entry.items():
                # pyobjc bridges Python bytes straight to NSData here.
                item.setData_forType_(raw, type_name)
            items.append(item)
        return bool(pasteboard.writeObjects_(items))
    except Exception:  # pragma: no cover - defensive
        _log.debug("clipboard restore failed", exc_info=True)
        return False


def _write_text(pasteboard: Any, text: str, transient: bool) -> bool:
    """Overwrite the pasteboard with `text`.

    When `transient` is set, the item is also tagged with the nspasteboard.org
    marker types so clipboard-history apps skip it. Those tags are advisory --
    see :data:`_MARKER_TYPES`.
    """
    appkit = _appkit()
    if appkit is None:  # pragma: no cover - checked by callers already
        return False
    try:
        # prepareForNewContentsWithOptions_ is clearContents plus the option to
        # keep this write on this machine, i.e. off Universal Clipboard. macOS 11+;
        # fall back for anything older.
        prepare = getattr(pasteboard, "prepareForNewContentsWithOptions_", None)
        if prepare is not None:
            prepare(appkit.NSPasteboardContentsCurrentHostOnly)
        else:  # pragma: no cover - macOS 10.x
            pasteboard.clearContents()

        item = appkit.NSPasteboardItem.alloc().init()
        if not item.setString_forType_(text, appkit.NSPasteboardTypeString):
            return False
        if transient:
            for marker in _MARKER_TYPES:
                # Marker types carry no payload; presence is the whole signal.
                # A rejection here is cosmetic, so it must not fail the write.
                try:
                    item.setData_forType_(b"", marker)
                except Exception:  # pragma: no cover - defensive
                    _log.debug("could not set marker %s", marker, exc_info=True)
        return bool(pasteboard.writeObjects_([item]))
    except Exception:  # pragma: no cover - defensive
        _log.debug("clipboard write failed", exc_info=True)
        return False


def _post_command_v() -> bool:
    """Post one synthetic Cmd+V to the system.

    Returns True if the events were posted. That is not proof they were delivered
    or acted upon -- see the module docstring on why success is unverifiable.
    """
    quartz = _quartz()
    if quartz is None:
        return False
    try:
        # A HID-system source makes our events look like they came from a real
        # keyboard, which is what applications and their key handlers expect.
        source = quartz.CGEventSourceCreate(quartz.kCGEventSourceStateHIDSystemState)

        key_down = quartz.CGEventCreateKeyboardEvent(source, _KEYCODE_V, True)
        key_up = quartz.CGEventCreateKeyboardEvent(source, _KEYCODE_V, False)
        if key_down is None or key_up is None:  # pragma: no cover - defensive
            return False

        # Set flags explicitly on both events. Assigning the exact mask rather
        # than OR-ing into whatever is currently held keeps a modifier the user
        # is still leaning on from turning this into a different shortcut. The
        # key-up carries the command flag too: releasing it first would make the
        # sequence read as Cmd being let go mid-chord.
        quartz.CGEventSetFlags(key_down, quartz.kCGEventFlagMaskCommand)
        quartz.CGEventSetFlags(key_up, quartz.kCGEventFlagMaskCommand)

        quartz.CGEventPost(quartz.kCGHIDEventTap, key_down)
        quartz.CGEventPost(quartz.kCGHIDEventTap, key_up)
        return True
    except Exception:  # pragma: no cover - defensive
        _log.debug("posting Cmd+V failed", exc_info=True)
        return False


def copy_to_clipboard(text: str) -> None:
    """Put `text` on the general pasteboard, replacing what was there.

    The public fallback for when pasting is impossible (secure input, missing
    Accessibility permission): the caller tells the user their dictation is on the
    clipboard and they paste it themselves.

    Unlike :func:`insert_text` this makes no attempt to preserve or restore the
    previous contents -- the text staying on the clipboard is the entire point.
    It is still marked transient so clipboard-history apps skip it. Never raises;
    a failure to reach the pasteboard is logged and swallowed.
    """
    appkit = _appkit()
    if appkit is None:
        return
    try:
        pasteboard = appkit.NSPasteboard.generalPasteboard()
    except Exception:  # pragma: no cover - defensive
        _log.debug("no general pasteboard", exc_info=True)
        return
    _write_text(pasteboard, text, transient=True)


def insert_text(
    text: str,
    paste_delay_ms: int = 120,
    restore_delay_ms: int = 400,
) -> bool:
    """Insert `text` at the user's cursor by pasting it, then restore the clipboard.

    Returns True on apparent success -- meaning the clipboard was written and the
    Cmd+V events were posted with the permissions needed for them to be delivered.
    It is not confirmation that the characters reached the screen; nothing on macOS
    reports that back. Returns False when pasting was blocked, in which case the
    clipboard is left exactly as it was found and the caller should fall back to
    :func:`copy_to_clipboard`.

    Never raises on the normal failure paths.

    `paste_delay_ms` is the pause between writing the clipboard and posting Cmd+V.
    The target application needs a moment to notice new pasteboard contents; too
    short and it pastes the *previous* clipboard, which is a spectacular bug from
    the user's point of view. `restore_delay_ms` is how long we wait afterwards
    before putting the old contents back, since the paste is handled
    asynchronously by the other application and restoring too eagerly yanks the
    text away before it reads it. The defaults come from Config
    (paste_delay_ms=120, clipboard_restore_ms=400) and are conservative on
    purpose -- the floor machine is a two-core Intel and a loaded Electron app can
    take its time.
    """
    if not text:
        # Nothing to insert. Say so rather than clearing the clipboard and firing
        # a paste that would insert emptiness.
        return False

    appkit = _appkit()
    if appkit is None:
        return False

    # Both checks happen BEFORE we touch the clipboard. If the paste cannot land
    # there is no reason to disturb the user's pasteboard on the way to failing.
    if secure_input_active() is True:
        _log.info(
            "secure input is active (password field or terminal secure keyboard "
            "entry); synthetic paste is blocked by macOS"
        )
        return False
    if accessibility_trusted() is False:
        _log.warning(
            "not trusted for Accessibility; synthetic events would be silently "
            "discarded. Grant access in System Settings > Privacy & Security > "
            "Accessibility (to the terminal application, when blurt is run from a shell)"
        )
        return False

    try:
        pasteboard = appkit.NSPasteboard.generalPasteboard()
    except Exception:  # pragma: no cover - defensive
        _log.debug("no general pasteboard", exc_info=True)
        return False

    # Snapshot first. A snapshot failure is survivable -- we paste anyway and
    # accept losing the old clipboard -- but it must be attempted before the write.
    saved = _snapshot(pasteboard)

    if not _write_text(pasteboard, text, transient=True):
        _log.warning("could not write dictated text to the clipboard")
        return False

    # Remember where the counter stands with our content on it. prepareForNewContents
    # bumps it and writeObjects_ does not bump it again, so read it after the write
    # rather than predicting it.
    try:
        our_change_count = int(pasteboard.changeCount())
    except Exception:  # pragma: no cover - defensive
        our_change_count = -1

    time.sleep(max(0.0, paste_delay_ms / 1000.0) + _SETTLE_S)

    posted = _post_command_v()
    if not posted:
        # Leave our text on the clipboard: pasting failed, so the user's best
        # remaining move is Cmd+V by hand, and restoring would take that away.
        _log.warning("could not post Cmd+V; dictated text left on the clipboard")
        return False

    if saved is None:
        # Nothing captured, so nothing to give back. The paste still happened.
        return True

    time.sleep(max(0.0, restore_delay_ms / 1000.0))

    try:
        current_change_count = int(pasteboard.changeCount())
    except Exception:  # pragma: no cover - defensive
        current_change_count = our_change_count

    if our_change_count >= 0 and current_change_count != our_change_count:
        # Someone wrote to the pasteboard after us. Their copy is newer and more
        # relevant than our stale snapshot; putting the snapshot back now would
        # destroy something the user just deliberately copied. Stand down.
        _log.debug(
            "clipboard changed after paste (%d -> %d); leaving it alone",
            our_change_count,
            current_change_count,
        )
        return True

    if not _restore(pasteboard, saved):
        _log.debug("could not restore previous clipboard contents")

    return True
