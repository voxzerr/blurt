# blurt v2 — native Swift design (Apple Silicon only)

**Status: recorded, not started. Nothing in this document has been built.**

This is a design record, not a plan of work. A prior research effort produced a
coherent native design that **cannot run on the machine blurt is currently
developed on** (2017 Intel MacBook, macOS 13, no Neural Engine). Every decision
below is written down so that it survives until an Apple Silicon machine exists
to build it on, and so that the reasoning is recoverable rather than
re-derived.

Read the status markers literally:

- **Decided** — settled, with a reason. Reopen only if the reason is shown wrong.
- **Open** — genuinely unresolved. Listed in §6. No answer is implied anywhere
  else in this document, and none should be invented while reading it.

v1 (Python) is not deprecated by this. See §7.

---

## 1. Why a rewrite is warranted — and only on Apple Silicon

A rewrite for its own sake would not be worth it. The justification rests on one
specific capability that does not exist on the current machine.

### 1.1 The core reason: actual-length audio processing

Whisper pads every input to a fixed 30-second window. A 1.5-second utterance
costs approximately the same as a 25-second one. This is not an implementation
detail we can optimize away — it is the shape of the model. v1 already documents
this in `blurt/types.py`, where `Transcript` carries a warning against computing
a realtime factor from `latency_seconds` and `audio_seconds`, because latency is
roughly constant regardless of how long the user actually spoke.

**Parakeet TDT 0.6B v3, running on the Apple Neural Engine via FluidAudio,
processes the actual audio length.** Short utterances are genuinely fast, not
uniformly fast. For a push-to-talk dictation tool, where the majority of
utterances are a handful of seconds, this is the entire difference between "it
feels instant" and "it feels like it is thinking."

Intel + Whisper will never feel instant, no matter how much the surrounding code
is optimized. That is the honest reason for a rewrite, and it is the only one
that is load-bearing.

### 1.2 Secondary benefits of Parakeet

These would not justify a rewrite alone, but they remove existing v1 problems:

- **Punctuation and capitalization are emitted natively.** v1's `cleanup.py`
  reconstructs sentence casing from raw lowercase-ish output using heuristics
  (`_apply_sentence_case`, `_is_terminal`, an abbreviation table). Much of that
  machinery exists to compensate for what the model does not give us.
- **No hallucinate-on-silence failure mode.** Whisper is known to emit confident
  invented text when given silence or near-silence. Parakeet does not exhibit
  this. For a tool triggered by a held key — where trailing silence is normal —
  this matters.
- **Size: ~115–131 MB.** Small enough that bundling is at least arguable
  (see §6.3).

### 1.3 Why this excludes Intel by construction

FluidAudio's ANE path requires **macOS 14+ and arm64**. The current development
machine is macOS 13 on x86_64 with no Neural Engine. It fails the requirement on
all three counts simultaneously.

This is not a soft performance gap that a slower machine merely experiences more
slowly. There is no ANE to schedule work onto. The exclusion is structural, and
it is the reason v1 must continue to exist rather than be replaced.

---

## 2. Architecture decisions

All **decided** unless marked otherwise.

### 2.1 Build system: SwiftPM only

No `.xcodeproj`. No Interface Builder. No storyboards.

The build is `swift build` plus a roughly 40-line shell script that assembles:

```
blurt.app/
  Contents/
    MacOS/          # the built binary
    Resources/
    Info.plist
```

followed by `codesign`. Xcode Command Line Tools are sufficient; the full Xcode
install is not required.

**Why:** a `.xcodeproj` is a large generated file that produces unreadable
diffs and merge conflicts, and it hides build configuration in a GUI. A shell
script that a person can read end to end is a better artifact for a project this
size. Interface Builder has the same problem in worse form — UI state stored in
XML that cannot be reviewed.

### 2.2 `BlurtCore` library target, no AppKit dependency

Core logic lives in a `BlurtCore` library target that **does not import AppKit**.
The app target depends on it; not the reverse.

What belongs in `BlurtCore`: the recording/transcription state machine, the
deterministic cleanup rules (§7), config parsing and validation, the `ASREngine`
protocol and its implementations.

**Why:** state machines and cleanup are exactly the parts that must be tested
exhaustively, and they are also the parts that AppKit would make impossible to
test headlessly. v1 proved the value of this — `cleanup.py` is a pure function
with no I/O, and it has the most thorough test coverage in the project precisely
because it needs no fixtures. Keeping AppKit out of the core target is the
mechanism that preserves that property in Swift.

### 2.3 Hotkey capture: `CGEvent.tapCreate`

**Decided**, and the alternatives are ruled out for concrete reasons.

```
CGEvent.tapCreate(
  tap:     .cghidEventTap,
  options: .kCGEventTapOptionDefault,   // active tap — can modify/suppress
  mask:    keyDown | keyUp | flagsChanged
)
```

Runs on a **dedicated background thread with its own `CFRunLoop`**.

Rejected alternatives:

| Approach | Why it fails |
|---|---|
| Carbon `RegisterEventHotKey` | Cannot register modifier-only combinations. A bare-modifier trigger (e.g. hold Right Option) is unrepresentable. |
| `NSEvent` global monitors | Observation only. Cannot suppress an event, so the trigger key would also reach the focused app. |
| `CGEventTap`, listen-only option | Sees events but cannot suppress. Same problem as above. |

Only a `CGEventTap` sees `.flagsChanged`, and `.flagsChanged` is what a
bare-modifier trigger fundamentally requires — there is no keyDown for holding a
modifier alone.

The dedicated thread with its own run loop is not optional. A tap installed on
the main run loop competes with UI work, and a callback that is late enough gets
the tap forcibly disabled by the system (§3.1).

### 2.4 Suppression scope: as narrow as possible

**Swallow only Esc, and only while armed.**

Everything else passes through untouched, including the trigger key itself where
that is possible.

**Why:** an active event tap sits in the path of every keystroke on the system.
The blast radius of a bug in the suppression logic is "the user's keyboard stops
working correctly," which is unacceptable in a background utility. A narrow,
explicitly-enumerated suppression rule is auditable; a general one is not.

### 2.5 Permissions: both Input Monitoring and Accessibility

blurt v2 needs **both**, and they are not interchangeable:

- **Input Monitoring** — required to *listen* to events.
- **Accessibility** — required to *suppress* events and to *post synthetic*
  events (the paste).

Preflight both before doing anything that depends on them:

- `CGPreflightListenEventAccess()`
- `AXIsProcessTrustedWithOptions` (with the prompt option under user control,
  not fired unconditionally at launch)

**Why explicitly:** these two permissions are separately granted, separately
revocable, and produce different symptoms when missing. A build with Input
Monitoring but not Accessibility appears to work — the hotkey fires — but the
paste silently does nothing. Diagnosing that without an explicit preflight is
miserable. Surface which grant is missing, by name.

### 2.6 Stable code-signing identity from day one

**Decided, and deliberately placed at day one rather than at release.**

Ad-hoc signing produces a **new CDHash on every build**. macOS keys TCC
permission grants to code identity, so every rebuild is treated as a new
application, and previously granted Input Monitoring and Accessibility
permissions silently evaporate.

The failure mode is that the app works, then stops working after a rebuild, with
no error and no dialog — permissions that were visibly checked in System
Settings simply stop taking effect. This makes the development loop feel haunted
and burns hours before the cause is identified.

Use a stable signing identity from the first build. Which identity is a separate
question and is **open** (§6.1) — but "stable" is decided regardless of how that
resolves.

### 2.7 `ASREngine` protocol from day one

Define the `ASREngine` protocol before writing the Parakeet implementation, with
**whisper.cpp as the documented fallback** — MIT licensed, **Metal backend, not
the CoreML backend**.

**Why a protocol immediately:** v1 already has this shape
(`blurt/types.py` defines the `ASREngine` interface, `blurt/engines/__init__.py`
selects among implementations), and the discipline it enforces is worth keeping:
an engine is constructed cheaply, asked `is_available()`, and only then `load()`ed.
That separation is what allows a "preparing…" state instead of a frozen UI.

Carry over v1's rule verbatim: **never silently substitute a backend the user did
not ask for.** An explicitly configured engine that is unavailable is an error
with a stated reason, not a quiet fallback. Only `auto` may choose.

**Why Metal and not CoreML for whisper.cpp:** noted as decided by prior research.
The CoreML backend adds a model-conversion step and its own failure surface for
what is already the fallback path.

---

## 3. Known hazards

Each of these cost real debugging time to identify. They are recorded so the
cost is paid once.

### 3.1 `CGEventTap` dies silently

Taps get disabled by the system on:

- callback timeout (the callback took too long — this is the common one)
- sleep / wake
- screen lock
- fast user switching

Two disable events exist: `kCGEventTapDisabledByTimeout` and
`kCGEventTapDisabledByUserInput`. **The disable callback is not guaranteed to
arrive.** A design that relies solely on being notified will eventually end up
with a dead tap and no indication.

Required mitigations, all three:

1. **Inline re-enable on both disable types**, handled inside the tap callback.
2. **A `CGEventTapIsEnabled` poll on roughly a 5-second interval**, as the
   backstop for the un-notified case.
3. **Full tap recreation on wake**, not just re-enable. After sleep/wake the tap
   handle itself can be unusable.

Symptom when unhandled: the hotkey works, then stops after the lid is closed
once, and restarting the app "fixes" it.

### 3.2 Secure Event Input is a hard stop

When Secure Event Input is active — password fields, and terminals with Secure
Keyboard Entry enabled — **both the hotkey and the paste are blocked**. This is
the OS protecting credential entry. It is working as intended.

**Do not attempt to engineer around this.** There is no supported bypass, and an
unsupported one would be a keylogger evasion technique.

The correct handling is to detect it and **surface it as a named state** in the
UI — the user needs to understand that dictation is unavailable *right now, in
this field*, and why. Silence here reads as a broken app.

### 3.3 Clipboard restore is a race

Paste-based injection means: save the clipboard, write the transcript, send
Cmd-V, restore the clipboard. **There is no transactional guarantee.** Another
application can read or write the pasteboard in the window between steps.

Mitigation: guard the restore with `NSPasteboard.changeCount`. If the change
count is not what we left it at, someone else has written to the pasteboard —
**do not restore**, because restoring would clobber their write.

This cannot be made fully correct, only made careful. v1 exposes
`clipboard_restore_ms` as a config knob for the same reason.

### 3.4 `AVAudioEngine` silently ignores the requested tap format

Installing a tap with a requested format of 16 kHz **does not guarantee 16 kHz**.
The engine can hand back 48 kHz without error and without complaint.

If that goes unnoticed, the audio is interpreted at the wrong rate — a **3x
time-stretched signal** — which the ASR model transcribes into confident,
fluent, completely wrong text. It does not look like an audio bug. It looks like
the model is bad.

**Always convert explicitly** with an `AVAudioConverter` from the format the tap
actually delivered to the format the engine requires. Read the real format off
the tap; never assume the requested one was honored.

### 3.5 Parakeet weights are CC-BY-4.0, not MIT

The model weights are **CC-BY-4.0**. This is not a permissive-code license and
it has real obligations:

- A **`NOTICE` file with NVIDIA attribution is mandatory.**
- **Weights must not be redistributed in the repository.**

This constrains §6.3 (bundle vs fetch) — bundling in a release is a
redistribution question that must be answered against the license terms, not
against convenience.

### 3.6 Parakeet cannot disambiguate capitalization-only proper nouns

Where capitalization is the *only* cue distinguishing a proper noun from a common
one, Parakeet has no way to choose correctly. It also has **no `initial_prompt`
equivalent** — there is no mechanism to bias it toward the user's vocabulary the
way Whisper allows.

Consequence: **the custom dictionary is a requirement, not a nicety.** v1 already
has this (`Config.dictionary`, applied last in `cleanup.py` so user entries win
over sentence casing — `{"github": "GitHub"}` survives rather than being
re-cased to "Github"). That design carries over directly and becomes more
important, not less.

### 3.7 License contamination risk

The most useful native references for this problem — **VoiceInk** and
**FluidVoice** — are **GPL-family**.

**Do not read their source while implementing v2.** Not for reference, not for
"just checking how they handled the tap." If blurt is to remain permissively
licensed, the implementation must not be derived from GPL code, and the cleanest
guarantee of that is not to have read it.

Documentation, public API references, and Apple's own materials are fine.

---

## 4. What this means for the state machine

Not a decision so much as a consequence worth recording: the hazards above imply
the app has more visible states than "idle / recording / transcribing." At
minimum it must be able to say:

- permissions missing (and **which** one — §2.5)
- event tap dead / recovering (§3.1)
- Secure Event Input active (§3.2)
- model not present / downloading (§6.3)

These belong in `BlurtCore` and should be unit-tested there, which is the whole
reason for §2.2.

---

## 5. Testing posture

`BlurtCore` is testable headlessly and should be tested that way: state machine
transitions, cleanup rules, config parsing and degradation.

The parts that cannot be unit-tested — the event tap, TCC grants, Secure Event
Input, pasteboard races — need a written manual check sequence instead. Pretending
they are covered is worse than admitting they are not.

---

## 6. Open questions

**These are unresolved. No answer is implied elsewhere in this document.**

### 6.1 Apple Developer account, or self-signed?

$99/yr for a Developer ID and notarization, versus self-signed with Gatekeeper
friction for anyone who installs it.

Note this is *separate* from §2.6 — signing must be **stable** either way. This
question is only about *which* identity, and it mostly turns on whether v2 is
ever distributed to anyone other than the author.

### 6.2 Default trigger: Right Option vs fn/Globe

v1 defaults to `right_option` (`Config.hotkey`). Whether v2 keeps that or moves
to fn/Globe is open. fn/Globe has its own system-level behaviors that may
conflict; Right Option is a known quantity from v1 but is a real modifier a user
might want.

### 6.3 Bundle the model in the release, or fetch on first run?

~115–131 MB. Bundling gives a working app with no network step; fetching keeps
the release small and avoids a redistribution question. **Constrained by §3.5** —
the CC-BY-4.0 terms must be checked before bundling is even an option.

### 6.4 Minimum macOS floor

FluidAudio's ANE path requires macOS 14+, so 14 is the floor *if Parakeet is
required*. Whether to set the floor higher (for newer APIs) or support 14
specifically is undecided.

### 6.5 English-first, or multilingual on day one?

Parakeet TDT 0.6B **v3** is the multilingual line. Whether v2 ships English-only
initially — and what that implies for the cleanup rules, which are currently
English-specific in v1 — is open.

---

## 7. Relationship to v1

**v1 is not superseded. It is the cross-architecture and Intel path.**

| | v1 (Python) | v2 (Swift) |
|---|---|---|
| Architecture | x86_64 and arm64 | **arm64 only** |
| macOS floor | 13 (current dev machine) | 14+ (§6.4 open) |
| Engine | faster-whisper, apple-speech | Parakeet (ANE), whisper.cpp fallback |
| Status | working | not started |

v2 is Apple-Silicon-only **by construction**, not by neglect (§1.3). The Intel
machine cannot run it at all, so v1 remains the only thing that works there and
must keep working.

### Shared concepts — port, do not redesign

Two pieces of v1 are the product of real thought and should be **carried across
rather than reinvented**:

**1. The deterministic cleanup rules** (`blurt/cleanup.py`)

The governing principle is **asymmetric risk**: failing to clean something is a
small annoyance the user fixes in a second; deleting a word the user actually
said is a silent corruption they may not notice until it matters. Every rule is
biased toward doing nothing when unsure.

Specifically worth preserving:

- `FILLERS` is a **closed list** of non-lexical sounds (`um`, `uh`, `er`…).
- `PROTECTED_DISCOURSE_WORDS` — `like`, `so`, `right`, `well`, `you know`,
  `actually`, `basically`, `i mean` — are **never** deleted as fillers. Each is
  grammatically load-bearing. v1 enforces this with an import-time tripwire that
  raises if the two sets ever intersect. Reproduce that check in Swift.
- Self-correction is **bounded** — never past a clause boundary, never more than
  `MAX_CORRECTION_TOKENS` (8), whichever is shorter.
- The dictionary applies **last**, so user entries beat sentence casing.
- The whole module is a **pure function**. Keep it that way in `BlurtCore`.

Note that Parakeet emitting native punctuation and capitalization (§1.2) means
some of this becomes less load-bearing — the sentence-casing and abbreviation
machinery in particular. That is a reason to *re-scope* the port, not to abandon
the principles. The filler and self-correction rules are about what the user
said, not about what the model formatted, and they stay relevant.

**2. The config schema** (`blurt/config.py`)

The schema itself and, more importantly, its **degradation behavior**:

> A bad config file must never stop the user from dictating.

- Missing file → defaults, write nothing, no output at all.
- Corrupt JSON → warn, move aside to `.bak`, use defaults.
- Unknown keys → ignored (forward compatibility).
- Bad value → warn, default **for that key only**.
- Saves are atomic — temp file in the *destination directory*, then `os.replace`.
  (`/tmp` is a separate volume on modern macOS, so a temp file there would break
  atomicity. The Swift equivalent has the same constraint.)
- Config is chmod 600 — it can hold a personal replacement dictionary.

Fields carry over largely as-is: `hotkey`, `cleanup_level`, `dictionary`,
`paste_delay_ms`, `clipboard_restore_ms`. Some become meaningless (`cpu_threads`,
`model` sizing — v1's `hardware.py` exists to size a Whisper model to a slow
Intel machine, which is not a v2 problem). `engine` gains new valid values.

Whether the two versions should read the *same* config file or separate ones is
**not decided** and should be settled before either is changed.

---

*Recorded 2026-07-20, against blurt v1 at commit `3e552aa`. Written from a prior
research effort; not validated on hardware, because no such hardware is
available yet.*
