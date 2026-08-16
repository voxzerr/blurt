# blurt

Local voice dictation for macOS. Hold a key, talk, release — cleaned-up text
appears wherever your cursor is. Nothing leaves your machine, there's no
account, and there's nothing to pay.

Built because dictation is worth $180/year of value and $0/year of cost.

## Requirements

- macOS 13 (Ventura) or newer
- Python 3.9 or newer — the system `/usr/bin/python3` is fine
- A microphone

That is the whole list. No Homebrew, no cmake, no ffmpeg, no Xcode. Command Line
Tools are enough, and you probably already have them. Every dependency installs
from a prebuilt wheel; `sounddevice` bundles its own PortAudio.

blurt runs on Intel and Apple Silicon.

## Install

```sh
git clone <clone-url> blurt
cd blurt
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then:

```sh
blurt doctor     # check the machine before trusting it
blurt            # start listening
```

That is the whole install. `faster-whisper` (>= 1.2) is a base dependency, not an
extra, because there is no second engine to fall back to: the `apple-speech`
backend blurt ships is a rejection stub that always reports itself unavailable —
three independent reasons, written out in
`blurt/engines/apple_speech_engine.py`, the first being that on an Intel Mac it
can quietly fall back to server-side recognition and upload your audio. An
install without faster-whisper therefore produced a program that started, probed
the hardware, and then died with `NoEngineAvailable`.

It used to live behind a `[whisper]` extra. If your notes still say
`pip install -e ".[whisper]"`, that command still works — the extra is
deliberately kept, and is now empty, so an instruction that used to be correct
stays a harmless no-op instead of becoming an error.

The first run of a given model downloads its weights (roughly 75 MB for
`base.en`) into `~/.cache/huggingface`. That is the only network call blurt ever
makes. Once cached, the model loads with `local_files_only=True` and the network
is never touched again.

### Optional extras

```sh
pip install -e ".[dev]"      # pytest, for running the test suite
pip install -e ".[speech]"   # pyobjc-framework-Speech
```

`dev` adds pytest. It is the one to install if you are going to change anything —
see [Development](#development).

`speech` installs the PyObjC binding for Apple's `SFSpeechRecognizer`. It does
**not** give you a working second engine. The `apple-speech` backend reports
itself unavailable whether or not the binding is present, and `blurt doctor`
prints the reason next to it. Install it only if you want to re-examine that
rejection on your own machine; nothing in blurt's normal operation uses it.

### A demo that needs no microphone

```sh
bash scripts/demo.sh
```

It exercises the parts of blurt that are pure logic: configuration resolution,
the deterministic cleanup pass, the "undo that" path that takes cleanup back, and
the learning loop that turns journalled transcripts into `dictionary` and
`initial_prompt` suggestions. No microphone, no model download, no macOS — it
runs on the checkout as-is.

Which also means it demonstrates nothing about the part you will actually wait
for. It does not record audio, does not load Whisper, and says nothing about
transcription accuracy or latency on your hardware. `blurt bench` is the only
thing that answers those, because it is the only one that runs the model.

## macOS permissions

blurt needs two grants. Neither is optional, and macOS will not always ask you
for them clearly.

**Microphone** — required to record anything. macOS usually prompts the first
time blurt opens the input device. If you miss the prompt, grant it at *System
Settings → Privacy & Security → Microphone*.

**Accessibility** — required to paste. blurt inserts text by posting a synthetic
Cmd+V, and macOS refuses to deliver synthetic key events from an untrusted
process. There is usually no prompt for this one; it just silently does nothing.
Grant it at *System Settings → Privacy & Security → Accessibility*.

### The honest caveat about running from a terminal

macOS attributes permission grants to the **application that owns the process**,
not to the script. If you launch blurt from Terminal or iTerm, the grant is
recorded against Terminal or iTerm — not against blurt. Practical consequences:

- You are approving *the terminal* for microphone and accessibility access, which
  is a broader grant than you may have intended. Everything else you run from
  that terminal inherits it.
- Switching terminal apps means granting again from scratch.
- Launching blurt some other way (a `launchd` job, an editor's integrated
  terminal) is a different application as far as macOS is concerned, and starts
  with no permissions.

There is no way around this short of shipping a signed `.app` bundle. It is a
real tradeoff, not an oversight — decide whether you are comfortable with it
before granting.

### Secure input

Password fields and some full-screen apps switch macOS into *secure input* mode,
which blocks synthetic keystrokes system-wide. blurt cannot paste while that is
active, and says so rather than dropping your text silently.

## Performance

This is the section most dictation tools are vague about, so here are the numbers
measured on the slowest machine blurt is expected to work on.

**Test machine:** 2017 Intel Core i7-7567U, 2 physical cores, 16 GB RAM,
macOS 13.7.8, faster-whisper with int8 quantization, 2 threads, model resident,
best-of-5.

| Model      | 3s of speech | 11s of speech | Verdict                          |
| ---------- | ------------ | ------------- | -------------------------------- |
| `tiny.en`  | ~1.15s       | —             | ~200ms faster, measurably worse   |
| `base.en`  | ~1.35s       | ~1.80s        | **Default on Intel**             |
| `small.en` | —            | ~8.7s         | Unusable here (+20s cold load)   |

**A three-second phrase is not meaningfully faster than an eleven-second one.**
That is not a measurement error. Whisper pads every input to a fixed 30-second
window before processing it, so the model does roughly the same work whether you
spoke for two seconds or twenty. Saying "yes" costs about what a full sentence
costs. Any design that assumes short phrases feel snappy is wrong on this
architecture, and blurt does not assume it.

### Thread count is not a free knob

Worth knowing if you are tempted to tune it. On the same machine, `base.en`:

| Threads                   | 3s of speech | 11s of speech |
| ------------------------- | ------------ | ------------- |
| 2 (physical cores)        | 1.35s        | 1.80s         |
| 4 (all hyperthreads)      | 3.86s        | 10.74s        |

Doubling the threads made it roughly **six times slower** on the longer clip.
Hyperthread siblings contend for the same AVX2/FMA execution ports, and the
damage lands hardest on tail latency — the part you actually experience as "this
app is unreliable". blurt defaults to physical cores for this reason. An earlier
revision of this project used logical cores and published numbers 3× worse than
the hardware was capable of.

### Why not `tiny.en` on slow machines?

It seems obvious that the smallest model should win on the slowest hardware. It
does not. Fixed pipeline overhead dominates, so `tiny.en` buys only ~200ms — and
pays for it in accuracy. On the same clip it produced *"And so am I fellow
Americans"* where `base.en` produced *"And so, my fellow Americans!"*. `base.en`
is both the floor and the ceiling on this tier.

### What this means for you

On an older 2-core Intel Mac, expect **about a second and a half** between
releasing the key and seeing text, for anything you would say in one breath.
Accurate, private, free — but you will notice the pause. It is not instant, and
this document will not tell you otherwise.

Two things that make it worse: a busy machine (on 2 physical cores, a running
Electron app roughly doubles latency) and a cold start (the first transcription
after launch pays a one-time model load).

Apple Silicon is substantially faster and `small.en` becomes affordable there, so
it is both quicker *and* more accurate. Put plainly: **Apple Silicon is the good
experience, Intel is the supported one.** blurt detects which you have and picks
accordingly.

### Two notes on the numbers above

The table was measured at 4 threads. blurt ships with `cpu_threads = 0` (auto),
which resolves to **physical** cores only — 2 on this machine, not 4 — because
hyperthread siblings contend for the same vector execution ports and measurably
hurt tail latency. The project's recorded 2-thread figures for `base.en` are
faster than the 4-thread figures above. The table is kept as the conservative
case.

Do not take any of this on faith. Measure your own machine:

```sh
blurt bench                 # record a sample and time it
blurt bench --synth         # skip the mic, measure compute only
blurt bench --seconds 3     # check the short-utterance case yourself
```

## Troubleshooting

**Start here:**

```sh
blurt doctor
```

`doctor` reports your hardware, which engines can actually run, which model it
would pick and why, and the state of both permissions. It briefly opens the
microphone, so it verifies that the mic grant is real rather than inferring it
from a settings pane. Most problems are diagnosed by reading its output.

Common cases:

| Symptom | Likely cause |
| ------- | ------------ |
| Nothing happens on keypress | Accessibility not granted to your terminal |
| Recording works, nothing pastes | Same, or secure input is active |
| Silence recorded | Microphone not granted, or the wrong input device is selected |
| No engine available | `faster-whisper` did not install, or does not import — `doctor` prints the error |
| Long delay on first use only | One-time model download, or cold model load |
| Every phrase is slow | Expected on Intel — see Performance |

If `doctor` looks clean and it still misbehaves, run `blurt config` to confirm
which settings are actually in effect and which file they came from, or
`blurt config get KEY` for one of them on its own.

## Configuration

Settings live at `~/.config/blurt/config.json` (or under `$XDG_CONFIG_HOME`).
The file is optional; defaults apply when it is missing. A corrupt file is
renamed to `config.json.bak` and defaults are used — a bad config never stops you
dictating.

You never have to write that file by hand:

```sh
blurt config                              # everything, and the path it came from
blurt config get history_enabled          # -> false
blurt config set history_enabled true     # turn on the journal 'blurt learn' reads
blurt config set cleanup_level standard
blurt config set hotkey right_ctrl
```

`get` prints the value and nothing else — no label, no quotes, no trailing
commentary — so it can be piped or captured: `[ "$(blurt config get
history_enabled)" = true ]` does what it looks like. Booleans print as
`true`/`false`, which is exactly the spelling `set` accepts back.

`set` takes one scalar setting at a time and validates it against the rules
`load_config` already enforces rather than a second copy of them — including the
numeric ranges it does not print. It serializes the change, reads it back through
the loader, and refuses anything that would not survive the round trip, so
`blurt config set sample_rate 3` is rejected outright instead of being saved and
then silently replaced by the default on every launch. A rejected value writes
nothing at all, which includes leaving the file non-existent on a fresh install.
On/off settings accept `true/false`, `yes/no`, `on/off` or `1/0`, in any case.

Two things `set` deliberately will not do:

- **It never persists a command-line override.** `blurt --cleanup standard config
  set history_enabled true` writes the journal setting and nothing else. `set`
  reloads the file from disk rather than saving the config the current run
  resolved, because `--cleanup` is a flag you passed to try something, not a
  setting you chose.
- **It refuses `dictionary`.** That setting holds many entries and is built by
  `blurt learn --apply` from your own transcripts, which knows more about what
  belongs in there than a single command-line word can.

Changes take effect the next time blurt starts.

Every setting, and what it is for:

| Field | Type | Default | What it does |
| ----- | ---- | ------- | ------------ |
| `engine` | string | `"auto"` | ASR backend: `auto`, `faster-whisper`, or `apple-speech`. `auto` picks the best available, which today means `faster-whisper` — `apple-speech` is the rejection stub described under [Install](#install) and never runs. |
| `model` | string | `"auto"` | Whisper model: `auto`, `tiny.en`, `base.en`, `small.en`, and larger. `auto` picks by hardware tier. |
| `hotkey` | string | `"right_option"` | Push-to-talk key. One of `right_option`, `left_option`, `right_cmd`, `right_ctrl`, `right_shift`, `left_cmd`, `left_ctrl`, `left_shift`. |
| `cleanup_level` | string | `"light"` | `none` (trim only), `light` (casing, stutters, non-lexical filler, dictionary), `standard` (adds spoken punctuation and bounded self-correction). |
| `sample_rate` | int | `16000` | Capture rate in Hz. Whisper wants 16 kHz; other rates are resampled. Clamped to 8000–48000. |
| `preroll_ms` | int | `500` | Audio kept from *before* the keypress, so a word started early is not clipped. |
| `min_hold_ms` | int | `200` | Holds shorter than this are treated as an accidental tap and discarded. |
| `paste_delay_ms` | int | `120` | Wait after writing the clipboard before sending Cmd+V, so the target app sees the new contents. |
| `clipboard_restore_ms` | int | `400` | Wait after pasting before restoring your previous clipboard. |
| `cpu_threads` | int | `0` | Threads for the ASR engine. `0` means auto: physical cores, never logical. Raising it past physical cores makes things worse. |
| `keep_raw_history` | bool | `true` | Keep the pre-cleanup transcript in memory for the session, so you can see what the model actually heard. It is also what "undo that" hands back — see [Undo](#undo-getting-the-raw-transcript-back). In-memory only; writing transcripts to disk is a separate, opt-in setting (`history_enabled`). |
| `dictionary` | object | `{}` | Literal replacements applied during cleanup, e.g. `{"kubernetes": "Kubernetes"}`. Useful for names and jargon the model gets wrong the same way every time. |
| `initial_prompt` | string | `""` | Vocabulary hint passed to Whisper — names, jargon, acronyms it keeps mishearing. A *matching* prompt measurably improves accuracy; a mismatched one can hurt, so it is empty until you fill it with your own words. |
| `assistant_enabled` | bool | `true` | Enable command mode (the voice assistant). |
| `assistant_hotkey` | string | `"right_cmd"` | Hold-to-command key. Same key names as `hotkey`; must differ from it. |
| `history_enabled` | bool | `false` | Write finished dictations to a journal on disk, so `blurt learn` can suggest `dictionary` and `initial_prompt` entries. **Off by default** — this is the only thing blurt writes to disk. Turn it on with `blurt config set history_enabled true`. See below. |
| `history_limit` | int | `2000` | Most journal records to keep. Older ones are dropped. |

Anything can be overridden for a single run without editing the file:

```sh
blurt --model tiny.en --cleanup standard
blurt --hotkey right_cmd run
```

Overrides are validated. A typo exits with an error rather than quietly running
something else.

## Learning your vocabulary

Two settings do almost all the work of making blurt recognise *your* words:
`dictionary` (fix a term after the fact) and `initial_prompt` (bias Whisper
toward a term before it listens). Both are empty until you fill them, and
neither is much use if you have to guess what to put in them.

`blurt learn` closes that loop. Turn on the journal:

```sh
blurt config set history_enabled true
```

That creates `~/.config/blurt/config.json` for you if it does not exist yet, and
prints what it is about to start doing. There is no JSON to author: asking
someone to hand-write a config file at a path that does not exist on a fresh
install, correctly, before they can try a feature is the same as not shipping the
feature — which is roughly what happened to this one until `config set` existed.

Then dictate normally for a few days, and:

```sh
blurt learn              # show what it found; changes nothing
blurt learn --apply      # review each suggestion and accept the ones you want
blurt learn --apply --yes  # accept the high-confidence ones without asking
blurt learn --forget     # delete the journal
```

A real report looks like this:

```
SUGGESTIONS (3: 2 high confidence, 1 worth a look)

  dictionary -- literal replacements applied during cleanup
    [high  ] github -> GitHub
             spelled 3 ways: github (4), GitHub (2), Github (2)
    [medium] kubernetis -> kubernetes
             seen 2 time(s); 1 edit(s) from 'kubernetes', which you said 8 times

  prompt -- vocabulary hints passed to Whisper before it listens
    [high  ] Priya
             always capitalized mid-sentence (8 times across 12 dictations)
```

It also tells you which of your existing dictionary entries have never matched
anything, which is the fastest way to find one whose key is subtly wrong.

### What it can and cannot learn

blurt does not know what you *meant* to say. The obvious way to find out — watch
what you edit after the text lands — means reading your keystrokes in other
applications, which is exactly the thing blurt refuses to be. So nothing here
infers your intent, and nothing is applied without you agreeing to it.

What is visible in the transcripts alone is still worth having:

- **Spelling variance.** The engine wrote `GitHub`, `github` and `Github`. One of
  those is what you wanted, and a dictionary entry pins it. The inconsistency is
  the evidence — no guess required.
- **Your proper nouns.** A word capitalized in the *middle* of a sentence is one
  the engine believes is a name. Yours are exactly what `initial_prompt` is for.
- **Your jargon.** Words you use often that most people do not.
- **Near-misses.** A rare token a couple of edits from one you say constantly.
  This is the one genuinely speculative rule, and it never rises above `medium`.

Every suggestion carries `high` or `medium`, and that split is the safety model:
`--yes` applies only `high`. A suggestion you decline costs a keystroke; a wrong
dictionary entry silently rewrites a word in every dictation from then on, and
you might not notice which setting did it.

### The honest cost

This is the only feature in blurt that writes your speech to disk, which is why
it is off until you switch it on rather than on until you notice.

The journal lives at `~/.local/share/blurt/history.jsonl` (or under
`$XDG_DATA_HOME`), mode `0600` inside a `0700` directory. It is one JSON object
per line, so you can read, grep or edit it with the tools you already have. It
never leaves your machine — blurt has no network path to send it down. Turning
raw history off (`blurt config set keep_raw_history false`) means only cleaned
text is journalled, which weakens some of the findings; `blurt learn` says so
rather than reporting less and looking confident about it.

`blurt learn --forget` deletes it. That is a plain unlink: on a copy-on-write
filesystem like APFS it does not reliably destroy the underlying blocks, so it is
not a secure erase and is not claimed to be. FileVault is what actually solves
that.

`blurt doctor` reports the journal's state — including when it is off — so you
never have to infer whether it is running.

## How it works

1. **You hold the hotkey.** A global listener sees the modifier go down.
2. **Capture starts — with a ring buffer.** The microphone is already running and
   writing into a circular buffer, so blurt recovers the 500 ms *preceding* your
   keypress. This is why the first syllable does not get cut off. Nobody presses
   the key and then starts talking; everyone starts talking and then notices.
3. **You release the key.** Anything shorter than `min_hold_ms` is discarded as
   an accidental tap.
4. **Whisper transcribes locally.** The audio goes to faster-whisper on your CPU.
   No network, no upload, no API key. This is the step you wait for.
5. **Deterministic cleanup.** A pure function fixes whitespace and sentence
   casing, collapses stutters, drops non-lexical filler, and applies your
   dictionary. Same input, same output, every time. No model is involved.
6. **Paste.** Your current clipboard is snapshotted, the text is written to the
   pasteboard, a synthetic Cmd+V is posted, and your clipboard is restored a
   moment later. The transient write is marked with the `org.nspasteboard.*`
   conventions so well-behaved clipboard managers ignore it, and is flagged
   host-only so it does not fly off to your other devices via Universal
   Clipboard.

Pasting rather than typing is deliberate. Synthetic per-character keystrokes are
slow, mangle non-ASCII text, and break in apps with input handling of their own.

## Command mode (the voice assistant)

blurt has a second hotkey — **Right Command** by default — that treats what you
say as a *command* instead of text to paste. Hold it, speak, release:

| Say | It does |
| --- | --- |
| "schedule lunch with Sam tomorrow at noon" | Creates a real calendar event |
| "remind me to call the dentist" | Creates a reminder |
| "set a timer for 5 minutes" | Starts a timer, notifies you when it's up |
| "open Safari" | Launches the app |
| "undo that" | Gives back the raw transcript of your last *dictation* — never of a command — see [Undo](#undo-getting-the-raw-transcript-back) |
| anything it doesn't recognise | Falls back to dictation — nothing is lost |

It is **fully local**. Calendar and reminders go through macOS EventKit on your
own machine; nothing is sent anywhere. The command hotkey is deliberately
separate from the dictation hotkey so a command is never mistaken for text you
wanted typed, and the reverse.

Intent matching is deterministic and conservative — "add milk" is treated as
dictation, not a calendar event, because it names no time and no calendar. The
natural-language time parser understands "tomorrow at 2:30", "next friday",
"in half an hour", "at noon", and similar.

First use of a calendar or reminder command triggers the macOS Calendar/Reminders
permission prompt, granted (like the mic) to the terminal that launched blurt.

**Asking questions** ("what's the capital of France", "summarise this") is
scaffolded but **off** — it needs a language model, which is the one thing that
can't run well locally on older Intel Macs. That will arrive as an explicitly
opt-in mode: a local model on Apple Silicon, or the Claude API with a clear,
announced carve-out to the "nothing leaves your machine" promise. It is off by
default and never sends anything anywhere until you turn it on.

Turn command mode off, or change its key, in the config (`assistant_enabled`,
`assistant_hotkey`).

## Undo: getting the raw transcript back

blurt cleans up every dictation before it types it, and does so by default
(`cleanup_level = light`). This is the mechanism that makes leaving that on
defensible: if the cleanup pass mangles a sentence, you can get back exactly what
the engine heard.

Hold the **command-mode** hotkey — Right Command by default — say "undo that",
release. blurt inserts the raw transcript at your cursor: the engine's output
before casing, stutter collapsing, filler removal or dictionary replacement.

It rides on command mode rather than having a hotkey of its own. That costs no
extra macOS permission, adds no third key to bind and get wrong, and reuses a
capture path that already works — see
[Command mode](#command-mode-the-voice-assistant).

You can watch this without a Mac. Section 4 of `bash scripts/demo.sh` puts a
canned transcript through the real cleanup pass, the real intent router and the
real `revert_last`, and names the three pieces it has to substitute — the ASR
engine, the paste layer, the hotkeys — in the code that substitutes them.

### It undoes your last dictation, never a command

Only things you dictated are candidates. Command-mode utterances are kept out of
the buffer the undo reads from, including the "undo that" itself.

That exclusion is load-bearing rather than tidiness. "undo that" is a capture
like any other, so if commands were recorded there, the newest entry at the
moment you asked would always be the undo — and blurt would paste the words
"undo that" into your document. It compounds, too: each retry would push the
dictation you actually wanted one slot further out of reach, so you could never
speak your way back to it.

So "dictate a sentence → ask for a timer → say undo that" gives you back the
sentence, not the timer. The on-disk journal (`history_enabled`) records both
modes; the undo buffer is deliberately narrower than the journal is.

### What it does not do

blurt inserts the raw text. It **cannot delete the cleaned text it typed a moment
earlier**, so you remove that yourself. It says so at the time rather than
leaving you to notice:

```
  reverting to raw transcript (delete the cleaned text above it):
```

The reason is that `blurt.inject` exposes pasting and nothing else. There is no
path for sending backspaces, and adding one would mean guessing where the caret
now sits in an application blurt cannot see — you may have clicked elsewhere,
typed something, or switched apps between the dictation and the undo. Guess wrong
by a few characters and the deletion eats a sentence somebody wrote by hand. Two
copies of a sentence on screen is a visible problem you fix in a second; silently
deleting the wrong text in someone else's document is the failure this whole
project is arranged to avoid.

### A blocked paste is reported, and costs you nothing

The undo counts as done only when macOS accepted the paste. When the paste is
refused — secure input is active, Accessibility is not granted — blurt says so,
and does **not** spend the undo on it:

```
  The revert did NOT go through -- nothing was inserted.
  This dictation is still revertible: say it again once pasting works.
```

Fix the cause, say "undo that" again, and it works. That matters more than it
sounds: the premise of command mode is that you are looking at some *other*
application, so a cheerful "Reverted to the raw transcript." notification over a
paste that never happened would be the only thing you saw, while the raw text sat
on a clipboard blurt overwrites on your next dictation. The feature would be gone
for that dictation, permanently, and it would have said so nowhere you were
looking.

One honest ceiling on the success case: "macOS accepted the paste" is not "the
characters appeared on screen". Nothing on macOS reports the second, so blurt
does not claim it. A refusal, on the other hand, macOS *does* report — which is
why the refusal is the case that gets handled and the success is only ever
claimed as far as it is known.

### The rest of the limits, stated up front

- **`keep_raw_history` must be true** (it is, by default). The raw text is what
  revert restores; with it off there is nothing to restore, and blurt says
  `cannot revert: raw history is disabled` instead of pasting something. This is
  the in-memory session history, not the on-disk journal — undo does not need
  `history_enabled`.
- **Reverting the same dictation twice is a no-op.** The second attempt reports
  "that dictation was already reverted" and inserts nothing.
- **Command mode must be on** (`assistant_enabled`, default true). With it off,
  or with its hotkey colliding with the dictation hotkey, the undo is simply
  unreachable — which is the honest outcome; there is no key that silently does
  nothing.
- It also declines, naming the reason each time, when nothing has been dictated
  yet, and when cleanup did not change that dictation at all.

### What counts as "undo that"

Two conditions, and both have to hold:

1. **The whole utterance is the command.** Matching is anchored at both ends and
   has no wildcard in it anywhere, so a sentence that merely *contains* one of
   these phrases is never a match.
2. **At most six words**, counted after punctuation is stripped.

What follows is a list, not a grammar. Phrases that look like near neighbours of
these often do not work, and that is deliberate — see the note under "raw text"
below.

**`undo` / `revert`**, alone or followed by one object from a closed list —
`that`, `this`, `it`, `last`, `the last`, `that one`, `this one`, `the last one`,
`the last thing`, `the last dictation`, `the last transcript`, `the cleanup`:

```
undo                        revert
undo that                   revert that
undo the last one           revert the last dictation
undo the last thing         revert the cleanup
```

"undo the migration" and "revert to the previous vendor" name objects that are
not on that list, so they are typed out.

**`scratch that`**, `scratch this`, `scratch it`, `scratch that one`. Not
"scratch the plan" — wrong object — and not "scratch that itch", where the anchor
is the entire difference between it and a real command.

**Corrections**: `that's` or `it's`, then `wrong` / `not right` /
`not what I said`, then `undo` or `revert`, then optionally `it` or `that`.

```
that's wrong, undo it
it's not right, revert
that's not what I said, undo
```

The third line is where the six-word ceiling bites. *"that's not what I said,
undo it"* is seven words and is typed out, not obeyed; dropping the trailing
"it" is what makes it a command.

**`never mind`** (or `nevermind`, or `nvm`) followed by `undo` or `revert`, and
optionally `it` / `that` / `that one`. Bare "never mind" is not an undo — it is
ordinary speech, so the explicit verb is required.

**Raw text — exactly these five, and nothing adjacent to them:**

```
use the raw text
use the raw transcript
give me the raw text
show me the raw text
paste the raw version
```

"paste the raw **text**" does not work. Neither does "use the raw version",
"give me the raw transcript", "use raw" or "i want raw". There is no rule
generating those five; they are typed out one at a time in
`blurt/assistant/intents.py`, and adding a sixth means answering one question
first: is there a plausible English sentence where somebody says exactly this, as
their whole utterance, and does not mean "undo my last dictation"? "I want raw"
is a sentence about sushi, or a file format, or a camera setting, so it is not on
the list. None of these five is load-bearing anyway — "undo that" is what people
actually say, and if one of the five ever fires on real dictation the fix is to
delete the line.

Anything above may be preceded by `please`, `hey`, `ok`/`okay`, `so`, `just`,
`um` or `uh`, and followed by `please`, as long as the whole thing stays inside
six words: *"please give me the raw text"* is a command, *"please give me the raw
text please"* is seven words and is typed out. Casing, spacing and trailing
punctuation do not matter — "Undo that." is "undo that".

Sentences that merely contain the word are dictated as text, which is what you
want. All of these get typed, not obeyed:

```
undo the last commit in git and force push
I need to undo the migration before the deploy, can you note that
we should revert to the previous vendor
the revert button is greyed out in the admin panel
scratch that itch
```

That strictness is the asymmetric-risk rule pointing the other way for once.
Everywhere else in blurt the cheap mistake is failing to act — you lose a second
and say it again. Here a false positive pastes a stale transcript into whatever
you happened to be typing, and for the reason above blurt cannot take it back. A
missed undo costs a repeat; a spurious one costs you a document.

## What it deliberately does not do

Each of these is a decision, not a missing feature.

**No LLM rewriting.** blurt will not send your transcript to a language model to
"clean it up". That changes your words into words you did not say, plausibly
enough that you may not catch it. Dictation should produce your sentences. The
cleanup pass is a set of narrow, auditable, deterministic rules, and when it is
unsure it does nothing.

**No network.** Beyond the one-time model download, blurt makes no network calls
at all. Once the model is cached, it loads with `local_files_only=True`.

**No telemetry.** No analytics, no crash reporting, no usage counters, no phoning
home. Nothing to opt out of, because there is nothing there.

**No account.** No sign-up, no licence key, no subscription, no cloud tier.

**Never presses Enter.** blurt inserts text and stops. It will not submit your
message, send your email, or run your command. An auto-send that fires one word
early is unrecoverable; a Return you press yourself never is.

**Never strips meaningful words.** Filler removal is limited to non-lexical
sounds — "um", "uh" and their relatives. Words like *like*, *so*, *well*,
*actually* and *right* are left exactly where you said them. They carry hedging,
tone and emphasis, and a tool that quietly deletes them is editing your voice
rather than transcribing it.

The governing principle throughout is asymmetric risk. Failing to clean something
costs you a second of editing. Deleting something you actually said is a silent
corruption you might not notice until it matters.

## Development

```sh
python3 -m pytest -q
```

No install required. `tests/conftest.py` puts the repository root on `sys.path`,
so `import blurt` resolves straight from the checkout — which is not a test
convenience, it is how blurt actually runs on the floor machine, where there is
no Homebrew, no virtualenv and no `pip install` of the project itself. You need
`pytest` and `numpy` and nothing else; `pip install -e ".[dev]"` gets pytest.

The suite is deliberately hardware-free: no microphone, no model download, no
network, no Accessibility or Microphone prompts. That is what lets it run
unattended, and on Linux. A test that needs a real device belongs behind a skip,
not in the default run.

### The Python 3.9 floor

3.9 is not the oldest version grudgingly tolerated — it is the version the
primary target is on. Apple's `/usr/bin/python3` on macOS 13 is 3.9.6, and blurt
is expected to work there with nothing installed but its own dependencies. Newer
syntax does not degrade on that interpreter; it raises `SyntaxError` at import,
which means the app does not start at all. So in every module under `blurt/`:

- `from __future__ import annotations` at the top, without exception.
- `typing.Dict` / `List` / `Optional` / `Tuple`, never a builtin generic
  evaluated at runtime (`x = dict[str, str]`, `cast(list[int], v)`).
- No PEP 604 unions — `Optional[X]`, not `X | None`.
- No `match` / `case`.

### CI

`.github/workflows/ci.yml` runs the suite on `ubuntu-latest` and `macos-latest`
across Python 3.9 – 3.13. The macOS legs skip 3.9, 3.10 and 3.12: there is no
arm64 macOS build of 3.9 to install, and what those legs are for is catching
platform assumptions rather than repeating version coverage. Seven jobs in total.

A second, cheaper job guards the floor without waiting on a 3.9 interpreter: it
byte-compiles every module, checks that each one carries the future import, and
greps for the constructs listed above.

CI never installs blurt itself, and a change should not make it need to. Two
reasons, either sufficient: the pyobjc wheels are macOS-only, so `pip install -e .`
cannot resolve on a Linux runner at all; and `faster-whisper` is a base
dependency now rather than an extra, so any install drags in CTranslate2 and its
shared objects on every leg of the matrix, none of which the suite imports.

## Credits and licence

blurt is MIT licensed. See [LICENSE](LICENSE).

**Speech recognition** is [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(MIT), a CTranslate2 reimplementation of OpenAI's Whisper inference. blurt calls
it as a library and vendors none of its code.

**Whisper models** are OpenAI's, released under the MIT licence. The converted
weights are downloaded from Hugging Face on first use. blurt does not
redistribute them.

**Other dependencies:** `sounddevice` (MIT, bundling PortAudio under the MIT
licence), `pynput` (LGPL 3.0, used as an unmodified library dependency), `numpy`
(BSD 3-Clause), and the PyObjC frameworks (MIT).

**On originality:** blurt contains no code copied from any other dictation
project — not from Whisper wrappers, not from commercial dictation tools, not
from GPL-licensed projects. Where a well-known approach was the right one —
hold-to-talk with a pre-roll ring buffer, paste-and-restore text injection — the
technique was studied and then implemented from scratch here. This is stated
plainly because local Whisper dictation is a crowded space and the question is a
fair one to ask.
