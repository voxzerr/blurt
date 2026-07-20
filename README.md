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
pip install -e ".[whisper]"
```

Then:

```sh
blurt doctor     # check the machine before trusting it
blurt            # start listening
```

The `[whisper]` extra pulls in `faster-whisper`, which is the engine you want.
Installing without it gives you the CLI and the Apple Speech engine but no local
Whisper.

The first run of a given model downloads its weights (roughly 75 MB for
`base.en`) into `~/.cache/huggingface`. That is the only network call blurt ever
makes. Once cached, the model loads with `local_files_only=True` and the network
is never touched again.

### Optional extras

```sh
pip install -e ".[whisper,speech]"   # adds the Apple Speech engine
pip install -e ".[whisper,dev]"      # adds pytest
```

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
| No engine available | `pip install -e ".[whisper]"` was skipped |
| Long delay on first use only | One-time model download, or cold model load |
| Every phrase is slow | Expected on Intel — see Performance |

If `doctor` looks clean and it still misbehaves, run `blurt config` to confirm
which settings are actually in effect and which file they came from.

## Configuration

Settings live at `~/.config/blurt/config.json` (or under `$XDG_CONFIG_HOME`).
The file is optional; defaults apply when it is missing. A corrupt file is
renamed to `config.json.bak` and defaults are used — a bad config never stops you
dictating.

| Field | Type | Default | What it does |
| ----- | ---- | ------- | ------------ |
| `engine` | string | `"auto"` | ASR backend: `auto`, `faster-whisper`, or `apple-speech`. `auto` picks the best available. |
| `model` | string | `"auto"` | Whisper model: `auto`, `tiny.en`, `base.en`, `small.en`, and larger. `auto` picks by hardware tier. |
| `hotkey` | string | `"right_option"` | Push-to-talk key. One of `right_option`, `left_option`, `right_cmd`, `right_ctrl`, `right_shift`, `left_cmd`, `left_ctrl`, `left_shift`. |
| `cleanup_level` | string | `"light"` | `none` (trim only), `light` (casing, stutters, non-lexical filler, dictionary), `standard` (adds spoken punctuation and bounded self-correction). |
| `sample_rate` | int | `16000` | Capture rate in Hz. Whisper wants 16 kHz; other rates are resampled. Clamped to 8000–48000. |
| `preroll_ms` | int | `500` | Audio kept from *before* the keypress, so a word started early is not clipped. |
| `min_hold_ms` | int | `200` | Holds shorter than this are treated as an accidental tap and discarded. |
| `paste_delay_ms` | int | `120` | Wait after writing the clipboard before sending Cmd+V, so the target app sees the new contents. |
| `clipboard_restore_ms` | int | `400` | Wait after pasting before restoring your previous clipboard. |
| `cpu_threads` | int | `0` | Threads for the ASR engine. `0` means auto: physical cores, never logical. Raising it past physical cores makes things worse. |
| `keep_raw_history` | bool | `true` | Keep the pre-cleanup transcript in memory for the session, so you can see what the model actually heard. Never written to disk. |
| `dictionary` | object | `{}` | Literal replacements applied during cleanup, e.g. `{"kubernetes": "Kubernetes"}`. Useful for names and jargon the model gets wrong the same way every time. |

Anything can be overridden for a single run without editing the file:

```sh
blurt --model tiny.en --cleanup standard
blurt --hotkey right_cmd run
```

Overrides are validated. A typo exits with an error rather than quietly running
something else.

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
