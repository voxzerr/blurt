#!/usr/bin/env bash
#
# blurt demo -- prove the text pipeline works, on any machine, right now.
#
#   bash scripts/demo.sh          narrated, with pauses between sections
#   NO_PAUSE=1 bash scripts/demo.sh   same output, no waiting (CI, piping to a file)
#
# Invoke it with `bash scripts/demo.sh` rather than `./scripts/demo.sh`. The
# executable bit is a single mode bit that git records and many checkouts,
# archives and copy-paste installs quietly drop; `bash <path>` works either way,
# so the instructions everywhere say that.
#
# WHAT THIS DEMONSTRATES, AND WHAT IT CANNOT
# ------------------------------------------
# blurt is two halves. The half below the microphone -- capture, Whisper, the
# global hotkey, the synthetic paste -- needs a Mac, a real mic, a downloaded
# model, and permissions a human has to grant in System Settings. None of that
# can run in a container, and pretending otherwise would be the one thing this
# script must not do.
#
# The other half is pure text: cleanup, the undo that takes cleanup back,
# configuration, the transcript journal and the learning loop that reads it. That
# half has no hardware in it at all, which is exactly why it was built that way --
# it is the part that can be proven rather than described. This script proves it
# end to end, and section 7 names every single thing it did not touch.
#
# Section 4 is the one place this script replaces a piece of blurt rather than
# running it, because the undo path ends at a macOS paste. Three seams are
# substituted there -- the ASR engine, the paste layer, the hotkeys -- each named
# in the snippet that does it, with the real cleanup pass, the real revert buffer,
# the real IntentRouter and the real revert_last running between them. Every line
# of output in that section came out of blurt's own code; none of it is printed
# to look like a result.
#
# WHY IT BUILDS ITS OWN HOME
# --------------------------
# Section 1 points XDG_CONFIG_HOME and XDG_DATA_HOME at a throwaway directory
# before anything else runs. blurt derives both of the files it ever writes --
# config.json and history.jsonl -- from exactly those two variables (see
# blurt/config.py:default_config_path and blurt/history.py:default_history_path),
# so redirecting them is total isolation rather than a best effort. This matters
# more here than in most demos: section 5 fills the transcript journal and
# section 6 deletes it, and a demo that did either of those to a real user's
# machine would be indefensible. The demo can only ever destroy its own data.
#
# HONESTY RULES THIS SCRIPT FOLLOWS
# ---------------------------------
#   * Every command is echoed before it runs, so what you read is what executed.
#   * stderr is never swallowed. If blurt warns about something, you see it.
#   * Nothing is installed, downloaded, or fetched. There is no network call in
#     here, and blurt has no code path that would make one for these commands.
#   * blurt is invoked as `python3 -m blurt` from the repo root, which is how the
#     test suite imports it, so this runs on a fresh clone with nothing installed.
#
# Portability notes: bash 3.2 (what macOS ships) has everything used here --
# no associative arrays, no `mapfile`, no `${var@Q}`. `mktemp -d` is given an
# explicit template because BSD mktemp and GNU mktemp disagree about defaults.

set -euo pipefail


# --------------------------------------------------------------------------- #
# Where we are
# --------------------------------------------------------------------------- #

# Resolved from the script's own location, never from the caller's cwd: this has
# to work identically from the repo root, from scripts/, and from a launcher that
# starts somewhere else entirely.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [ ! -d "${REPO_ROOT}/blurt" ]; then
    echo "demo.sh: cannot find the blurt package next to ${SCRIPT_DIR}" >&2
    echo "         (expected ${REPO_ROOT}/blurt). Run this from a blurt checkout." >&2
    exit 1
fi

# `python3 -m blurt` resolves the package from the current directory, so the demo
# needs no install, no venv and no PYTHONPATH -- the same import path the tests use.
cd "${REPO_ROOT}"

PY_BIN="${PYTHON:-python3}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
    echo "demo.sh: ${PY_BIN} not found on PATH" >&2
    exit 1
fi


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #

# Colour only when stdout is a terminal that wants it. Piping this into a file or
# a CI log has to produce clean text, because a wall of escape sequences in a
# build log is worse than no colour at all. NO_COLOR is honoured as well; it is
# the closest thing to a standard for this.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    C_OFF=$'\033[0m'
    C_BAR=$'\033[38;5;39m'
    C_HEAD=$'\033[1;38;5;39m'
    C_CMD=$'\033[1;32m'
    C_SRC=$'\033[2;37m'
    C_SAY=$'\033[0m'
    C_KEY=$'\033[1;33m'
    C_NOTE=$'\033[2m'
else
    C_OFF=""; C_BAR=""; C_HEAD=""; C_CMD=""; C_SRC=""; C_SAY=""; C_KEY=""; C_NOTE=""
fi

BAR="────────────────────────────────────────────────────────────────────────"

# Between sections, so a human watching live can actually read one before the
# next scrolls past. NO_PAUSE=1 removes every one of them: CI has no eyes, and a
# test that takes an extra minute to sleep is a test people stop running.
pause() {
    if [ "${NO_PAUSE:-0}" != "1" ]; then
        sleep "${1:-1.2}"
    fi
}

banner() {
    local number="$1"
    shift
    printf '\n%s%s%s\n' "${C_BAR}" "${BAR}" "${C_OFF}"
    printf '%s  %s.  %s%s\n' "${C_HEAD}" "${number}" "$*" "${C_OFF}"
    printf '%s%s%s\n\n' "${C_BAR}" "${BAR}" "${C_OFF}"
}

say() {
    printf '%s%s%s\n' "${C_SAY}" "$*" "${C_OFF}"
}

note() {
    printf '%s%s%s\n' "${C_NOTE}" "$*" "${C_OFF}"
}

# Echo the command, then run it. The echo is the point: a demo whose output you
# cannot map back to a command you could type yourself is a screenshot, not a
# proof. "$*" is safe to display here because every command in this script is
# built from arguments without embedded whitespace.
run() {
    printf '\n%s$ %s%s\n' "${C_CMD}" "$*" "${C_OFF}"
    "$@"
}

# Same contract as run(), for a python snippet fed on stdin: print the exact
# source that is about to execute, then execute that source and nothing else.
# The snippets are short on purpose -- they are meant to be read, and a reader
# who cannot check them has to take the output on faith.
run_py() {
    local comment="$1"
    local source line
    source="$(cat)"

    printf '\n%s$ %s - <<%sPY%s%s   %s# %s%s\n' \
        "${C_CMD}" "${PY_BIN}" "'" "'" "${C_OFF}" "${C_NOTE}" "${comment}" "${C_OFF}"
    # Piped rather than fed through a here-string or here-document: those expand
    # `$` and backticks, so a snippet containing either would be DISPLAYED
    # differently from how it RUNS. That is the one bug this helper must not
    # have, since its entire job is to show you what is about to execute.
    printf '%s\n' "${source}" | while IFS= read -r line; do
        printf '%s%s%s\n' "${C_SRC}" "${line}" "${C_OFF}"
    done
    printf '%sPY%s\n' "${C_CMD}" "${C_OFF}"

    printf '%s\n' "${source}" | "${PY_BIN}" -
}


# --------------------------------------------------------------------------- #
# Opening: what this is, before anything runs
# --------------------------------------------------------------------------- #

printf '\n%s%s%s\n' "${C_HEAD}" "  blurt -- demo" "${C_OFF}"
printf '%s%s%s\n' "${C_BAR}" "${BAR}" "${C_OFF}"
say ""
say "  This demo runs the half of blurt that is pure text: cleanup, the \"undo"
say "  that\" escape hatch that makes cleanup-by-default defensible,"
say "  configuration, the opt-in transcript journal, and the learning loop"
say "  that turns that journal into a personal vocabulary."
say ""
say "  It needs no microphone, no downloaded model, no pynput, no pyobjc and"
say "  no macOS permission, so it runs on this machine exactly as it runs on"
say "  a Mac -- and everything you are about to see is really executing."
say ""
say "  ${C_KEY}Live dictation is NOT demonstrated here.${C_OFF}${C_SAY} That half needs a Mac with a"
say "  microphone and the faster-whisper model on disk. Section 7 lists"
say "  precisely what was left out and how to try it there."
say ""
note "  (NO_PAUSE=1 skips the pauses between sections.)"

pause 2.0


# --------------------------------------------------------------------------- #
# 1. SETUP
# --------------------------------------------------------------------------- #

banner 1 "SETUP -- a throwaway home, so your real config is untouchable"

say "blurt writes two files -- config.json and history.jsonl -- and it finds"
say "both of them through XDG_CONFIG_HOME and XDG_DATA_HOME. Point those at a"
say "temporary directory and the demo is unable to read or write your real"
say "settings or your real transcripts: not by convention, by path resolution."

DEMO_HOME="$(mktemp -d "${TMPDIR:-/tmp}/blurt-demo.XXXXXXXX")"

# Registered on the line after mktemp, before anything else can fail, so there
# is no window in which the directory exists and nothing is responsible for
# removing it. `set -e` makes an early exit likely enough to care about, and EXIT
# covers the ordinary end, the failed command, and Ctrl-C alike. The status is
# captured and returned so a demo that dies half way still dies with the exit
# code that killed it rather than the exit code of `rm`.
_demo_cleanup() {
    local status=$?
    if [ -n "${DEMO_HOME:-}" ] && [ -d "${DEMO_HOME}" ]; then
        rm -rf "${DEMO_HOME}"
        printf '\n%sremoved the demo home: %s%s\n' "${C_NOTE}" "${DEMO_HOME}" "${C_OFF}"
    fi
    return "${status}"
}
trap _demo_cleanup EXIT

export XDG_CONFIG_HOME="${DEMO_HOME}/config"
export XDG_DATA_HOME="${DEMO_HOME}/data"
mkdir -p "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}"

printf '\n'
printf '  repo root        : %s%s%s\n' "${C_KEY}" "${REPO_ROOT}" "${C_OFF}"
printf '  demo home        : %s%s%s\n' "${C_KEY}" "${DEMO_HOME}" "${C_OFF}"
printf '  XDG_CONFIG_HOME  : %s%s%s\n' "${C_KEY}" "${XDG_CONFIG_HOME}" "${C_OFF}"
printf '  XDG_DATA_HOME    : %s%s%s\n' "${C_KEY}" "${XDG_DATA_HOME}" "${C_OFF}"
printf '\n'
note "That directory is deleted when this script exits, however it exits."

pause


# --------------------------------------------------------------------------- #
# 2. CONFIG
# --------------------------------------------------------------------------- #

banner 2 "CONFIG -- resolved defaults, and changing one without editing JSON"

say "There is no config file yet. blurt prints what it would use anyway, says"
say "so, and resolves the 'auto' settings against this actual machine."

run "${PY_BIN}" -m blurt config

pause

say ""
say "The transcript journal is off, as it always is until someone turns it on."
say "'config set' is the supported way to do that -- no hand-authored JSON at a"
say "path that does not exist yet."

run "${PY_BIN}" -m blurt config set history_enabled true

run "${PY_BIN}" -m blurt config get history_enabled

say ""
note "'get' prints the value alone, unlabelled and unquoted, so a script can"
note "capture it. That 'true' came back off disk: the setting stuck."
say ""
note "The \"initial_prompt is empty\" line above is stderr, and it is a real"
note "cosmetic wart rather than a problem with your config: config.json now"
note "exists and stores initial_prompt as \"\", which is also its default, and"
note "the loader announces the fallback instead of staying quiet about a value"
note "that did not actually change. It goes away in section 5, the moment the"
note "prompt is non-empty. This demo does not hide stderr, so you get to see it."

pause


# --------------------------------------------------------------------------- #
# 3. CLEANUP
# --------------------------------------------------------------------------- #

banner 3 "CLEANUP -- the deterministic pass between Whisper and your cursor"

say "blurt.cleanup.clean() is a pure function: no model, no randomness, no I/O."
say "The same raw transcript always produces the same text, which is the only"
say "reason it is allowed to run without the user reviewing it first."
say ""
say "Below, five raw ASR outputs at each of the three levels. Watch row 5"
say "especially -- it is the one where cleanup does nothing on purpose."

run_py "blurt.cleanup.clean() at all three levels" <<'PY'
from blurt.cleanup import clean

DICTIONARY = {"github": "GitHub"}

EXAMPLES = [
    ("filler sounds and a stutter", "um so the the deploy uh finished cleanly"),
    ("spoken punctuation", "open the pull request comma then ping me period"),
    ("sentence casing", "we shipped it. the rollout looked fine"),
    ("dictionary substitution", "the github action failed again"),
    ("RESTRAINT: like / so / actually",
     "actually, I like how the retry logic works, so we should keep it"),
]

for number, (label, raw) in enumerate(EXAMPLES, 1):
    print("")
    print("  %d. %s" % (number, label))
    print("     raw       | %s" % raw)
    for level in ("none", "light", "standard"):
        print("     %-9s | %s" % (level, clean(raw, level, DICTIONARY)))
PY

say ""
say "Read row 1 and row 5 together, because between them they are the whole"
say "product promise."
say ""
say "  Row 1: 'um' and 'uh' are gone and 'the the' collapsed -- but 'so'"
say "         survived. It is not a filler; it is a conjunction."
say "  Row 5: 'actually', 'like' and 'so' are all still there, untouched, at"
say "         every level. blurt will not delete them and never learns to."
say ""
say "That is deliberate and it is enforced in code: blurt/cleanup.py keeps a"
say "PROTECTED_DISCOURSE_WORDS set and raises at import time if any of those"
say "words is ever added to the filler list. Every one of them is load-bearing"
say "English -- 'I like it', 'so I left', \"it's actually cheaper\" -- and there"
say "is no way to tell the filler use from the meaning-bearing use without a"
say "parser we do not have."
say ""
note "Failing to clean something costs you a second. Deleting a word you"
note "actually said is silent and unrecoverable. Every rule is biased toward"
note "the first failure."

pause


# --------------------------------------------------------------------------- #
# 4. UNDO
# --------------------------------------------------------------------------- #

banner 4 "UNDO -- \"undo that\", the way back out of section 3"

say "Section 3 argues that cleanup is conservative enough to be allowed to run"
say "by default. This section is the other half of that argument: that it is"
say "reversible. Conservative is not the same as never wrong."
say ""
say "On a Mac you hold the ${C_KEY}command${C_OFF}${C_SAY} hotkey -- Right Command by default, the"
say "second key, not the dictation one -- say \"undo that\", and release. blurt"
say "pastes the raw transcript: exactly what the engine heard, before casing,"
say "stutter collapsing, filler removal or your dictionary touched it."
say ""
say "From the microphone inwards that path is ordinary software, so it can run"
say "here. Three pieces of it cannot, and each is replaced ${C_KEY}by name${C_OFF}${C_SAY} in the code"
say "below rather than being papered over:"
say ""
say "  ${C_KEY}the ASR engine${C_OFF}${C_SAY}     needs model weights on disk. Replaced by an object"
say "                     returning a canned string, so \"what Whisper heard\" is"
say "                     a literal you can read in the source."
say "  ${C_KEY}the paste layer${C_OFF}${C_SAY}    needs pyobjc and Accessibility. Replaced by a"
say "                     recorder that prints what it was handed and reports"
say "                     success -- which is all insert_text ever reports."
say "  ${C_KEY}the two hotkeys${C_OFF}${C_SAY}    need pynput. Not built at all; the capture handler"
say "                     is called directly, which is exactly what a hotkey"
say "                     callback does the moment you release the key."
say ""
say "Everything between them is the shipping code: the real cleanup pass, the"
say "real revert buffer, the real IntentRouter with the real RevertHandler in"
say "front of it, and the real BlurtApp.revert_last."

run_py "four utterances through the real capture handler, router and revert" <<'PY'
import blurt.app as app_module
from blurt.app import BlurtApp
from blurt.assistant import build_default_router, system_actions
from blurt.config import Config

# ---- the three macOS seams. Nothing else in this snippet is a stand-in. ----
PASTED = []


def paste(text, paste_delay_ms=120, restore_delay_ms=400):
    """Stands in for blurt.inject.insert_text: pasteboard write + Cmd-V."""
    PASTED.append(text)
    print("    >> paste layer was handed: %r" % text)
    return True


class CannedEngine(object):
    """Stands in for faster-whisper: returns whatever we say was heard."""

    name = "canned"
    text = ""

    def transcribe(self, pcm, sample_rate):
        return self.text


class Capture(object):
    """Stands in for the recorder's PCM buffer; only .shape is ever read."""

    shape = (16000,)


app_module.insert_text = paste
system_actions.notify = lambda title, message: None   # osascript: macOS only

engine = CannedEngine()
app = BlurtApp(Config())            # stock config: cleanup_level "light"
app._engine = engine
app._engine_label = "canned transcript"
# The identical call BlurtApp._build_assistant makes. Only its hotkey is skipped.
app._router = build_default_router(
    dictate_fallback=app._deliver_as_result,
    revert_fn=app.revert_last,
)


def hold(key, mode, words):
    """What a hotkey callback does on release: hand the capture to the app."""
    print("")
    print("  [hold the %s key and say] %r" % (key, words))
    engine.text = words
    app._handle_capture(Capture(), False, False, mode)
    print("  revert buffer now holds: %r" % [t.cleaned for t in app.history])


hold("DICTATION", "dictate", "um we need to go go go on this one")
hold("COMMAND", "assistant",
     "I need to undo the migration before the deploy, can you note that")
hold("COMMAND", "assistant", "undo that")
hold("COMMAND", "assistant", "undo that")

print("")
print("everything the paste layer was handed, in order:")
for number, text in enumerate(PASTED, 1):
    print("  %d. %r" % (number, text))
PY

pause 2.0

say ""
say "Four things happened there, and every one of them is the point."
say ""
say "  ${C_KEY}1. cleanup ate an emphasis.${C_OFF}${C_SAY} \"go go go\" became \"go\". The stutter"
say "     collapser cannot tell a disfluency from three deliberate words, and"
say "     that is a real limit rather than a bug this demo dressed up: the"
say "     sentence you dictated is not the sentence that landed."
say "  ${C_KEY}2. the false positive did not fire.${C_OFF}${C_SAY} \"I need to undo the migration"
say "     before the deploy, can you note that\" was said in COMMAND mode, which"
say "     is the only place the revert matcher gets a vote at all -- and it was"
say "     typed out verbatim. That sentence is the whole engineering argument."
say "  ${C_KEY}3. \"undo that\" gave back the raw text.${C_OFF}${C_SAY} Not the cleaned line, not the"
say "     command: the original \"um we need to go go go on this one\"."
say "  ${C_KEY}4. the second \"undo that\" was refused.${C_OFF}${C_SAY} One revert per dictation, and"
say "     blurt says why on stderr instead of pasting a second copy."
say ""
say "Watch the revert buffer line between the utterances. It never changes when"
say "a COMMAND is spoken -- neither the sentence about the migration nor the"
say "\"undo that\" itself is ever in it. That is deliberate, and blurt/app.py"
say "spends a paragraph on why: a command that entered the buffer would become"
say "the target of the next revert, so saying \"undo that\" twice would paste the"
say "words \"undo that\" into your document. Only dictations are revertible; the"
say "on-disk journal is where a record of both modes belongs."

pause

say ""
say "Now the matcher on its own, with no app and no paste anywhere near it --"
say "just the yes/no question, asked of seventeen phrases."

run_py "RevertHandler.match(): command, or text to be typed?" <<'PY'
from blurt.assistant.intents import RevertHandler

# The callable is never invoked here: matching is pure inspection, and only the
# caller ever decides to cross the line into executing what was matched.
handler = RevertHandler(lambda: True)

MEANT_AS_A_COMMAND = [
    "undo",
    "undo that",
    "scratch that",
    "revert the last dictation",
    "that's wrong, undo it",
    "never mind undo",
    "give me the raw text",
    "please undo that",
    "Undo that.",
]

MEANT_AS_TEXT = [
    "I need to undo the migration before the deploy, can you note that",
    "undo the last commit in git and force push",
    "we should revert to the previous vendor",
    "the revert button is greyed out in the admin panel",
    "scratch that itch",
    "please give me the raw text please",
    "paste the raw text",
    "i want raw",
]

for label, phrases in (("SPOKEN AS A COMMAND", MEANT_AS_A_COMMAND),
                       ("SPOKEN AS TEXT", MEANT_AS_TEXT)):
    print("")
    print("  %s" % label)
    for phrase in phrases:
        action = handler.match(phrase)
        if action is None:
            verdict = "dictated verbatim"
        else:
            verdict = "revert (%.2f)" % action.confidence
        print("    %-18s | %s" % (verdict, phrase))
PY

say ""
say "Read the second block. Two of those sentences OPEN with a revert verb, and"
say "\"i want raw\" is three words -- shorter than phrases in the first block that"
say "do match. All eight are typed out as text anyway. Three mechanisms do that,"
say "and they are independent on purpose:"
say ""
say "  ${C_KEY}the anchor${C_OFF}${C_SAY}       the pattern is ^...\$ with no wildcard anywhere in it,"
say "                   so a match is never a substring of a longer sentence."
say "                   \"scratch that\" is a command; \"scratch that itch\" is one"
say "                   word longer and is not. The anchor is the whole"
say "                   difference between them."
say "  ${C_KEY}the ceiling${C_OFF}${C_SAY}      at most six words, counted after normalization."
say "                   \"please give me the raw text please\" satisfies the"
say "                   pattern and is turned down on word count alone -- a"
say "                   second, cruder net that keeps holding if the first is"
say "                   ever loosened."
say "  ${C_KEY}closed lists${C_OFF}${C_SAY}     \"paste the raw version\" is accepted and \"paste the"
say "                   raw text\" is not, which looks arbitrary and is not:"
say "                   there is no grammar generating those. The five raw-text"
say "                   phrasings are typed out one at a time in"
say "                   blurt/assistant/intents.py, so the vocabulary can only"
say "                   grow by someone adding a line and answering for it."
say ""
note "This is the one place in blurt where the asymmetry runs backwards."
note "Everywhere else, failing to act costs a second and a repeat. Here, acting"
note "wrongly pastes a stale transcript into whatever you were typing -- and"
note "blurt.inject can paste but cannot delete, so it stays there. A missed undo"
note "costs a repeat; a spurious one costs you a document. Hence a vocabulary"
note "small enough to be listed on one screen."

pause


# --------------------------------------------------------------------------- #
# 5. LEARNING LOOP
# --------------------------------------------------------------------------- #

banner 5 "LEARNING LOOP -- blurt reads its own journal and proposes a vocabulary"

say "Journalling is on now, so here are six days of dictation about a fictional"
say "project. These are written straight into the journal with the same"
say "blurt.history.append_record() the dictation worker calls, so what 'learn'"
say "reads next is indistinguishable from a real week of talking."
say ""
say "The lines are ordinary work chatter with four things hidden in them --"
say "see if you can spot them before blurt does."

run_py "seed 40 dictations into the journal in the demo home" <<'PY'
import time

from blurt.cleanup import clean
from blurt.history import HistoryRecord, append_record, default_history_path

# What the engine produced, warts and all: inconsistent casing of a product
# name, a colleague's name mid-sentence, one misrecognition, some jargon, and
# the fillers a real transcript is full of.
LINES = [
    "The github action failed on the um staging cluster again.",
    "Naomi said the GitHub token expired overnight.",
    "The Github issue about the flaky test is still open.",
    "I asked Naomi to look at the webhook retries before standup.",
    "We should uh ask Naomi once github is back up.",
    "The kubernetes rollout for Halyard finished cleanly.",
    "Restart kubernetes and check the pods on the staging cluster.",
    "The kubernetes scheduling is fine, the disk is just slow.",
    "The kubernetis node is is flapping again this morning.",
    "Halyard is now doing the backfill in batches of five thousand.",
    "The backfill for Halyard finished in about forty minutes.",
    "I told Naomi the backfill would need a second pass.",
    "The webhook delivery is retried um three times with backoff.",
    "Our webhook endpoint is returning a five oh two under load.",
    "The grafana dashboard shows throughput dropping after midnight.",
    "Check grafana before you page anyone about throughput.",
    "The throughput on the ingest path is uh about nine thousand a second.",
    "Halyard writes to postgres and then fans out to the queue.",
    "The postgres replica lag went up during the backfill.",
    "We moved the postgres connection pool into the sidecar.",
    "Naomi rewrote the idempotency key so the retries are safe.",
    "The idempotency check is what makes the webhook replay harmless.",
    "I pushed the branch to github and um opened a draft pull request.",
    "The github review is blocked on the failing integration test.",
    "Ask Naomi whether the Halyard cutover is still on for Friday.",
    "The Halyard migration needs kubernetes to drain the old nodes.",
    "We are running kubernetes one point twenty nine in staging.",
    "The grafana alert fired twice during the the postgres failover.",
    "I want the throughput graph next to the error rate in grafana.",
    "Halyard should reject the duplicate webhook without an error.",
    "The backfill job is idempotent so we can just run it again.",
    "Naomi is writing the runbook for the Halyard cutover.",
    "The staging cluster lost a node and kubernetes rescheduled everything.",
    "I left a comment on the github pull request about the uh retries.",
    "The postgres vacuum is what caused the throughput dip.",
    "We should page Naomi if the webhook queue backs up again.",
    "The Halyard dashboard in grafana needs the new latency panel.",
    "I am going to rerun the backfill after the uh postgres upgrade.",
    "The kubernetes operator restarts the ingest pods on config change.",
    "Naomi wants the github checks green before the Halyard cutover.",
]

# Spread evenly across the last six days, so the "span" the report prints is
# the one the narration claims.
DAYS = 6.0
start = time.time() - DAYS * 86400.0
step = DAYS * 86400.0 / len(LINES)

for index, raw in enumerate(LINES):
    append_record(
        HistoryRecord(
            timestamp=start + index * step,
            mode="dictate",
            raw=raw,                       # what the engine heard
            cleaned=clean(raw, "light"),   # what was actually typed
            engine="faster-whisper base.en",
            audio_seconds=3.4,
            latency_seconds=0.9,
        ),
        limit=0,
    )

print("wrote %d records to %s" % (len(LINES), default_history_path()))
PY

pause

say ""
say "Now let blurt read it. This is read-only: 'learn' with no flags cannot"
say "change anything, which is what makes it safe to run out of curiosity."

run "${PY_BIN}" -m blurt learn

pause 2.0

say ""
say "Fourteen suggestions, from four different kinds of evidence -- and the"
say "confidence attached to each one is the whole story:"
say ""
say "  ${C_KEY}github -> GitHub${C_OFF}${C_SAY}   high. Not a guess about the right spelling -- the"
say "                     engine used three, so it is provably inconsistent."
say "  ${C_KEY}Halyard, Naomi${C_OFF}${C_SAY}     high. Capitalized mid-sentence every single time."
say "                     Whisper emits casing, so a mid-sentence capital is"
say "                     the engine telling us it thinks this is a name."
say "  ${C_KEY}kubernetis${C_OFF}${C_SAY}         medium. One edit from a word said 7 times. That is"
say "                     an inference, not an observation, so it stays medium"
say "                     and --yes will refuse to touch it."
say "  ${C_KEY}the jargon list${C_OFF}${C_SAY}    medium. Real vocabulary, but prompt budget is small"
say "                     and spending it is a judgement call, not a fact."
say ""
say "--apply --yes takes the high-confidence findings and nothing else, ever."

run "${PY_BIN}" -m blurt learn --apply --yes

pause

say ""
say "It said what it skipped, and it skipped it. Here is what actually landed"
say "in the config file:"

run "${PY_BIN}" -m blurt config

pause

say ""
say "One dictionary entry and a two-word initial_prompt, both built entirely"
say "from words this user really said. The eleven medium findings are still"
say "sitting there waiting for a human, which is where they belong."
say ""
say "Run it again. A learning loop that re-proposes what it already applied is"
say "a loop that trains people to stop reading it."

run "${PY_BIN}" -m blurt learn

say ""
note "Same journal, same 40 records -- and now 0 high-confidence suggestions."
note "The three that were applied are gone from the list; the medium ones that"
note "were not applied are all still offered. Idempotent, and honest about it."

pause


# --------------------------------------------------------------------------- #
# 6. PRIVACY
# --------------------------------------------------------------------------- #

banner 6 "PRIVACY -- the journal is deletable, and the deletion is described honestly"

say "The journal is the one thing blurt keeps on disk that can contain anything"
say "you have ever said out loud. It is off by default, and there is a single"
say "command to destroy it."

run_py "confirm the journal exists before deleting it" <<'PY'
from blurt.history import default_history_path, history_size

path = default_history_path()
print("journal path : %s" % path)
print("exists       : %s" % path.exists())
print("records      : %d" % history_size())
PY

run "${PY_BIN}" -m blurt learn --forget

run_py "confirm it is gone" <<'PY'
from blurt.history import default_history_path, history_size

path = default_history_path()
print("journal path : %s" % path)
print("exists       : %s" % path.exists())
print("records      : %d" % history_size())
PY

say ""
say "Note what blurt refused to claim. It unlinked the file, and then said in"
say "the same breath that on a copy-on-write filesystem that is not a secure"
say "erase -- because on APFS overwriting in place does not reliably destroy"
say "the old blocks, so a scrub would be theatre. It points at FileVault, which"
say "actually solves the problem."
say ""
note "Deleting the journal does not touch the config: the dictionary and prompt"
note "learned from it are yours to keep, and nothing has to be re-learned."

pause


# --------------------------------------------------------------------------- #
# 7. WHAT THIS DID NOT SHOW
# --------------------------------------------------------------------------- #

banner 7 "WHAT THIS DID NOT SHOW"

say "Everything above really ran, and section 4 named the three things it stood"
say "in for as it stood in for them. These five did not run at all, and could"
say "not, because every one needs hardware or a macOS permission that does not"
say "exist in a container:"
say ""
say "  ${C_KEY}The global hotkey${C_OFF}${C_SAY}      pynput watching for the push-to-talk key while"
say "                         another app has focus. Needs macOS Accessibility"
say "                         permission, granted by a human in System Settings."
say "  ${C_KEY}Microphone capture${C_OFF}${C_SAY}     sounddevice opening a real input device. A"
say "                         denied app receives silence, not an error, which"
say "                         is why 'blurt doctor' opens the mic to check."
say "  ${C_KEY}Whisper transcription${C_OFF}${C_SAY}  faster-whisper with model weights on disk"
say "                         (~75 MB for base.en, downloaded once)."
say "  ${C_KEY}Clipboard paste${C_OFF}${C_SAY}        pyobjc writing the pasteboard and synthesizing"
say "                         Cmd-V into whatever app you were typing in. This"
say "                         is the seam section 4 replaced with a recorder;"
say "                         the undo you watched proved the decision to paste,"
say "                         never the paste itself."
say "  ${C_KEY}Command mode's key${C_OFF}${C_SAY}     the second hotkey, and the EventKit calls behind"
say "                         timers, reminders and calendar events. The router"
say "                         that key feeds is what ran in section 4; the key,"
say "                         and everything except undo that it dispatches to,"
say "                         did not."
say ""
say "On a Mac, run these two commands in this order:"
say ""
printf '    %s$ blurt doctor%s   %s# hardware, engines, permissions, dependencies%s\n' \
    "${C_CMD}" "${C_OFF}" "${C_NOTE}" "${C_OFF}"
printf '    %s$ blurt run%s      %s# hold the key, talk, release%s\n' \
    "${C_CMD}" "${C_OFF}" "${C_NOTE}" "${C_OFF}"
say ""
say "Then dictate one sentence, hold ${C_KEY}Right Command${C_OFF}${C_SAY} and say \"undo that\". That"
say "is section 4 with the three real seams back in place, and it is the fastest"
say "way to find out whether your Accessibility grant genuinely works: the undo"
say "either pastes your raw words or tells you the paste was refused, and blurt"
say "keeps that dictation revertible so you can try again once it is fixed."
say ""
say "Run 'doctor' first and read all of it. blurt's failure modes on macOS are"
say "almost all permission problems that produce silence rather than errors --"
say "a hotkey that never fires, a mic that returns zeros, a paste the system"
say "discards. None of those raise an exception, so 'doctor' exists to turn"
say "every one of them into text you can read."
say ""
printf '%s%s%s\n' "${C_BAR}" "${BAR}" "${C_OFF}"
say ""
say "The learning loop you just watched needs nothing but your own transcripts."
say "To start it for real, on your machine:"
say ""
printf '    %s$ blurt config set history_enabled true%s\n' "${C_CMD}" "${C_OFF}"
printf '    %s$ blurt learn%s                            %s# after a few days of dictating%s\n' \
    "${C_CMD}" "${C_OFF}" "${C_NOTE}" "${C_OFF}"
printf '    %s$ blurt learn --forget%s                   %s# whenever you want it gone%s\n' \
    "${C_CMD}" "${C_OFF}" "${C_NOTE}" "${C_OFF}"
say ""
