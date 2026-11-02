"""Word-boundary keyword matching for the intent classifiers.

The classifiers score a query by looking for keyword signals in it. Doing that
with plain `signal in text` matches signals embedded inside unrelated words,
which silently misroutes queries:

    "end"  fires inside "vendor"   -> every vendor question became termination
    "own"  fires inside "unknown"
    "ip"   fires inside "equipment"

The signal lists are deliberately truncated stems ("terminat", "royalt",
"licens") so they match inflections, which rules out whole-word matching. The
fix is to anchor only the START of the signal to a word boundary: "end" then
matches end / ends / ending but not vendor.
"""

import re
from functools import lru_cache


@lru_cache(maxsize=None)
def signal_pattern(signal: str):
    """Compiled boundary-anchored pattern for one signal (cached)."""
    return re.compile(r"\b" + re.escape(signal))


def signal_in(text: str, signal: str) -> bool:
    """True if `signal` starts at a word boundary in `text`."""
    return signal_pattern(signal).search(text) is not None


def any_signal(text: str, signals) -> bool:
    """True if any signal starts at a word boundary in `text`."""
    return any(signal_pattern(s).search(text) for s in signals)


def count_signals(text: str, signals) -> int:
    """How many signals start at a word boundary in `text`."""
    return sum(1 for s in signals if signal_pattern(s).search(text))
