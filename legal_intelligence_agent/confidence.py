"""One definition of what a clause's confidence is, and what the gate accepts.

The pipeline advertises a hard rule: "Remove any result where confidence is
below 0.6" (prompts/v1.0.0_system.txt, README "Anti-Hallucination Measures").
Four places implemented that rule differently, and three of them invented a
number when the model did not report one:

  agent_core.analyze        missing confidence -> 0.7   (clears the gate)
  agent_core.validate       missing confidence -> 0.0   (correct)
  MockLLMClient.validate    missing confidence -> 0.0   (correct)
  ClaudeLLMClient.validate  missing confidence -> 1.0   (clears the gate)
  app.py clause formatting  missing or "sentinel" -> 0.62-0.97 for display

The net effect was that a clause the model never scored could reach the user
displayed as 97% confident, and the 0.6 filter the README cites as an
anti-hallucination measure could not reject it. The app layer even tried to
detect the upstream 0.7 default by treating {0.7, 0.8, 0.65} as sentinels,
which also discarded genuine model confidences that happened to equal them.

The rule here: an absent confidence is absent. It is never a number.

  - model_confidence() returns what the model reported, or None.
  - passes_threshold() accepts only a real value at or above the threshold,
    so an unscored clause is never treated as a vetted one.
  - estimated_confidence() exists for display only. Callers must store it
    under a key other than "confidence" so that, by construction, it cannot
    be read back as a model score or satisfy passes_threshold().
"""

CONFIDENCE_THRESHOLD = 0.6


def model_confidence(clause):
    """The confidence the model actually reported for this clause, or None.

    None covers both "the field is absent" and "the field is present but not a
    number". Neither is a confidence, and guessing a value for either is what
    this module exists to stop.
    """
    raw = clause.get("confidence")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def passes_threshold(clause, threshold=CONFIDENCE_THRESHOLD):
    """True only if the model reported a confidence at or above `threshold`.

    A clause with no usable confidence fails. That is the point: the gate's
    job is to drop results the model did not vouch for, and defaulting a
    missing value to anything >= threshold inverts it.
    """
    confidence = model_confidence(clause)
    return confidence is not None and confidence >= threshold


def partition_by_threshold(clauses, threshold=CONFIDENCE_THRESHOLD):
    """Split clauses into (kept, dropped_missing, dropped_low).

    The gate used to drop silently, which made its behaviour unobservable: you
    could not tell a contract with few relevant clauses from one where the
    model declined to score them. Separating "no confidence reported" from
    "scored below the bar" matters because they mean different things -- the
    first is a model or parsing problem, the second is the filter working.
    """
    kept, dropped_missing, dropped_low = [], [], []
    for clause in clauses or []:
        confidence = model_confidence(clause)
        if confidence is None:
            dropped_missing.append(clause)
        elif confidence < threshold:
            dropped_low.append(clause)
        else:
            kept.append(clause)
    return kept, dropped_missing, dropped_low


def drop_counts(dropped_missing, dropped_low):
    """The two counters every validate() reports, so traces record them."""
    return {
        "dropped_missing_confidence": len(dropped_missing),
        "dropped_low_confidence": len(dropped_low),
    }


def estimated_confidence(risk_level, triggered=False):
    """Display-only stand-in for a clause the model did not score.

    Returned so the UI can show something ordered and sensible rather than a
    blank, NOT so the clause can pass a confidence gate. Keep it out of the
    "confidence" key; see confidence_for_display().
    """
    risk = str(risk_level or "MEDIUM").upper()
    if triggered:
        return {
            "CRITICAL": 0.97,
            "HIGH": 0.92,
            "MEDIUM": 0.85,
        }.get(risk, 0.78)
    return {
        "CRITICAL": 0.88,
        "HIGH": 0.80,
        "MEDIUM": 0.72,
    }.get(risk, 0.62)


def confidence_for_display(clause):
    """Split a clause's confidence into a reported value and a display value.

    Returns (confidence, display_confidence, is_estimated):

      confidence         what the model reported, or None. The only value any
                         gate or filter should ever look at.
      display_confidence a number the UI can always render and sort on.
      is_estimated       True when display_confidence was derived rather than
                         reported, so the UI can label it instead of passing
                         a guess off as the model's own score.
    """
    reported = model_confidence(clause)
    if reported is not None:
        return reported, reported, False

    estimate = estimated_confidence(
        clause.get("risk_level"),
        bool(clause.get("triggered", False)),
    )
    return None, estimate, True
