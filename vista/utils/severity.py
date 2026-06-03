"""Severity-specific utilities for the VISTA challenge pipeline.

Severity is modelled as an ordered vocabulary per object category.
Higher-index entries represent more severe states.
"""
from __future__ import annotations

# ── Severity vocabularies (ordered: index = severity rank) ────────────────────

VEHICLE_SEVERITY: list[str] = [
    "undamaged",
    "minor damage",
    "heavily damaged",
    "overturned",
    "on fire",
]

PERSON_SEVERITY: list[str] = [
    "standing",
    "running",
    "helping",
    "calling for help",
    "injured, sitting",
    "injured, lying",
    "unconscious",
]

EMERGENCY_VEHICLE_CAPTIONS: list[str] = [
    "ambulance on scene",
    "fire truck on scene",
    "police on scene",
    "emergency vehicle",
]

_VEHICLE_RANK: dict[str, int] = {s: i for i, s in enumerate(VEHICLE_SEVERITY)}
_PERSON_RANK:  dict[str, int] = {s: i for i, s in enumerate(PERSON_SEVERITY)}


# ── Severity aggregation ──────────────────────────────────────────────────────

def peak_severity(captions: list[str | None], category: str) -> str | None:
    """Return the most severe caption observed in a track's history.

    Uses a conservative "peak observed" strategy — once a critical state has
    been detected it is retained, which is appropriate for emergency dispatch.

    Args:
        captions:  List of captions emitted for this track (oldest first).
        category:  Canonical challenge category: "car", "person", or
                   "emergency_vehicle".

    Returns:
        The highest-severity caption, or ``None`` if the list is empty.
    """
    non_null = [c for c in captions if c is not None]
    if not non_null:
        return None

    rank = _PERSON_RANK if category == "person" else _VEHICLE_RANK
    scored = [(rank.get(c, -1), c) for c in non_null]
    return max(scored, key=lambda x: x[0])[1]


# ── Zero-shot prompts (per category) ─────────────────────────────────────────

def vehicle_severity_prompt() -> str:
    """Prompt for judging vehicle damage level from a single crop."""
    return (
        "You are analysing a cropped image of a vehicle from a UAV accident scene. "
        "Describe the damage level of this vehicle in 2-4 words. "
        "Pick the single best match from: "
        "undamaged | minor damage | heavily damaged | overturned | on fire. "
        "Reply with ONLY the chosen label, nothing else."
    )


def person_severity_prompt() -> str:
    """Prompt for judging a person's injury/status from a single crop."""
    return (
        "You are analysing a cropped image of a person from a UAV accident scene. "
        "Describe this person's visible status in 2-4 words. "
        "Pick the single best match from: "
        "standing | running | helping | calling for help | "
        "injured, sitting | injured, lying | unconscious. "
        "Reply with ONLY the chosen label, nothing else."
    )


def emergency_vehicle_prompt() -> str:
    """Prompt for distinguishing emergency vehicles from regular ones."""
    return (
        "You are analysing a cropped image of a vehicle from a UAV accident scene. "
        "Is this vehicle an emergency vehicle (ambulance, fire truck, police car)? "
        "If yes, reply with one of: ambulance on scene | fire truck on scene | police on scene. "
        "If no, reply with: not emergency vehicle. "
        "Reply with ONLY the chosen label, nothing else."
    )


# ── Initial severity from YOLO class label ───────────────────────────────────

#: Maps YOLO class names (from VistaCrash fine-tuned model) to an initial
#: severity caption used before the VLM refines it.
YOLO_INITIAL_CAPTION: dict[str, str] = {
    "crashed car":  "heavily damaged",  # VistaCrash fine-tuned model (space variant)
    "crashed_car":  "heavily damaged",  # underscore variant (same class, safe fallback)
    "car":          "undamaged",
    "person":       "standing",
    "truck":        "undamaged",
    "bus":          "undamaged",
    "motorcycle":   "undamaged",
    "bicycle":      "undamaged",
}

#: Maps YOLO class names to the canonical challenge category.
YOLO_TO_CATEGORY: dict[str, str] = {
    "crashed car":  "car",   # VistaCrash fine-tuned model (space variant)
    "crashed_car":  "car",   # underscore variant
    "car":          "car",
    "person":       "person",
    "truck":        "car",       # may be upgraded to "emergency_vehicle" by VLM
    "bus":          "car",
    "motorcycle":   "car",
    "bicycle":      "car",
}

# ── Vocabulary enforcement ────────────────────────────────────────────────────

_ALL_VALID: set[str] = (
    set(VEHICLE_SEVERITY) | set(PERSON_SEVERITY) | set(EMERGENCY_VEHICLE_CAPTIONS)
)

# Fuzzy fallback map: common Qwen paraphrases → canonical label
_SNAP_MAP: dict[str, str] = {
    # vehicle damage variants
    "crashed":              "heavily damaged",
    "crashed vehicle":      "heavily damaged",
    "accident vehicle":     "heavily damaged",
    "accident-involved":    "heavily damaged",
    "accident-related":     "heavily damaged",
    "accident-relevant":    "heavily damaged",
    "involved":             "heavily damaged",
    "involved vehicle":     "heavily damaged",
    "involved vehicles":    "heavily damaged",
    "crashed/involved":     "heavily damaged",
    "crashed/collision":    "heavily damaged",
    "crashed vehicles":     "heavily damaged",
    "normal":               "undamaged",
    "normal vehicle":       "undamaged",
    "car":                  "undamaged",
    "vehicle":              "undamaged",
    "vehicles":             "undamaged",
    "motorcycle":           "undamaged",
    # person variants
    "people":               "standing",
    "person":               "standing",
    "person standing":      "standing",
    "people walking":       "standing",
    "person walking":       "standing",
    "injured sitting":      "injured, sitting",
    "injured lying":        "injured, lying",
    # emergency variants
    "emergency":            "emergency vehicle",
    "emergency response":   "emergency vehicle",
    "emergency responder":  "emergency vehicle",
    "emergency personnel":  "emergency vehicle",
    "emergency workers":    "emergency vehicle",
    "emergency personnel attending": "emergency vehicle",
}


def snap_to_vocabulary(label: str) -> str:
    """Map a raw Qwen label to the nearest valid vocabulary entry.

    First checks for an exact match (case-insensitive), then tries the
    fuzzy fallback map. If neither matches, returns the label unchanged
    (peak_severity will treat it as rank -1, so it won't override a valid label).
    """
    normalised = label.strip().lower()
    if normalised in _ALL_VALID:
        return normalised
    return _SNAP_MAP.get(normalised, label)
