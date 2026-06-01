from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


EVENT_TYPES = ("convective", "stratiform", "mixed")


def _as_rainrate_array(rainrate_sequence: Any) -> np.ndarray:
    """Convert an event to a finite numeric rain-rate array."""
    rainrate = np.asarray(rainrate_sequence, dtype=np.float32)
    if rainrate.size == 0:
        raise ValueError("rainrate_sequence must contain at least one value.")
    return np.nan_to_num(rainrate, nan=0.0, posinf=0.0, neginf=0.0)


def classify_event_type(
    rainrate_sequence: Any,
    *,
    rain_threshold: float = 1.0,
    convective_intensity_threshold: float = 20.0,
    convective_extent_threshold: float = 0.1,
    stratiform_intensity_threshold: float = 10.0,
    stratiform_extent_threshold: float = 0.3,
) -> str:
    """Classify a precipitation event as convective, stratiform, or mixed."""
    rainrate = _as_rainrate_array(rainrate_sequence)
    max_intensity = float(np.max(rainrate))
    spatial_extent = float(np.mean(rainrate > rain_threshold))

    if max_intensity > convective_intensity_threshold and spatial_extent < convective_extent_threshold:
        return "convective"
    if max_intensity < stratiform_intensity_threshold and spatial_extent > stratiform_extent_threshold:
        return "stratiform"
    return "mixed"


def event_intensity_summary(rainrate_sequence: Any, *, rain_threshold: float = 1.0) -> dict[str, float]:
    """Return intensity features used for classification and filtering."""
    rainrate = _as_rainrate_array(rainrate_sequence)
    rainy_pixels = rainrate[rainrate > rain_threshold]

    return {
        "max_intensity": float(np.max(rainrate)),
        "mean_intensity": float(np.mean(rainrate)),
        "rainy_mean_intensity": float(np.mean(rainy_pixels)) if rainy_pixels.size else 0.0,
        "spatial_extent": float(np.mean(rainrate > rain_threshold)),
    }


def filter_events_by_intensity(
    events: Iterable[Any],
    *,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
    intensity_stat: str = "max_intensity",
    rain_threshold: float = 1.0,
) -> list[Any]:
    """Filter precipitation events by an intensity statistic.

    Events can be raw rain-rate arrays or dictionaries containing a
    ``rainrate_sequence`` key. The original event objects are returned.
    """
    if intensity_stat not in {"max_intensity", "mean_intensity", "rainy_mean_intensity"}:
        raise ValueError("intensity_stat must be one of: max_intensity, mean_intensity, rainy_mean_intensity.")

    filtered_events = []
    for event in events:
        sequence = event.get("rainrate_sequence") if isinstance(event, dict) else event
        summary = event_intensity_summary(sequence, rain_threshold=rain_threshold)
        intensity = summary[intensity_stat]

        if min_intensity is not None and intensity < min_intensity:
            continue
        if max_intensity is not None and intensity > max_intensity:
            continue
        filtered_events.append(event)

    return filtered_events


def create_balanced_test_set(
    events: Iterable[Any],
    *,
    target_event_types: tuple[str, ...] = EVENT_TYPES,
    samples_per_type: int | None = None,
    random_state: int | None = 42,
    sequence_key: str = "rainrate_sequence",
) -> list[Any]:
    """Create a test set with the same number of events per event type.

    Dictionary events may include a precomputed ``event_type``. Otherwise the
    type is computed from ``sequence_key``. Raw array events are also accepted.
    """
    unknown_types = set(target_event_types) - set(EVENT_TYPES)
    if unknown_types:
        raise ValueError(f"Unknown target event types: {sorted(unknown_types)}")

    grouped_events: dict[str, list[Any]] = defaultdict(list)
    for event in events:
        if isinstance(event, dict):
            event_type = event.get("event_type")
            if event_type is None:
                event_type = classify_event_type(event[sequence_key])
        else:
            event_type = classify_event_type(event)

        if event_type in target_event_types:
            grouped_events[event_type].append(event)

    available_counts = [len(grouped_events[event_type]) for event_type in target_event_types]
    if not available_counts or min(available_counts) == 0:
        missing = [event_type for event_type in target_event_types if not grouped_events[event_type]]
        raise ValueError(f"Cannot create a balanced test set; missing event types: {missing}")

    n_per_type = min(available_counts) if samples_per_type is None else samples_per_type
    if n_per_type <= 0:
        raise ValueError("samples_per_type must be positive.")
    if any(len(grouped_events[event_type]) < n_per_type for event_type in target_event_types):
        raise ValueError("Not enough events to satisfy samples_per_type for every target event type.")

    rng = np.random.default_rng(random_state)
    balanced_events = []
    for event_type in target_event_types:
        type_events = grouped_events[event_type]
        selected_indices = rng.choice(len(type_events), size=n_per_type, replace=False)
        balanced_events.extend(type_events[index] for index in selected_indices)

    rng.shuffle(balanced_events)
    return balanced_events
