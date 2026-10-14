import statistics
from typing import Any
from measure import measured, unknown, MIN_SAMPLES_FOR_STATIONARITY, STATIONARITY_TOL

def is_stationary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < MIN_SAMPLES_FOR_STATIONARITY:
        return unknown(
            "is_stationary",
            why="too few samples to divide into thirds"
        )

    overall_median = statistics.median(samples)

    if overall_median <= 0:
        return unknown(
            "is_stationary",
            why="median is not positive"
        )

    k = len(samples) // 3

    first_third = samples[:k]
    last_third = samples[-k:]

    first_median = statistics.median(first_third)
    last_median = statistics.median(last_third)

    drift = last_median - first_median
    relative_drift = abs(drift) / overall_median

    if drift > 0:
        direction = "slower"
    elif drift < 0:
        direction = "faster"
    else:
        direction = "flat"

    stationary = relative_drift <= STATIONARITY_TOL

    return measured(
        stationary,
        first_third_median_ms=round(first_median, 4),
        last_third_median_ms=round(last_median, 4),
        drift_ms=round(drift, 4),
        drift_relative=round(relative_drift, 4),
        direction=direction,
        tolerance=STATIONARITY_TOL
    )