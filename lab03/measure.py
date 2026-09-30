from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()

    times = []

    for _ in range(repeats):
        start = bench.clock()

        bench.workload.run()
        bench.workload.synchronize()

        end = bench.clock()

        elapsed_ms = (end - start) / 1_000_000.0
        times.append(elapsed_ms)

    return times


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown("warmup", why="not enough samples")

    midpoint = len(samples) // 2
    second_half = samples[midpoint:]
    settled_median = statistics.median(second_half)

    if settled_median <= 0:
        return unknown("warmup", why="invalid settled median")

    threshold = settled_median * (1 + WARMUP_TOL)

    count = 0

    for sample in samples:
        if sample > threshold:
            count += 1
        else:
            break

    retained = len(samples) - count

    return measured(
        count,
        source="leading prefix above (1 + 0.5) x median of the run's second half",
        settled_rate_ms=settled_median,
        threshold_ms=threshold,
        tolerance=WARMUP_TOL,
        retained=retained,
    )

    




def summarize(samples: list[float]) -> dict[str, Any]:

    if not samples:
        return {"n": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "p50": None,
                "p95": None,
                "p99": None,}

    samples.sort()
    mean = statistics.fmean(samples)
    min_num = min(samples)
    max_num = max(samples)
    if len(samples) > 2:
        std = statistics.stdev(samples)
    else:
        std = 0.0

    h50 = (len(samples) - 1) * 0.5
    i50 = int(h50)

    if i50 + 1 < len(samples):
        value50 = samples[i50] + (h50 - i50) * (
            samples[i50 + 1] - samples[i50]
        )
    else:
        value50 = samples[i50]

    h95 = (len(samples) - 1) * (95 / 100)
    i95 = int(h95)

    if i95 + 1 < len(samples):
        value95 = samples[i95] + (h95 - i95) * (
            samples[i95 + 1] - samples[i95]
        )
    else:
        value95 = samples[i95]

    h99 = (len(samples) - 1) * (99 / 100)
    i99 = int(h99)

    if i99 + 1 < len(samples):
        value99 = samples[i99] + (h99 - i99) * (
            samples[i99 + 1] - samples[i99]
        )
    else:
        value99 = samples[i99]


    return {
        "n": len(samples),
        "mean": round(mean,4),
        "std": round(std,4),
        "min": round(min_num,4),
        "max": round(max_num,4),
        "p50": round(value50,4),
        "p95": round(value95,4),
        "p99": round(value99,4),
    }

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 20:
        return unknown("is_multimodal", "not enough samples")

    samples.sort()
    low_5_cutoff = int(len(samples) * 0.05)
    high_5_cutoff = int(len(samples) * 0.95)
    trimmed_samples = samples[low_5_cutoff:high_5_cutoff]

    gaps = [j - i for i, j in zip(trimmed_samples[:-1], trimmed_samples[1:])]

    if not gaps:
        return unknown("is_multimodal", "not enough samples remain after trimming to calculate gaps")

    median_gap = statistics.median(gaps)

    if median_gap <= 0:
        return unknown("is_multimodal", "the timer resolution is too coarse")

    widest_gap = max(gaps)
    ratio = widest_gap / median_gap
    split_point = gaps.index(widest_gap)
    left = split_point + 1
    right = len(trimmed_samples) - left

    size = len(trimmed_samples)

    left_percent = left / size
    right_percent = right / size

    value_bool = (
        ratio >= 20.0
        and left_percent >= 0.10
        and right_percent >= 0.10
    )
    return measured(value_bool,
             source="widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side",
             gap_ratio=round(ratio,4),
             widest_gap_ms=round(widest_gap,4),
             typical_gap_ms=round(median_gap,4),
             modes=[{"n":left,"share": round(left_percent,4),"median_ms":round(statistics.median(trimmed_samples[:left]),4)},
                    {"n":right,"share": round(right_percent,4), "median_ms":round(statistics.median(trimmed_samples[left:]),4)}])



# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    result = bench.runner(["nvpmodel", "-q"])

    if not result.ok:
        return unknown(
            "nvpmodel -q",
            why=f"could not run nvpmodel: {result.error}",
        )

    if result.returncode != 0:
        return unknown(
            "nvpmodel -q",
            why=f"nvpmodel returned exit code {result.returncode}",
        )

    lines = result.stdout.splitlines()

    power_mode = None
    mode_index = None

    for i, line in enumerate(lines):
        if "NV Power Mode:" in line:
            power_mode = line.split("NV Power Mode:", 1)[1].strip()

            if i + 1 < len(lines):
                try:
                    mode_index = int(lines[i + 1].strip())
                except ValueError:
                    return unknown(
                        "nvpmodel -q",
                        why="could not parse power mode index",
                    )

            break

    if power_mode is None or mode_index is None:
        return unknown(
            "nvpmodel -q",
            why="could not parse power mode",
        )

    min_path = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
    max_path = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"

    min_freq = read_text(bench.telemetry, min_path)
    max_freq = read_text(bench.telemetry, max_path)

    if min_freq is None or max_freq is None:
        jetson_clocks = None
        clocks_source = unknown(f"{min_path} vs {max_path}","could not read CPU frequency")
    else:
        jetson_clocks = int(min_freq) == int(max_freq)

        clocks_source = measured(
            f"scaling_min_freq={min_freq}, scaling_max_freq={max_freq}",
            source=f"{min_path} vs {max_path}",
        )

    return measured(
        power_mode,
        source="nvpmodel -q",
        mode_index=mode_index,
        jetson_clocks=jetson_clocks,
        jetson_clocks_source=clocks_source,
    )




def probe_telemetry(bench: Bench) -> dict[str, Any]:
    # ------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------

    src = "sys/devices/virtual/thermal/*/temp"
    base = bench.telemetry / "sys/devices/virtual/thermal"
    zone_paths = list(base.glob("thermal_zone*"))

    highest_temp = None
    highest_zone = None
    zones_read = 0

    for zone_path in sorted(zone_paths):
        type_path = zone_path / "type"
        temp_path = zone_path / "temp"

        try:
            with open(type_path, "rb") as f:
                type_data = f.read()

            with open(temp_path, "rb") as f:
                temp_data = f.read()

            if type_data is None or temp_data is None:
                continue

            zone_type = type_data.decode(
                errors="replace"
            ).strip("\x00").strip()

            raw_temp = temp_data.decode(
                errors="replace"
            ).strip("\x00").strip()

        except (OSError, UnicodeDecodeError):
            continue

        if not zone_type or not raw_temp:
            continue

        try:
            temp_c = int(raw_temp) / 1000.0
        except ValueError:
            continue

        if int(raw_temp) <= -1000:
            continue

        zones_read += 1

        if highest_temp is None or temp_c > highest_temp:
            highest_temp = temp_c
            highest_zone = zone_type

    if highest_temp is None:
        temperature = unknown(
            src,
            why="no readable thermal zones found",
        )
    else:
        temperature = measured(
            round(highest_temp,4),
            source=src,
            zone=highest_zone,
            zones_read=zones_read,
        )
    # ------------------------------------------------------------
    # Power Draw
    # ------------------------------------------------------------

    power_candidates = (
        "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
        "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
        "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
    )

    power_result = read_first(
        bench.telemetry,
        power_candidates,
    )

    if power_result is None:
        power_mw = unknown(
            " | ".join(power_candidates),
            why="none of the documented INA3221 rail paths could be read",
        )
    else:
        power_path, raw_power = power_result

        try:
            power_value = int(raw_power)

            power_mw = measured(
                power_value,
                source=power_path,
            )

        except ValueError:
            power_mw = unknown(
                power_path,
                why="power value was not an integer",
            )

    # ------------------------------------------------------------
    # GPU Load
    # ------------------------------------------------------------

    gpu_candidates = (
        "sys/devices/platform/gpu.0/load",
        "sys/devices/gpu.0/load",
    )

    gpu_result = read_first(
        bench.telemetry,
        gpu_candidates,
    )

    if gpu_result is None:
        gpu_utilization = unknown(
            " | ".join(gpu_candidates),
            why="no GPU load file could be read",
        )
    else:
        gpu_path, raw_load = gpu_result

        try:
            gpu_load = int(raw_load)

            gpu_utilization = measured(
                round(gpu_load / 10.0,4),
                source=gpu_path,
                units="per-mille / 10",
            )

        except ValueError:
            gpu_utilization = unknown(
                gpu_path,
                why="GPU load value was not an integer",
            )

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    return {
        "temperature_c": temperature,
        "power_mw": power_mw,
        "gpu_utilization_percent": gpu_utilization,
    }
## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)