from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sig


SCALAR_COLUMN_ORDER = [
    "voltage",
    "current",
    "frequency",
    "active_power",
    "reactive_power",
    "voltage_a",
    "voltage_b",
    "voltage_c",
    "current_a",
    "current_b",
    "current_c",
    "active_power_a",
    "active_power_b",
    "active_power_c",
    "reactive_power_a",
    "reactive_power_b",
    "reactive_power_c",
    "voltage_zero_sequence",
    "voltage_positive_sequence",
    "voltage_negative_sequence",
    "current_zero_sequence",
    "current_positive_sequence",
    "current_negative_sequence",
    "voltage_zero_unbalance",
    "voltage_negative_unbalance",
    "current_zero_unbalance",
    "current_negative_unbalance",
]

SCALAR_DISPLAY_LABELS = {
    "voltage": "RMS voltage or DC",
    "current": "RMS current or DC",
    "frequency": "Frequency",
    "active_power": "Active power",
    "reactive_power": "Reactive power",
    "voltage_a": "RMS voltage A or DC",
    "voltage_b": "RMS voltage B or DC",
    "voltage_c": "RMS voltage C or DC",
    "current_a": "RMS current A or DC",
    "current_b": "RMS current B or DC",
    "current_c": "RMS current C or DC",
    "active_power_a": "Active power A",
    "active_power_b": "Active power B",
    "active_power_c": "Active power C",
    "reactive_power_a": "Reactive power A",
    "reactive_power_b": "Reactive power B",
    "reactive_power_c": "Reactive power C",
    "voltage_zero_sequence": "Zero-sequence voltage",
    "voltage_positive_sequence": "Positive-sequence voltage",
    "voltage_negative_sequence": "Negative-sequence voltage",
    "current_zero_sequence": "Zero-sequence current",
    "current_positive_sequence": "Positive-sequence current",
    "current_negative_sequence": "Negative-sequence current",
    "voltage_zero_unbalance": "Voltage zero-sequence unbalance",
    "voltage_negative_unbalance": "Voltage negative-sequence unbalance",
    "current_zero_unbalance": "Current zero-sequence unbalance",
    "current_negative_unbalance": "Current negative-sequence unbalance",
}

SINUSOIDAL_DISPLAY_LABELS = {
    "voltage": "Voltage",
    "current": "Current",
    "voltage_a": "Voltage A",
    "voltage_b": "Voltage B",
    "voltage_c": "Voltage C",
    "current_a": "Current A",
    "current_b": "Current B",
    "current_c": "Current C",
}

USER_SELECTION_TO_INTERNAL = {
    "sinusoidal": {
        "Time": "time",
        "Voltage": "voltage",
        "Current": "current",
        "Voltage A": "voltage_a",
        "Voltage B": "voltage_b",
        "Voltage C": "voltage_c",
        "Current A": "current_a",
        "Current B": "current_b",
        "Current C": "current_c",
        "Ignore signal": "ignore",
    },
    "scalar": {
        "Time": "time",
        "RMS voltage or DC": "voltage",
        "RMS current or DC": "current",
        "Frequency": "frequency",
        "Active power": "active_power",
        "Reactive power": "reactive_power",
        "RMS voltage A or DC": "voltage_a",
        "RMS voltage B or DC": "voltage_b",
        "RMS voltage C or DC": "voltage_c",
        "RMS current A or DC": "current_a",
        "RMS current B or DC": "current_b",
        "RMS current C or DC": "current_c",
        "Active power A": "active_power_a",
        "Active power B": "active_power_b",
        "Active power C": "active_power_c",
        "Reactive power A": "reactive_power_a",
        "Reactive power B": "reactive_power_b",
        "Reactive power C": "reactive_power_c",
        "Zero-sequence voltage": "voltage_zero_sequence",
        "Positive-sequence voltage": "voltage_positive_sequence",
        "Negative-sequence voltage": "voltage_negative_sequence",
        "Zero-sequence current": "current_zero_sequence",
        "Positive-sequence current": "current_positive_sequence",
        "Negative-sequence current": "current_negative_sequence",
        "Voltage zero-sequence unbalance": "voltage_zero_unbalance",
        "Voltage negative-sequence unbalance": "voltage_negative_unbalance",
        "Current zero-sequence unbalance": "current_zero_unbalance",
        "Current negative-sequence unbalance": "current_negative_unbalance",
        "Ignore signal": "ignore",
    },
}

IGNORE_SELECTION = "Ignore signal"

TEST_SYNC_FIELD = {
    "step_test": "step_quantity",
    "fault_test": "fault_quantity",
    "ramp_test": "ramp_quantity",
}

TEST_QUANTITY_TO_INTERNAL = {
    "Voltage": "voltage",
    "Current": "current",
    "Frequency": "frequency",
    "Active power": "active_power",
    "Reactive power": "reactive_power",
    "Voltage A": "voltage_a",
    "Voltage B": "voltage_b",
    "Voltage C": "voltage_c",
    "Current A": "current_a",
    "Current B": "current_b",
    "Current C": "current_c",
    "Positive-sequence voltage": "voltage_positive_sequence",
    "Negative-sequence voltage": "voltage_negative_sequence",
    "Zero-sequence voltage": "voltage_zero_sequence",
    "Positive-sequence current": "current_positive_sequence",
    "Negative-sequence current": "current_negative_sequence",
    "Zero-sequence current": "current_zero_sequence",
}

TEST_TARGET_FIELDS = {
    "step_test": ("step_quantity", "pre_step_percent", "post_step_percent"),
    "fault_test": ("fault_quantity", "pre_fault_percent", "during_fault_percent"),
    "ramp_test": ("ramp_quantity", "pre_ramp_percent", "post_ramp_percent"),
}


@dataclass
class ValidationResult:
    signal_type: str
    sync_label: str
    sync_column: str
    scalar_experimental: pd.DataFrame
    scalar_model: pd.DataFrame
    waveform_experimental: pd.DataFrame | None
    waveform_model: pd.DataFrame | None
    zoom_experimental: pd.Series
    zoom_model: pd.Series
    zoom_time_s: float


@dataclass
class WindowSegment:
    name: str
    start_s: float
    end_s: float


def sniff_csv_options(path: Path) -> tuple[str, csv.Dialect]:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as csv_file:
                sample = csv_file.read(4096)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
                except csv.Error:
                    dialect = csv.excel
                return encoding, dialect
        except UnicodeDecodeError:
            continue

    raise ValueError("It was not possible to identify the CSV file format.")


def read_csv_headers(path: Path) -> list[str]:
    encoding, dialect = sniff_csv_options(path)
    with path.open("r", encoding=encoding, newline="") as csv_file:
        csv_reader = csv.reader(csv_file, dialect=dialect)
        for row in csv_reader:
            if not any(cell.strip() for cell in row):
                continue
            return [cell.strip() or f"Column {index}" for index, cell in enumerate(row, start=1)]

    raise ValueError("It was not possible to read the header of the selected CSV file.")


def validate_import_configuration(
    signal_type: str,
    headers_by_side: dict[str, list[str]],
    selections_by_side: dict[str, list[str]],
) -> list[str]:
    errors: list[str] = []
    sinusoidal_modes: dict[str, str] = {}

    for side in ("experimental", "model"):
        headers = headers_by_side[side]
        selections = normalize_user_selections(selections_by_side[side])

        if not headers:
            errors.append(f"Import the CSV for {translate_side(side)}.")
            continue

        if len(headers) != len(selections):
            errors.append(
                f"Classify all columns in the file for {translate_side(side)} before continuing."
            )
            continue

        internal_counts = _count_internal_selection(signal_type, selections)
        if internal_counts["time"] != 1:
            errors.append(
                f"The file for {translate_side(side)} must have exactly one column marked as Time."
            )

        if signal_type == "sinusoidal":
            has_single_phase = internal_counts["voltage"] == 1 and internal_counts["current"] == 1
            has_three_phase = all(
                internal_counts[required] == 1
                for required in ("voltage_a", "voltage_b", "voltage_c", "current_a", "current_b", "current_c")
            )
            if not has_single_phase and not has_three_phase:
                errors.append(
                    f"The file for {translate_side(side)} must contain either Voltage and Current, "
                    "or Voltage A/B/C and Current A/B/C."
                )
            elif has_three_phase:
                sinusoidal_modes[side] = "three_phase"
            else:
                sinusoidal_modes[side] = "single_phase"
            for internal_name in SINUSOIDAL_DISPLAY_LABELS:
                if internal_counts.get(internal_name, 0) > 1:
                    errors.append(
                        f"The file for {translate_side(side)} cannot have more than one column marked as {SINUSOIDAL_DISPLAY_LABELS[internal_name]}."
                    )
        else:
            for internal_name in SCALAR_COLUMN_ORDER:
                if internal_counts.get(internal_name, 0) > 1:
                    errors.append(
                        f"The file for {translate_side(side)} cannot have more than one column marked as {SCALAR_DISPLAY_LABELS[internal_name]}."
                    )

    if signal_type == "scalar":
        common_columns = _common_scalar_columns(selections_by_side)
        if not common_columns:
            errors.append(
                "Scalar files must have at least one quantity in common besides time."
            )
    elif signal_type == "sinusoidal" and len(set(sinusoidal_modes.values())) > 1:
        errors.append(
            "Experimental and model waveform files must use the same electrical structure "
            "(both single-phase or both three-phase)."
        )

    return errors


def run_validation_pipeline(
    signal_type: str,
    test_key: str,
    form_values: dict[str, str],
    filter_signals: bool,
    file_paths: dict[str, Path],
    headers_by_side: dict[str, list[str]],
    selections_by_side: dict[str, list[str]],
) -> ValidationResult:
    datasets = {
        side: load_selected_dataset(
            path=file_paths[side],
            headers=headers_by_side[side],
            selections=selections_by_side[side],
            signal_type=signal_type,
        )
        for side in ("experimental", "model")
    }

    if signal_type == "sinusoidal":
        experimental_waveform, model_waveform = standardize_sampling_pair(
            datasets["experimental"],
            datasets["model"],
        )
        experimental_waveform = trim_to_complete_cycles(experimental_waveform)
        model_waveform = trim_to_complete_cycles(model_waveform)

        reference_voltage_column = waveform_reference_voltage_column(experimental_waveform)
        nominal_grid_freq = estimate_nominal_grid_frequency(
            experimental_waveform["time"].to_numpy(),
            experimental_waveform[reference_voltage_column].to_numpy(),
        )

        scalar_experimental = calculate_scalar_signals(experimental_waveform, nominal_grid_freq)
        scalar_model = calculate_scalar_signals(model_waveform, nominal_grid_freq)

        aligned = synchronize_datasets(
            test_key=test_key,
            form_values=form_values,
            signal_type=signal_type,
            scalar_experimental=scalar_experimental,
            scalar_model=scalar_model,
            waveform_experimental=experimental_waveform,
            waveform_model=model_waveform,
        )
        aligned = trim_scalar_edges_after_sync(
            result=aligned,
            nominal_grid_freq=nominal_grid_freq,
        )
    else:
        scalar_experimental, scalar_model = standardize_sampling_pair(
            datasets["experimental"],
            datasets["model"],
        )
        scalar_experimental = augment_scalar_dataset(scalar_experimental)
        scalar_model = augment_scalar_dataset(scalar_model)
        if filter_signals:
            scalar_experimental = filter_scalar_dataset(scalar_experimental)
            scalar_model = filter_scalar_dataset(scalar_model)
        aligned = synchronize_datasets(
            test_key=test_key,
            form_values=form_values,
            signal_type=signal_type,
            scalar_experimental=scalar_experimental,
            scalar_model=scalar_model,
            waveform_experimental=None,
            waveform_model=None,
        )

    return aligned


def augment_scalar_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    augmented = dataset.copy()
    phase_voltage_columns = ["voltage_a", "voltage_b", "voltage_c"]
    phase_current_columns = ["current_a", "current_b", "current_c"]
    phase_active_power_columns = ["active_power_a", "active_power_b", "active_power_c"]
    phase_reactive_power_columns = ["reactive_power_a", "reactive_power_b", "reactive_power_c"]

    if "voltage" not in augmented.columns and all(column in augmented.columns for column in phase_voltage_columns):
        augmented["voltage"] = augmented[phase_voltage_columns].mean(axis=1)
    if "current" not in augmented.columns and all(column in augmented.columns for column in phase_current_columns):
        augmented["current"] = augmented[phase_current_columns].mean(axis=1)
    if "active_power" not in augmented.columns and all(column in augmented.columns for column in phase_active_power_columns):
        augmented["active_power"] = augmented[phase_active_power_columns].sum(axis=1)
    if "reactive_power" not in augmented.columns and all(column in augmented.columns for column in phase_reactive_power_columns):
        augmented["reactive_power"] = augmented[phase_reactive_power_columns].sum(axis=1)

    return augmented


def filter_scalar_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    if "time" not in dataset.columns or len(dataset) < 11:
        return dataset.copy()

    filtered = dataset.copy()
    time_values = filtered["time"].to_numpy(dtype=float)
    dt = np.diff(time_values)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    if valid_dt.size == 0:
        return filtered

    sample_rate_hz = float(1.0 / np.median(valid_dt))
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        return filtered

    nyquist_hz = sample_rate_hz / 2.0
    cutoff_hz = min(max(sample_rate_hz * 0.015, 1.0), nyquist_hz * 0.12)
    if cutoff_hz <= 0 or cutoff_hz >= nyquist_hz:
        return filtered

    try:
        sos = sig.butter(
            N=3,
            Wn=cutoff_hz,
            btype="lowpass",
            fs=sample_rate_hz,
            output="sos",
        )
    except ValueError:
        return filtered

    for column in filtered.columns:
        if column == "time":
            continue
        values = filtered[column].to_numpy(dtype=float)
        if len(values) < 11 or not np.all(np.isfinite(values)):
            continue
        median_window = min(
            21,
            max(5, int(len(values) * 0.01) | 1),
        )
        if median_window >= len(values):
            median_window = len(values) - 1 if len(values) % 2 == 0 else len(values)
        prefiltered = values
        if median_window >= 5:
            prefiltered = (
                pd.Series(values)
                .rolling(window=median_window, center=True, min_periods=1)
                .median()
                .to_numpy(dtype=float)
            )
        try:
            filtered[column] = sig.sosfiltfilt(sos, prefiltered)
        except ValueError:
            window = min(11, len(values) if len(values) % 2 == 1 else len(values) - 1)
            if window >= 3:
                filtered[column] = (
                    pd.Series(prefiltered)
                    .rolling(window=window, center=True, min_periods=1)
                    .mean()
                    .to_numpy(dtype=float)
                )

    return filtered


def trim_scalar_edges_after_sync(
    result: ValidationResult,
    nominal_grid_freq: float,
) -> ValidationResult:
    cycle_duration_s = 1.0 / nominal_grid_freq
    edge_trim_s = 2.0 * cycle_duration_s

    trimmed_experimental, trimmed_model = trim_pair_edges_by_time(
        experimental=result.scalar_experimental,
        model=result.scalar_model,
        trim_start_s=edge_trim_s,
        trim_end_s=edge_trim_s,
    )

    trimmed_experimental, trimmed_model = drop_partial_edge_windows(
        experimental=trimmed_experimental,
        model=trimmed_model,
    )

    normalized_experimental, normalized_model = normalize_pair_duration(
        experimental=trimmed_experimental,
        model=trimmed_model,
    )

    result.scalar_experimental = normalized_experimental
    result.scalar_model = normalized_model
    if result.waveform_experimental is not None and result.waveform_model is not None:
        waveform_experimental, waveform_model = trim_pair_edges_by_time(
            experimental=result.waveform_experimental,
            model=result.waveform_model,
            trim_start_s=edge_trim_s,
            trim_end_s=edge_trim_s,
        )
        waveform_experimental, waveform_model = normalize_pair_duration(
            waveform_experimental,
            waveform_model,
        )
        result.waveform_experimental = waveform_experimental
        result.waveform_model = waveform_model
    return result


def load_selected_dataset(
    path: Path,
    headers: list[str],
    selections: list[str],
    signal_type: str,
) -> pd.DataFrame:
    encoding, dialect = sniff_csv_options(path)
    raw_df = pd.read_csv(
        path,
        encoding=encoding,
        sep=dialect.delimiter,
        dtype=str,
        keep_default_na=False,
    )

    selected_indexes: list[int] = []
    selected_internal_names: list[str] = []
    seen_internal: set[str] = set()
    mapping = USER_SELECTION_TO_INTERNAL[signal_type]
    normalized_selections = normalize_user_selections(selections)

    for index, selection in enumerate(normalized_selections):
        internal_name = mapping.get(selection, "ignore")
        if internal_name == "ignore":
            continue

        if internal_name in seen_internal:
            raise ValueError(f"The quantity '{selection}' was selected more than once.")

        selected_indexes.append(index)
        selected_internal_names.append(internal_name)
        seen_internal.add(internal_name)

    if not selected_indexes:
        raise ValueError("No valid quantity was selected for import.")

    dataset = raw_df.iloc[:, selected_indexes].copy()
    dataset.columns = selected_internal_names
    parsed_dataset = pd.DataFrame(index=dataset.index)
    for column in dataset.columns:
        if column == "time":
            parsed_dataset[column] = parse_time_series(dataset[column])
        else:
            parsed_dataset[column] = parse_numeric_series(dataset[column])

    dataset = parsed_dataset.dropna()

    if "time" not in dataset.columns:
        raise ValueError("The import must contain one column marked as Time.")

    dataset = dataset.sort_values("time").drop_duplicates(subset="time")
    if len(dataset) < 4:
        raise ValueError("The file must contain at least 4 valid samples.")

    return dataset.reset_index(drop=True)


def standardize_sampling_pair(
    experimental: pd.DataFrame,
    model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_dt = estimate_sample_time(experimental["time"].to_numpy())
    model_dt = estimate_sample_time(model["time"].to_numpy())
    target_dt = max(exp_dt, model_dt)

    return (
        resample_to_time_step(experimental, target_dt),
        resample_to_time_step(model, target_dt),
    )


def estimate_sample_time(time_values: np.ndarray) -> float:
    diffs = np.diff(time_values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("It was not possible to identify the signal sampling.")
    return float(np.median(diffs))


def resample_to_time_step(dataset: pd.DataFrame, target_dt: float) -> pd.DataFrame:
    time_values = dataset["time"].to_numpy(dtype=float)
    start = time_values[0]
    end = time_values[-1]
    if end <= start:
        raise ValueError("The CSV time axis must be strictly increasing.")

    sample_count = max(int(np.floor((end - start) / target_dt)) + 1, 2)
    resampled_time = start + np.arange(sample_count) * target_dt
    if resampled_time[-1] > end:
        resampled_time = resampled_time[resampled_time <= end]
    if len(resampled_time) < 2:
        raise ValueError("It was not possible to resample the signal using the provided time axis.")

    resampled = {"time": resampled_time}
    for column in dataset.columns:
        if column == "time":
            continue
        resampled[column] = np.interp(resampled_time, time_values, dataset[column].to_numpy(dtype=float))

    return pd.DataFrame(resampled)


def trim_to_complete_cycles(dataset: pd.DataFrame) -> pd.DataFrame:
    reference_voltage_column = waveform_reference_voltage_column(dataset)
    voltage = dataset[reference_voltage_column].to_numpy(dtype=float)
    time_values = dataset["time"].to_numpy(dtype=float)
    fs = 1.0 / estimate_sample_time(time_values)

    sos = sig.butter(N=5, Wn=[30, 80], btype="bandpass", fs=fs, output="sos")
    filtered_voltage = sig.sosfiltfilt(sos, voltage)

    zero_crossings = np.where(np.diff(np.sign(filtered_voltage)) > 0)[0] + 1
    if len(zero_crossings) < 2:
        raise ValueError("It was not possible to identify complete cycles in the voltage signal.")

    start = int(zero_crossings[0])
    end = int(zero_crossings[-1])
    trimmed = dataset.iloc[start:end].copy().reset_index(drop=True)
    trimmed["time"] = trimmed["time"] - float(trimmed["time"].iloc[0])
    return trimmed


def waveform_reference_voltage_column(dataset: pd.DataFrame) -> str:
    if "voltage" in dataset.columns:
        return "voltage"
    if "voltage_a" in dataset.columns:
        return "voltage_a"
    raise ValueError("The imported waveform must contain a voltage reference signal.")


def waveform_has_three_phase(dataset: pd.DataFrame) -> bool:
    return all(
        column in dataset.columns
        for column in ("voltage_a", "voltage_b", "voltage_c", "current_a", "current_b", "current_c")
    )


def estimate_nominal_grid_frequency(time_values: np.ndarray, voltage_values: np.ndarray) -> float:
    centered = voltage_values - np.mean(voltage_values)
    zero_crossings = np.where(np.diff(np.sign(centered)) > 0)[0] + 1
    if len(zero_crossings) < 2:
        return 60.0

    periods = np.diff(time_values[zero_crossings])
    periods = periods[periods > 0]
    if len(periods) == 0:
        return 60.0

    estimated_frequency = 1.0 / float(np.median(periods))
    return 50.0 if abs(estimated_frequency - 50.0) < abs(estimated_frequency - 60.0) else 60.0


def calculate_scalar_signals(waveform_df: pd.DataFrame, nominal_grid_freq: float) -> pd.DataFrame:
    del nominal_grid_freq

    time_values = waveform_df["time"].to_numpy(dtype=float)
    reference_voltage_column = waveform_reference_voltage_column(waveform_df)
    voltage_values = waveform_df[reference_voltage_column].to_numpy(dtype=float)
    half_cycle_boundaries = detect_half_cycle_boundaries(time_values, voltage_values)
    if len(half_cycle_boundaries) < 3:
        raise ValueError("It was not possible to identify enough cycles to calculate scalar quantities.")

    is_three_phase = waveform_has_three_phase(waveform_df)
    phase_pairs = (
        (("voltage_a", "current_a"), "a"),
        (("voltage_b", "current_b"), "b"),
        (("voltage_c", "current_c"), "c"),
    )
    if not is_three_phase:
        current_values = waveform_df["current"].to_numpy(dtype=float)

    cycle_rows: list[dict[str, float]] = []
    for start_boundary, end_boundary in zip(half_cycle_boundaries[:-2], half_cycle_boundaries[2:]):
        start_index = start_boundary["index"]
        end_index = end_boundary["index"]
        if end_index - start_index < 2:
            continue

        time_segment = time_values[start_index : end_index + 1].copy()
        time_segment[0] = start_boundary["time"]
        time_segment[-1] = end_boundary["time"]

        voltage_segment = voltage_values[start_index : end_index + 1]
        cycle_duration = float(end_boundary["time"] - start_boundary["time"])
        if cycle_duration <= 0:
            continue

        frequency = 1.0 / cycle_duration
        if is_three_phase:
            row: dict[str, float] = {
                "time": (start_boundary["time"] + end_boundary["time"]) / 2.0,
                "frequency": frequency,
            }
            voltage_phasors: list[complex] = []
            current_phasors: list[complex] = []
            voltage_rms_values: list[float] = []
            current_rms_values: list[float] = []
            active_powers: list[float] = []
            reactive_powers: list[float] = []

            for (voltage_column, current_column), phase_suffix in phase_pairs:
                phase_voltage_segment = waveform_df[voltage_column].to_numpy(dtype=float)[start_index : end_index + 1]
                phase_current_segment = waveform_df[current_column].to_numpy(dtype=float)[start_index : end_index + 1]
                phase_voltage_rms = rms_over_time(time_segment, phase_voltage_segment, cycle_duration)
                phase_current_rms = rms_over_time(time_segment, phase_current_segment, cycle_duration)
                phase_active_power = average_over_time(
                    time_segment,
                    phase_voltage_segment * phase_current_segment,
                    cycle_duration,
                )
                phase_voltage_phasor = fundamental_rms_phasor(time_segment, phase_voltage_segment, frequency)
                phase_current_phasor = fundamental_rms_phasor(time_segment, phase_current_segment, frequency)
                phase_reactive_power = float(np.imag(phase_voltage_phasor * np.conjugate(phase_current_phasor)))

                row[f"voltage_{phase_suffix}"] = phase_voltage_rms
                row[f"current_{phase_suffix}"] = phase_current_rms
                row[f"active_power_{phase_suffix}"] = phase_active_power
                row[f"reactive_power_{phase_suffix}"] = phase_reactive_power
                voltage_phasors.append(phase_voltage_phasor)
                current_phasors.append(phase_current_phasor)
                voltage_rms_values.append(phase_voltage_rms)
                current_rms_values.append(phase_current_rms)
                active_powers.append(phase_active_power)
                reactive_powers.append(phase_reactive_power)

            voltage_sequence = symmetrical_components(np.asarray(voltage_phasors, dtype=complex))
            current_sequence = symmetrical_components(np.asarray(current_phasors, dtype=complex))
            voltage_positive = float(abs(voltage_sequence[1]))
            current_positive = float(abs(current_sequence[1]))
            row.update(
                {
                    "voltage": float(np.mean(voltage_rms_values)),
                    "current": float(np.mean(current_rms_values)),
                    "active_power": float(np.sum(active_powers)),
                    "reactive_power": float(np.sum(reactive_powers)),
                    "voltage_zero_sequence": float(abs(voltage_sequence[0])),
                    "voltage_positive_sequence": voltage_positive,
                    "voltage_negative_sequence": float(abs(voltage_sequence[2])),
                    "current_zero_sequence": float(abs(current_sequence[0])),
                    "current_positive_sequence": current_positive,
                    "current_negative_sequence": float(abs(current_sequence[2])),
                    "voltage_zero_unbalance": (
                        float(abs(voltage_sequence[0]) / voltage_positive * 100.0)
                        if voltage_positive > 0
                        else float("nan")
                    ),
                    "voltage_negative_unbalance": (
                        float(abs(voltage_sequence[2]) / voltage_positive * 100.0)
                        if voltage_positive > 0
                        else float("nan")
                    ),
                    "current_zero_unbalance": (
                        float(abs(current_sequence[0]) / current_positive * 100.0)
                        if current_positive > 0
                        else float("nan")
                    ),
                    "current_negative_unbalance": (
                        float(abs(current_sequence[2]) / current_positive * 100.0)
                        if current_positive > 0
                        else float("nan")
                    ),
                }
            )
            cycle_rows.append(row)
            continue

        current_segment = current_values[start_index : end_index + 1]
        voltage_rms = rms_over_time(time_segment, voltage_segment, cycle_duration)
        current_rms = rms_over_time(time_segment, current_segment, cycle_duration)
        active_power = average_over_time(time_segment, voltage_segment * current_segment, cycle_duration)
        voltage_fundamental = fundamental_rms_phasor(time_segment, voltage_segment, frequency)
        current_fundamental = fundamental_rms_phasor(time_segment, current_segment, frequency)
        reactive_power = float(np.imag(voltage_fundamental * np.conjugate(current_fundamental)))

        cycle_rows.append(
            {
                "time": (start_boundary["time"] + end_boundary["time"]) / 2.0,
                "voltage": voltage_rms,
                "current": current_rms,
                "frequency": frequency,
                "active_power": active_power,
                "reactive_power": reactive_power,
            }
        )

    if not cycle_rows:
        raise ValueError("It was not possible to calculate the quantities cycle by cycle for the imported signal.")

    scalar_df = pd.DataFrame(cycle_rows)
    scalar_df = scalar_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    scalar_df["time"] = scalar_df["time"] - float(scalar_df["time"].iloc[0])
    return scalar_df


def symmetrical_components(abc_phasors: np.ndarray) -> np.ndarray:
    if abc_phasors.shape[0] != 3:
        raise ValueError("Symmetrical components require three phase phasors.")
    a120 = np.exp(1j * 2.0 * np.pi / 3.0)
    a240 = np.exp(1j * 4.0 * np.pi / 3.0)
    transform = (1.0 / 3.0) * np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, a120, a240],
            [1.0, a240, a120],
        ],
        dtype=complex,
    )
    return transform @ abc_phasors


def detect_cycle_boundaries(time_values: np.ndarray, voltage_values: np.ndarray) -> list[dict[str, float]]:
    fs = 1.0 / estimate_sample_time(time_values)
    sos = sig.butter(N=4, Wn=[30, 80], btype="bandpass", fs=fs, output="sos")
    filtered_voltage = sig.sosfiltfilt(sos, voltage_values)

    boundaries: list[dict[str, float]] = []
    for index in range(len(filtered_voltage) - 1):
        current_value = filtered_voltage[index]
        next_value = filtered_voltage[index + 1]
        if current_value < 0 <= next_value and next_value != current_value:
            ratio = -current_value / (next_value - current_value)
            crossing_time = float(
                time_values[index] + ratio * (time_values[index + 1] - time_values[index])
            )
            boundaries.append({"index": index, "time": crossing_time})

    return boundaries


def detect_half_cycle_boundaries(time_values: np.ndarray, voltage_values: np.ndarray) -> list[dict[str, float]]:
    fs = 1.0 / estimate_sample_time(time_values)
    sos = sig.butter(N=4, Wn=[30, 80], btype="bandpass", fs=fs, output="sos")
    filtered_voltage = sig.sosfiltfilt(sos, voltage_values)

    boundaries: list[dict[str, float]] = []
    for index in range(len(filtered_voltage) - 1):
        current_value = filtered_voltage[index]
        next_value = filtered_voltage[index + 1]
        if np.signbit(current_value) != np.signbit(next_value) and next_value != current_value:
            ratio = -current_value / (next_value - current_value)
            crossing_time = float(
                time_values[index] + ratio * (time_values[index + 1] - time_values[index])
            )
            boundaries.append({"index": index, "time": crossing_time})

    return boundaries


def rms_over_time(time_values: np.ndarray, signal_values: np.ndarray, duration: float) -> float:
    mean_square = np.trapezoid(np.square(signal_values), time_values) / duration
    return float(np.sqrt(max(mean_square, 0.0)))


def average_over_time(time_values: np.ndarray, signal_values: np.ndarray, duration: float) -> float:
    return float(np.trapezoid(signal_values, time_values) / duration)


def fundamental_rms_phasor(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    frequency_hz: float,
) -> complex:
    relative_time = time_values - float(time_values[0])
    angular_frequency = 2.0 * np.pi * frequency_hz
    cosine_projection = (2.0 / relative_time[-1]) * np.trapezoid(
        signal_values * np.cos(angular_frequency * relative_time),
        relative_time,
    )
    sine_projection = (2.0 / relative_time[-1]) * np.trapezoid(
        signal_values * np.sin(angular_frequency * relative_time),
        relative_time,
    )
    peak_complex = complex(cosine_projection, -sine_projection)
    return peak_complex / np.sqrt(2.0)


def get_test_target_levels(test_key: str, form_values: dict[str, str]) -> tuple[float, float]:
    if test_key not in TEST_TARGET_FIELDS:
        return 1.0, 1.0

    _, pre_field, event_field = TEST_TARGET_FIELDS[test_key]
    pre_target = safe_percent_value(form_values.get(pre_field), default=1.0)
    event_target = safe_percent_value(form_values.get(event_field), default=pre_target)
    return pre_target, event_target


def safe_percent_value(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        parsed_value = float(str(value).replace(",", ".")) / 100.0
    except ValueError:
        return default
    if parsed_value == 0:
        return default
    return parsed_value


def normalize_signal_to_target(signal_values: np.ndarray, pre_target_pu: float) -> np.ndarray:
    base_value = initial_signal_mean(signal_values, sample_count=3)
    return signal_values / base_value * pre_target_pu


def initial_signal_mean(signal_values: np.ndarray, sample_count: int = 3) -> float:
    values = np.asarray(signal_values[:sample_count], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    mean_value = float(np.mean(values))
    if mean_value == 0:
        return 1.0
    return mean_value


def build_combined_pu_signal(
    validation_result: ValidationResult,
    column: str,
    pre_target_pu: float,
) -> np.ndarray:
    experimental_values = validation_result.scalar_experimental[column].to_numpy(dtype=float)
    model_values = validation_result.scalar_model[column].to_numpy(dtype=float)
    experimental_pu = normalize_signal_to_target(experimental_values, pre_target_pu)
    model_pu = normalize_signal_to_target(model_values, pre_target_pu)
    return (experimental_pu + model_pu) / 2.0


def find_transition_index(
    signal_values: np.ndarray,
    start_level: float,
    end_level: float,
    fraction: float,
    start_index: int,
) -> int | None:
    delta = end_level - start_level
    if abs(delta) < 1e-9:
        return None

    threshold_level = start_level + delta * fraction
    direction = 1.0 if delta > 0 else -1.0
    sustained_points = 2
    max_index = len(signal_values) - sustained_points
    for index in range(max(start_index, 0), max_index + 1):
        window = signal_values[index : index + sustained_points]
        if np.all(direction * (window - threshold_level) >= 0):
            return index
    return None


def find_event_departure_index(
    signal_values: np.ndarray,
    baseline_level: float,
    target_level: float,
    start_index: int = 0,
) -> int | None:
    delta = target_level - baseline_level
    if abs(delta) < 1e-9:
        return None

    direction = 1.0 if delta > 0 else -1.0
    departure_threshold = baseline_level + delta * 0.06
    sustained_points = 3
    max_index = len(signal_values) - sustained_points
    for index in range(max(start_index, 0), max_index + 1):
        window = signal_values[index : index + sustained_points]
        if np.all(direction * (window - departure_threshold) >= 0):
            return index
    return None


def transition_tolerance(
    start_level: float,
    end_level: float,
    tolerance_percent: float = 5.0,
) -> float:
    amplitude = abs(end_level - start_level)
    reference_level = max(abs(start_level), abs(end_level), 1e-6)
    tolerance_fraction = tolerance_percent / 100.0 if tolerance_percent > 0 else 0.05
    return max(amplitude * tolerance_fraction, reference_level * 0.002, 0.0005)


def signal_has_significant_transient(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    start_index: int,
    tolerance: float,
    min_transition_percent: float = 1.0,
    max_duration_s: float = 1.0,
) -> bool:
    if len(signal_values) == 0:
        return False
    start_index = min(max(start_index, 0), len(signal_values) - 1)
    pre_start = max(0, start_index - sustained_points_count(signal_values))
    baseline_window = signal_values[pre_start : start_index + 1]
    if baseline_window.size == 0:
        baseline_window = signal_values[: start_index + 1]
    baseline_level = float(np.median(baseline_window))
    max_end_index = transient_max_end_index(time_values, start_index, max_duration_s=max_duration_s)
    transient_window = signal_values[start_index : max_end_index + 1]
    if transient_window.size == 0:
        return False
    deviation = float(np.max(np.abs(transient_window - baseline_level)))
    spread = float(np.max(transient_window) - np.min(transient_window))
    min_transition_pu = max(min_transition_percent / 100.0, 0.0)
    significance_threshold = max(min_transition_pu, tolerance * 0.25, 0.0025)
    return deviation >= significance_threshold or spread >= significance_threshold * 1.25


def transient_max_end_index(
    time_values: np.ndarray,
    start_index: int,
    max_duration_s: float = 1.0,
) -> int:
    if len(time_values) == 0:
        return 0
    start_index = min(max(start_index, 0), len(time_values) - 1)
    limit_time = float(time_values[start_index]) + max_duration_s
    limit_index = int(np.searchsorted(time_values, limit_time, side="right") - 1)
    return min(max(limit_index, start_index + 1), len(time_values) - 1)


def sustained_points_count(signal_values: np.ndarray) -> int:
    return max(4, min(8, len(signal_values) // 10 or 4))


def find_sustained_band_entry(
    signal_values: np.ndarray,
    target_level: float,
    tolerance: float,
    start_index: int,
    min_points: int,
) -> int | None:
    max_index = len(signal_values) - min_points
    for index in range(max(start_index, 0), max_index + 1):
        window = signal_values[index : index + min_points]
        if np.all(np.abs(window - target_level) <= tolerance):
            return index
    return None


def find_persistent_band_entry(
    signal_values: np.ndarray,
    target_level: float,
    tolerance: float,
    start_index: int,
    max_end_index: int,
) -> int | None:
    min_points = sustained_points_count(signal_values)
    max_index = min(len(signal_values) - min_points, max_end_index - min_points + 1)
    for index in range(max(start_index, 0), max_index + 1):
        window = signal_values[index : index + min_points]
        if not np.all(np.abs(window - target_level) <= tolerance):
            continue
        tail = signal_values[index : max_end_index + 1]
        if tail.size < min_points:
            continue
        inside_tail = np.abs(tail - target_level) <= tolerance
        if np.mean(inside_tail) >= 0.9 and np.all(inside_tail[-min_points:]):
            return index
    return None


def build_window_segments(
    time_values: np.ndarray,
    boundaries: list[tuple[str, int]],
) -> list[WindowSegment]:
    if len(time_values) == 0:
        return [WindowSegment("Janela unica", 0.0, 0.0)]

    clamped_boundaries: list[tuple[str, int]] = []
    max_index = len(time_values) - 1
    previous_index = 0
    for name, raw_index in boundaries:
        index = min(max(raw_index, previous_index), max_index)
        clamped_boundaries.append((name, index))
        previous_index = index

    segments: list[WindowSegment] = []
    for (name, start_index), (_, end_index) in zip(clamped_boundaries[:-1], clamped_boundaries[1:]):
        segments.append(WindowSegment(name, float(time_values[start_index]), float(time_values[end_index])))
    return segments


def detect_event_start_index(
    signal_values: np.ndarray,
    pre_target_pu: float,
    event_target_pu: float,
    test_key: str,
) -> int:
    del test_key
    event_start_index = find_event_departure_index(
        signal_values=signal_values,
        baseline_level=pre_target_pu,
        target_level=event_target_pu,
        start_index=0,
    )
    return 0 if event_start_index is None else event_start_index


def detect_settling_index(
    signal_values: np.ndarray,
    target_level: float,
    tolerance: float,
    start_index: int,
    max_end_index: int | None = None,
) -> int | None:
    min_points = sustained_points_count(signal_values)
    max_index = len(signal_values) - min_points
    if max_end_index is not None:
        max_index = min(max_index, max_end_index - min_points + 1)
    for index in range(max(start_index, 0), max_index + 1):
        window = signal_values[index : index + min_points]
        center_value = float(np.median(window))
        spread = float(np.max(np.abs(window - center_value)))
        slope = float(abs(window[-1] - window[0]))
        confirm_end = min(index + 3 * min_points, len(signal_values))
        confirm_window = signal_values[index:confirm_end]
        confirm_spread = float(np.max(np.abs(confirm_window - target_level)))
        if (
            abs(center_value - target_level) <= tolerance * 0.85
            and spread <= tolerance * 0.6
            and slope <= tolerance * 0.5
            and confirm_spread <= tolerance * 0.9
        ):
            return index
    return None


def detect_limited_settling_index(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    target_level: float,
    tolerance: float,
    start_index: int,
    max_duration_s: float = 1.0,
) -> int:
    if len(signal_values) == 0:
        return 0
    start_index = min(max(start_index, 0), len(signal_values) - 1)
    max_end_index = transient_max_end_index(time_values, start_index, max_duration_s=max_duration_s)
    settling_index = find_persistent_band_entry(
        signal_values=signal_values,
        target_level=target_level,
        tolerance=tolerance,
        start_index=start_index,
        max_end_index=max_end_index,
    )
    if settling_index is not None:
        return min(settling_index, max_end_index)
    settling_index = detect_settling_index(
        signal_values=signal_values,
        target_level=target_level,
        tolerance=tolerance,
        start_index=start_index,
        max_end_index=max_end_index,
    )
    if settling_index is None:
        settling_index = find_sustained_band_entry(
            signal_values=signal_values,
            target_level=target_level,
            tolerance=tolerance,
            start_index=start_index,
            min_points=sustained_points_count(signal_values),
        )
        if settling_index is not None and settling_index > max_end_index:
            settling_index = None
    if settling_index is None:
        return max_end_index
    return min(settling_index, max_end_index)


def detect_limited_stable_index(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    tolerance: float,
    start_index: int,
    max_duration_s: float = 1.0,
) -> int:
    if len(signal_values) == 0:
        return 0
    start_index = min(max(start_index, 0), len(signal_values) - 1)
    max_end_index = transient_max_end_index(time_values, start_index, max_duration_s=max_duration_s)
    stable_window = find_first_stable_window(
        signal_values=signal_values,
        start_index=start_index,
        tolerance=tolerance,
        max_end_index=max_end_index,
    )
    if stable_window is None:
        return max_end_index
    return min(max(stable_window[0], start_index), max_end_index)


def find_first_stable_window(
    signal_values: np.ndarray,
    start_index: int,
    tolerance: float,
    max_end_index: int | None = None,
) -> tuple[int, float] | None:
    min_points = sustained_points_count(signal_values)
    max_index = len(signal_values) - min_points
    if max_end_index is not None:
        max_index = min(max_index, max_end_index - min_points + 1)
    for index in range(max(start_index, 0), max_index + 1):
        window = signal_values[index : index + min_points]
        center_value = float(np.median(window))
        spread = float(np.max(np.abs(window - center_value)))
        slope = float(abs(window[-1] - window[0]))
        confirm_end = min(index + 3 * min_points, len(signal_values))
        confirm_window = signal_values[index:confirm_end]
        confirm_spread = float(np.max(np.abs(confirm_window - center_value)))
        if (
            spread <= tolerance * 0.6
            and slope <= tolerance * 0.45
            and confirm_spread <= tolerance * 0.85
        ):
            return index, center_value
    return None


def detect_recovery_start_index(
    signal_values: np.ndarray,
    during_target_pu: float,
    pre_target_pu: float,
    start_index: int,
) -> int | None:
    departure_index = find_event_departure_index(
        signal_values=signal_values,
        baseline_level=during_target_pu,
        target_level=pre_target_pu,
        start_index=start_index,
    )
    if departure_index is not None:
        return departure_index

    return find_transition_index(
        signal_values=signal_values,
        start_level=during_target_pu,
        end_level=pre_target_pu,
        fraction=0.05,
        start_index=start_index,
    )


def build_test_windows(
    test_key: str,
    form_values: dict[str, str],
    validation_result: ValidationResult,
    tolerance_percent: float = 5.0,
    min_transition_percent: float = 1.0,
) -> dict[str, list[WindowSegment]]:
    time_values = validation_result.scalar_experimental["time"].to_numpy(dtype=float)
    if len(time_values) < 2:
        return {
            column: [WindowSegment("Single window", 0.0, 0.0)]
            for column in SCALAR_COLUMN_ORDER
            if column in validation_result.scalar_experimental.columns and column in validation_result.scalar_model.columns
        }

    if test_key == "steady_state":
        return {
            column: [WindowSegment("Steady state", float(time_values[0]), float(time_values[-1]))]
            for column in SCALAR_COLUMN_ORDER
            if column in validation_result.scalar_experimental.columns and column in validation_result.scalar_model.columns
        }

    sync_column = validation_result.sync_column
    pre_target_pu, event_target_pu = get_test_target_levels(test_key, form_values)
    sync_signal = build_combined_pu_signal(
        validation_result=validation_result,
        column=sync_column,
        pre_target_pu=pre_target_pu,
    )
    sync_smoothed_signal = (
        pd.Series(sync_signal)
        .rolling(window=5, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    event_start_index = detect_event_start_index(
        signal_values=sync_smoothed_signal,
        pre_target_pu=pre_target_pu,
        event_target_pu=event_target_pu,
        test_key=test_key,
    )

    column_signals: dict[str, np.ndarray] = {}
    for column in SCALAR_COLUMN_ORDER:
        if column not in validation_result.scalar_experimental.columns or column not in validation_result.scalar_model.columns:
            continue
        column_signals[column] = (
            pd.Series(build_combined_pu_signal(validation_result, column, pre_target_pu))
            .rolling(window=5, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )

    windows_by_column: dict[str, list[WindowSegment]] = {}
    if test_key in ("step_test", "ramp_test"):
        windows_by_column = build_transition_windows_by_column(
            time_values=time_values,
            validation_result=validation_result,
            pre_target_pu=pre_target_pu,
            event_target_pu=event_target_pu,
            event_start_index=event_start_index,
            test_key=test_key,
            sync_column=sync_column,
            column_signals=column_signals,
            tolerance_percent=tolerance_percent,
            min_transition_percent=min_transition_percent,
        )
    elif test_key == "fault_test":
        windows_by_column = build_fault_windows_by_column(
            time_values=time_values,
            validation_result=validation_result,
            pre_target_pu=pre_target_pu,
            during_target_pu=event_target_pu,
            event_start_index=event_start_index,
            sync_signal=sync_smoothed_signal,
            sync_column=sync_column,
            column_signals=column_signals,
            tolerance_percent=tolerance_percent,
            min_transition_percent=min_transition_percent,
        )
    else:
        for column in SCALAR_COLUMN_ORDER:
            if column in validation_result.scalar_experimental.columns and column in validation_result.scalar_model.columns:
                windows_by_column[column] = [WindowSegment("Single window", float(time_values[0]), float(time_values[-1]))]

    return windows_by_column


def build_transition_windows_by_column(
    time_values: np.ndarray,
    validation_result: ValidationResult,
    pre_target_pu: float,
    event_target_pu: float,
    event_start_index: int,
    test_key: str,
    sync_column: str,
    column_signals: dict[str, np.ndarray],
    tolerance_percent: float,
    min_transition_percent: float,
) -> dict[str, list[WindowSegment]]:
    tolerance = transition_tolerance(pre_target_pu, event_target_pu, tolerance_percent=tolerance_percent)
    sample_count = len(time_values)
    start_index = min(max(event_start_index, 0), sample_count - 1)

    sync_signal = column_signals.get(sync_column)
    if sync_signal is not None:
        base_end_index = detect_limited_settling_index(
            time_values=time_values,
            signal_values=sync_signal,
            target_level=event_target_pu,
            tolerance=tolerance,
            start_index=min(start_index + 1, sample_count - 1),
        )
    else:
        base_end_index = min(start_index + 1, sample_count - 1)
    base_end_index = min(max(base_end_index, start_index + 1), sample_count - 1)

    window_names = (
        ("Pre-step", "Step transient", "Post-step")
        if test_key == "step_test"
        else ("Pre-ramp", "Ramp", "Post-ramp")
    )

    windows_by_column: dict[str, list[WindowSegment]] = {}
    for column in SCALAR_COLUMN_ORDER:
        if column not in validation_result.scalar_experimental.columns or column not in validation_result.scalar_model.columns:
            continue
        signal_values = column_signals.get(column)
        if signal_values is None:
            end_index = base_end_index
            boundaries = [
                (window_names[0], 0),
                (window_names[1], start_index),
                (window_names[2], end_index),
                ("End", sample_count - 1),
            ]
        elif column == sync_column:
            end_index = base_end_index
            boundaries = [
                (window_names[0], 0),
                (window_names[1], start_index),
                (window_names[2], end_index),
                ("End", sample_count - 1),
            ]
        elif signal_has_significant_transient(
            time_values=time_values,
            signal_values=signal_values,
            start_index=start_index,
            tolerance=tolerance,
            min_transition_percent=min_transition_percent,
        ):
            end_index = detect_limited_stable_index(
                time_values=time_values,
                signal_values=signal_values,
                tolerance=tolerance,
                start_index=min(start_index + 1, sample_count - 1),
            )
            end_index = min(max(end_index, start_index + 1), sample_count - 1)
            boundaries = [
                (window_names[0], 0),
                (window_names[1], start_index),
                (window_names[2], end_index),
                ("End", sample_count - 1),
            ]
        else:
            boundaries = [
                (window_names[0], 0),
                (window_names[2], start_index),
                ("End", sample_count - 1),
            ]
        windows_by_column[column] = build_window_segments(time_values, boundaries)
    return windows_by_column


def build_fault_windows_by_column(
    time_values: np.ndarray,
    validation_result: ValidationResult,
    pre_target_pu: float,
    during_target_pu: float,
    event_start_index: int,
    sync_signal: np.ndarray,
    sync_column: str,
    column_signals: dict[str, np.ndarray],
    tolerance_percent: float,
    min_transition_percent: float,
) -> dict[str, list[WindowSegment]]:
    sample_count = len(time_values)
    start_index = min(max(event_start_index, 0), sample_count - 1)
    tolerance = transition_tolerance(pre_target_pu, during_target_pu, tolerance_percent=tolerance_percent)

    base_entry_end = detect_limited_settling_index(
        time_values=time_values,
        signal_values=sync_signal,
        target_level=during_target_pu,
        tolerance=tolerance,
        start_index=min(start_index + 1, sample_count - 1),
    )
    base_entry_end = min(max(base_entry_end, start_index + 1), sample_count - 1)

    recovery_start_index = detect_recovery_start_index(
        signal_values=sync_signal,
        during_target_pu=during_target_pu,
        pre_target_pu=pre_target_pu,
        start_index=base_entry_end,
    )
    if recovery_start_index is None:
        recovery_start_index = base_entry_end
    recovery_start_index = min(max(recovery_start_index, start_index + 1), sample_count - 1)

    base_exit_end = detect_limited_settling_index(
        time_values=time_values,
        signal_values=sync_signal,
        target_level=pre_target_pu,
        tolerance=tolerance,
        start_index=min(recovery_start_index + 1, sample_count - 1),
    )
    base_exit_end = min(max(base_exit_end, recovery_start_index + 1), sample_count - 1)

    windows_by_column: dict[str, list[WindowSegment]] = {}
    for column in SCALAR_COLUMN_ORDER:
        if column not in validation_result.scalar_experimental.columns or column not in validation_result.scalar_model.columns:
            continue
        signal_values = column_signals.get(column)
        if signal_values is None:
            entry_end_index = base_entry_end
            exit_end_index = base_exit_end
        elif column == sync_column:
            entry_end_index = base_entry_end
            exit_end_index = base_exit_end
        else:
            if signal_has_significant_transient(
                time_values=time_values,
                signal_values=signal_values,
                start_index=start_index,
                tolerance=tolerance,
                min_transition_percent=min_transition_percent,
            ):
                entry_end_index = detect_limited_stable_index(
                    time_values=time_values,
                    signal_values=signal_values,
                    tolerance=tolerance,
                    start_index=min(start_index + 1, sample_count - 1),
                )
                entry_end_index = min(max(entry_end_index, start_index + 1), sample_count - 1)
            else:
                entry_end_index = min(start_index + 1, sample_count - 1)

            if signal_has_significant_transient(
                time_values=time_values,
                signal_values=signal_values,
                start_index=recovery_start_index,
                tolerance=tolerance,
                min_transition_percent=min_transition_percent,
            ):
                exit_end_index = detect_limited_stable_index(
                    time_values=time_values,
                    signal_values=signal_values,
                    tolerance=tolerance,
                    start_index=min(recovery_start_index + 1, sample_count - 1),
                )
                exit_end_index = min(max(exit_end_index, recovery_start_index + 1), sample_count - 1)
            else:
                exit_end_index = min(recovery_start_index + 1, sample_count - 1)

        entry_has_transient = signal_values is not None and signal_has_significant_transient(
            time_values=time_values,
            signal_values=signal_values,
            start_index=start_index,
            tolerance=tolerance,
            min_transition_percent=min_transition_percent,
        )
        exit_has_transient = signal_values is not None and signal_has_significant_transient(
            time_values=time_values,
            signal_values=signal_values,
            start_index=recovery_start_index,
            tolerance=tolerance,
            min_transition_percent=min_transition_percent,
        )

        boundaries: list[tuple[str, int]] = [("Pre-disturbance", 0)]
        if entry_has_transient:
            boundaries.append(("Disturbance transient 1", start_index))
            boundaries.append(("During disturbance", entry_end_index))
        else:
            boundaries.append(("During disturbance", start_index))

        if exit_has_transient:
            boundaries.append(("Disturbance transient 2", recovery_start_index))
            boundaries.append(("Post-disturbance", exit_end_index))
        else:
            boundaries.append(("Post-disturbance", recovery_start_index))

        boundaries.append(("End", sample_count - 1))
        windows_by_column[column] = build_window_segments(time_values, boundaries)

    return windows_by_column


def measurement_window_times(
    time_index: pd.TimedeltaIndex,
    reference_split: pd.TimedeltaIndex,
) -> np.ndarray:
    boundaries = [time_index[0]]
    boundaries.extend(reference_split.to_list())
    boundaries.append(time_index[-1])

    centers = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        centers.append((start.total_seconds() + end.total_seconds()) / 2.0)

    return np.asarray(centers, dtype=float)


def synchronize_datasets(
    test_key: str,
    form_values: dict[str, str],
    signal_type: str,
    scalar_experimental: pd.DataFrame,
    scalar_model: pd.DataFrame,
    waveform_experimental: pd.DataFrame | None,
    waveform_model: pd.DataFrame | None,
) -> ValidationResult:
    if test_key == "steady_state":
        sync_column = "voltage"
        sync_label = "Sinusoidal voltage"
        waveform_sync_column = None
        if waveform_experimental is not None and waveform_model is not None:
            waveform_sync_column = waveform_reference_voltage_column(waveform_experimental)
            time_shift_s = estimate_time_shift_by_correlation(
                waveform_experimental["time"].to_numpy(),
                waveform_experimental[waveform_sync_column].to_numpy(),
                waveform_model["time"].to_numpy(),
                waveform_model[waveform_reference_voltage_column(waveform_model)].to_numpy(),
            )
        elif sync_column in scalar_experimental.columns and sync_column in scalar_model.columns:
            time_shift_s = estimate_time_shift_by_correlation(
                scalar_experimental["time"].to_numpy(),
                scalar_experimental[sync_column].to_numpy(),
                scalar_model["time"].to_numpy(),
                scalar_model[sync_column].to_numpy(),
            )
            sync_label = SCALAR_DISPLAY_LABELS[sync_column]
        else:
            raise ValueError("It was not possible to synchronize the steady-state test without a valid voltage signal.")

        exp_zoom, model_zoom, zoom_time_s = shift_and_trim_series(
            pd.Series(
                (
                    waveform_experimental[waveform_sync_column or waveform_reference_voltage_column(waveform_experimental)].to_numpy()
                    if waveform_experimental is not None
                    else scalar_experimental[sync_column].to_numpy()
                ),
                index=(
                    waveform_experimental["time"].to_numpy()
                    if waveform_experimental is not None
                    else scalar_experimental["time"].to_numpy()
                ),
            ),
            pd.Series(
                (
                    waveform_model[waveform_reference_voltage_column(waveform_model)].to_numpy()
                    if waveform_model is not None
                    else scalar_model[sync_column].to_numpy()
                ),
                index=(
                    waveform_model["time"].to_numpy()
                    if waveform_model is not None
                    else scalar_model["time"].to_numpy()
                ),
            ),
            time_shift_s,
            center_time_s=0.0,
            zoom_half_window_s=0.08,
        )
    else:
        sync_column = TEST_QUANTITY_TO_INTERNAL[form_values[TEST_SYNC_FIELD[test_key]]]
        if sync_column not in scalar_experimental.columns or sync_column not in scalar_model.columns:
            raise ValueError(
                "The quantity selected for synchronization must exist in the imported signals."
            )

        pre_target_pu, event_target_pu = get_test_target_levels(test_key, form_values)
        exp_event_time = detect_event_time_from_targets(
            scalar_experimental["time"].to_numpy(),
            scalar_experimental[sync_column].to_numpy(),
            pre_target_pu=pre_target_pu,
            event_target_pu=event_target_pu,
            test_key=test_key,
        )
        model_event_time = detect_event_time_from_targets(
            scalar_model["time"].to_numpy(),
            scalar_model[sync_column].to_numpy(),
            pre_target_pu=pre_target_pu,
            event_target_pu=event_target_pu,
            test_key=test_key,
        )
        time_shift_s = model_event_time - exp_event_time
        sync_label = SCALAR_DISPLAY_LABELS[sync_column]
        exp_zoom, model_zoom, zoom_time_s = shift_and_trim_series(
            pd.Series(scalar_experimental[sync_column].to_numpy(), index=scalar_experimental["time"].to_numpy()),
            pd.Series(scalar_model[sync_column].to_numpy(), index=scalar_model["time"].to_numpy()),
            time_shift_s,
            center_time_s=exp_event_time,
            zoom_half_window_s=max(_time_span(scalar_experimental) * 0.1, 0.2),
        )

    aligned_scalar_experimental, aligned_scalar_model = shift_and_trim_dataframe_pair(
        scalar_experimental,
        scalar_model,
        time_shift_s,
    )
    aligned_scalar_experimental, aligned_scalar_model = normalize_pair_duration(
        aligned_scalar_experimental,
        aligned_scalar_model,
    )

    aligned_waveform_experimental = None
    aligned_waveform_model = None
    if waveform_experimental is not None and waveform_model is not None:
        aligned_waveform_experimental, aligned_waveform_model = shift_and_trim_dataframe_pair(
            waveform_experimental,
            waveform_model,
            time_shift_s,
        )
        aligned_waveform_experimental, aligned_waveform_model = normalize_pair_duration(
            aligned_waveform_experimental,
            aligned_waveform_model,
        )

    return ValidationResult(
        signal_type=signal_type,
        sync_label=sync_label,
        sync_column=sync_column,
        scalar_experimental=aligned_scalar_experimental,
        scalar_model=aligned_scalar_model,
        waveform_experimental=aligned_waveform_experimental,
        waveform_model=aligned_waveform_model,
        zoom_experimental=exp_zoom,
        zoom_model=model_zoom,
        zoom_time_s=zoom_time_s,
    )


def estimate_time_shift_by_correlation(
    exp_time: np.ndarray,
    exp_values: np.ndarray,
    model_time: np.ndarray,
    model_values: np.ndarray,
) -> float:
    exp_dt = estimate_sample_time(exp_time)
    model_dt = estimate_sample_time(model_time)
    if not np.isclose(exp_dt, model_dt, rtol=1e-3, atol=1e-6):
        raise ValueError("The signals must have the same sampling rate for synchronization.")

    exp_centered = exp_values - np.mean(exp_values)
    model_centered = model_values - np.mean(model_values)
    correlation = sig.correlate(model_centered, exp_centered, mode="full")
    lags = sig.correlation_lags(len(model_centered), len(exp_centered), mode="full")
    lag_samples = int(lags[int(np.argmax(correlation))])
    return float(lag_samples * exp_dt)


def detect_event_time(time_values: np.ndarray, signal_values: np.ndarray) -> float:
    if len(signal_values) < 5:
        return float(time_values[0])

    window_length = min(len(signal_values) if len(signal_values) % 2 == 1 else len(signal_values) - 1, 31)
    if window_length < 5:
        smoothed = signal_values
    else:
        smoothed = sig.savgol_filter(signal_values, window_length=window_length, polyorder=2)

    derivative = np.gradient(smoothed, time_values)
    event_index = int(np.argmax(np.abs(derivative)))
    return float(time_values[event_index])


def detect_event_time_from_targets(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    pre_target_pu: float,
    event_target_pu: float,
    test_key: str,
) -> float:
    normalized_signal = normalize_signal_to_target(signal_values, pre_target_pu)
    smoothed_signal = (
        pd.Series(normalized_signal)
        .rolling(window=3, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )

    event_index = detect_event_start_index(
        signal_values=smoothed_signal,
        pre_target_pu=pre_target_pu,
        event_target_pu=event_target_pu,
        test_key=test_key,
    )
    if event_index is None:
        return detect_event_time(time_values, signal_values)
    return float(time_values[event_index])


def shift_and_trim_dataframe_pair(
    experimental: pd.DataFrame,
    model: pd.DataFrame,
    time_shift_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experimental_aligned = experimental.copy()
    model_aligned = model.copy()
    model_aligned["time"] = model_aligned["time"] - time_shift_s

    overlap_start = max(
        float(experimental_aligned["time"].min()),
        float(model_aligned["time"].min()),
    )
    overlap_end = min(
        float(experimental_aligned["time"].max()),
        float(model_aligned["time"].max()),
    )
    if overlap_end <= overlap_start:
        raise ValueError("There was no temporal overlap after synchronization.")

    experimental_aligned = experimental_aligned[
        experimental_aligned["time"].between(overlap_start, overlap_end)
    ].copy()
    model_aligned = model_aligned[
        model_aligned["time"].between(overlap_start, overlap_end)
    ].copy()

    experimental_aligned["time"] = experimental_aligned["time"] - overlap_start
    model_aligned["time"] = model_aligned["time"] - overlap_start

    experimental_aligned = experimental_aligned.reset_index(drop=True)
    model_aligned = model_aligned.reset_index(drop=True)
    return experimental_aligned, model_aligned


def trim_pair_edges_by_time(
    experimental: pd.DataFrame,
    model: pd.DataFrame,
    trim_start_s: float,
    trim_end_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_duration = min(
        float(experimental["time"].max()),
        float(model["time"].max()),
    )
    start_time = max(0.0, trim_start_s)
    end_time = max_duration - max(0.0, trim_end_s)

    if end_time <= start_time:
        return experimental.copy(), model.copy()

    trimmed_experimental = experimental[
        experimental["time"].between(start_time, end_time)
    ].copy()
    trimmed_model = model[
        model["time"].between(start_time, end_time)
    ].copy()

    if trimmed_experimental.empty or trimmed_model.empty:
        return experimental.copy(), model.copy()

    trimmed_experimental["time"] = trimmed_experimental["time"] - start_time
    trimmed_model["time"] = trimmed_model["time"] - start_time

    return (
        trimmed_experimental.reset_index(drop=True),
        trimmed_model.reset_index(drop=True),
    )


def drop_partial_edge_windows(
    experimental: pd.DataFrame,
    model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(experimental) <= 2 or len(model) <= 2:
        return experimental, model

    trimmed_experimental = experimental.iloc[1:-1].copy().reset_index(drop=True)
    trimmed_model = model.iloc[1:-1].copy().reset_index(drop=True)
    if trimmed_experimental.empty or trimmed_model.empty:
        return experimental, model

    return normalize_pair_duration(trimmed_experimental, trimmed_model)


def normalize_pair_duration(
    experimental: pd.DataFrame,
    model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_start = max(
        float(experimental["time"].min()),
        float(model["time"].min()),
    )
    common_end = min(
        float(experimental["time"].max()),
        float(model["time"].max()),
    )
    if common_end <= common_start:
        return experimental.reset_index(drop=True), model.reset_index(drop=True)

    experimental_window = experimental[
        experimental["time"].between(common_start, common_end)
    ].copy()
    model_window = model[
        model["time"].between(common_start, common_end)
    ].copy()
    if len(experimental_window) < 2 or len(model_window) < 2:
        return experimental_window.reset_index(drop=True), model_window.reset_index(drop=True)

    target_dt = max(
        estimate_sample_time(experimental_window["time"].to_numpy(dtype=float)),
        estimate_sample_time(model_window["time"].to_numpy(dtype=float)),
    )
    sample_count = max(int(np.floor((common_end - common_start) / target_dt)) + 1, 2)
    common_time = common_start + np.arange(sample_count) * target_dt
    common_time = common_time[common_time <= common_end]
    if len(common_time) < 2:
        common_time = np.array([common_start, common_end], dtype=float)

    normalized_experimental = interpolate_dataframe_to_timebase(
        experimental_window,
        common_time,
    )
    normalized_model = interpolate_dataframe_to_timebase(
        model_window,
        common_time,
    )

    normalized_experimental["time"] = normalized_experimental["time"] - common_start
    normalized_model["time"] = normalized_model["time"] - common_start

    return normalized_experimental.reset_index(drop=True), normalized_model.reset_index(drop=True)


def interpolate_dataframe_to_timebase(
    dataset: pd.DataFrame,
    common_time: np.ndarray,
) -> pd.DataFrame:
    source_time = dataset["time"].to_numpy(dtype=float)
    interpolated = {"time": common_time.copy()}
    for column in dataset.columns:
        if column == "time":
            continue
        interpolated[column] = np.interp(
            common_time,
            source_time,
            dataset[column].to_numpy(dtype=float),
        )
    return pd.DataFrame(interpolated)


def shift_and_trim_series(
    experimental: pd.Series,
    model: pd.Series,
    time_shift_s: float,
    center_time_s: float,
    zoom_half_window_s: float,
) -> tuple[pd.Series, pd.Series, float]:
    exp_df = pd.DataFrame({"time": experimental.index.to_numpy(dtype=float), "value": experimental.to_numpy(dtype=float)})
    model_df = pd.DataFrame({"time": model.index.to_numpy(dtype=float), "value": model.to_numpy(dtype=float)})
    aligned_exp, aligned_model = shift_and_trim_dataframe_pair(exp_df, model_df, time_shift_s)

    shifted_center = center_time_s - max(float(exp_df["time"].min()), float(model_df["time"].min()) - time_shift_s)
    shifted_center = min(
        max(shifted_center, float(aligned_exp["time"].min())),
        float(aligned_exp["time"].max()),
    )
    zoom_start = max(float(aligned_exp["time"].min()), shifted_center - zoom_half_window_s)
    zoom_end = min(float(aligned_exp["time"].max()), shifted_center + zoom_half_window_s)

    exp_zoom = aligned_exp[aligned_exp["time"].between(zoom_start, zoom_end)]
    model_zoom = aligned_model[aligned_model["time"].between(zoom_start, zoom_end)]

    return (
        pd.Series(exp_zoom["value"].to_numpy(), index=exp_zoom["time"].to_numpy()),
        pd.Series(model_zoom["value"].to_numpy(), index=model_zoom["time"].to_numpy()),
        shifted_center,
    )


def _time_span(dataset: pd.DataFrame) -> float:
    return float(dataset["time"].max() - dataset["time"].min())


def _window_length_for_frequency(nominal_grid_freq: float) -> int:
    if nominal_grid_freq == 60.0:
        return 12
    if nominal_grid_freq == 50.0:
        return 10
    raise ValueError("The nominal frequency must be 50 or 60 Hz.")


def _count_internal_selection(signal_type: str, selections: list[str]) -> dict[str, int]:
    counts = {internal_name: 0 for internal_name in USER_SELECTION_TO_INTERNAL[signal_type].values()}
    for selection in normalize_user_selections(selections):
        internal_name = USER_SELECTION_TO_INTERNAL[signal_type].get(selection, "ignore")
        counts[internal_name] = counts.get(internal_name, 0) + 1
    return counts


def _common_scalar_columns(selections_by_side: dict[str, list[str]]) -> list[str]:
    experimental_internal = {
        USER_SELECTION_TO_INTERNAL["scalar"].get(selection)
        for selection in normalize_user_selections(selections_by_side["experimental"])
    }
    model_internal = {
        USER_SELECTION_TO_INTERNAL["scalar"].get(selection)
        for selection in normalize_user_selections(selections_by_side["model"])
    }

    common_columns = []
    for internal_name in SCALAR_COLUMN_ORDER:
        if internal_name in experimental_internal and internal_name in model_internal:
            common_columns.append(internal_name)

    return common_columns


def translate_side(side: str) -> str:
    return "experimental signals" if side == "experimental" else "model signals"


def normalize_user_selections(selections: list[str]) -> list[str]:
    return [selection if selection else IGNORE_SELECTION for selection in selections]


def parse_time_series(series: pd.Series) -> pd.Series:
    cleaned = clean_string_series(series)
    non_empty = cleaned.dropna()
    if non_empty.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    text_values = non_empty.astype(str)
    looks_like_datetime = text_values.str.contains(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True).any()
    looks_like_datetime = looks_like_datetime or text_values.str.contains(r"^\d{1,2}:\d{2}(:\d{2})?", regex=True).any()
    looks_like_datetime = looks_like_datetime or text_values.str.contains(r"[Tt].*\d{1,2}:\d{2}", regex=True).any()
    if looks_like_datetime:
        datetime_values = pd.to_datetime(non_empty, errors="coerce")
        if datetime_values.notna().all():
            base_time = datetime_values.iloc[0]
            seconds = (datetime_values - base_time).dt.total_seconds()
            result = pd.Series(np.nan, index=series.index, dtype=float)
            result.loc[non_empty.index] = seconds.to_numpy(dtype=float)
            return result

    return parse_numeric_series(series)


def parse_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = clean_string_series(series)
    normalized = cleaned.map(normalize_numeric_string)
    return pd.to_numeric(normalized, errors="coerce")


def clean_string_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("\u00a0", "", regex=False).str.strip()
    cleaned = cleaned.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "null": np.nan,
            "NULL": np.nan,
        }
    )
    return cleaned


def normalize_numeric_string(value: object) -> object:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(" ", "")
    if not text:
        return np.nan

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "")
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
        return text

    if "," in text:
        comma_number = text.count(",")
        if comma_number > 1:
            text = text.replace(".", "")
            last_comma = text.rfind(",")
            text = text[:last_comma].replace(",", "") + "." + text[last_comma + 1:]
            return text

        if "e" in text.lower():
            return text.replace(",", ".")

        integer_part, decimal_part = text.split(",", 1)
        if decimal_part.isdigit() and len(decimal_part) == 3 and integer_part.lstrip("+-").isdigit():
            return text.replace(",", "")

        return text.replace(",", ".")

    if "." in text and text.count(".") > 1:
        last_dot = text.rfind(".")
        text = text[:last_dot].replace(".", "") + "." + text[last_dot + 1:]

    return text


def split_samples(data, cycles_in_window):
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    elif isinstance(data, pd.Series):
        pass
    else:
        raise TypeError("'data' should be a pandas Series or Dataframe")

    if cycles_in_window % 0.5 != 0:
        raise ValueError("'cycles_in_window' must be a multiple of 0.5")

    ts = data.index[1] - data.index[0]
    fs = 1 / ts.total_seconds()
    sos = sig.butter(N=5, Wn=[30, 80], btype="bandpass", fs=fs, output="sos")
    sos_butter = np.vstack([sos])
    grid_voltage_samples_filtered_v = sig.sosfilt(sos_butter, data)
    zero_crossings_position = np.where(np.diff(np.sign(grid_voltage_samples_filtered_v)))[0] + 1
    last_zc_window = np.asarray(range(0, len(zero_crossings_position), int(cycles_in_window * 2)))
    split_window_position = []
    for i in range(0, len(last_zc_window)):
        split_window_position.append(zero_crossings_position[last_zc_window[i]])
    return data.index[split_window_position]


def select_window(data, splits, window):
    if not (isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)):
        raise TypeError("'data' should be a pandas Series or Dataframe")

    if isinstance(window, int):
        if window >= 1:
            w_slice = [window - 1, window + 1]
        elif window <= -1:
            w_slice = [len(splits) + window - 1, len(splits) + window + 1]
    else:
        if window[0] >= 1 and window[1] >= 1:
            w_slice = [window[0] - 1, window[1] + 1]
        elif window[0] <= -1 and window[1] <= -1:
            w_slice = [len(splits) + window[0] - 1, len(splits) + window[1] + 1]
        elif window[0] >= 1 and window[1] <= -1:
            w_slice = [window[0] - 1, len(splits) + window[1] + 1]
    idx = splits[w_slice[0]:w_slice[1]]
    idx_iloc1 = data.index.get_loc(idx[0])
    idx_iloc2 = data.index.get_loc(idx[-1])
    return data.iloc[idx_iloc1:idx_iloc2]


def map_rms(split_samples_columns: pd.DataFrame):
    rms_windows_columns = pd.DataFrame(
        np.full_like(split_samples_columns, np.sqrt(np.mean(np.power(split_samples_columns, 2)))),
        split_samples_columns.index,
    )

    return rms_windows_columns


def fundamental_component_measurement(
    samples: pd.DataFrame,
    nominal_grid_freq: float,
    reference_split=None,
):
    if nominal_grid_freq == 60.0:
        window_len = 12
    elif nominal_grid_freq == 50.0:
        window_len = 10
    else:
        raise ValueError("The freq should be 50 or 60 Hz")

    if reference_split is None:
        indexes = split_samples(samples, window_len)
    else:
        indexes = reference_split
    split_window_position = samples.index.get_indexer(indexes)
    indexes = indexes.insert(0, samples.index[0])
    indexes = indexes.insert(len(indexes), samples.index[-1])
    grid_frequency_each_window_Hz = window_len / np.subtract(
        indexes[1:].total_seconds(),
        indexes[:-1].total_seconds(),
    )
    split_samples_columns = np.split(samples, split_window_position)

    component_amplitudes_ck = []
    phase_angle = []
    for j in range(0, len(grid_frequency_each_window_Hz)):
        w_rads_p_s = np.asarray(
            2
            * np.pi
            * grid_frequency_each_window_Hz[j]
            * split_samples_columns[j].index.total_seconds()
        )
        real_part_ak = float(np.dot(
            np.transpose(split_samples_columns[j].values),
            np.cos(w_rads_p_s),
        ) / len(split_samples_columns[j]))
        imaginary_part_bk = float(np.dot(
            np.transpose(split_samples_columns[j].values),
            np.sin(w_rads_p_s),
        ) / len(split_samples_columns[j]))
        component_amplitudes_ck.append(
            np.sqrt((2 * real_part_ak) ** 2 + (2 * imaginary_part_bk) ** 2)
        )
        if imaginary_part_bk < 0:
            phase_angle.append(math.pi + math.atan(real_part_ak / imaginary_part_bk))
        elif imaginary_part_bk > 0:
            phase_angle.append(math.atan(real_part_ak / imaginary_part_bk))
        elif imaginary_part_bk == 0 and real_part_ak > 0:
            phase_angle.append(math.pi)
        elif imaginary_part_bk == 0 and real_part_ak < 0:
            phase_angle.append(-math.pi)
        else:
            phase_angle.append(0)
    rms_comp = np.array(component_amplitudes_ck).reshape((1, len(component_amplitudes_ck))) / np.sqrt(2)
    return rms_comp, phase_angle, grid_frequency_each_window_Hz


def calc_rms_phasor_from_waveform(amplitude_VorA: np.array, reference_angle_rad: np.array):
    phasor = amplitude_VorA * np.sin(reference_angle_rad) - 1j * amplitude_VorA * np.cos(reference_angle_rad)
    return phasor


def calc_zero_pos_neg_seq_from_abc_phasor(Vabc_phasor_vector: np.array):
    a120, a240 = np.exp(1j * 2 * np.pi / 3), np.exp(1j * 4 * np.pi / 3)
    T_abc_to_zpn = 1 / 3 * np.array(
        [
            [1, 1, 1],
            [1, a120, a240],
            [1, a240, a120],
        ]
    )
    Vzpn_phasor_vector = T_abc_to_zpn @ Vabc_phasor_vector
    return Vzpn_phasor_vector
