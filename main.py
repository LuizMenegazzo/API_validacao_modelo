from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import textwrap

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from app import AssessmentProject
from app.project_storage import list_saved_assessment_projects, load_assessment_project, save_assessment_project
from app.signal_processing import (
    SCALAR_COLUMN_ORDER,
    SCALAR_DISPLAY_LABELS,
    SINUSOIDAL_DISPLAY_LABELS,
    TEST_QUANTITY_TO_INTERNAL,
    ValidationResult,
    WindowSegment,
    build_test_windows,
    detect_cycle_boundaries,
    read_csv_headers,
    run_validation_pipeline,
    validate_import_configuration,
)


class ModelComparisonApp:
    WATERMARK_TEXT = "Model comparison software developed by Luiz Fernando Menegazzo - UFSM"
    SYSTEM_CONFIGURATION_OPTIONS = (
        "Single-phase",
        "Three-phase, 3-wire",
        "Three-phase, 4-wire",
    )
    EQUIVALENT_INITIAL_FINAL_MESSAGE = "Does not apply: equivalent initial and final values"
    NO_SIGNIFICANT_OSCILLATION_MESSAGE = "No significant oscillation in the segment"
    HIGHER_IS_BETTER_METRICS = {"transient_pearson_r", "transient_r2"}
    TIME_METRIC_IDS = {
        "transient_rise_fall_time",
        "transient_settling_time",
        "transient_response_time",
        "transient_reaction_time",
        "transient_delay_time",
    }
    TRANSIENT_FEATURE_METRIC_MAP = {
        "transient_overshoot_undershoot": ("overshoot_pct", "Overshoot/Undershoot"),
        "transient_rise_fall_time": ("rise_fall_time_s", "Rise/Fall time"),
        "transient_settling_time": ("settling_time_s", "Settling time"),
        "transient_response_time": ("response_time_s", "Response time"),
        "transient_reaction_time": ("reaction_time_s", "Reaction time"),
        "transient_relative_overshoot_undershoot": ("overshoot_pct", "Overshoot/Undershoot"),
        "transient_relative_rise_fall_time": ("rise_fall_time_s", "Rise/Fall time"),
        "transient_relative_settling_time": ("settling_time_s", "Settling time"),
        "transient_relative_response_time": ("response_time_s", "Response time"),
        "transient_relative_reaction_time": ("reaction_time_s", "Reaction time"),
    }
    RELATIVE_TRANSIENT_FEATURE_METRICS = {
        "transient_relative_overshoot_undershoot",
        "transient_relative_rise_fall_time",
        "transient_relative_settling_time",
        "transient_relative_response_time",
        "transient_relative_reaction_time",
    }
    RELATIVE_TRANSIENT_FEATURE_BASE_METRIC = {
        "transient_relative_overshoot_undershoot": "transient_overshoot_undershoot",
        "transient_relative_rise_fall_time": "transient_rise_fall_time",
        "transient_relative_settling_time": "transient_settling_time",
        "transient_relative_response_time": "transient_response_time",
        "transient_relative_reaction_time": "transient_reaction_time",
    }
    NOT_IMPLEMENTED_METRICS = {
        "transient_iae",
        "transient_ise",
        "transient_itae",
        "transient_itse",
    }
    DEFAULT_LIMITS = {
        "phase_error": (7, 15),
        "thd_total_error": (1.5, 3),
        "harmonic_even_amplitude_mean_error": (0.5, 1),
        "harmonic_odd_amplitude_mean_error": (0.5, 1),
        "frequency_mean_error": (0.3, 0.6),
        "max_harmonic_error": (0.6, 1.2),
        "transient_overshoot_undershoot": (3.5, 7),
        "transient_rise_fall_time": (100, 200),
        "transient_response_time": (100, 200),
        "transient_reaction_time": (100, 200),
        "transient_settling_time": (200, 300),
        "transient_relative_overshoot_undershoot": (10, 25),
        "transient_relative_rise_fall_time": (10, 25),
        "transient_relative_response_time": (10, 25),
        "transient_relative_reaction_time": (10, 25),
        "transient_relative_settling_time": (10, 25),
        "transient_delay_time": (200, 500),
        "transient_pearson_r": (0.85, 0.65),
        "transient_r2": (0.85, 0.65),
    }
    TRANSIENT_NORMALIZATION_MODES = (
        ("mean_value", "Normalize by mean value"),
        ("event_amplitude", "Normalize by event amplitude"),
        (
            "origin_event_percent",
            "Normalize by % of the event amplitude of the quantity that originated the event",
        ),
    )
    SCORE_MODE_OPTIONS = (
        ("continuous", "Continuous by metric value"),
        ("threshold", "Threshold-based bands"),
    )
    WINDOW_AGGREGATION_OPTIONS = (
        ("average", "Average of windows"),
        ("worst", "Worst window"),
        ("duration_weighted", "Weighted average by window duration"),
    )
    SCORE_CATEGORY_DEFINITIONS = (
        {
            "id": "waveform_fidelity",
            "title": "Waveform fidelity",
            "adjusted_variants": ("phase_adjusted",),
            "metrics": (
                "waveform_mean_error",
                "waveform_nrmse",
                "tve",
                "phase_error",
                "frequency_mean_error",
            ),
        },
        {
            "id": "spectral_fidelity",
            "title": "Spectral and harmonic fidelity",
            "adjusted_variants": ("phase_adjusted",),
            "metrics": (
                "fundamental_amplitude_error",
                "thd_total_error",
                "harmonic_even_amplitude_mean_error",
                "harmonic_odd_amplitude_mean_error",
                "max_harmonic_error",
            ),
        },
        {
            "id": "steady_state_operating",
            "title": "Steady-state operating point accuracy",
            "adjusted_variants": (),
            "metrics": (
                "steady_mean_error",
                "steady_nrmse",
                "steady_residual_rms",
                "steady_ssi",
            ),
        },
        {
            "id": "steady_state_variability",
            "title": "Steady-state oscillation and variability",
            "adjusted_variants": (),
            "metrics": (
                "steady_std_error",
                "steady_peak_to_peak_error",
            ),
        },
        {
            "id": "transient_magnitude",
            "title": "Transient magnitude accuracy",
            "adjusted_variants": ("delay_adjusted",),
            "metrics": (
                "transient_mean_error",
                "transient_nrmse",
                "transient_overshoot_undershoot",
                "transient_relative_overshoot_undershoot",
            ),
        },
        {
            "id": "transient_timing",
            "title": "Transient timing accuracy",
            "adjusted_variants": ("delay_adjusted",),
            "metrics": (
                "transient_rise_fall_time",
                "transient_relative_rise_fall_time",
                "transient_settling_time",
                "transient_relative_settling_time",
                "transient_response_time",
                "transient_relative_response_time",
                "transient_reaction_time",
                "transient_relative_reaction_time",
                "transient_delay_time",
            ),
        },
        {
            "id": "transient_similarity",
            "title": "Shape transient similarity - extra indices",
            "adjusted_variants": ("delay_adjusted",),
            "metrics": (
                "transient_pearson_r",
                "transient_r2",
            ),
        },
    )

    SIGNAL_IMPORT_DEFINITIONS = {
        "sinusoidal": {
            "screen_title": "Upload Sinusoidal Signals",
            "subtitle": "Select one .csv file for the experimental signals and another for the model signals.",
            "classification_options": (
                "Time",
                "Voltage",
                "Current",
                "Voltage A",
                "Voltage B",
                "Voltage C",
                "Current A",
                "Current B",
                "Current C",
                "Ignore signal",
            ),
            "result_title": "Validation with sinusoidal signals",
        },
        "scalar": {
            "screen_title": "Upload Scalar Signals",
            "subtitle": "Select one .csv file for the experimental quantities and another for the model quantities.",
            "classification_options": (
                "Time",
                "RMS voltage or DC",
                "RMS current or DC",
                "Frequency",
                "Active power",
                "Reactive power",
                "RMS voltage A or DC",
                "RMS voltage B or DC",
                "RMS voltage C or DC",
                "RMS current A or DC",
                "RMS current B or DC",
                "RMS current C or DC",
                "Active power A",
                "Active power B",
                "Active power C",
                "Reactive power A",
                "Reactive power B",
                "Reactive power C",
                "Zero-sequence voltage",
                "Positive-sequence voltage",
                "Negative-sequence voltage",
                "Zero-sequence current",
                "Positive-sequence current",
                "Negative-sequence current",
                "Voltage zero-sequence unbalance",
                "Voltage negative-sequence unbalance",
                "Current zero-sequence unbalance",
                "Current negative-sequence unbalance",
                "Ignore signal",
            ),
            "result_title": "Validation with scalar signals",
        },
    }

    TEST_DEFINITIONS = {
        "steady_state": {
            "button_label": "Steady-state tests",
            "screen_title": "Steady-state tests",
            "fields": [
                {
                    "name": "nominal_power_w",
                    "label": "Equipment nominal power in W",
                    "kind": "entry",
                },
                {
                    "name": "nominal_voltage_v",
                    "label": "Equipment nominal voltage in V",
                    "kind": "entry",
                },
                {
                    "name": "system_configuration",
                    "label": "Electrical system",
                    "kind": "combobox",
                    "options": SYSTEM_CONFIGURATION_OPTIONS,
                },
                {
                    "name": "test_power_percent",
                    "label": "Test power as % of nominal power",
                    "kind": "entry",
                },
            ],
        },
        "step_test": {
            "button_label": "Step test",
            "screen_title": "Step test",
            "fields": [
                {
                    "name": "nominal_power_w",
                    "label": "Equipment nominal power in W",
                    "kind": "entry",
                },
                {
                    "name": "nominal_voltage_v",
                    "label": "Equipment nominal voltage in V",
                    "kind": "entry",
                },
                {
                    "name": "system_configuration",
                    "label": "Electrical system",
                    "kind": "combobox",
                    "options": SYSTEM_CONFIGURATION_OPTIONS,
                },
                {
                    "name": "step_quantity",
                    "label": "Quantity where the step occurred",
                    "kind": "combobox",
                    "options": (
                        "Voltage",
                        "Current",
                        "Frequency",
                        "Active power",
                        "Reactive power",
                        "Voltage A",
                        "Voltage B",
                        "Voltage C",
                        "Current A",
                        "Current B",
                        "Current C",
                        "Positive-sequence voltage",
                        "Negative-sequence voltage",
                        "Zero-sequence voltage",
                        "Positive-sequence current",
                        "Negative-sequence current",
                        "Zero-sequence current",
                    ),
                },
                {
                    "name": "pre_step_percent",
                    "label": "Pre-step value of the selected quantity in %",
                    "kind": "entry",
                },
                {
                    "name": "post_step_percent",
                    "label": "Post-step value of the selected quantity in %",
                    "kind": "entry",
                },
            ],
        },
        "fault_test": {
            "button_label": "Transient disturbance test",
            "screen_title": "Transient disturbance test",
            "fields": [
                {
                    "name": "nominal_power_w",
                    "label": "Equipment nominal power in W",
                    "kind": "entry",
                },
                {
                    "name": "nominal_voltage_v",
                    "label": "Equipment nominal voltage in V",
                    "kind": "entry",
                },
                {
                    "name": "system_configuration",
                    "label": "Electrical system",
                    "kind": "combobox",
                    "options": SYSTEM_CONFIGURATION_OPTIONS,
                },
                {
                    "name": "fault_quantity",
                    "label": "Quantity where the disturbance occurred",
                    "kind": "combobox",
                    "options": (
                        "Voltage",
                        "Current",
                        "Frequency",
                        "Active power",
                        "Reactive power",
                        "Voltage A",
                        "Voltage B",
                        "Voltage C",
                        "Current A",
                        "Current B",
                        "Current C",
                        "Positive-sequence voltage",
                        "Negative-sequence voltage",
                        "Zero-sequence voltage",
                        "Positive-sequence current",
                        "Negative-sequence current",
                        "Zero-sequence current",
                    ),
                },
                {
                    "name": "pre_fault_percent",
                    "label": "Pre-disturbance value of the selected quantity in %",
                    "kind": "entry",
                },
                {
                    "name": "during_fault_percent",
                    "label": "During-disturbance value of the selected quantity in %",
                    "kind": "entry",
                },
            ],
        },
        "ramp_test": {
            "button_label": "Ramp test",
            "screen_title": "Ramp test",
            "fields": [
                {
                    "name": "nominal_power_w",
                    "label": "Equipment nominal power in W",
                    "kind": "entry",
                },
                {
                    "name": "nominal_voltage_v",
                    "label": "Equipment nominal voltage in V",
                    "kind": "entry",
                },
                {
                    "name": "system_configuration",
                    "label": "Electrical system",
                    "kind": "combobox",
                    "options": SYSTEM_CONFIGURATION_OPTIONS,
                },
                {
                    "name": "ramp_quantity",
                    "label": "Quantity where the ramp occurred",
                    "kind": "combobox",
                    "options": (
                        "Voltage",
                        "Current",
                        "Frequency",
                        "Active power",
                        "Reactive power",
                        "Voltage A",
                        "Voltage B",
                        "Voltage C",
                        "Current A",
                        "Current B",
                        "Current C",
                        "Positive-sequence voltage",
                        "Negative-sequence voltage",
                        "Zero-sequence voltage",
                        "Positive-sequence current",
                        "Negative-sequence current",
                        "Zero-sequence current",
                    ),
                },
                {
                    "name": "pre_ramp_percent",
                    "label": "Pre-ramp value of the selected quantity in %",
                    "kind": "entry",
                },
                {
                    "name": "post_ramp_percent",
                    "label": "Post-ramp value of the selected quantity in %",
                    "kind": "entry",
                },
            ],
        },
    }

    METRIC_GROUP_DEFINITIONS = (
        {
            "id": "waveform",
            "title": "Voltage and current waveform metrics",
            "subtitle": "Available only when the signals were imported as sinusoidal waveforms.",
            "signal_type": "sinusoidal",
            "sections": (
                {
                    "title": "Main metrics",
                    "metrics": (
                        ("waveform_mean_error", "Normalized mean error"),
                        ("waveform_nrmse", "Normalized root mean square error (NRMSE)"),
                        ("fundamental_amplitude_error", "Relative error of fundamental amplitude"),
                        ("phase_error", "Phase delay"),
                        ("thd_total_error", "Total THD error"),
                        ("harmonic_even_amplitude_mean_error", "Mean error of even harmonic amplitudes (up to harmonic 32)"),
                        ("harmonic_odd_amplitude_mean_error", "Mean error of odd harmonic amplitudes (from harmonic 3 to harmonic 33)"),
                        ("frequency_mean_error", "Mean frequency error"),
                    ),
                },
                {
                    "title": "Extra metrics",
                    "metrics": (
                        ("tve", "TVE (total vector error)"),
                        ("max_harmonic_error", "Maximum harmonic error"),
                    ),
                },
            ),
        },
        {
            "id": "steady_scalar",
            "title": "Steady-state scalar signal metrics",
            "subtitle": "These metrics will be applied separately to each steady-state window, such as pre-disturbance, during-disturbance, and post-disturbance.",
            "sections": (
                {
                    "title": "Main metrics",
                    "metrics": (
                        ("steady_mean_error", "Normalized mean error"),
                        ("steady_nrmse", "Normalized root mean square error (NRMSE)"),
                        ("steady_residual_rms", "Normalized residual RMS"),
                        ("steady_ssi", "SSI error"),
                        ("steady_std_error", "Relative standard deviation error"),
                        ("steady_peak_to_peak_error", "Relative peak-to-peak error (if oscillation exists in steady state)"),
                    ),
                },
            ),
        },
        {
            "id": "transient_scalar",
            "title": "Transient scalar signal metrics",
            "subtitle": "These metrics will be evaluated separately in each transient window, such as Transient 1 and Transient 2.",
            "test_keys": ("step_test", "fault_test", "ramp_test"),
            "sections": (
                {
                    "title": "Main metrics",
                    "metrics": (
                        ("transient_mean_error", "Normalized mean error"),
                        ("transient_nrmse", "Normalized root mean square error (NRMSE)"),
                        ("transient_overshoot_undershoot", "Signal difference: Overshoot/Undershoot"),
                        ("transient_relative_overshoot_undershoot", "Relative signal difference (%): Overshoot/Undershoot"),
                        ("transient_rise_fall_time", "Signal difference: Rise/Fall time"),
                        ("transient_relative_rise_fall_time", "Relative signal difference (%): Rise/Fall time"),
                        ("transient_settling_time", "Signal difference: Settling time"),
                        ("transient_relative_settling_time", "Relative signal difference (%): Settling time"),
                        ("transient_response_time", "Signal difference: Response time"),
                        ("transient_relative_response_time", "Relative signal difference (%): Response time"),
                        ("transient_reaction_time", "Signal difference: Reaction time"),
                        ("transient_relative_reaction_time", "Relative signal difference (%): Reaction time"),
                        ("transient_delay_time", "Delay time"),
                    ),
                },
                {
                    "title": "Extra metrics",
                    "metrics": (
                        ("transient_pearson_r", "Pearson r"),
                        ("transient_r2", "Coefficient of determination (R²)"),
                        ("transient_iae", "Normalized IAE"),
                        ("transient_ise", "Normalized ISE"),
                        ("transient_itae", "Normalized ITAE"),
                        ("transient_itse", "Normalized ITSE"),
                    ),
                },
            ),
        },
    )

    METRIC_UNIT_LABELS = {
        "waveform_mean_error": "%",
        "waveform_nrmse": "%",
        "fundamental_amplitude_error": "%",
        "phase_error": "degrees",
        "thd_total_error": "%",
        "harmonic_even_amplitude_mean_error": "%",
        "harmonic_odd_amplitude_mean_error": "%",
        "frequency_mean_error": "%",
        "tve": "%",
        "max_harmonic_error": "%",
        "steady_mean_error": "%",
        "steady_nrmse": "%",
        "steady_residual_rms": "%",
        "steady_ssi": "%",
        "steady_std_error": "%",
        "steady_peak_to_peak_error": "%",
        "transient_mean_error": "%",
        "transient_nrmse": "%",
        "transient_overshoot_undershoot": "%",
        "transient_relative_overshoot_undershoot": "%",
        "transient_rise_fall_time": "ms",
        "transient_relative_rise_fall_time": "%",
        "transient_settling_time": "ms",
        "transient_relative_settling_time": "%",
        "transient_response_time": "ms",
        "transient_relative_response_time": "%",
        "transient_reaction_time": "ms",
        "transient_relative_reaction_time": "%",
        "transient_delay_time": "ms",
        "transient_pearson_r": "adim.",
        "transient_r2": "adim.",
        "transient_iae": "%",
        "transient_ise": "%",
        "transient_itae": "%",
        "transient_itse": "%",
    }

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Model Comparison")
        self.root.geometry("1200x820")
        self.root.minsize(960, 700)
        self.root.configure(bg="#f4f6fb")
        self.ui_style = ttk.Style(self.root)
        try:
            self.ui_style.theme_use("clam")
        except tk.TclError:
            pass
        self.ui_style.configure("Codex.TNotebook", background="#f4f6fb", borderwidth=0, tabmargins=(8, 8, 8, 0))
        self.ui_style.configure(
            "Codex.TNotebook.Tab",
            font=("Segoe UI", 10, "bold"),
            padding=(18, 10),
        )
        self.ui_style.map(
            "Codex.TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#e8eef7")],
            foreground=[("selected", "#17324d"), ("active", "#17324d")],
        )

        self.selected_files: dict[str, dict[str, Path | None]] = {
            signal_type: {"experimental": None, "model": None}
            for signal_type in self.SIGNAL_IMPORT_DEFINITIONS
        }
        self.column_names: dict[str, dict[str, list[str]]] = {
            signal_type: {"experimental": [], "model": []}
            for signal_type in self.SIGNAL_IMPORT_DEFINITIONS
        }
        self.column_selection_values: dict[str, dict[str, list[str]]] = {
            signal_type: {"experimental": [], "model": []}
            for signal_type in self.SIGNAL_IMPORT_DEFINITIONS
        }
        self.file_labels: dict[str, tk.Label] = {}
        self.column_frames: dict[str, tk.Frame] = {}
        self.validation_button: tk.Button | None = None
        self.validation_status_label: tk.Label | None = None
        self.current_flow_mode: str | None = None
        self.current_test_key: str | None = None
        self.current_signal_type: str | None = None
        self.current_assessment_project: AssessmentProject | None = None
        self.project_test_case_count_var = tk.StringVar(value="1")
        self.project_signal_type_var = tk.StringVar(value="scalar")
        self.project_filter_signals_var = tk.BooleanVar(value=False)
        self.project_test_case_states: list[dict[str, object]] = []
        self.project_test_case_definitions: list[dict[str, object]] = []
        self.project_upload_files: dict[tuple[int, str, int], Path | None] = {}
        self.project_upload_headers: dict[tuple[int, str, int], list[str]] = {}
        self.project_upload_labels: dict[tuple[int, str, int], tk.Label] = {}
        self.project_classification_vars: dict[tuple[object, ...], list[tk.StringVar]] = {}
        self.project_run_strategy_var = tk.StringVar(value="all_x_all")
        self.project_runs_preview: list[dict[str, object]] = []
        self.project_processing_task_state: dict[str, object] | None = None
        self.saved_project_selection_var = tk.StringVar()
        self.loading_status_label: tk.Label | None = None
        self.loading_percent_label: tk.Label | None = None
        self.current_project_storage_path: Path | None = None
        self.current_project_report: dict[str, object] | None = None
        self.last_result: ValidationResult | None = None
        self.current_analysis_result: ValidationResult | None = None
        self.loading_window: tk.Toplevel | None = None
        self.loading_progress: ttk.Progressbar | None = None
        self.validation_task_result: tuple[str, object] | None = None
        self.current_figure = None
        self.current_figure_canvas: FigureCanvasTkAgg | None = None
        self.current_scroll_canvas: tk.Canvas | None = None
        self.current_window_segments: dict[str, list[WindowSegment]] = {}
        self.window_boundary_lines: list[object] = []
        self.window_drag_state: dict[str, object] | None = None
        self.last_transient_debug_data: dict[str, list[dict[str, object]]] = {}
        self.transient_delay_cache: dict[str, float] = {}
        self.test_form_vars = self._create_test_form_vars()
        self.project_form_vars = self._create_project_form_vars()
        self.project_global_form_vars = self._create_project_global_form_vars()
        self.metric_selection_state = self._create_metric_selection_state()
        self.metric_threshold_entries: dict[str, dict[str, tk.Entry]] = {}
        self.metric_special_options = self._create_metric_special_options()
        self.signal_filter_options = {
            signal_type: tk.BooleanVar(value=False)
            for signal_type in self.SIGNAL_IMPORT_DEFINITIONS
        }
        self.window_tolerance_percent_var = tk.StringVar(value="5")
        self.min_transition_percent_var = tk.StringVar(value="1")
        self.sync_analysis_mode_var = tk.StringVar(value="per_unit_independent")

        self._build_start_mode_screen()

    def _create_test_form_vars(self) -> dict[str, dict[str, tk.StringVar]]:
        form_vars: dict[str, dict[str, tk.StringVar]] = {}
        for test_key, definition in self.TEST_DEFINITIONS.items():
            form_vars[test_key] = {
                field["name"]: tk.StringVar()
                for field in definition["fields"]
            }
        return form_vars

    def _create_project_form_vars(self) -> dict[str, tk.StringVar]:
        return {
            "model_name": tk.StringVar(),
            "model_version": tk.StringVar(),
            "notes": tk.StringVar(),
        }

    def _create_project_global_form_vars(self) -> dict[str, tk.StringVar]:
        return {
            "nominal_power_w": tk.StringVar(),
            "nominal_voltage_v": tk.StringVar(),
            "system_configuration": tk.StringVar(value=self.SYSTEM_CONFIGURATION_OPTIONS[0]),
        }

    def _project_test_type_options(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (test_key, str(definition["screen_title"]))
            for test_key, definition in self.TEST_DEFINITIONS.items()
        )

    def _project_test_type_label_to_key(self) -> dict[str, str]:
        return {label: key for key, label in self._project_test_type_options()}

    def _project_test_type_key_to_label(self) -> dict[str, str]:
        return {key: label for key, label in self._project_test_type_options()}

    def _new_project_test_case_state(self, case_number: int) -> dict[str, object]:
        test_type_label = self._project_test_type_key_to_label().get("steady_state", "Steady-state tests")
        return {
            "test_name": tk.StringVar(value=f"Test case {case_number}"),
            "test_type": tk.StringVar(value=test_type_label),
            "test_power_percent": tk.StringVar(),
            "event_quantity": tk.StringVar(value="Voltage"),
            "pre_event_percent": tk.StringVar(),
            "event_percent": tk.StringVar(),
            "experimental_capture_count": tk.StringVar(value="1"),
            "model_capture_count": tk.StringVar(value="1"),
            "dynamic_frame": None,
        }

    def _create_metric_selection_state(self) -> dict[str, dict[str, dict[str, object]]]:
        state: dict[str, dict[str, dict[str, object]]] = {}
        for group in self.METRIC_GROUP_DEFINITIONS:
            group_state: dict[str, dict[str, object]] = {}
            for section in group["sections"]:
                for metric_id, _metric_label in section["metrics"]:
                    group_state[metric_id] = {
                        "selected": tk.BooleanVar(value=False),
                        "weight": tk.StringVar(value="1"),
                        "limits": {
                            "good": tk.StringVar(),
                            "acceptable": tk.StringVar(),
                        },
                    }
            state[group["id"]] = group_state
        return state

    def _create_metric_special_options(self) -> dict[str, tk.Variable]:
        return {
            "waveform_adjust_phase_error": tk.BooleanVar(value=False),
            "transient_adjust_reaction_time": tk.BooleanVar(value=False),
            "transient_normalization_mode": tk.StringVar(value="event_amplitude"),
            "score_mode": tk.StringVar(value="continuous"),
            "window_aggregation_mode": tk.StringVar(value="average"),
        }

    def _clear_screen(self) -> None:
        self.current_figure = None
        self.current_figure_canvas = None
        self.current_scroll_canvas = None
        for widget in self.root.winfo_children():
            widget.destroy()

    def _add_screen_watermark(self, parent: tk.Widget, bg: str) -> None:
        label = tk.Label(
            parent,
            text=self.WATERMARK_TEXT,
            font=("Segoe UI", 9),
            bg=bg,
            fg="#8aa0b8",
            anchor="e",
            justify="right",
        )
        label.pack(fill="x", pady=(0, 12))

    def _create_scrollable_screen(
        self,
        *,
        bg: str = "#f4f6fb",
        padx: int = 24,
        pady: int = 24,
    ) -> tuple[tk.Frame, tk.Frame]:
        outer = tk.Frame(self.root, bg=bg)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        container = tk.Frame(canvas, bg=bg, padx=padx, pady=pady)
        canvas_window = canvas.create_window((0, 0), window=container, anchor="nw")

        def _on_container_configure(_event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        def _on_mousewheel(event) -> None:
            if self.current_scroll_canvas is canvas:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        container.bind("<Configure>", _on_container_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.current_scroll_canvas = canvas
        self._add_screen_watermark(container, bg)
        return outer, container

    def _build_start_mode_screen(self) -> None:
        self.current_flow_mode = None
        self._clear_screen()

        container = tk.Frame(self.root, bg="#f4f6fb", padx=32, pady=32)
        container.pack(fill="both", expand=True)
        self._add_screen_watermark(container, "#f4f6fb")

        title = tk.Label(
            container,
            text="Choose the analysis mode",
            font=("Segoe UI", 24, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(pady=(40, 12))

        subtitle = tk.Label(
            container,
            text="Start a single comparison test or create a complete model validation project.",
            font=("Segoe UI", 12),
            bg="#f4f6fb",
            fg="#49657f",
        )
        subtitle.pack(pady=(0, 32))

        buttons_frame = tk.Frame(container, bg="#f4f6fb")
        buttons_frame.pack(expand=True)

        single_test_button = tk.Button(
            buttons_frame,
            text="Single test analysis",
            font=("Segoe UI", 13, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            width=38,
            height=3,
            bd=0,
            cursor="hand2",
            command=self._start_single_test_flow,
        )
        single_test_button.pack(pady=12)

        complete_project_button = tk.Button(
            buttons_frame,
            text="Complete model validation",
            font=("Segoe UI", 13, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            width=38,
            height=3,
            bd=0,
            cursor="hand2",
            command=self._build_project_entry_screen,
        )
        complete_project_button.pack(pady=12)

    def _build_project_entry_screen(self) -> None:
        self.current_flow_mode = "assessment_project"
        self._clear_screen()

        _, container = self._create_scrollable_screen(padx=28, pady=28)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_start_mode_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Model validation",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(8, 6))

        subtitle = tk.Label(
            container,
            text="Create a new model validation project or load one of the saved results.",
            font=("Segoe UI", 11),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1080,
        )
        subtitle.pack(anchor="w", pady=(0, 24))

        actions_card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=20, pady=20)
        actions_card.pack(fill="x", pady=(0, 14))

        new_button = tk.Button(
            actions_card,
            text="Create new model validation",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._build_project_creation_screen,
        )
        new_button.pack(anchor="w")

        saved_projects = list_saved_assessment_projects()
        saved_card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=20, pady=20)
        saved_card.pack(fill="x")

        saved_title = tk.Label(
            saved_card,
            text="Load saved results",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#17324d",
        )
        saved_title.pack(anchor="w")

        if not saved_projects:
            tk.Label(
                saved_card,
                text="No saved model validation project was found yet.",
                font=("Segoe UI", 10),
                bg="white",
                fg="#64748b",
            ).pack(anchor="w", pady=(10, 0))
            return

        option_map: dict[str, str] = {}
        option_labels: list[str] = []
        for project in saved_projects:
            label = str(project["model_name"])
            if project["model_version"]:
                label += f" | {project['model_version']}"
            if project["date_updated"]:
                label += f" | {project['date_updated']}"
            if project["has_processed_report"]:
                label += " | saved report"
            option_map[label] = str(project["project_id"])
            option_labels.append(label)

        selected_label = option_labels[0]
        self.saved_project_selection_var.set(selected_label)

        combo = ttk.Combobox(
            saved_card,
            textvariable=self.saved_project_selection_var,
            values=option_labels,
            state="readonly",
            width=100,
            font=("Segoe UI", 10),
        )
        combo.pack(fill="x", pady=(12, 10))
        combo.set(selected_label)

        load_button = tk.Button(
            saved_card,
            text="Load saved model validation",
            font=("Segoe UI", 11, "bold"),
            bg="#0f766e",
            fg="white",
            activebackground="#115e59",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=lambda current_map=option_map: self._load_saved_project_from_selection(current_map),
        )
        load_button.pack(anchor="e")

    def _start_single_test_flow(self) -> None:
        self.current_flow_mode = "single_test"
        self._build_test_selection_screen()

    def _build_test_selection_screen(self) -> None:
        self.current_flow_mode = "single_test"
        self._clear_screen()

        container = tk.Frame(self.root, bg="#f4f6fb", padx=32, pady=32)
        container.pack(fill="both", expand=True)
        self._add_screen_watermark(container, "#f4f6fb")

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_start_mode_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Select the comparative test",
            font=("Segoe UI", 24, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(pady=(40, 12))

        subtitle = tk.Label(
            container,
            text="Choose which type of test will be used before importing the signals.",
            font=("Segoe UI", 12),
            bg="#f4f6fb",
            fg="#49657f",
        )
        subtitle.pack(pady=(0, 32))

        buttons_frame = tk.Frame(container, bg="#f4f6fb")
        buttons_frame.pack(expand=True)

        for test_key, definition in self.TEST_DEFINITIONS.items():
            button = tk.Button(
                buttons_frame,
                text=definition["button_label"],
                font=("Segoe UI", 13, "bold"),
                bg="#1d4ed8",
                fg="white",
                activebackground="#1e40af",
                activeforeground="white",
                width=38,
                height=3,
                bd=0,
                cursor="hand2",
                command=lambda selected_key=test_key: self._build_test_form_screen(
                    selected_key
                ),
            )
            button.pack(pady=12)

    def _build_project_creation_screen(self) -> None:
        self.current_flow_mode = "assessment_project"
        self._clear_screen()

        _, container = self._create_scrollable_screen(padx=28, pady=28)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_start_mode_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Create model validation project",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(8, 6))

        subtitle = tk.Label(
            container,
            text="Define the model information to start a complete model validation project.",
            font=("Segoe UI", 11),
            bg="#f4f6fb",
            fg="#49657f",
        )
        subtitle.pack(anchor="w", pady=(0, 24))

        card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=24, pady=24)
        card.pack(fill="both", expand=True)

        fields = (
            ("model_name", "Model name", "entry"),
            ("model_version", "Version", "entry"),
            ("notes", "Notes", "text"),
        )
        for field_name, field_label, field_kind in fields:
            field_frame = tk.Frame(card, bg="white")
            field_frame.pack(fill="x", pady=(0, 16))

            label = tk.Label(
                field_frame,
                text=field_label,
                font=("Segoe UI", 11, "bold"),
                bg="white",
                fg="#17324d",
                anchor="w",
            )
            label.pack(fill="x", pady=(0, 6))

            if field_kind == "text":
                text_widget = tk.Text(
                    field_frame,
                    font=("Segoe UI", 11),
                    relief="solid",
                    bd=1,
                    height=6,
                    wrap="word",
                )
                text_widget.insert("1.0", self.project_form_vars[field_name].get())

                def _sync_notes(*_args, widget=text_widget, variable=self.project_form_vars[field_name]) -> None:
                    variable.set(widget.get("1.0", "end-1c"))

                text_widget.bind("<KeyRelease>", _sync_notes)
                text_widget.pack(fill="x")
            else:
                entry = tk.Entry(
                    field_frame,
                    textvariable=self.project_form_vars[field_name],
                    font=("Segoe UI", 11),
                    relief="solid",
                    bd=1,
                )
                entry.pack(fill="x", ipady=6)

        continue_button = tk.Button(
            card,
            text="Create project",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=16,
            pady=12,
            cursor="hand2",
            command=self._create_assessment_project_and_continue,
        )
        continue_button.pack(anchor="e", pady=(10, 0))

    def _create_assessment_project_and_continue(self) -> None:
        model_name = self.project_form_vars["model_name"].get().strip()
        if not model_name:
            messagebox.showerror(
                "Incomplete form",
                "Fill in the model name before continuing.",
            )
            return

        model_version = self.project_form_vars["model_version"].get().strip()
        notes = self.project_form_vars["notes"].get().strip()
        self.current_assessment_project = AssessmentProject.create(
            project_name=f"{model_name} model validation",
            model_name=model_name,
            model_version=model_version,
            notes=notes,
        )
        self._autosave_current_project()
        self._build_project_overview_screen()

    def _hydrate_project_runtime_state(self, project: AssessmentProject) -> None:
        self.current_assessment_project = project
        self.current_project_storage_path = save_assessment_project(project)
        self.current_project_report = None
        self.project_form_vars["model_name"].set(project.model_name)
        self.project_form_vars["model_version"].set(project.model_version)
        self.project_form_vars["notes"].set(project.notes)

        global_settings = dict(project.global_settings)
        self.project_global_form_vars["nominal_power_w"].set(str(global_settings.get("nominal_power_w", "")))
        self.project_global_form_vars["nominal_voltage_v"].set(str(global_settings.get("nominal_voltage_v", "")))
        self.project_global_form_vars["system_configuration"].set(
            str(global_settings.get("system_configuration", self.SYSTEM_CONFIGURATION_OPTIONS[0])).strip()
            or self.SYSTEM_CONFIGURATION_OPTIONS[0]
        )
        self.project_filter_signals_var.set(bool(global_settings.get("filter_signals_for_analysis", False)))

        analysis_representation = str(global_settings.get("analysis_representation", "per_unit_independent")).strip()
        self.sync_analysis_mode_var.set("real" if analysis_representation in {"real_values", "real"} else analysis_representation)
        self.window_tolerance_percent_var.set(str(global_settings.get("transient_end_tolerance_percent", "5")))
        self.min_transition_percent_var.set(str(global_settings.get("minimum_signal_variation_percent", "1")))
        self.project_signal_type_var.set(str(global_settings.get("project_signal_type", "scalar")).strip() or "scalar")

        saved_cases = list(project.test_cases or [])
        self.project_test_case_definitions = []
        self.project_test_case_states = []
        self.project_test_case_count_var.set(str(max(len(saved_cases), 1)))
        for index, case in enumerate(saved_cases, start=1):
            test_type = str(case.get("test_type", "steady_state"))
            form_values = dict(case.get("form_values", {}))
            state = self._new_project_test_case_state(index)
            state["test_name"].set(str(case.get("case_name", f"Test case {index}")))
            state["test_type"].set(self._project_test_type_key_to_label().get(test_type, self._project_test_type_key_to_label()["steady_state"]))
            state["experimental_capture_count"].set(str(case.get("experimental_capture_count", 1)))
            state["model_capture_count"].set(str(case.get("model_capture_count", 1)))
            if test_type == "steady_state":
                state["test_power_percent"].set(str(form_values.get("test_power_percent", "")))
            elif test_type == "step_test":
                state["event_quantity"].set(str(form_values.get("step_quantity", "Voltage")))
                state["pre_event_percent"].set(str(form_values.get("pre_step_percent", "")))
                state["event_percent"].set(str(form_values.get("post_step_percent", "")))
            elif test_type == "fault_test":
                state["event_quantity"].set(str(form_values.get("fault_quantity", "Voltage")))
                state["pre_event_percent"].set(str(form_values.get("pre_fault_percent", "")))
                state["event_percent"].set(str(form_values.get("during_fault_percent", "")))
            elif test_type == "ramp_test":
                state["event_quantity"].set(str(form_values.get("ramp_quantity", "Voltage")))
                state["pre_event_percent"].set(str(form_values.get("pre_ramp_percent", "")))
                state["event_percent"].set(str(form_values.get("post_ramp_percent", "")))
            self.project_test_case_states.append(state)
            self.project_test_case_definitions.append(
                {
                    "case_index": index - 1,
                    "case_name": str(case.get("case_name", f"Test case {index}")),
                    "test_type": test_type,
                    "experimental_capture_count": int(case.get("experimental_capture_count", 1)),
                    "model_capture_count": int(case.get("model_capture_count", 1)),
                    "form_values": form_values,
                }
            )
        self._apply_project_metric_settings_to_runtime()

    def _load_saved_project_from_selection(self, option_map: dict[str, str]) -> None:
        selected_label = self.saved_project_selection_var.get().strip()
        project_id = option_map.get(selected_label)
        if not project_id:
            messagebox.showerror("Load error", "Select a saved model validation before continuing.")
            return
        try:
            project = load_assessment_project(project_id)
        except Exception as exc:
            messagebox.showerror("Load error", f"It was not possible to load the saved model validation.\n\n{exc}")
            return

        self._hydrate_project_runtime_state(project)
        if self._project_has_processed_cases(project):
            rebuilt_report = self._build_project_report_data(list(project.test_cases or []))
            project.global_settings["processed_report"] = rebuilt_report
            self.current_assessment_project = project
            self._autosave_current_project()
            self._build_project_report_screen(rebuilt_report)
            return
        processed_report = project.global_settings.get("processed_report")
        if isinstance(processed_report, dict) and processed_report:
            self._build_project_report_screen(processed_report)
            return
        self._build_project_overview_screen()

    def _project_has_processed_cases(self, project: AssessmentProject) -> bool:
        for case in list(project.test_cases or []):
            if not isinstance(case, dict):
                continue
            runs = case.get("runs")
            if not isinstance(runs, list) or not runs:
                continue
            for run in runs:
                if not isinstance(run, dict):
                    continue
                if isinstance(run.get("score_sections"), list) and run["score_sections"]:
                    return True
        return False

    def _build_project_overview_screen(self) -> None:
        if self.current_assessment_project is None:
            self._build_project_creation_screen()
            return

        self._clear_screen()
        _, container = self._create_scrollable_screen(padx=28, pady=28)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_project_creation_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Model validation project created",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(8, 6))

        subtitle = tk.Label(
            container,
            text="The project structure is ready. The next implementation step will be adding test cases to this model validation workflow.",
            font=("Segoe UI", 11),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=980,
        )
        subtitle.pack(anchor="w", pady=(0, 24))

        card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=24, pady=24)
        card.pack(fill="x")

        project = self.current_assessment_project
        details = (
            ("Model name", project.model_name),
            ("Version", project.model_version or "-"),
            ("Created at", project.date_created),
            ("Test cases", str(project.test_case_count)),
            ("Electrical system", str(project.global_settings.get("system_configuration", self.SYSTEM_CONFIGURATION_OPTIONS[0]))),
        )
        for label_text, value_text in details:
            row = tk.Frame(card, bg="white")
            row.pack(fill="x", pady=4)
            label = tk.Label(
                row,
                text=f"{label_text}:",
                font=("Segoe UI", 10, "bold"),
                bg="white",
                fg="#17324d",
                width=16,
                anchor="w",
            )
            label.pack(side="left")
            value = tk.Label(
                row,
                text=value_text,
                font=("Segoe UI", 10),
                bg="white",
                fg="#49657f",
                anchor="w",
                justify="left",
            )
            value.pack(side="left")

        if project.notes:
            notes_title = tk.Label(
                card,
                text="Notes",
                font=("Segoe UI", 10, "bold"),
                bg="white",
                fg="#17324d",
            )
            notes_title.pack(anchor="w", pady=(14, 4))
            notes_value = tk.Label(
                card,
                text=project.notes,
                font=("Segoe UI", 10),
                bg="white",
                fg="#49657f",
                anchor="w",
                justify="left",
                wraplength=980,
            )
            notes_value.pack(anchor="w")

        actions = tk.Frame(container, bg="#f4f6fb")
        actions.pack(fill="x", pady=(18, 0))

        configure_button = tk.Button(
            actions,
            text="Configure test cases",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._build_project_test_case_setup_screen,
        )
        configure_button.pack(side="right")

        if self.current_project_storage_path is not None:
            storage_label = tk.Label(
                container,
                text=f"Saved project file: {self.current_project_storage_path}",
                font=("Segoe UI", 9),
                bg="#f4f6fb",
                fg="#64748b",
                justify="left",
                wraplength=1080,
            )
            storage_label.pack(anchor="w", pady=(12, 0))

    def _build_project_test_case_setup_screen(self) -> None:
        if self.current_assessment_project is None:
            self._build_project_creation_screen()
            return

        self._clear_screen()
        _, container = self._create_scrollable_screen(padx=28, pady=28)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_project_overview_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Configure test cases",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(8, 6))

        subtitle = tk.Label(
            container,
            text=(
                "Define how many different tests will be performed, the operating conditions of each test, "
                "and how many experimental and model captures belong to each case."
            ),
            font=("Segoe UI", 11),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1080,
        )
        subtitle.pack(anchor="w", pady=(0, 24))

        config_card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=24, pady=24)
        config_card.pack(fill="x", pady=(0, 16))

        count_row = tk.Frame(config_card, bg="white")
        count_row.pack(fill="x")

        count_label = tk.Label(
            count_row,
            text="Number of different tests",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#17324d",
        )
        count_label.pack(side="left")

        count_entry = tk.Entry(
            count_row,
            textvariable=self.project_test_case_count_var,
            font=("Segoe UI", 11),
            relief="solid",
            bd=1,
            width=8,
        )
        count_entry.pack(side="left", padx=(12, 0), ipady=6)

        rebuild_button = tk.Button(
            count_row,
            text="Generate forms",
            font=("Segoe UI", 10, "bold"),
            bg="#dbeafe",
            fg="#1e3a8a",
            activebackground="#bfdbfe",
            activeforeground="#1e3a8a",
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
            command=lambda: self._rebuild_project_test_case_states(preserve_existing=True),
        )
        rebuild_button.pack(side="left", padx=(12, 0))

        self.project_test_case_cards_container = tk.Frame(container, bg="#f4f6fb")
        self.project_test_case_cards_container.pack(fill="both", expand=True)

        self._rebuild_project_test_case_states(preserve_existing=True)

        footer = tk.Frame(container, bg="#f4f6fb")
        footer.pack(fill="x", pady=(18, 0))

        continue_button = tk.Button(
            footer,
            text="Continue to file upload",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._validate_project_test_cases_and_continue,
        )
        continue_button.pack(side="right")

    def _rebuild_project_test_case_states(self, preserve_existing: bool) -> None:
        try:
            requested_count = max(int(self.project_test_case_count_var.get().strip()), 1)
        except ValueError:
            requested_count = max(len(self.project_test_case_states), 1)
            self.project_test_case_count_var.set(str(requested_count))

        existing_states = self.project_test_case_states if preserve_existing else []
        updated_states: list[dict[str, object]] = []
        for index in range(requested_count):
            if index < len(existing_states):
                updated_states.append(existing_states[index])
            else:
                updated_states.append(self._new_project_test_case_state(index + 1))
        self.project_test_case_states = updated_states

        for widget in self.project_test_case_cards_container.winfo_children():
            widget.destroy()

        for index, state in enumerate(self.project_test_case_states):
            self._build_project_test_case_card(self.project_test_case_cards_container, index, state)

    def _build_project_test_case_card(self, parent: tk.Frame, index: int, state: dict[str, object]) -> None:
        card = tk.Frame(parent, bg="white", bd=1, relief="solid", padx=20, pady=20)
        card.pack(fill="x", pady=(0, 14))

        title = tk.Label(
            card,
            text=f"Test case {index + 1}",
            font=("Segoe UI", 13, "bold"),
            bg="white",
            fg="#17324d",
        )
        title.pack(anchor="w")

        row_one = tk.Frame(card, bg="white")
        row_one.pack(fill="x", pady=(12, 10))
        row_two = tk.Frame(card, bg="white")
        row_two.pack(fill="x", pady=(0, 10))

        self._build_project_labeled_entry(row_one, "Case name", state["test_name"])
        test_type_combo = self._build_project_labeled_combobox(
            row_one,
            "Test type",
            state["test_type"],
            [label for _key, label in self._project_test_type_options()],
        )
        selected_label = str(state["test_type"].get()).strip()
        if selected_label:
            test_type_combo.set(selected_label)

        self._build_project_labeled_entry(row_two, "Experimental captures", state["experimental_capture_count"], width=8)
        self._build_project_labeled_entry(row_two, "Model captures", state["model_capture_count"], width=8)

        dynamic_frame = tk.Frame(card, bg="white")
        dynamic_frame.pack(fill="x", pady=(4, 0))
        state["dynamic_frame"] = dynamic_frame

        def _on_type_change(*_args, case_state=state) -> None:
            self._render_project_test_case_dynamic_fields(case_state)

        state["test_type"].trace_add("write", _on_type_change)
        self._render_project_test_case_dynamic_fields(state)

    def _build_project_labeled_entry(
        self,
        parent: tk.Frame,
        label_text: str,
        variable: tk.StringVar,
        *,
        width: int = 18,
    ) -> None:
        frame = tk.Frame(parent, bg="white")
        frame.pack(side="left", padx=(0, 14))
        label = tk.Label(
            frame,
            text=label_text,
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#17324d",
        )
        label.pack(anchor="w")
        entry = tk.Entry(
            frame,
            textvariable=variable,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            width=width,
        )
        entry.pack(anchor="w", ipady=5)

    def _build_project_labeled_combobox(
        self,
        parent: tk.Frame,
        label_text: str,
        variable: tk.StringVar,
        options: list[str],
    ):
        frame = tk.Frame(parent, bg="white")
        frame.pack(side="left", padx=(0, 14))
        label = tk.Label(
            frame,
            text=label_text,
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#17324d",
        )
        label.pack(anchor="w")
        combo = ttk.Combobox(
            frame,
            textvariable=variable,
            values=options,
            state="readonly",
            width=28,
            font=("Segoe UI", 10),
        )
        combo.pack(anchor="w", ipady=4)
        return combo

    def _render_project_test_case_dynamic_fields(self, state: dict[str, object]) -> None:
        dynamic_frame = state.get("dynamic_frame")
        if not isinstance(dynamic_frame, tk.Frame):
            return
        for widget in dynamic_frame.winfo_children():
            widget.destroy()

        selected_value = str(state["test_type"].get()).strip()
        test_type_key = self._project_test_type_label_to_key().get(selected_value, selected_value)

        if test_type_key == "steady_state":
            self._build_project_labeled_entry(
                dynamic_frame,
                "Test power (% of nominal power)",
                state["test_power_percent"],
            )
            return

        quantity_options = [
            "Voltage",
            "Current",
            "Frequency",
            "Active power",
            "Reactive power",
            "Voltage A",
            "Voltage B",
            "Voltage C",
            "Current A",
            "Current B",
            "Current C",
            "Positive-sequence voltage",
            "Negative-sequence voltage",
            "Zero-sequence voltage",
            "Positive-sequence current",
            "Negative-sequence current",
            "Zero-sequence current",
        ]
        quantity_combo = self._build_project_labeled_combobox(
            dynamic_frame,
            "Quantity where the event occurred",
            state["event_quantity"],
            quantity_options,
        )
        if not str(state["event_quantity"].get()).strip() or str(state["event_quantity"].get()).strip() not in quantity_options:
            quantity_combo.set(quantity_options[0])

        self._build_project_labeled_entry(
            dynamic_frame,
            "Pre-event value (%)",
            state["pre_event_percent"],
        )

        event_label_map = {
            "step_test": "Post-step value (%)",
            "fault_test": "During-disturbance value (%)",
            "ramp_test": "Post-ramp value (%)",
        }
        self._build_project_labeled_entry(
            dynamic_frame,
            event_label_map.get(test_type_key, "Event value (%)"),
            state["event_percent"],
        )

    def _validate_project_test_cases_and_continue(self) -> None:
        case_definitions: list[dict[str, object]] = []
        for index, state in enumerate(self.project_test_case_states):
            test_type_value = str(state["test_type"].get()).strip()
            test_type_key = self._project_test_type_label_to_key().get(test_type_value, test_type_value)
            case_name = str(state["test_name"].get()).strip() or f"Test case {index + 1}"
            experimental_count = str(state["experimental_capture_count"].get()).strip()
            model_count = str(state["model_capture_count"].get()).strip()

            missing_fields = []
            if not experimental_count:
                missing_fields.append("Experimental captures")
            if not model_count:
                missing_fields.append("Model captures")

            if test_type_key == "steady_state":
                test_power_percent = str(state["test_power_percent"].get()).strip()
                if not test_power_percent:
                    missing_fields.append("Test power")
            else:
                event_quantity = str(state["event_quantity"].get()).strip()
                pre_event_percent = str(state["pre_event_percent"].get()).strip()
                event_percent = str(state["event_percent"].get()).strip()
                if not event_quantity:
                    missing_fields.append("Event quantity")
                if not pre_event_percent:
                    missing_fields.append("Pre-event value")
                if not event_percent:
                    missing_fields.append("Event value")

            if missing_fields:
                messagebox.showerror(
                    "Incomplete test case",
                    f"Fill in all fields for {case_name} before continuing.",
                )
                return

            try:
                experimental_count_value = max(int(float(experimental_count.replace(",", "."))), 1)
                model_count_value = max(int(float(model_count.replace(",", "."))), 1)
            except ValueError:
                messagebox.showerror(
                    "Invalid capture count",
                    f"Use integer values for the capture counts in {case_name}.",
                )
                return

            form_values = {
            }
            if test_type_key == "steady_state":
                form_values["test_power_percent"] = str(state["test_power_percent"].get()).strip()
            else:
                event_quantity = str(state["event_quantity"].get()).strip()
                if test_type_key == "step_test":
                    form_values.update(
                        {
                            "step_quantity": event_quantity,
                            "pre_step_percent": str(state["pre_event_percent"].get()).strip(),
                            "post_step_percent": str(state["event_percent"].get()).strip(),
                        }
                    )
                elif test_type_key == "fault_test":
                    form_values.update(
                        {
                            "fault_quantity": event_quantity,
                            "pre_fault_percent": str(state["pre_event_percent"].get()).strip(),
                            "during_fault_percent": str(state["event_percent"].get()).strip(),
                        }
                    )
                elif test_type_key == "ramp_test":
                    form_values.update(
                        {
                            "ramp_quantity": event_quantity,
                            "pre_ramp_percent": str(state["pre_event_percent"].get()).strip(),
                            "post_ramp_percent": str(state["event_percent"].get()).strip(),
                        }
                    )

            case_definitions.append(
                {
                    "case_index": index,
                    "case_name": case_name,
                    "test_type": test_type_key,
                    "experimental_capture_count": experimental_count_value,
                    "model_capture_count": model_count_value,
                    "form_values": form_values,
                }
            )

        self.project_test_case_definitions = case_definitions
        self._build_project_file_upload_screen()

    def _build_project_file_upload_screen(self) -> None:
        if self.current_assessment_project is None or not self.project_test_case_definitions:
            self._build_project_test_case_setup_screen()
            return

        self._clear_screen()
        self.project_upload_labels = {}
        _, container = self._create_scrollable_screen(padx=28, pady=28)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_project_test_case_setup_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Upload test files",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(8, 6))

        subtitle = tk.Label(
            container,
            text="Choose whether all files will be imported as scalar signals or as sinusoidal voltage and current waveforms, then upload every capture for each test case.",
            font=("Segoe UI", 11),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1080,
        )
        subtitle.pack(anchor="w", pady=(0, 18))

        mode_card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=20, pady=16)
        mode_card.pack(fill="x", pady=(0, 14))

        mode_label = tk.Label(
            mode_card,
            text="Signal type for all files in this project stage",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#17324d",
        )
        mode_label.pack(anchor="w", pady=(0, 8))

        for value, label in (("sinusoidal", "Sinusoidal voltage and current"), ("scalar", "Scalar signals")):
            radio = tk.Radiobutton(
                mode_card,
                text=label,
                variable=self.project_signal_type_var,
                value=value,
                font=("Segoe UI", 10),
                bg="white",
                fg="#17324d",
                activebackground="white",
                activeforeground="#17324d",
                anchor="w",
            )
            radio.pack(anchor="w")

        for case_definition in self.project_test_case_definitions:
            case_index = int(case_definition["case_index"])
            case_card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=20, pady=20)
            case_card.pack(fill="x", pady=(0, 14))

            header = tk.Label(
                case_card,
                text=f"{case_definition['case_name']} - {self.TEST_DEFINITIONS[case_definition['test_type']]['screen_title']}",
                font=("Segoe UI", 13, "bold"),
                bg="white",
                fg="#17324d",
            )
            header.pack(anchor="w", pady=(0, 12))

            grid = tk.Frame(case_card, bg="white")
            grid.pack(fill="x")
            grid.grid_columnconfigure(0, weight=1)
            grid.grid_columnconfigure(1, weight=1)

            self._build_project_upload_group(
                grid,
                row=0,
                column=0,
                side="experimental",
                case_definition=case_definition,
            )
            self._build_project_upload_group(
                grid,
                row=0,
                column=1,
                side="model",
                case_definition=case_definition,
            )

        footer = tk.Frame(container, bg="#f4f6fb")
        footer.pack(fill="x", pady=(18, 0))

        continue_button = tk.Button(
            footer,
            text="Continue to column classification",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._validate_project_uploads_and_continue,
        )
        continue_button.pack(side="right")

    def _build_project_upload_group(
        self,
        parent: tk.Frame,
        *,
        row: int,
        column: int,
        side: str,
        case_definition: dict[str, object],
    ) -> None:
        side_title = "Experimental captures" if side == "experimental" else "Model captures"
        frame = tk.Frame(parent, bg="#f8fbff", bd=1, relief="solid", padx=14, pady=14)
        frame.grid(row=row, column=column, padx=8, pady=8, sticky="nsew")

        title = tk.Label(
            frame,
            text=side_title,
            font=("Segoe UI", 11, "bold"),
            bg="#f8fbff",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(0, 8))

        capture_count = int(case_definition[f"{side}_capture_count"])
        case_index = int(case_definition["case_index"])
        for capture_index in range(capture_count):
            row_frame = tk.Frame(frame, bg="#f8fbff")
            row_frame.pack(fill="x", pady=(0, 10))

            button = tk.Button(
                row_frame,
                text=f"Upload {side_title[:-1]} {capture_index + 1}",
                font=("Segoe UI", 10, "bold"),
                bg="#dbeafe",
                fg="#1e3a8a",
                activebackground="#bfdbfe",
                activeforeground="#1e3a8a",
                bd=0,
                padx=12,
                pady=8,
                cursor="hand2",
                command=lambda current_case=case_index, current_side=side, current_capture=capture_index: self._select_project_capture_file(
                    current_case,
                    current_side,
                    current_capture,
                ),
            )
            button.pack(side="left")

            label = tk.Label(
                row_frame,
                text="No file selected",
                font=("Segoe UI", 9),
                bg="#f8fbff",
                fg="#7c8da1",
                anchor="w",
                justify="left",
            )
            label.pack(side="left", padx=(12, 0))
            self.project_upload_labels[(case_index, side, capture_index)] = label

    def _select_project_capture_file(self, case_index: int, side: str, capture_index: int) -> None:
        path_str = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            headers = read_csv_headers(path)
        except ValueError as exc:
            messagebox.showerror("Invalid file", str(exc))
            return

        key = (case_index, side, capture_index)
        self.project_upload_files[key] = path
        self.project_upload_headers[key] = headers
        label = self.project_upload_labels.get(key)
        if label is not None:
            label.config(text=path.name, fg="#0f766e")

    def _validate_project_uploads_and_continue(self) -> None:
        for case_definition in self.project_test_case_definitions:
            case_index = int(case_definition["case_index"])
            for side in ("experimental", "model"):
                capture_count = int(case_definition[f"{side}_capture_count"])
                for capture_index in range(capture_count):
                    if self.project_upload_files.get((case_index, side, capture_index)) is None:
                        messagebox.showerror(
                            "Missing files",
                            f"Upload all files for {case_definition['case_name']} before continuing.",
                        )
                        return
        self._build_project_classification_screen()

    def _project_grouped_headers(self, side: str) -> tuple[bool, list[tuple[tuple[object, ...], str, list[str], list[tuple[int, int]]]]]:
        collected: list[tuple[int, int, list[str]]] = []
        for case_definition in self.project_test_case_definitions:
            case_index = int(case_definition["case_index"])
            capture_count = int(case_definition[f"{side}_capture_count"])
            for capture_index in range(capture_count):
                headers = self.project_upload_headers.get((case_index, side, capture_index), [])
                collected.append((case_index, capture_index, headers))

        if not collected:
            return False, []

        all_same = all(headers == collected[0][2] for _case_index, _capture_index, headers in collected)
        if all_same:
            file_refs = [(case_index, capture_index) for case_index, capture_index, _headers in collected]
            return True, [((side, "shared"), f"All {side} files", collected[0][2], file_refs)]

        groups = []
        for case_index, capture_index, headers in collected:
            case_name = str(self.project_test_case_definitions[case_index]["case_name"])
            groups.append(
                (
                    (side, case_index, capture_index),
                    f"{case_name} - {side.capitalize()} capture {capture_index + 1}",
                    headers,
                    [(case_index, capture_index)],
                )
            )
        return False, groups

    def _build_project_classification_screen(self) -> None:
        self._clear_screen()
        self.project_classification_vars = {}
        signal_type = str(self.project_signal_type_var.get()).strip() or "scalar"
        options = self.SIGNAL_IMPORT_DEFINITIONS[signal_type]["classification_options"]

        _, container = self._create_scrollable_screen(padx=28, pady=28)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_project_file_upload_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Column classification",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(8, 6))

        subtitle = tk.Label(
            container,
            text=(
                "If all uploaded files on one side have the same column names, the classification is shared. "
                "If any file differs, that file must be classified separately."
            ),
            font=("Segoe UI", 11),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1080,
        )
        subtitle.pack(anchor="w", pady=(0, 18))

        for side in ("experimental", "model"):
            _shared, groups = self._project_grouped_headers(side)
            section_card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=20, pady=20)
            section_card.pack(fill="x", pady=(0, 14))

            section_title = tk.Label(
                section_card,
                text="Experimental files" if side == "experimental" else "Model files",
                font=("Segoe UI", 13, "bold"),
                bg="white",
                fg="#17324d",
            )
            section_title.pack(anchor="w", pady=(0, 12))

            for block_key, block_title, headers, _file_refs in groups:
                block = tk.Frame(section_card, bg="#f8fbff", bd=1, relief="solid", padx=14, pady=14)
                block.pack(fill="x", pady=(0, 10))

                block_title_label = tk.Label(
                    block,
                    text=block_title,
                    font=("Segoe UI", 11, "bold"),
                    bg="#f8fbff",
                    fg="#17324d",
                )
                block_title_label.pack(anchor="w", pady=(0, 10))

                variables = [tk.StringVar(value="Ignore signal") for _header in headers]
                self.project_classification_vars[block_key] = variables

                for column_index, header in enumerate(headers):
                    row = tk.Frame(block, bg="#f8fbff")
                    row.pack(fill="x", pady=(0, 8))

                    label = tk.Label(
                        row,
                        text=f"Column {column_index + 1} - {header}",
                        font=("Segoe UI", 10),
                        bg="#f8fbff",
                        fg="#17324d",
                        anchor="w",
                        justify="left",
                    )
                    label.pack(side="left", fill="x", expand=True)

                    combo = ttk.Combobox(
                        row,
                        textvariable=variables[column_index],
                        values=options,
                        state="readonly",
                        width=28,
                        font=("Segoe UI", 10),
                    )
                    combo.pack(side="right")
                    combo.set("Ignore signal")

        footer = tk.Frame(container, bg="#f4f6fb")
        footer.pack(fill="x", pady=(18, 0))

        continue_button = tk.Button(
            footer,
            text="Save test cases",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._save_project_test_cases_from_classification,
        )
        continue_button.pack(side="right")

    def _save_project_test_cases_from_classification(self) -> None:
        if self.current_assessment_project is None:
            self._build_project_creation_screen()
            return

        signal_type = str(self.project_signal_type_var.get()).strip() or "scalar"
        experimental_shared, experimental_groups = self._project_grouped_headers("experimental")
        model_shared, model_groups = self._project_grouped_headers("model")

        def _group_payload(groups, shared: bool) -> list[dict[str, object]]:
            payload = []
            for block_key, block_title, headers, file_refs in groups:
                selections = [str(variable.get()).strip() or "Ignore signal" for variable in self.project_classification_vars.get(block_key, [])]
                payload.append(
                    {
                        "shared": shared,
                        "title": block_title,
                        "headers": list(headers),
                        "selections": selections,
                        "file_refs": [{"case_index": case_index, "capture_index": capture_index} for case_index, capture_index in file_refs],
                    }
                )
            return payload

        saved_cases: list[dict[str, object]] = []
        for case_definition in self.project_test_case_definitions:
            case_index = int(case_definition["case_index"])
            saved_cases.append(
                {
                    "case_index": case_index,
                    "case_name": case_definition["case_name"],
                    "test_type": case_definition["test_type"],
                    "signal_type": signal_type,
                    "form_values": dict(case_definition["form_values"]),
                    "experimental_capture_count": case_definition["experimental_capture_count"],
                    "model_capture_count": case_definition["model_capture_count"],
                    "uploaded_files": {
                        "experimental": [
                            str(self.project_upload_files[(case_index, "experimental", capture_index)])
                            for capture_index in range(int(case_definition["experimental_capture_count"]))
                        ],
                        "model": [
                            str(self.project_upload_files[(case_index, "model", capture_index)])
                            for capture_index in range(int(case_definition["model_capture_count"]))
                        ],
                    },
                }
            )

        self.current_assessment_project.test_cases = saved_cases
        self.current_assessment_project.touch()
        self.current_assessment_project.global_settings["project_signal_type"] = signal_type
        self.current_assessment_project.global_settings["project_column_classification"] = {
            "experimental": _group_payload(experimental_groups, experimental_shared),
            "model": _group_payload(model_groups, model_shared),
        }
        self.current_assessment_project.global_settings["project_column_classification_shared"] = {
            "experimental": experimental_shared,
            "model": model_shared,
        }
        self._autosave_current_project()
        self._build_project_global_settings_screen()

    def _project_has_transient_cases(self) -> bool:
        return any(
            str(case_definition.get("test_type", "")) in {"step_test", "fault_test", "ramp_test"}
            for case_definition in self.project_test_case_definitions
        )

    def _autosave_current_project(self) -> None:
        if self.current_assessment_project is None:
            return
        self.current_project_storage_path = save_assessment_project(self.current_assessment_project)

    def _build_project_global_settings_screen(self) -> None:
        if self.current_assessment_project is None:
            self._build_project_creation_screen()
            return

        self._clear_screen()
        self.metric_threshold_entries = {}
        _, container = self._create_scrollable_screen(padx=20, pady=20)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 10))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_project_classification_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Project global settings",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            container,
            text=(
                "Define the global settings that will be used across all model validation test cases, "
                "including nominal values, metrics, plot representation, signal filtering, "
                "transient detection tolerances, and score configuration."
            ),
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1100,
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        defaults_button = tk.Button(
            container,
            text="Import default values",
            font=("Segoe UI", 10, "bold"),
            bg="#0f766e",
            fg="white",
            activebackground="#115e59",
            activeforeground="white",
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
            command=self._import_default_metric_limits,
        )
        defaults_button.pack(anchor="e", pady=(0, 12))

        setup_card = tk.Frame(container, bg="#eef4fb", bd=1, relief="solid", padx=18, pady=18)
        setup_card.pack(fill="x", pady=(0, 14))

        setup_title = tk.Label(
            setup_card,
            text="Analysis setup",
            font=("Segoe UI", 13, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        setup_title.pack(anchor="w")

        nominal_row = tk.Frame(setup_card, bg="#eef4fb")
        nominal_row.pack(fill="x", pady=(12, 10))
        self._build_project_labeled_entry(
            nominal_row,
            "Nominal power (W)",
            self.project_global_form_vars["nominal_power_w"],
        )
        self._build_project_labeled_entry(
            nominal_row,
            "Nominal voltage (V)",
            self.project_global_form_vars["nominal_voltage_v"],
        )

        system_row = tk.Frame(setup_card, bg="#eef4fb")
        system_row.pack(fill="x", pady=(0, 10))
        system_label = tk.Label(
            system_row,
            text="Electrical system",
            font=("Segoe UI", 10, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        system_label.pack(anchor="w", pady=(0, 4))
        system_combo = ttk.Combobox(
            system_row,
            textvariable=self.project_global_form_vars["system_configuration"],
            values=self.SYSTEM_CONFIGURATION_OPTIONS,
            state="readonly",
            font=("Segoe UI", 10),
        )
        if not self.project_global_form_vars["system_configuration"].get().strip():
            system_combo.set(self.SYSTEM_CONFIGURATION_OPTIONS[0])
        system_combo.pack(fill="x", ipady=3)

        filter_checkbox = tk.Checkbutton(
            setup_card,
            text="Filter signals for analysis (zero-phase bidirectional filter)",
            variable=self.project_filter_signals_var,
            onvalue=True,
            offvalue=False,
            font=("Segoe UI", 10),
            bg="#eef4fb",
            fg="#17324d",
            activebackground="#eef4fb",
            activeforeground="#17324d",
            selectcolor="white",
            anchor="w",
        )
        filter_checkbox.pack(anchor="w", pady=(2, 0))

        filter_help = tk.Label(
            setup_card,
            text="The selected filter setting will be applied to the project runs during analysis.",
            font=("Segoe UI", 9),
            bg="#eef4fb",
            fg="#64748b",
            justify="left",
            wraplength=1040,
        )
        filter_help.pack(anchor="w", pady=(4, 10))

        representation_title = tk.Label(
            setup_card,
            text="Representation used for the calculations",
            font=("Segoe UI", 10, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        representation_title.pack(anchor="w")

        representation_options = (
            ("real_values", "Real values"),
            ("per_unit_independent", "PU with independent base for experimental and model"),
            ("per_unit_reference", "PU using the experimental base for both signals"),
        )
        for option_value, option_label in representation_options:
            radio = tk.Radiobutton(
                setup_card,
                text=option_label,
                variable=self.sync_analysis_mode_var,
                value=option_value,
                font=("Segoe UI", 10),
                bg="#eef4fb",
                fg="#17324d",
                activebackground="#eef4fb",
                activeforeground="#17324d",
                anchor="w",
                justify="left",
            )
            radio.pack(anchor="w")

        tolerance_row = tk.Frame(setup_card, bg="#eef4fb")
        tolerance_row.pack(fill="x", pady=(12, 0))
        self._build_project_labeled_entry(
            tolerance_row,
            "Transient end tolerance (%)",
            self.window_tolerance_percent_var,
            width=10,
        )
        self._build_project_labeled_entry(
            tolerance_row,
            "Minimum signal variation to consider transient window (%)",
            self.min_transition_percent_var,
            width=10,
        )

        scoring_card = tk.Frame(container, bg="#eef4fb", bd=1, relief="solid", padx=18, pady=18)
        scoring_card.pack(fill="x", pady=(0, 14))

        scoring_title = tk.Label(
            scoring_card,
            text="Scoring configuration",
            font=("Segoe UI", 13, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        scoring_title.pack(anchor="w")

        scoring_note = tk.Label(
            scoring_card,
            text=(
                "Scores are calculated from 0 to 100 for each category and quantity. "
                "You can choose the scoring method, the window aggregation mode, and the weight of each metric."
            ),
            font=("Segoe UI", 10),
            bg="#eef4fb",
            fg="#49657f",
            justify="left",
            wraplength=1040,
        )
        scoring_note.pack(anchor="w", pady=(4, 12))

        scoring_mode_row = tk.Frame(scoring_card, bg="#eef4fb")
        scoring_mode_row.pack(fill="x", pady=(0, 10))

        scoring_mode_label = tk.Label(
            scoring_mode_row,
            text="Scoring mode:",
            font=("Segoe UI", 10, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        scoring_mode_label.pack(side="left")

        scoring_mode_combo = ttk.Combobox(
            scoring_mode_row,
            textvariable=self.metric_special_options["score_mode"],
            state="readonly",
            width=36,
            values=[label for _key, label in self.SCORE_MODE_OPTIONS],
        )
        scoring_mode_combo.pack(side="left", padx=(10, 0))
        selected_score_mode = str(self.metric_special_options["score_mode"].get()).strip()
        score_mode_label = next(
            (label for key, label in self.SCORE_MODE_OPTIONS if key == selected_score_mode),
            self.SCORE_MODE_OPTIONS[0][1],
        )
        scoring_mode_combo.set(score_mode_label)

        aggregation_row = tk.Frame(scoring_card, bg="#eef4fb")
        aggregation_row.pack(fill="x")

        aggregation_label = tk.Label(
            aggregation_row,
            text="Window aggregation:",
            font=("Segoe UI", 10, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        aggregation_label.pack(side="left")

        aggregation_combo = ttk.Combobox(
            aggregation_row,
            textvariable=self.metric_special_options["window_aggregation_mode"],
            state="readonly",
            width=36,
            values=[label for _key, label in self.WINDOW_AGGREGATION_OPTIONS],
        )
        aggregation_combo.pack(side="left", padx=(10, 0))
        selected_aggregation = str(self.metric_special_options["window_aggregation_mode"].get()).strip()
        aggregation_mode_label = next(
            (label for key, label in self.WINDOW_AGGREGATION_OPTIONS if key == selected_aggregation),
            self.WINDOW_AGGREGATION_OPTIONS[0][1],
        )
        aggregation_combo.set(aggregation_mode_label)

        project_signal_type = str(self.current_assessment_project.global_settings.get("project_signal_type", self.project_signal_type_var.get())).strip()
        has_transient_cases = self._project_has_transient_cases()
        for group in self.METRIC_GROUP_DEFINITIONS:
            is_unavailable = False
            unavailable_message = ""

            if group.get("signal_type") == "sinusoidal" and project_signal_type != "sinusoidal":
                is_unavailable = True
                unavailable_message = (
                    "These metrics are unavailable because the project files were imported "
                    "as scalar signals rather than sinusoidal waveforms."
                )

            allowed_test_keys = group.get("test_keys")
            if (
                not is_unavailable
                and allowed_test_keys is not None
                and not has_transient_cases
            ):
                is_unavailable = True
                unavailable_message = (
                    "These metrics are unavailable because the configured project test cases "
                    "do not include transient tests at this stage."
                )

            if is_unavailable:
                info_card = tk.Frame(container, bg="#eef4fb", bd=1, relief="solid", padx=18, pady=18)
                info_card.pack(fill="x", pady=(0, 14))
                group_title = tk.Label(
                    info_card,
                    text=group["title"],
                    font=("Segoe UI", 13, "bold"),
                    bg="#eef4fb",
                    fg="#17324d",
                )
                group_title.pack(anchor="w")
                group_note = tk.Label(
                    info_card,
                    text=unavailable_message,
                    font=("Segoe UI", 10),
                    bg="#eef4fb",
                    fg="#49657f",
                    justify="left",
                    wraplength=1040,
                )
                group_note.pack(anchor="w", pady=(6, 0))
                continue

            self._build_metric_group_card(container, group)

        footer = tk.Frame(container, bg="#f4f6fb")
        footer.pack(fill="x", pady=(8, 0))

        footer_text = tk.Label(
            footer,
            text="Select the desired metrics, define the project global settings, and click Save settings to continue.",
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#64748b",
            anchor="w",
        )
        footer_text.pack(side="left")

        continue_button = tk.Button(
            footer,
            text="Save settings",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._save_project_global_settings,
        )
        continue_button.pack(side="right")

    def _serialize_metric_selection_state(self) -> dict[str, dict[str, dict[str, object]]]:
        serialized: dict[str, dict[str, dict[str, object]]] = {}
        for group_id, group_metrics in self.metric_selection_state.items():
            serialized[group_id] = {}
            for metric_id, metric_state in group_metrics.items():
                serialized[group_id][metric_id] = {
                    "selected": bool(metric_state["selected"].get()),
                    "weight": str(metric_state["weight"].get()).strip(),
                    "limits": {
                        "good": str(metric_state["limits"]["good"].get()).strip(),
                        "acceptable": str(metric_state["limits"]["acceptable"].get()).strip(),
                    },
                }
        return serialized

    def _serialize_metric_special_options(self) -> dict[str, object]:
        return {
            "waveform_adjust_phase_error": bool(self.metric_special_options["waveform_adjust_phase_error"].get()),
            "transient_adjust_reaction_time": bool(self.metric_special_options["transient_adjust_reaction_time"].get()),
            "transient_normalization_mode": self._selected_transient_normalization_mode(),
            "score_mode": self._selected_score_mode(),
            "window_aggregation_mode": self._selected_window_aggregation_mode(),
        }

    def _save_project_global_settings(self) -> None:
        if self.current_assessment_project is None:
            self._build_project_creation_screen()
            return

        nominal_power = self.project_global_form_vars["nominal_power_w"].get().strip()
        nominal_voltage = self.project_global_form_vars["nominal_voltage_v"].get().strip()
        if not nominal_power or not nominal_voltage:
            messagebox.showerror(
                "Incomplete global settings",
                "Fill in the nominal power and nominal voltage before continuing.",
            )
            return

        self.current_assessment_project.global_settings.update(
            {
                "nominal_power_w": nominal_power,
                "nominal_voltage_v": nominal_voltage,
                "system_configuration": str(self.project_global_form_vars["system_configuration"].get()).strip()
                or self.SYSTEM_CONFIGURATION_OPTIONS[0],
                "filter_signals_for_analysis": bool(self.project_filter_signals_var.get()),
                "analysis_representation": str(self.sync_analysis_mode_var.get()).strip(),
                "transient_end_tolerance_percent": str(self.window_tolerance_percent_var.get()).strip(),
                "minimum_signal_variation_percent": str(self.min_transition_percent_var.get()).strip(),
                "metric_selection": self._serialize_metric_selection_state(),
                "metric_special_options": self._serialize_metric_special_options(),
            }
        )
        self.current_assessment_project.touch()
        self._autosave_current_project()
        self._build_project_run_definition_screen()

    def _build_project_run_definition_screen(self) -> None:
        if self.current_assessment_project is None:
            self._build_project_creation_screen()
            return

        self.project_runs_preview = self._generate_project_runs_preview()
        self._clear_screen()
        _, container = self._create_scrollable_screen(padx=28, pady=28)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_project_global_settings_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Define project runs",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(8, 6))

        subtitle = tk.Label(
            container,
            text=(
                "Review how the experimental and model captures will be combined for each test case. "
                "The default strategy is all x all."
            ),
            font=("Segoe UI", 11),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1080,
        )
        subtitle.pack(anchor="w", pady=(0, 18))

        strategy_card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=20, pady=16)
        strategy_card.pack(fill="x", pady=(0, 14))

        strategy_title = tk.Label(
            strategy_card,
            text="Run strategy",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#17324d",
        )
        strategy_title.pack(anchor="w", pady=(0, 8))

        strategy_combo = ttk.Combobox(
            strategy_card,
            textvariable=self.project_run_strategy_var,
            state="readonly",
            values=["all_x_all"],
            width=26,
        )
        strategy_combo.pack(anchor="w")
        strategy_combo.set("all_x_all")

        strategy_note = tk.Label(
            strategy_card,
            text="Every experimental capture is crossed with every model capture of the same test case.",
            font=("Segoe UI", 9),
            bg="white",
            fg="#64748b",
            justify="left",
            wraplength=1040,
        )
        strategy_note.pack(anchor="w", pady=(8, 0))

        preview_card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=20, pady=20)
        preview_card.pack(fill="x", pady=(0, 14))

        preview_title = tk.Label(
            preview_card,
            text="Run preview",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#17324d",
        )
        preview_title.pack(anchor="w", pady=(0, 10))

        table = tk.Frame(preview_card, bg="white")
        table.pack(fill="x")

        headers = ["Test case", "Experimental captures", "Model captures", "Generated runs"]
        for column_index, header_text in enumerate(headers):
            header = tk.Label(
                table,
                text=header_text,
                font=("Segoe UI", 9, "bold"),
                bg="#e8eef7",
                fg="#17324d",
                bd=1,
                relief="solid",
                padx=8,
                pady=6,
            )
            header.grid(row=0, column=column_index, sticky="nsew")
            table.grid_columnconfigure(column_index, weight=1)

        for row_index, preview in enumerate(self.project_runs_preview, start=1):
            row_values = [
                str(preview["case_name"]),
                str(preview["experimental_capture_count"]),
                str(preview["model_capture_count"]),
                str(preview["run_count"]),
            ]
            for column_index, cell_value in enumerate(row_values):
                cell = tk.Label(
                    table,
                    text=cell_value,
                    font=("Segoe UI", 9),
                    bg="white",
                    fg="#334155",
                    bd=1,
                    relief="solid",
                    padx=8,
                    pady=6,
                )
                cell.grid(row=row_index, column=column_index, sticky="nsew")

        total_runs = sum(int(preview["run_count"]) for preview in self.project_runs_preview)
        total_label = tk.Label(
            preview_card,
            text=f"Total generated runs: {total_runs}",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#17324d",
        )
        total_label.pack(anchor="e", pady=(12, 0))

        footer = tk.Frame(container, bg="#f4f6fb")
        footer.pack(fill="x", pady=(8, 0))

        generate_button = tk.Button(
            footer,
            text="Generate model indices",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._start_project_processing,
        )
        generate_button.pack(side="right")

    def _generate_project_runs_preview(self) -> list[dict[str, object]]:
        previews: list[dict[str, object]] = []
        for case_definition in self.project_test_case_definitions:
            experimental_count = int(case_definition["experimental_capture_count"])
            model_count = int(case_definition["model_capture_count"])
            previews.append(
                {
                    "case_index": int(case_definition["case_index"]),
                    "case_name": str(case_definition["case_name"]),
                    "experimental_capture_count": experimental_count,
                    "model_capture_count": model_count,
                    "run_count": experimental_count * model_count,
                }
            )
        return previews

    def _project_classification_for_capture(
        self,
        side: str,
        case_index: int,
        capture_index: int,
    ) -> tuple[list[str], list[str]]:
        if self.current_assessment_project is None:
            return [], []
        classification = self.current_assessment_project.global_settings.get("project_column_classification", {})
        for group in classification.get(side, []):
            if group.get("shared"):
                return list(group.get("headers", [])), list(group.get("selections", []))
            for file_ref in group.get("file_refs", []):
                if (
                    int(file_ref.get("case_index", -1)) == case_index
                    and int(file_ref.get("capture_index", -1)) == capture_index
                ):
                    return list(group.get("headers", [])), list(group.get("selections", []))
        return [], []

    def _populate_form_vars_for_project_case(self, test_key: str, form_values: dict[str, str]) -> None:
        for current_test_key, variables in self.test_form_vars.items():
            for variable in variables.values():
                variable.set("")
        for field_name, field_value in form_values.items():
            if field_name in self.test_form_vars[test_key]:
                self.test_form_vars[test_key][field_name].set(str(field_value))

    def _apply_project_metric_settings_to_runtime(self) -> None:
        if self.current_assessment_project is None:
            return
        global_settings = self.current_assessment_project.global_settings
        metric_selection = global_settings.get("metric_selection", {})
        for group_id, group_metrics in self.metric_selection_state.items():
            for metric_id, metric_state in group_metrics.items():
                saved_state = metric_selection.get(group_id, {}).get(metric_id, {})
                metric_state["selected"].set(bool(saved_state.get("selected", False)))
                metric_state["weight"].set(str(saved_state.get("weight", "1")))
                metric_state["limits"]["good"].set(str(saved_state.get("limits", {}).get("good", "")))
                metric_state["limits"]["acceptable"].set(str(saved_state.get("limits", {}).get("acceptable", "")))

        metric_special = global_settings.get("metric_special_options", {})
        self.metric_special_options["waveform_adjust_phase_error"].set(bool(metric_special.get("waveform_adjust_phase_error", False)))
        self.metric_special_options["transient_adjust_reaction_time"].set(bool(metric_special.get("transient_adjust_reaction_time", False)))
        self.metric_special_options["transient_normalization_mode"].set(str(metric_special.get("transient_normalization_mode", "event_amplitude")))
        self.metric_special_options["score_mode"].set(str(metric_special.get("score_mode", "continuous")))
        self.metric_special_options["window_aggregation_mode"].set(str(metric_special.get("window_aggregation_mode", "average")))

        self.project_filter_signals_var.set(bool(global_settings.get("filter_signals_for_analysis", False)))
        analysis_representation = str(global_settings.get("analysis_representation", "per_unit_independent")).strip()
        self.sync_analysis_mode_var.set("real" if analysis_representation in {"real_values", "real"} else analysis_representation)
        self.window_tolerance_percent_var.set(str(global_settings.get("transient_end_tolerance_percent", "5")))
        self.min_transition_percent_var.set(str(global_settings.get("minimum_signal_variation_percent", "1")))

    def _start_project_processing(self) -> None:
        if self.current_assessment_project is None:
            self._build_project_creation_screen()
            return

        self._apply_project_metric_settings_to_runtime()
        self.project_processing_task_state = {
            "done": False,
            "progress_percent": 0.0,
            "status_text": "Preparing runs...",
            "result": None,
            "error": None,
        }
        self._show_loading_indicator(
            title_text="Generating model indices",
            subtitle_text="Processing all runs, calculating the metrics, and consolidating the model validation report.",
            determinate=True,
        )
        worker = threading.Thread(
            target=self._run_project_processing_worker,
            daemon=True,
        )
        worker.start()
        self.root.after(150, self._poll_project_processing_task)

    def _poll_project_processing_task(self) -> None:
        if self.project_processing_task_state is None:
            return
        state = self.project_processing_task_state
        if self.loading_progress is not None:
            self.loading_progress.configure(value=float(state.get("progress_percent", 0.0)))
        if self.loading_status_label is not None:
            self.loading_status_label.config(text=str(state.get("status_text", "")))
        if self.loading_percent_label is not None:
            self.loading_percent_label.config(text=f"{float(state.get('progress_percent', 0.0)):.1f}%")

        if not bool(state.get("done", False)):
            self.root.after(150, self._poll_project_processing_task)
            return

        self._hide_loading_indicator()
        error = state.get("error")
        result = state.get("result")
        self.project_processing_task_state = None
        if error is not None:
            messagebox.showerror("Project processing error", str(error))
            return
        self._build_project_report_screen(result)

    def _run_project_processing_worker(self) -> None:
        if self.project_processing_task_state is None:
            return
        try:
            report = self._process_assessment_project()
            self.project_processing_task_state["result"] = report
            self.project_processing_task_state["done"] = True
            self.project_processing_task_state["progress_percent"] = 100.0
            self.project_processing_task_state["status_text"] = "Processing complete."
        except Exception as exc:
            self.project_processing_task_state["error"] = exc
            self.project_processing_task_state["done"] = True

    def _process_assessment_project(self) -> dict[str, object]:
        if self.current_assessment_project is None:
            raise ValueError("There is no active assessment project.")

        all_runs = sum(preview["run_count"] for preview in self._generate_project_runs_preview())
        processed_counter = 0
        project_signal_type = str(self.current_assessment_project.global_settings.get("project_signal_type", self.project_signal_type_var.get())).strip()
        processed_cases: list[dict[str, object]] = []

        for case_definition in self.project_test_case_definitions:
            case_runs = self._generate_runs_for_case(case_definition)
            processed_runs: list[dict[str, object]] = []
            for run_definition in case_runs:
                processed_counter += 1
                if self.project_processing_task_state is not None:
                    self.project_processing_task_state["status_text"] = (
                        f"Processing {case_definition['case_name']} - run {processed_counter} of {all_runs}"
                    )
                    self.project_processing_task_state["progress_percent"] = processed_counter / max(all_runs, 1) * 100.0
                processed_runs.append(
                    self._process_project_run(
                        case_definition=case_definition,
                        run_definition=run_definition,
                        signal_type=project_signal_type,
                    )
                )
            processed_cases.append({**dict(case_definition), "runs": processed_runs})

        self.current_assessment_project.test_cases = processed_cases
        report = self._build_project_report_data(processed_cases)
        self.current_assessment_project.global_settings["processed_report"] = report
        self.current_assessment_project.touch()
        self._autosave_current_project()
        return report

    def _generate_runs_for_case(self, case_definition: dict[str, object]) -> list[dict[str, object]]:
        runs: list[dict[str, object]] = []
        case_index = int(case_definition["case_index"])
        experimental_count = int(case_definition["experimental_capture_count"])
        model_count = int(case_definition["model_capture_count"])
        run_index = 0
        for experimental_capture_index in range(experimental_count):
            for model_capture_index in range(model_count):
                run_index += 1
                runs.append(
                    {
                        "run_id": f"case_{case_index + 1}_run_{run_index}",
                        "experimental_capture_index": experimental_capture_index,
                        "model_capture_index": model_capture_index,
                        "experimental_file": str(self.project_upload_files[(case_index, "experimental", experimental_capture_index)]),
                        "model_file": str(self.project_upload_files[(case_index, "model", model_capture_index)]),
                    }
                )
        return runs

    def _process_project_run(
        self,
        *,
        case_definition: dict[str, object],
        run_definition: dict[str, object],
        signal_type: str,
    ) -> dict[str, object]:
        case_index = int(case_definition["case_index"])
        test_key = str(case_definition["test_type"])
        form_values = dict(case_definition["form_values"])
        form_values["nominal_power_w"] = str(self.project_global_form_vars["nominal_power_w"].get()).strip()
        form_values["nominal_voltage_v"] = str(self.project_global_form_vars["nominal_voltage_v"].get()).strip()
        form_values["system_configuration"] = (
            str(self.project_global_form_vars["system_configuration"].get()).strip()
            or self.SYSTEM_CONFIGURATION_OPTIONS[0]
        )

        exp_headers, exp_selections = self._project_classification_for_capture(
            "experimental",
            case_index,
            int(run_definition["experimental_capture_index"]),
        )
        model_headers, model_selections = self._project_classification_for_capture(
            "model",
            case_index,
            int(run_definition["model_capture_index"]),
        )
        errors = validate_import_configuration(
            signal_type=signal_type,
            headers_by_side={"experimental": exp_headers, "model": model_headers},
            selections_by_side={"experimental": exp_selections, "model": model_selections},
        )
        if errors:
            raise ValueError(f"Invalid classification in {case_definition['case_name']}: {'; '.join(errors)}")

        previous_test_key = self.current_test_key
        previous_last_result = self.last_result
        previous_current_analysis_result = self.current_analysis_result
        previous_window_segments = self.current_window_segments

        self.current_test_key = test_key
        self._populate_form_vars_for_project_case(test_key, form_values)

        result = run_validation_pipeline(
            signal_type=signal_type,
            test_key=test_key,
            form_values=form_values,
            filter_signals=bool(self.project_filter_signals_var.get()),
            file_paths={
                "experimental": Path(str(run_definition["experimental_file"])),
                "model": Path(str(run_definition["model_file"])),
            },
            headers_by_side={"experimental": exp_headers, "model": model_headers},
            selections_by_side={"experimental": exp_selections, "model": model_selections},
        )
        self.last_result = result
        self.current_analysis_result = self._build_active_validation_result(result)
        self.current_window_segments = build_test_windows(
            test_key=test_key,
            form_values=form_values,
            validation_result=self.current_analysis_result,
            tolerance_percent=self._get_window_tolerance_percent(),
            min_transition_percent=self._get_min_transition_percent(),
        )
        group_results = self._calculate_selected_metric_results()
        score_sections = self._build_score_sections(group_results)

        run_payload = {
            "run_id": str(run_definition["run_id"]),
            "experimental_capture_index": int(run_definition["experimental_capture_index"]),
            "model_capture_index": int(run_definition["model_capture_index"]),
            "experimental_file": str(run_definition["experimental_file"]),
            "model_file": str(run_definition["model_file"]),
            "test_description": self._build_test_description(),
            "sync_label": result.sync_label,
            "group_results": group_results,
            "score_sections": score_sections,
        }

        self.current_test_key = previous_test_key
        self.last_result = previous_last_result
        self.current_analysis_result = previous_current_analysis_result
        self.current_window_segments = previous_window_segments
        return run_payload

    def _score_sections_to_records(self, score_sections: list[dict[str, object]]) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for section in score_sections:
            variant = str(section.get("variant", "original"))
            for row in section.get("rows", []):
                category = str(row.get("category", ""))
                for quantity, score in dict(row.get("values", {})).items():
                    try:
                        numeric_score = float(score)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(numeric_score):
                        continue
                    records.append(
                        {
                            "variant": variant,
                            "category": category,
                            "quantity": str(quantity),
                            "score": numeric_score,
                        }
                    )
        return records

    def _aggregate_project_score_records(
        self,
        records: list[dict[str, object]],
        group_key: str | None = None,
    ) -> list[dict[str, object]]:
        buckets: dict[tuple[str, str, str, str], list[float]] = {}
        for record in records:
            scope = str(record.get(group_key, "overall")) if group_key else "overall"
            bucket_key = (
                scope,
                str(record["variant"]),
                str(record["category"]),
                str(record["quantity"]),
            )
            buckets.setdefault(bucket_key, []).append(float(record["score"]))

        aggregated: list[dict[str, object]] = []
        for (scope, variant, category, quantity), values in buckets.items():
            values_array = np.asarray(values, dtype=float)
            aggregated.append(
                {
                    "group_key": scope,
                    "variant": variant,
                    "category": category,
                    "quantity": quantity,
                    "mean_score": float(np.mean(values_array)),
                    "median_score": float(np.median(values_array)),
                    "min_score": float(np.min(values_array)),
                    "max_score": float(np.max(values_array)),
                    "run_count": int(len(values_array)),
                }
            )
        return aggregated

    def _overall_project_score(self, aggregated_scores: list[dict[str, object]]) -> float:
        original_entries = [entry for entry in aggregated_scores if str(entry["variant"]) == "original"]
        if not original_entries:
            return float("nan")
        return float(np.mean([float(entry["mean_score"]) for entry in original_entries]))

    def _best_and_worst_score_entries(self, aggregated_scores: list[dict[str, object]]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        original_entries = [entry for entry in aggregated_scores if str(entry["variant"]) == "original"]
        if not original_entries:
            return None, None
        return (
            max(original_entries, key=lambda entry: float(entry["mean_score"])),
            min(original_entries, key=lambda entry: float(entry["mean_score"])),
        )

    def _project_aggregated_scores_to_sections(self, aggregated_scores: list[dict[str, object]]) -> list[dict[str, object]]:
        sections: list[dict[str, object]] = []
        variant_titles = {
            "original": "Original scores",
            "phase_adjusted": "Phase-delay-corrected scores",
            "delay_adjusted": "Delay-adjusted scores",
        }
        for variant_key in ("original", "phase_adjusted", "delay_adjusted"):
            variant_entries = [
                entry
                for entry in aggregated_scores
                if str(entry["variant"]) == variant_key and str(entry["group_key"]) == "overall"
            ]
            if not variant_entries:
                continue
            rows: list[dict[str, object]] = []
            available_quantities: set[str] = set()
            for category_definition in self.SCORE_CATEGORY_DEFINITIONS:
                category_name = str(category_definition["title"])
                category_entries = [entry for entry in variant_entries if str(entry["category"]) == category_name]
                if not category_entries:
                    continue
                quantity_scores = {}
                for entry in category_entries:
                    quantity = str(entry["quantity"])
                    quantity_scores[quantity] = float(entry["mean_score"])
                    available_quantities.add(quantity)
                rows.append({"category": category_name, "values": quantity_scores})
            sections.append(
                {
                    "variant": variant_key,
                    "title": variant_titles.get(variant_key, variant_key),
                    "quantities": [
                        quantity
                        for quantity in SCALAR_COLUMN_ORDER
                        if quantity in available_quantities
                    ],
                    "rows": rows,
                }
            )
        return sections

    def _build_project_report_data(self, project_cases: list[dict[str, object]]) -> dict[str, object]:
        run_records: list[dict[str, object]] = []
        test_case_summaries: list[dict[str, object]] = []
        for case in project_cases:
            case_records: list[dict[str, object]] = []
            for run in case.get("runs", []):
                for record in self._score_sections_to_records(list(run.get("score_sections", []))):
                    enriched_record = {
                        **record,
                        "test_type": str(case["test_type"]),
                        "test_case_name": str(case["case_name"]),
                        "run_id": str(run["run_id"]),
                    }
                    run_records.append(enriched_record)
                    case_records.append(enriched_record)
            test_case_summaries.append(
                {
                    "case_name": str(case["case_name"]),
                    "test_type": str(case["test_type"]),
                    "run_count": len(case.get("runs", [])),
                    "scores": self._aggregate_project_score_records(case_records),
                }
            )

        aggregated_scores = self._aggregate_project_score_records(run_records)
        scores_by_test_type = self._aggregate_project_score_records(run_records, group_key="test_type")
        best_entry, worst_entry = self._best_and_worst_score_entries(aggregated_scores)

        return {
            "model_name": self.current_assessment_project.model_name,
            "model_version": self.current_assessment_project.model_version,
            "project_id": self.current_assessment_project.project_id,
            "system_configuration": str(
                self.current_assessment_project.global_settings.get(
                    "system_configuration",
                    self.SYSTEM_CONFIGURATION_OPTIONS[0],
                )
            ),
            "total_test_cases": len(project_cases),
            "total_runs": sum(len(case.get("runs", [])) for case in project_cases),
            "overall_score": self._overall_project_score(aggregated_scores),
            "best_entry": best_entry,
            "worst_entry": worst_entry,
            "aggregated_scores": aggregated_scores,
            "scores_by_test_type": scores_by_test_type,
            "scores_by_test_case": test_case_summaries,
            "score_sections": self._project_aggregated_scores_to_sections(aggregated_scores),
        }

    def _format_project_score(self, value: object) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(numeric_value):
            return "-"
        return f"{numeric_value:.3f}"

    def _project_best_worst_summary(self, best_entry: object, worst_entry: object) -> str:
        messages = []
        if isinstance(best_entry, dict):
            messages.append(
                f"Best-performing area: {best_entry['category']} in {self._score_quantity_label(str(best_entry['quantity']))} with score {float(best_entry['mean_score']):.3f}."
            )
        if isinstance(worst_entry, dict):
            messages.append(
                f"Weakest area: {worst_entry['category']} in {self._score_quantity_label(str(worst_entry['quantity']))} with score {float(worst_entry['mean_score']):.3f}."
            )
        return " ".join(messages) if messages else "No consolidated result is available yet."

    def _create_embedded_scrollable_area(self, parent: tk.Frame) -> tuple[tk.Frame, tk.Frame]:
        outer = tk.Frame(parent, bg="#f4f6fb")
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, bg="#f4f6fb", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        container = tk.Frame(canvas, bg="#f4f6fb", padx=10, pady=10)
        window_id = canvas.create_window((0, 0), window=container, anchor="nw")
        container.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))
        return outer, container

    def _aggregate_scores_by_dimension(self, aggregated_scores: list[dict[str, object]], dimension: str) -> list[dict[str, object]]:
        buckets: dict[str, list[float]] = {}
        for entry in aggregated_scores:
            if str(entry.get("variant")) != "original":
                continue
            key = str(entry.get(dimension, ""))
            buckets.setdefault(key, []).append(float(entry["mean_score"]))
        summary_rows: list[dict[str, object]] = []
        for key, values in buckets.items():
            values_array = np.asarray(values, dtype=float)
            summary_rows.append(
                {
                    "key": key,
                    "mean_score": float(np.mean(values_array)),
                    "median_score": float(np.median(values_array)),
                    "count": int(len(values_array)),
                }
            )
        summary_rows.sort(key=lambda row: float(row["mean_score"]), reverse=True)
        return summary_rows

    def _build_project_grouped_summary(
        self,
        aggregated_scores: list[dict[str, object]],
        *,
        outer_dimension: str,
        inner_dimension: str,
    ) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        for entry in aggregated_scores:
            if str(entry.get("variant")) != "original":
                continue
            outer_key = str(entry.get(outer_dimension, "")).strip()
            if not outer_key:
                continue
            grouped.setdefault(outer_key, []).append(dict(entry))

        sections: list[dict[str, object]] = []
        for outer_key, entries in grouped.items():
            rows = []
            for entry in entries:
                rows.append(
                    {
                        "label": str(entry.get(inner_dimension, "")),
                        "mean_score": float(entry.get("mean_score", float("nan"))),
                        "median_score": float(entry.get("median_score", float("nan"))),
                        "run_count": int(entry.get("run_count", 0)),
                    }
                )
            rows.sort(key=lambda item: (np.nan_to_num(item["mean_score"], nan=-1.0)), reverse=True)
            sections.append({"title": outer_key, "rows": rows})
        sections.sort(key=lambda section: str(section["title"]))
        return sections

    def _build_project_dimension_summary_card(self, parent: tk.Frame, title: str, summary_rows: list[dict[str, object]], label_formatter) -> None:
        card = tk.Frame(parent, bg="white", bd=1, relief="solid", padx=18, pady=18)
        card.pack(fill="x", pady=(0, 14))
        tk.Label(card, text=title, font=("Segoe UI", 13, "bold"), bg="white", fg="#17324d").pack(anchor="w", pady=(0, 10))

        table = tk.Frame(card, bg="white")
        table.pack(fill="x")
        headers = ["Item", "Mean score", "Median score", "Count"]
        for column_index, header_text in enumerate(headers):
            header = tk.Label(
                table,
                text=header_text,
                font=("Segoe UI", 9, "bold"),
                bg="#e8eef7",
                fg="#17324d",
                bd=1,
                relief="solid",
                padx=8,
                pady=6,
            )
            header.grid(row=0, column=column_index, sticky="nsew")
            table.grid_columnconfigure(column_index, weight=1)

        for row_index, summary in enumerate(summary_rows, start=1):
            row_values = [
                str(label_formatter(str(summary["key"]))),
                f"{float(summary['mean_score']):.3f}",
                f"{float(summary['median_score']):.3f}",
                str(summary["count"]),
            ]
            for column_index, cell_value in enumerate(row_values):
                bg_color = "white"
                fg_color = "#334155"
                if column_index == 1:
                    bg_color, fg_color = self._score_value_color(float(summary["mean_score"]))
                cell = tk.Label(
                    table,
                    text=cell_value,
                    font=("Segoe UI", 9),
                    bg=bg_color,
                    fg=fg_color,
                    bd=1,
                    relief="solid",
                    padx=8,
                    pady=6,
                )
                cell.grid(row=row_index, column=column_index, sticky="nsew")

    def _build_project_grouped_tables_card(
        self,
        parent: tk.Frame,
        card_title: str,
        grouped_sections: list[dict[str, object]],
        *,
        section_title_formatter,
        row_label_formatter,
    ) -> None:
        card = tk.Frame(parent, bg="white", bd=1, relief="solid", padx=18, pady=18)
        card.pack(fill="x", pady=(0, 14))
        tk.Label(card, text=card_title, font=("Segoe UI", 13, "bold"), bg="white", fg="#17324d").pack(anchor="w", pady=(0, 10))

        if not grouped_sections:
            tk.Label(
                card,
                text="No consolidated data are available for this view.",
                font=("Segoe UI", 10),
                bg="white",
                fg="#64748b",
            ).pack(anchor="w")
            return

        for section in grouped_sections:
            section_frame = tk.Frame(card, bg="white")
            section_frame.pack(fill="x", pady=(0, 14))
            tk.Label(
                section_frame,
                text=str(section_title_formatter(str(section["title"]))),
                font=("Segoe UI", 11, "bold"),
                bg="white",
                fg="#17324d",
            ).pack(anchor="w", pady=(0, 6))
            self._build_project_grouped_rows_table(
                section_frame,
                list(section["rows"]),
                row_label_formatter=row_label_formatter,
            )

    def _build_project_grouped_rows_table(
        self,
        parent: tk.Frame,
        rows: list[dict[str, object]],
        *,
        row_label_formatter,
    ) -> None:
        table = tk.Frame(parent, bg="white")
        table.pack(fill="x")
        headers = ["Item", "Mean score", "Median score", "Runs"]
        for column_index, header_text in enumerate(headers):
            header = tk.Label(
                table,
                text=header_text,
                font=("Segoe UI", 9, "bold"),
                bg="#e8eef7",
                fg="#17324d",
                bd=1,
                relief="solid",
                padx=8,
                pady=6,
            )
            header.grid(row=0, column=column_index, sticky="nsew")
            table.grid_columnconfigure(column_index, weight=1)

        for row_index, row in enumerate(rows, start=1):
            mean_score = float(row.get("mean_score", float("nan")))
            row_values = [
                str(row_label_formatter(str(row.get("label", "")))),
                "-" if not np.isfinite(mean_score) else f"{mean_score:.3f}",
                "-" if not np.isfinite(float(row.get("median_score", float("nan")))) else f"{float(row['median_score']):.3f}",
                str(row.get("run_count", "-")),
            ]
            for column_index, cell_value in enumerate(row_values):
                bg_color = "white"
                fg_color = "#334155"
                if column_index == 1 and np.isfinite(mean_score):
                    bg_color, fg_color = self._score_value_color(mean_score)
                cell = tk.Label(
                    table,
                    text=cell_value,
                    font=("Segoe UI", 9),
                    bg=bg_color,
                    fg=fg_color,
                    bd=1,
                    relief="solid",
                    padx=8,
                    pady=6,
                    justify="left" if column_index == 0 else "center",
                    anchor="w" if column_index == 0 else "center",
                )
                cell.grid(row=row_index, column=column_index, sticky="nsew")

    def _build_project_report_screen(self, report: dict[str, object]) -> None:
        self.current_project_report = dict(report)
        self._clear_screen()
        container = tk.Frame(self.root, bg="#f4f6fb", padx=20, pady=20)
        container.pack(fill="both", expand=True)
        self._add_screen_watermark(container, "#f4f6fb")

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 10))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_project_run_definition_screen,
        )
        back_button.pack(side="left")

        home_button = tk.Button(
            top_bar,
            text="Home",
            font=("Segoe UI", 10, "bold"),
            bg="#cbd5e1",
            fg="#17324d",
            activebackground="#b8c6d8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_start_mode_screen,
        )
        home_button.pack(side="left", padx=(10, 0))

        export_button = tk.Button(
            top_bar,
            text="Generate report",
            font=("Segoe UI", 10, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._generate_project_report_pdf,
        )
        export_button.pack(side="right")

        save_button = tk.Button(
            top_bar,
            text="Save results",
            font=("Segoe UI", 10, "bold"),
            bg="#0f766e",
            fg="white",
            activebackground="#115e59",
            activeforeground="white",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._save_project_results,
        )
        save_button.pack(side="right", padx=(0, 10))

        title = tk.Label(
            container,
            text="Complete model validation report",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            container,
            text="The final model validation report is organized in tabs for easier review.",
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#49657f",
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        notebook_frame = tk.Frame(container, bg="#f4f6fb")
        notebook_frame.pack(fill="both", expand=True)

        notebook = ttk.Notebook(notebook_frame, style="Codex.TNotebook")
        notebook.pack(fill="both", expand=True)

        executive_tab = tk.Frame(notebook, bg="#f4f6fb")
        notebook.add(executive_tab, text="Executive summary")
        _, executive_container = self._create_embedded_scrollable_area(executive_tab)
        executive_card = tk.Frame(executive_container, bg="white", bd=1, relief="solid", padx=18, pady=18)
        executive_card.pack(fill="x", pady=(0, 14))
        for label_text, value_text in (
            ("Model name", report.get("model_name", "-")),
            ("Version", report.get("model_version", "-") or "-"),
            ("Electrical system", report.get("system_configuration", "-") or "-"),
            ("Total test cases", str(report.get("total_test_cases", 0))),
            ("Total runs", str(report.get("total_runs", 0))),
            ("Overall score", self._format_project_score(report.get("overall_score", float("nan")))),
        ):
            row = tk.Frame(executive_card, bg="white")
            row.pack(fill="x", pady=4)
            tk.Label(row, text=f"{label_text}:", font=("Segoe UI", 10, "bold"), bg="white", fg="#17324d", width=16, anchor="w").pack(side="left")
            tk.Label(row, text=str(value_text), font=("Segoe UI", 10), bg="white", fg="#49657f", anchor="w").pack(side="left")
        findings_card = tk.Frame(executive_container, bg="white", bd=1, relief="solid", padx=18, pady=18)
        findings_card.pack(fill="x", pady=(0, 14))
        tk.Label(findings_card, text="Main findings", font=("Segoe UI", 13, "bold"), bg="white", fg="#17324d").pack(anchor="w")
        tk.Label(
            findings_card,
            text=self._project_best_worst_summary(report.get("best_entry"), report.get("worst_entry")),
            font=("Segoe UI", 10),
            bg="white",
            fg="#49657f",
            justify="left",
            wraplength=1040,
        ).pack(anchor="w", pady=(8, 0))
        if self.current_project_storage_path is not None:
            tk.Label(
                executive_container,
                text=f"Saved project file: {self.current_project_storage_path}",
                font=("Segoe UI", 9),
                bg="#f4f6fb",
                fg="#64748b",
                justify="left",
                wraplength=1040,
            ).pack(anchor="w")

        test_type_tab = tk.Frame(notebook, bg="#f4f6fb")
        notebook.add(test_type_tab, text="By test type")
        _, test_type_container = self._create_embedded_scrollable_area(test_type_tab)
        self._build_project_grouped_tables_card(
            test_type_container,
            "Scores by test type",
            self._build_project_grouped_summary(
                list(report.get("scores_by_test_type", [])),
                outer_dimension="group_key",
                inner_dimension="category",
            ),
            section_title_formatter=lambda value: self.TEST_DEFINITIONS.get(value, {}).get("screen_title", value),
            row_label_formatter=lambda value: value,
        )

        quantity_tab = tk.Frame(notebook, bg="#f4f6fb")
        notebook.add(quantity_tab, text="By quantity")
        _, quantity_container = self._create_embedded_scrollable_area(quantity_tab)
        self._build_project_grouped_tables_card(
            quantity_container,
            "Scores by quantity",
            self._build_project_grouped_summary(
                list(report.get("aggregated_scores", [])),
                outer_dimension="quantity",
                inner_dimension="category",
            ),
            section_title_formatter=self._score_quantity_label,
            row_label_formatter=lambda value: value,
        )

        category_tab = tk.Frame(notebook, bg="#f4f6fb")
        notebook.add(category_tab, text="By category")
        _, category_container = self._create_embedded_scrollable_area(category_tab)
        self._build_project_grouped_tables_card(
            category_container,
            "Scores by category",
            self._build_project_grouped_summary(
                list(report.get("aggregated_scores", [])),
                outer_dimension="category",
                inner_dimension="quantity",
            ),
            section_title_formatter=lambda value: value,
            row_label_formatter=self._score_quantity_label,
        )

        test_case_tab = tk.Frame(notebook, bg="#f4f6fb")
        notebook.add(test_case_tab, text="Test cases")
        _, test_case_container = self._create_embedded_scrollable_area(test_case_tab)
        for case_summary in report.get("scores_by_test_case", []):
            card = tk.Frame(test_case_container, bg="white", bd=1, relief="solid", padx=18, pady=18)
            card.pack(fill="x", pady=(0, 14))
            tk.Label(card, text=str(case_summary["case_name"]), font=("Segoe UI", 13, "bold"), bg="white", fg="#17324d").pack(anchor="w")
            tk.Label(
                card,
                text=f"{self.TEST_DEFINITIONS[str(case_summary['test_type'])]['screen_title']} | Runs: {case_summary['run_count']}",
                font=("Segoe UI", 10),
                bg="white",
                fg="#49657f",
            ).pack(anchor="w", pady=(4, 10))
            self._build_project_grouped_rows_table(
                card,
                [
                    {
                        "label": f"{entry.get('category', '')} - {self._score_quantity_label(str(entry.get('quantity', '')))} ({entry.get('variant', '')})",
                        "mean_score": float(entry.get("mean_score", float("nan"))),
                        "median_score": float(entry.get("median_score", float("nan"))),
                        "run_count": int(entry.get("run_count", 0)),
                    }
                    for entry in list(case_summary.get("scores", []))
                ],
                row_label_formatter=lambda value: value,
            )

        charts_tab = tk.Frame(notebook, bg="#f4f6fb")
        notebook.add(charts_tab, text="Radar charts")
        _, charts_container = self._create_embedded_scrollable_area(charts_tab)
        charts_card = tk.Frame(charts_container, bg="white", bd=1, relief="solid", padx=18, pady=18)
        charts_card.pack(fill="both", expand=True, pady=(0, 14))
        self._build_score_summary_card(charts_card, "Complete model validation", list(report.get("score_sections", [])))

    def _save_project_results(self) -> None:
        if self.current_assessment_project is None:
            self._build_project_creation_screen()
            return
        self.current_assessment_project.touch()
        self._autosave_current_project()
        if self.current_project_storage_path is None:
            messagebox.showerror("Save error", "It was not possible to save the current project results.")
            return
        messagebox.showinfo(
            "Results saved",
            f"The model validation results were saved successfully.\n\nFile: {self.current_project_storage_path}",
        )

    def _generate_project_report_pdf(self) -> None:
        report = self.current_project_report
        if not isinstance(report, dict) or not report:
            messagebox.showerror("Export error", "There is no processed model validation report to export.")
            return

        model_name = str(report.get("model_name", "model_validation")).strip() or "model_validation"
        model_version = str(report.get("model_version", "")).strip()
        safe_name = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in model_name)
        safe_version = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in model_version)
        default_name = f"{safe_name}_report.pdf" if not safe_version else f"{safe_name}_{safe_version}_report.pdf"

        target_path = filedialog.asksaveasfilename(
            title="Save model validation report",
            defaultextension=".pdf",
            filetypes=(("PDF files", "*.pdf"), ("All files", "*.*")),
            initialfile=default_name,
        )
        if not target_path:
            return

        try:
            self._export_project_report_pdf(report, Path(target_path))
        except Exception as exc:
            messagebox.showerror("Export error", f"It was not possible to generate the PDF report.\n\n{exc}")
            return

        messagebox.showinfo(
            "Report generated",
            f"The PDF report was generated successfully.\n\nFile: {target_path}",
        )

    def _export_project_report_pdf(self, report: dict[str, object], target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(target_path) as pdf:
            metadata = pdf.infodict()
            metadata["Title"] = f"Model validation report - {report.get('model_name', 'Model')}"
            metadata["Author"] = "Model comparison software - Luiz Fernando Menegazzo - UFSM"
            metadata["Subject"] = "Complete model validation report"

            self._pdf_add_project_executive_summary(pdf, report)

            test_type_sections = self._build_project_grouped_summary(
                list(report.get("scores_by_test_type", [])),
                outer_dimension="group_key",
                inner_dimension="category",
            )
            self._pdf_add_grouped_sections(
                pdf,
                page_title="Scores by test type",
                sections=test_type_sections,
                section_title_formatter=lambda value: self.TEST_DEFINITIONS.get(value, {}).get("screen_title", value),
                row_label_formatter=lambda value: value,
            )

            quantity_sections = self._build_project_grouped_summary(
                list(report.get("aggregated_scores", [])),
                outer_dimension="quantity",
                inner_dimension="category",
            )
            self._pdf_add_grouped_sections(
                pdf,
                page_title="Scores by quantity",
                sections=quantity_sections,
                section_title_formatter=self._score_quantity_label,
                row_label_formatter=lambda value: value,
            )

            category_sections = self._build_project_grouped_summary(
                list(report.get("aggregated_scores", [])),
                outer_dimension="category",
                inner_dimension="quantity",
            )
            self._pdf_add_grouped_sections(
                pdf,
                page_title="Scores by category",
                sections=category_sections,
                section_title_formatter=lambda value: value,
                row_label_formatter=self._score_quantity_label,
            )

            self._pdf_add_test_case_sections(pdf, list(report.get("scores_by_test_case", [])))
            self._pdf_add_radar_pages(pdf, list(report.get("score_sections", [])))

    def _pdf_add_project_executive_summary(self, pdf: PdfPages, report: dict[str, object]) -> None:
        figure = Figure(figsize=(11.69, 8.27), dpi=150)
        axis = figure.add_subplot(111)
        axis.axis("off")

        axis.text(0.02, 0.95, "Complete model validation report", fontsize=20, fontweight="bold", color="#17324d", va="top")
        axis.text(0.02, 0.90, "Executive summary", fontsize=12, color="#49657f", va="top")

        summary_rows = [
            ("Model name", str(report.get("model_name", "-"))),
            ("Version", str(report.get("model_version", "-") or "-")),
            ("Electrical system", str(report.get("system_configuration", "-") or "-")),
            ("Total test cases", str(report.get("total_test_cases", 0))),
            ("Total runs", str(report.get("total_runs", 0))),
            ("Overall score", self._format_project_score(report.get("overall_score", float("nan")))),
        ]
        y_position = 0.82
        for label_text, value_text in summary_rows:
            axis.text(0.04, y_position, f"{label_text}:", fontsize=11, fontweight="bold", color="#17324d", va="top")
            axis.text(0.24, y_position, value_text, fontsize=11, color="#334155", va="top")
            y_position -= 0.055

        findings_text = self._project_best_worst_summary(report.get("best_entry"), report.get("worst_entry"))
        axis.text(0.02, 0.50, "Main findings", fontsize=13, fontweight="bold", color="#17324d", va="top")
        axis.text(
            0.04,
            0.45,
            textwrap.fill(findings_text, width=110),
            fontsize=10,
            color="#334155",
            va="top",
            linespacing=1.5,
        )

        if self.current_project_storage_path is not None:
            axis.text(
                0.02,
                0.09,
                textwrap.fill(f"Saved project file: {self.current_project_storage_path}", width=120),
                fontsize=9,
                color="#64748b",
                va="top",
            )

        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    def _pdf_add_grouped_sections(
        self,
        pdf: PdfPages,
        *,
        page_title: str,
        sections: list[dict[str, object]],
        section_title_formatter,
        row_label_formatter,
    ) -> None:
        if not sections:
            self._pdf_add_message_page(pdf, page_title, "No consolidated data are available for this section.")
            return

        for section in sections:
            rows = []
            for row in list(section.get("rows", [])):
                mean_score = float(row.get("mean_score", float("nan")))
                median_score = float(row.get("median_score", float("nan")))
                rows.append(
                    {
                        "label": str(row_label_formatter(str(row.get("label", "")))),
                        "mean_score": mean_score,
                        "median_score": median_score,
                        "runs": int(row.get("run_count", 0)),
                    }
                )
            self._pdf_add_rows_table_pages(
                pdf,
                page_title=page_title,
                section_title=str(section_title_formatter(str(section.get("title", "")))),
                rows=rows,
            )

    def _pdf_add_test_case_sections(self, pdf: PdfPages, test_case_summaries: list[dict[str, object]]) -> None:
        if not test_case_summaries:
            self._pdf_add_message_page(pdf, "Test cases", "No processed test case was found in the report.")
            return

        for case_summary in test_case_summaries:
            rows = []
            for entry in list(case_summary.get("scores", [])):
                rows.append(
                    {
                        "label": (
                            f"{entry.get('category', '')} - "
                            f"{self._score_quantity_label(str(entry.get('quantity', '')))} "
                            f"({entry.get('variant', '')})"
                        ),
                        "mean_score": float(entry.get("mean_score", float("nan"))),
                        "median_score": float(entry.get("median_score", float("nan"))),
                        "runs": int(entry.get("run_count", 0)),
                    }
                )
            test_type = str(case_summary.get("test_type", ""))
            section_title = (
                f"{case_summary.get('case_name', 'Test case')} | "
                f"{self.TEST_DEFINITIONS.get(test_type, {}).get('screen_title', test_type)} | "
                f"Runs: {case_summary.get('run_count', 0)}"
            )
            self._pdf_add_rows_table_pages(
                pdf,
                page_title="Test cases",
                section_title=section_title,
                rows=rows,
            )

    def _pdf_add_rows_table_pages(
        self,
        pdf: PdfPages,
        *,
        page_title: str,
        section_title: str,
        rows: list[dict[str, object]],
        rows_per_page: int = 18,
    ) -> None:
        if not rows:
            self._pdf_add_message_page(pdf, page_title, f"No rows are available for: {section_title}")
            return

        for start_index in range(0, len(rows), rows_per_page):
            page_rows = rows[start_index:start_index + rows_per_page]
            table_rows = []
            score_cells: dict[tuple[int, int], tuple[str, str]] = {}
            for row_index, row in enumerate(page_rows, start=1):
                mean_score = float(row.get("mean_score", float("nan")))
                median_score = float(row.get("median_score", float("nan")))
                table_rows.append(
                    [
                        textwrap.fill(str(row.get("label", "")), width=52),
                        "-" if not np.isfinite(mean_score) else f"{mean_score:.3f}",
                        "-" if not np.isfinite(median_score) else f"{median_score:.3f}",
                        str(row.get("runs", "-")),
                    ]
                )
                if np.isfinite(mean_score):
                    score_cells[(row_index, 1)] = self._score_value_color(mean_score)

            figure = Figure(figsize=(11.69, 8.27), dpi=150)
            axis = figure.add_subplot(111)
            axis.axis("off")

            axis.text(0.02, 0.97, page_title, fontsize=17, fontweight="bold", color="#17324d", va="top")
            subtitle = section_title
            if len(rows) > rows_per_page:
                subtitle += f" | Rows {start_index + 1}-{start_index + len(page_rows)} of {len(rows)}"
            axis.text(0.02, 0.92, textwrap.fill(subtitle, width=120), fontsize=10, color="#49657f", va="top")

            table = axis.table(
                cellText=table_rows,
                colLabels=["Item", "Mean score", "Median score", "Runs"],
                loc="upper left",
                cellLoc="center",
                colLoc="center",
                bbox=[0.02, 0.05, 0.96, 0.82],
                colWidths=[0.52, 0.16, 0.16, 0.12],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            for (row_index, column_index), cell in table.get_celld().items():
                cell.set_edgecolor("#cbd5e1")
                cell.set_linewidth(0.8)
                if row_index == 0:
                    cell.set_facecolor("#e8eef7")
                    cell.get_text().set_color("#17324d")
                    cell.get_text().set_fontweight("bold")
                else:
                    cell.set_facecolor("white")
                    cell.get_text().set_color("#334155")
                    if column_index == 0:
                        cell.get_text().set_ha("left")
                        cell.PAD = 0.02
                    bg_fg = score_cells.get((row_index, column_index))
                    if bg_fg is not None:
                        cell.set_facecolor(bg_fg[0])
                        cell.get_text().set_color(bg_fg[1])

            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    def _pdf_add_message_page(self, pdf: PdfPages, title: str, message: str) -> None:
        figure = Figure(figsize=(11.69, 8.27), dpi=150)
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.text(0.02, 0.95, title, fontsize=18, fontweight="bold", color="#17324d", va="top")
        axis.text(0.02, 0.88, textwrap.fill(message, width=120), fontsize=11, color="#334155", va="top")
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    def _pdf_add_radar_pages(self, pdf: PdfPages, score_sections: list[dict[str, object]]) -> None:
        quantity_radar_charts = self._build_quantity_radar_charts(score_sections)
        quantity_radar_figure = self._build_score_radar_figure(
            quantity_radar_charts,
            "Radar charts by quantity",
        )
        if quantity_radar_figure is not None:
            pdf.savefig(quantity_radar_figure, bbox_inches="tight")
            plt.close(quantity_radar_figure)
        else:
            self._pdf_add_message_page(pdf, "Radar charts by quantity", "No radar chart data are available.")

        category_radar_charts = self._build_category_radar_charts(score_sections)
        category_radar_figure = self._build_score_radar_figure(
            category_radar_charts,
            "Radar charts by score category",
        )
        if category_radar_figure is not None:
            pdf.savefig(category_radar_figure, bbox_inches="tight")
            plt.close(category_radar_figure)
        else:
            self._pdf_add_message_page(pdf, "Radar charts by score category", "No radar chart data are available.")

    def _build_project_test_case_summary_screen(self, experimental_shared: bool, model_shared: bool) -> None:
        if self.current_assessment_project is None:
            self._build_project_overview_screen()
            return

        self._clear_screen()
        _, container = self._create_scrollable_screen(padx=28, pady=28)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_project_classification_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Test cases saved",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(8, 6))

        subtitle = tk.Label(
            container,
            text="The project now contains the configured test cases, uploaded files, and column classifications. The next step will be processing and consolidating the runs.",
            font=("Segoe UI", 11),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1080,
        )
        subtitle.pack(anchor="w", pady=(0, 24))

        card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=24, pady=24)
        card.pack(fill="x")

        summary_rows = (
            ("Configured test cases", str(self.current_assessment_project.test_case_count)),
            ("Project signal type", str(self.current_assessment_project.global_settings.get("project_signal_type", "-"))),
            ("Electrical system", str(self.current_assessment_project.global_settings.get("system_configuration", self.SYSTEM_CONFIGURATION_OPTIONS[0]))),
            ("Shared experimental classification", "Yes" if experimental_shared else "No"),
            ("Shared model classification", "Yes" if model_shared else "No"),
        )
        for label_text, value_text in summary_rows:
            row = tk.Frame(card, bg="white")
            row.pack(fill="x", pady=4)
            label = tk.Label(
                row,
                text=f"{label_text}:",
                font=("Segoe UI", 10, "bold"),
                bg="white",
                fg="#17324d",
                width=28,
                anchor="w",
            )
            label.pack(side="left")
            value = tk.Label(
                row,
                text=value_text,
                font=("Segoe UI", 10),
                bg="white",
                fg="#49657f",
                anchor="w",
            )
            value.pack(side="left")

        return_button = tk.Button(
            container,
            text="Return to project overview",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._build_project_overview_screen,
        )
        return_button.pack(anchor="e", pady=(18, 0))

    def _build_test_form_screen(self, test_key: str) -> None:
        self.current_test_key = test_key
        self._clear_screen()

        definition = self.TEST_DEFINITIONS[test_key]
        form_vars = self.test_form_vars[test_key]

        _, container = self._create_scrollable_screen(padx=28, pady=28)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_test_selection_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text=definition["screen_title"],
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(8, 6))

        subtitle = tk.Label(
            container,
            text="Fill in the test parameters to continue to signal import.",
            font=("Segoe UI", 11),
            bg="#f4f6fb",
            fg="#49657f",
        )
        subtitle.pack(anchor="w", pady=(0, 24))

        card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=24, pady=24)
        card.pack(fill="both", expand=True)

        for field in definition["fields"]:
            field_frame = tk.Frame(card, bg="white")
            field_frame.pack(fill="x", pady=(0, 16))

            label = tk.Label(
                field_frame,
                text=field["label"],
                font=("Segoe UI", 11, "bold"),
                bg="white",
                fg="#17324d",
                anchor="w",
                justify="left",
                wraplength=900,
            )
            label.pack(fill="x", pady=(0, 6))

            if field["kind"] == "entry":
                entry = tk.Entry(
                    field_frame,
                    textvariable=form_vars[field["name"]],
                    font=("Segoe UI", 11),
                    relief="solid",
                    bd=1,
                )
                entry.pack(fill="x", ipady=6)
            else:
                combo = ttk.Combobox(
                    field_frame,
                    textvariable=form_vars[field["name"]],
                    values=field["options"],
                    state="readonly",
                    font=("Segoe UI", 11),
                )
                if not form_vars[field["name"]].get().strip() and field["options"]:
                    combo.set(field["options"][0])
                combo.pack(fill="x", ipady=4)

        continue_button = tk.Button(
            card,
            text="Continue",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=16,
            pady=12,
            cursor="hand2",
            command=self._validate_test_form_and_continue,
        )
        continue_button.pack(anchor="e", pady=(10, 0))

    def _build_import_choice_screen(self) -> None:
        self._clear_screen()

        container = tk.Frame(self.root, bg="#f4f6fb", padx=32, pady=32)
        container.pack(fill="both", expand=True)
        self._add_screen_watermark(container, "#f4f6fb")

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._go_back_to_test_form,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Signal import",
            font=("Segoe UI", 24, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(pady=(20, 12))

        subtitle = tk.Label(
            container,
            text="Choose how you want to import the data for comparison.",
            font=("Segoe UI", 12),
            bg="#f4f6fb",
            fg="#49657f",
        )
        subtitle.pack(pady=(0, 32))

        buttons_frame = tk.Frame(container, bg="#f4f6fb")
        buttons_frame.pack(expand=True)

        sinusoidal_button = tk.Button(
            buttons_frame,
            text="Import sinusoidal signals",
            font=("Segoe UI", 13, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            width=38,
            height=3,
            bd=0,
            cursor="hand2",
            command=self._build_sinusoidal_upload_screen,
        )
        sinusoidal_button.pack(pady=12)

        scalar_button = tk.Button(
            buttons_frame,
            text="Import scalar signals",
            font=("Segoe UI", 13, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            width=38,
            height=3,
            bd=0,
            cursor="hand2",
            command=self._build_scalar_upload_screen,
        )
        scalar_button.pack(pady=12)

    def _go_back_to_test_form(self) -> None:
        if self.current_test_key is None:
            self._build_test_selection_screen()
            return
        self._build_test_form_screen(self.current_test_key)

    def _validate_test_form_and_continue(self) -> None:
        if self.current_test_key is None:
            self._build_test_selection_screen()
            return

        missing_fields = [
            field["label"]
            for field in self.TEST_DEFINITIONS[self.current_test_key]["fields"]
            if not self.test_form_vars[self.current_test_key][field["name"]].get().strip()
        ]
        if missing_fields:
            messagebox.showerror(
                "Incomplete form",
                "Fill in all test fields before continuing.",
            )
            return

        self._build_import_choice_screen()

    def _build_sinusoidal_upload_screen(self) -> None:
        self._build_signal_upload_screen("sinusoidal")

    def _build_scalar_upload_screen(self) -> None:
        self._build_signal_upload_screen("scalar")

    def _build_signal_upload_screen(self, signal_type: str) -> None:
        self.current_signal_type = signal_type
        self._clear_screen()
        self.file_labels = {}
        self.column_frames = {}
        self.validation_button = None
        self.validation_status_label = None
        signal_definition = self.SIGNAL_IMPORT_DEFINITIONS[signal_type]

        _, container = self._create_scrollable_screen(padx=28, pady=28)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 18))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_import_choice_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text=signal_definition["screen_title"],
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w", pady=(8, 6))

        subtitle = tk.Label(
            container,
            text=signal_definition["subtitle"],
            font=("Segoe UI", 11),
            bg="#f4f6fb",
            fg="#49657f",
        )
        subtitle.pack(anchor="w", pady=(0, 24))

        filter_frame = tk.Frame(container, bg="#f4f6fb")
        filter_frame.pack(fill="x", pady=(0, 14))

        filter_checkbox = tk.Checkbutton(
            filter_frame,
            text="Filter signals for analysis (zero-phase bidirectional filter)",
            variable=self.signal_filter_options[signal_type],
            onvalue=True,
            offvalue=False,
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#17324d",
            activebackground="#f4f6fb",
            activeforeground="#17324d",
            selectcolor="white",
            anchor="w",
        )
        filter_checkbox.pack(anchor="w")

        filter_help = tk.Label(
            filter_frame,
            text=(
                "The filter acts on the scalar quantities used for synchronization, windows, and metrics. "
                "The original sinusoidal waveforms remain unchanged."
            ),
            font=("Segoe UI", 9),
            bg="#f4f6fb",
            fg="#64748b",
            justify="left",
            wraplength=1100,
        )
        filter_help.pack(anchor="w", pady=(4, 0))

        groups_frame = tk.Frame(container, bg="#f4f6fb")
        groups_frame.pack(fill="both", expand=True)
        groups_frame.grid_columnconfigure(0, weight=1)
        groups_frame.grid_columnconfigure(1, weight=1)

        self._create_upload_group(
            parent=groups_frame,
            column=0,
            side="experimental",
            title="Experimental signals",
            button_text="Import experimental signals",
        )
        self._create_upload_group(
            parent=groups_frame,
            column=1,
            side="model",
            title="Model signals",
            button_text="Import model signals",
        )

        actions_frame = tk.Frame(container, bg="#f4f6fb")
        actions_frame.pack(fill="x", pady=(18, 0))

        self.validation_status_label = tk.Label(
            actions_frame,
            text="Import both files and classify all columns to continue.",
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#64748b",
            anchor="w",
            justify="left",
        )
        self.validation_status_label.pack(side="left", fill="x", expand=True)

        self.validation_button = tk.Button(
            actions_frame,
            text="Continue model validation",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            state="disabled",
            command=self._start_validation,
        )
        self.validation_button.pack(side="right")

        self._update_validation_button_state()

    def _create_upload_group(
        self,
        parent: tk.Frame,
        column: int,
        side: str,
        title: str,
        button_text: str,
    ) -> None:
        card = tk.Frame(parent, bg="white", bd=1, relief="solid", padx=20, pady=20)
        card.grid(row=0, column=column, sticky="nsew", padx=10, pady=10)

        heading = tk.Label(
            card,
            text=title,
            font=("Segoe UI", 15, "bold"),
            bg="white",
            fg="#17324d",
        )
        heading.pack(anchor="w", pady=(0, 16))

        button = tk.Button(
            card,
            text=button_text,
            font=("Segoe UI", 11, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            bd=0,
            padx=12,
            pady=12,
            cursor="hand2",
            command=lambda selected_side=side: self._select_csv_file(selected_side),
        )
        button.pack(fill="x", pady=(0, 6))

        current_signal_type = self.current_signal_type or "sinusoidal"
        selected_file = self.selected_files[current_signal_type][side]
        label_text = selected_file.name if selected_file is not None else "No file selected"
        label_color = "#0f766e" if selected_file is not None else "#64748b"

        file_label = tk.Label(
            card,
            text=label_text,
            font=("Segoe UI", 10),
            bg="white",
            fg=label_color,
            anchor="w",
            justify="left",
            wraplength=420,
        )
        file_label.pack(fill="x", pady=(0, 14))
        self.file_labels[side] = file_label

        columns_frame = tk.Frame(card, bg="white")
        columns_frame.pack(fill="x", expand=True)
        self.column_frames[side] = columns_frame
        self._render_column_selectors(side)

    def _select_csv_file(self, side: str) -> None:
        current_signal_type = self.current_signal_type or "sinusoidal"
        filename = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv")],
        )
        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() != ".csv":
            messagebox.showerror(
                "Invalid format",
                "Only .csv files are accepted.",
            )
            return

        try:
            headers = read_csv_headers(path)
        except ValueError as error:
            messagebox.showerror("Error importing CSV", str(error))
            return

        self.selected_files[current_signal_type][side] = path
        self.column_names[current_signal_type][side] = headers
        default_selection = self.SIGNAL_IMPORT_DEFINITIONS[current_signal_type][
            "classification_options"
        ][-1]
        self.column_selection_values[current_signal_type][side] = [
            default_selection for _ in headers
        ]

        self.file_labels[side].config(text=path.name, fg="#0f766e")
        self._render_column_selectors(side)
        self._update_validation_button_state()

    def _render_column_selectors(self, side: str) -> None:
        current_signal_type = self.current_signal_type or "sinusoidal"
        columns_frame = self.column_frames[side]
        for widget in columns_frame.winfo_children():
            widget.destroy()

        headers = self.column_names[current_signal_type][side]
        selections = self.column_selection_values[current_signal_type][side]
        if not headers:
            return

        description = tk.Label(
            columns_frame,
            text="Classify each column in the imported file:",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#17324d",
            anchor="w",
        )
        description.pack(fill="x", pady=(4, 12))

        options = self.SIGNAL_IMPORT_DEFINITIONS[current_signal_type]["classification_options"]
        for index, header in enumerate(headers, start=1):
            row_frame = tk.Frame(columns_frame, bg="white")
            row_frame.pack(fill="x", pady=(0, 10))

            label = tk.Label(
                row_frame,
                text=f'Column {index} - "{header}"',
                font=("Segoe UI", 10),
                bg="white",
                fg="#334155",
                anchor="w",
                justify="left",
                wraplength=380,
            )
            label.pack(fill="x", pady=(0, 4))

            combo_var = tk.StringVar(value=selections[index - 1])
            combo = ttk.Combobox(
                row_frame,
                textvariable=combo_var,
                values=options,
                state="readonly",
                font=("Segoe UI", 10),
            )
            combo.pack(fill="x")
            combo.bind(
                "<<ComboboxSelected>>",
                lambda _event, selected_side=side, column_index=index - 1, var=combo_var: self._update_column_selection(
                    selected_side,
                    column_index,
                    var.get(),
                ),
            )

    def _update_column_selection(self, side: str, column_index: int, selected_value: str) -> None:
        current_signal_type = self.current_signal_type or "sinusoidal"
        self.column_selection_values[current_signal_type][side][column_index] = selected_value
        self._update_validation_button_state()

    def _update_validation_button_state(self) -> None:
        current_signal_type = self.current_signal_type
        if current_signal_type is None or self.validation_button is None:
            return

        headers_by_side = self.column_names[current_signal_type]
        selections_by_side = self.column_selection_values[current_signal_type]
        errors = validate_import_configuration(
            signal_type=current_signal_type,
            headers_by_side=headers_by_side,
            selections_by_side=selections_by_side,
        )

        if errors:
            self.validation_button.config(state="disabled")
            if self.validation_status_label is not None:
                self.validation_status_label.config(text=errors[0], fg="#b91c1c")
            return

        self.validation_button.config(state="normal")
        if self.validation_status_label is not None:
            self.validation_status_label.config(
                text="Files and classifications are ready. You can now continue validation.",
                fg="#0f766e",
            )

    def _start_validation(self) -> None:
        if self.current_test_key is None or self.current_signal_type is None:
            messagebox.showerror("Incomplete flow", "Select the test and signal type before validating.")
            return

        current_signal_type = self.current_signal_type
        headers_by_side = self.column_names[current_signal_type]
        selections_by_side = self.column_selection_values[current_signal_type]
        file_paths = self.selected_files[current_signal_type]

        errors = validate_import_configuration(
            signal_type=current_signal_type,
            headers_by_side=headers_by_side,
            selections_by_side=selections_by_side,
        )
        if errors:
            messagebox.showerror("Incomplete import", "\n".join(errors))
            return

        if file_paths["experimental"] is None or file_paths["model"] is None:
            messagebox.showerror("Missing files", "Import both CSV files before continuing.")
            return

        payload = {
            "signal_type": current_signal_type,
            "test_key": self.current_test_key,
            "form_values": self._get_current_form_values(),
            "filter_signals": self.signal_filter_options[current_signal_type].get(),
            "file_paths": {
                "experimental": file_paths["experimental"],
                "model": file_paths["model"],
            },
            "headers_by_side": headers_by_side,
            "selections_by_side": selections_by_side,
        }

        self.validation_task_result = None
        self._show_loading_indicator()
        if self.validation_button is not None:
            self.validation_button.config(state="disabled")

        worker = threading.Thread(
            target=self._run_validation_worker,
            args=(payload,),
            daemon=True,
        )
        worker.start()
        self.root.after(150, self._poll_validation_task)

    def _run_validation_worker(self, payload: dict[str, object]) -> None:
        try:
            result = run_validation_pipeline(
                signal_type=payload["signal_type"],
                test_key=payload["test_key"],
                form_values=payload["form_values"],
                filter_signals=bool(payload["filter_signals"]),
                file_paths=payload["file_paths"],
                headers_by_side=payload["headers_by_side"],
                selections_by_side=payload["selections_by_side"],
            )
            self.validation_task_result = ("success", result)
        except Exception as error:
            self.validation_task_result = ("error", error)

    def _poll_validation_task(self) -> None:
        if self.validation_task_result is None:
            self.root.after(150, self._poll_validation_task)
            return

        task_status, payload = self.validation_task_result
        self.validation_task_result = None
        self._hide_loading_indicator()
        self._update_validation_button_state()

        if task_status == "error":
            messagebox.showerror("Validation error", str(payload))
            return

        result = payload
        self.last_result = result
        self._build_validation_result_screen(result)

    def _show_loading_indicator(
        self,
        *,
        title_text: str = "Analyzing signals",
        subtitle_text: str = "Standardizing sampling, calculating quantities, and synchronizing the signals.",
        determinate: bool = False,
    ) -> None:
        self.root.config(cursor="watch")
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Processing")
        loading_window.geometry("420x170")
        loading_window.resizable(False, False)
        loading_window.configure(bg="#f4f6fb")
        loading_window.transient(self.root)
        loading_window.grab_set()
        loading_window.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = tk.Frame(loading_window, bg="#f4f6fb", padx=24, pady=24)
        frame.pack(fill="both", expand=True)

        title = tk.Label(
            frame,
            text=title_text,
            font=("Segoe UI", 16, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            frame,
            text=subtitle_text,
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=360,
        )
        subtitle.pack(anchor="w", pady=(8, 12))

        progress = ttk.Progressbar(
            frame,
            mode="determinate" if determinate else "indeterminate",
            length=340,
            maximum=100.0,
        )
        progress.pack(fill="x")
        if not determinate:
            progress.start(10)

        status_label = tk.Label(
            frame,
            text="Starting...",
            font=("Segoe UI", 9),
            bg="#f4f6fb",
            fg="#64748b",
            justify="left",
            wraplength=360,
        )
        status_label.pack(anchor="w", pady=(10, 0))

        percent_label = tk.Label(
            frame,
            text="0%",
            font=("Segoe UI", 10, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        percent_label.pack(anchor="e", pady=(4, 0))

        self.loading_window = loading_window
        self.loading_progress = progress
        self.loading_status_label = status_label
        self.loading_percent_label = percent_label
        self.root.update_idletasks()
        self.root.update()

    def _hide_loading_indicator(self) -> None:
        self.root.config(cursor="")
        if self.loading_progress is not None:
            self.loading_progress.stop()
            self.loading_progress = None
        self.loading_status_label = None
        self.loading_percent_label = None
        if self.loading_window is not None:
            self.loading_window.grab_release()
            self.loading_window.destroy()
            self.loading_window = None

    def _get_current_form_values(self) -> dict[str, str]:
        if self.current_test_key is None:
            return {}
        return {
            name: variable.get().strip()
            for name, variable in self.test_form_vars[self.current_test_key].items()
        }

    def _get_window_tolerance_percent(self) -> float:
        tolerance_value = self._safe_float(self.window_tolerance_percent_var.get())
        if tolerance_value <= 0:
            return 5.0
        return tolerance_value

    def _get_min_transition_percent(self) -> float:
        threshold_value = self._safe_float(self.min_transition_percent_var.get())
        if threshold_value <= 0:
            return 1.0
        return threshold_value

    def _build_validation_result_screen(self, result: ValidationResult) -> None:
        self._clear_screen()
        self.current_analysis_result = None
        current_signal_type = self.current_signal_type or result.signal_type
        _, container = self._create_scrollable_screen(padx=14, pady=14)
        self._add_screen_watermark(container, "#f4f6fb")

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 8))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=lambda: self._build_signal_upload_screen(current_signal_type),
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text=self.SIGNAL_IMPORT_DEFINITIONS[current_signal_type]["result_title"],
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            container,
            text=(
                f"Synchronization performed using {result.sync_label}. "
                "Review the plots below and choose which representation will be used in the next steps and calculations."
            ),
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1100,
        )
        subtitle.pack(anchor="w", pady=(4, 8))

        mode_frame = tk.Frame(container, bg="#eef4fb", bd=1, relief="solid", padx=14, pady=12)
        mode_frame.pack(fill="x", pady=(0, 8))

        mode_title = tk.Label(
            mode_frame,
            text="Representation to use in the next steps and calculations:",
            font=("Segoe UI", 10, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        mode_title.pack(anchor="w")

        radio_row = tk.Frame(mode_frame, bg="#eef4fb")
        radio_row.pack(fill="x", pady=(8, 0))

        for mode_value, mode_label in (
            ("real", "Real values"),
            ("per_unit_independent", "PU with independent base for experimental and model"),
            ("per_unit_reference", "PU using the experimental base for both signals"),
        ):
            radio = tk.Radiobutton(
                radio_row,
                text=mode_label,
                variable=self.sync_analysis_mode_var,
                value=mode_value,
                bg="#eef4fb",
                fg="#17324d",
                activebackground="#eef4fb",
                activeforeground="#17324d",
                selectcolor="white",
                font=("Segoe UI", 10),
            )
            radio.pack(anchor="w", pady=2)

        chart_frame = tk.Frame(container, bg="#f4f6fb")
        chart_frame.pack(fill="both", expand=True, pady=(0, 4))

        figure = self._build_validation_figure(result)

        self.current_figure = figure
        self.current_figure_canvas = FigureCanvasTkAgg(figure, master=chart_frame)
        self.current_figure_canvas.draw()
        self.current_figure_canvas.get_tk_widget().pack(fill="both", expand=True)

        tolerance_frame = tk.Frame(container, bg="#eef4fb", bd=1, relief="solid", padx=14, pady=12)
        tolerance_frame.pack(fill="x", pady=(8, 0))

        tolerance_label = tk.Label(
            tolerance_frame,
            text="Transient-end tolerance for window detection (% of event amplitude):",
            font=("Segoe UI", 10, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        tolerance_label.pack(side="left")

        tolerance_entry = tk.Entry(
            tolerance_frame,
            textvariable=self.window_tolerance_percent_var,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            width=10,
        )
        tolerance_entry.pack(side="left", padx=(10, 8), ipady=4)

        tolerance_unit = tk.Label(
            tolerance_frame,
            text="%",
            font=("Segoe UI", 10),
            bg="#eef4fb",
            fg="#49657f",
        )
        tolerance_unit.pack(side="left")

        transition_frame = tk.Frame(container, bg="#eef4fb", bd=1, relief="solid", padx=14, pady=12)
        transition_frame.pack(fill="x", pady=(8, 0))

        transition_label = tk.Label(
            transition_frame,
            text="Minimum signal variation to consider a transient window:",
            font=("Segoe UI", 10, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        transition_label.pack(side="left")

        transition_entry = tk.Entry(
            transition_frame,
            textvariable=self.min_transition_percent_var,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            width=10,
        )
        transition_entry.pack(side="left", padx=(10, 8), ipady=4)

        transition_unit = tk.Label(
            transition_frame,
            text="%",
            font=("Segoe UI", 10),
            bg="#eef4fb",
            fg="#49657f",
        )
        transition_unit.pack(side="left")

        transition_help = tk.Label(
            transition_frame,
            text="If the variation of a quantity is smaller than this value, that quantity will not have transient windows.",
            font=("Segoe UI", 9),
            bg="#eef4fb",
            fg="#64748b",
            justify="left",
        )
        transition_help.pack(side="right")

        actions_frame = tk.Frame(container, bg="#f4f6fb")
        actions_frame.pack(fill="x", pady=(8, 0))

        info = tk.Label(
            actions_frame,
            text=(
                "Each row shows the quantity in real scale, PU with independent bases, "
                "and PU using the experimental base for both signals."
            ),
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#64748b",
            anchor="w",
        )
        info.pack(side="left")

        confirm_button = tk.Button(
            actions_frame,
            text="Confirm synchronization",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._confirm_synchronization,
        )
        confirm_button.pack(side="right")

    def _build_validation_figure(self, result: ValidationResult) -> Figure:
        form_values = self._get_current_form_values()
        pu_experimental, pu_model, pu_targets = self._build_per_unit_data(result, form_values)
        shared_pu_experimental, shared_pu_model, _shared_targets = self._build_reference_based_per_unit_data(result, form_values)
        available_columns = [
            column
            for column in SCALAR_COLUMN_ORDER
            if column in result.scalar_experimental.columns and column in result.scalar_model.columns
        ]
        subplot_count = len(available_columns) + 1
        figure = Figure(figsize=(23, max(4.8, subplot_count * 1.9)), dpi=100)
        figure.patch.set_facecolor("#f4f6fb")

        for plot_index, column in enumerate(available_columns, start=1):
            axis = figure.add_subplot(subplot_count, 3, (plot_index - 1) * 3 + 1)
            axis_pu = figure.add_subplot(subplot_count, 3, (plot_index - 1) * 3 + 2)
            axis_pu_reference = figure.add_subplot(subplot_count, 3, (plot_index - 1) * 3 + 3)

            exp_values = result.scalar_experimental[column].to_numpy(dtype=float)
            model_values = result.scalar_model[column].to_numpy(dtype=float)
            exp_plot_time, exp_plot_values = self._build_smooth_plot_series(
                result.scalar_experimental["time"].to_numpy(dtype=float),
                exp_values,
            )
            model_plot_time, model_plot_values = self._build_smooth_plot_series(
                result.scalar_model["time"].to_numpy(dtype=float),
                model_values,
            )
            axis.plot(
                exp_plot_time,
                exp_plot_values,
                label="Experimental",
                color="#1d4ed8",
                linewidth=1.5,
            )
            axis.plot(
                model_plot_time,
                model_plot_values,
                label="Model",
                color="#dc2626",
                linewidth=1.5,
                alpha=0.85,
            )
            axis.set_title(SCALAR_DISPLAY_LABELS[column], loc="left", fontsize=11, fontweight="bold")
            if plot_index == subplot_count - 1:
                axis.set_xlabel("Time (s)")
            axis.grid(alpha=0.25)
            axis.legend(loc="upper right")
            self._set_time_axis_limits(
                axis,
                result.scalar_experimental["time"].to_numpy(dtype=float),
                result.scalar_model["time"].to_numpy(dtype=float),
            )
            self._set_axis_margin(
                axis,
                column,
                exp_values,
                model_values,
                form_values,
                result,
                is_per_unit=False,
                target_initial_pu=pu_targets.get(column, 1.0),
            )

            exp_pu_values = pu_experimental[column].to_numpy(dtype=float)
            model_pu_values = pu_model[column].to_numpy(dtype=float)
            axis_pu.plot(
                pu_experimental["time"],
                exp_pu_values,
                label="Experimental",
                color="#1d4ed8",
                linewidth=1.5,
            )
            axis_pu.plot(
                pu_model["time"],
                model_pu_values,
                label="Model",
                color="#dc2626",
                linewidth=1.5,
                alpha=0.85,
            )
            axis_pu.set_title(
                f"{SCALAR_DISPLAY_LABELS[column]} in PU",
                loc="left",
                fontsize=11,
                fontweight="bold",
            )
            if plot_index == subplot_count - 1:
                axis_pu.set_xlabel("Time (s)")
            axis_pu.grid(alpha=0.25)
            axis_pu.legend(loc="upper right")
            self._set_time_axis_limits(
                axis_pu,
                pu_experimental["time"].to_numpy(dtype=float),
                pu_model["time"].to_numpy(dtype=float),
            )
            self._set_axis_margin(
                axis_pu,
                column,
                exp_pu_values,
                model_pu_values,
                form_values,
                result,
                is_per_unit=True,
                target_initial_pu=pu_targets.get(column, 1.0),
            )

            exp_shared_values = shared_pu_experimental[column].to_numpy(dtype=float)
            model_shared_values = shared_pu_model[column].to_numpy(dtype=float)
            axis_pu_reference.plot(
                shared_pu_experimental["time"],
                exp_shared_values,
                label="Experimental",
                color="#1d4ed8",
                linewidth=1.5,
            )
            axis_pu_reference.plot(
                shared_pu_model["time"],
                model_shared_values,
                label="Model",
                color="#dc2626",
                linewidth=1.5,
                alpha=0.85,
            )
            axis_pu_reference.set_title(
                f"{SCALAR_DISPLAY_LABELS[column]} in PU (experimental base)",
                loc="left",
                fontsize=11,
                fontweight="bold",
            )
            if plot_index == subplot_count - 1:
                axis_pu_reference.set_xlabel("Time (s)")
            axis_pu_reference.grid(alpha=0.25)
            axis_pu_reference.legend(loc="upper right")
            self._set_time_axis_limits(
                axis_pu_reference,
                shared_pu_experimental["time"].to_numpy(dtype=float),
                shared_pu_model["time"].to_numpy(dtype=float),
            )
            self._set_axis_margin(
                axis_pu_reference,
                column,
                exp_shared_values,
                model_shared_values,
                form_values,
                result,
                is_per_unit=True,
                target_initial_pu=pu_targets.get(column, 1.0),
            )

        zoom_axis = figure.add_subplot(subplot_count, 3, (subplot_count - 1) * 3 + 1)
        zoom_axis_pu = figure.add_subplot(subplot_count, 3, (subplot_count - 1) * 3 + 2)
        zoom_axis_pu_reference = figure.add_subplot(subplot_count, 3, (subplot_count - 1) * 3 + 3)
        zoom_target = pu_targets.get(result.sync_column, 1.0)
        zoom_exp_time, zoom_exp_values = self._build_smooth_plot_series(
            result.zoom_experimental.index.to_numpy(dtype=float),
            result.zoom_experimental.values,
        )
        zoom_model_time, zoom_model_values = self._build_smooth_plot_series(
            result.zoom_model.index.to_numpy(dtype=float),
            result.zoom_model.values,
        )
        zoom_axis.plot(
            zoom_exp_time,
            zoom_exp_values,
            label="Experimental",
            color="#1d4ed8",
            linewidth=1.8,
        )
        zoom_axis.plot(
            zoom_model_time,
            zoom_model_values,
            label="Model",
            color="#dc2626",
            linewidth=1.8,
            alpha=0.85,
        )
        zoom_axis.axvline(result.zoom_time_s, color="#0f172a", linestyle="--", linewidth=1)
        zoom_axis.set_title(
            f"Synchronization zoom by {result.sync_label}",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        zoom_axis.set_xlabel("Time (s)")
        zoom_axis.grid(alpha=0.25)
        zoom_axis.legend(loc="upper right")
        self._set_time_axis_limits(
            zoom_axis,
            result.zoom_experimental.index.to_numpy(dtype=float),
            result.zoom_model.index.to_numpy(dtype=float),
        )
        self._set_axis_margin(
            zoom_axis,
            result.sync_column,
            result.zoom_experimental.to_numpy(dtype=float),
            result.zoom_model.to_numpy(dtype=float),
            form_values,
            result,
            is_per_unit=False,
            target_initial_pu=zoom_target,
        )

        zoom_exp_base = self._series_initial_mean(result.zoom_experimental, sample_count=3)
        zoom_model_base = self._series_initial_mean(result.zoom_model, sample_count=3)
        zoom_exp_pu = result.zoom_experimental.to_numpy(dtype=float) / zoom_exp_base * zoom_target
        zoom_model_pu = result.zoom_model.to_numpy(dtype=float) / zoom_model_base * zoom_target
        zoom_axis_pu.plot(
            result.zoom_experimental.index,
            zoom_exp_pu,
            label="Experimental",
            color="#1d4ed8",
            linewidth=1.8,
        )
        zoom_axis_pu.plot(
            result.zoom_model.index,
            zoom_model_pu,
            label="Model",
            color="#dc2626",
            linewidth=1.8,
            alpha=0.85,
        )
        zoom_axis_pu.axvline(result.zoom_time_s, color="#0f172a", linestyle="--", linewidth=1)
        zoom_axis_pu.set_title(
            f"PU synchronization zoom by {result.sync_label}",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        zoom_axis_pu.set_xlabel("Time (s)")
        zoom_axis_pu.grid(alpha=0.25)
        zoom_axis_pu.legend(loc="upper right")
        self._set_time_axis_limits(
            zoom_axis_pu,
            result.zoom_experimental.index.to_numpy(dtype=float),
            result.zoom_model.index.to_numpy(dtype=float),
        )
        self._set_axis_margin(
            zoom_axis_pu,
            result.sync_column,
            zoom_exp_pu,
            zoom_model_pu,
            form_values,
            result,
            is_per_unit=True,
            target_initial_pu=zoom_target,
        )

        zoom_exp_reference_base = self._series_initial_mean(result.zoom_experimental, sample_count=3)
        zoom_exp_pu_reference = result.zoom_experimental.to_numpy(dtype=float) / zoom_exp_reference_base * zoom_target
        zoom_model_pu_reference = result.zoom_model.to_numpy(dtype=float) / zoom_exp_reference_base * zoom_target
        zoom_axis_pu_reference.plot(
            result.zoom_experimental.index,
            zoom_exp_pu_reference,
            label="Experimental",
            color="#1d4ed8",
            linewidth=1.8,
        )
        zoom_axis_pu_reference.plot(
            result.zoom_model.index,
            zoom_model_pu_reference,
            label="Model",
            color="#dc2626",
            linewidth=1.8,
            alpha=0.85,
        )
        zoom_axis_pu_reference.axvline(result.zoom_time_s, color="#0f172a", linestyle="--", linewidth=1)
        zoom_axis_pu_reference.set_title(
            f"PU synchronization zoom by {result.sync_label} (experimental base)",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        zoom_axis_pu_reference.set_xlabel("Time (s)")
        zoom_axis_pu_reference.grid(alpha=0.25)
        zoom_axis_pu_reference.legend(loc="upper right")
        self._set_time_axis_limits(
            zoom_axis_pu_reference,
            result.zoom_experimental.index.to_numpy(dtype=float),
            result.zoom_model.index.to_numpy(dtype=float),
        )
        self._set_axis_margin(
            zoom_axis_pu_reference,
            result.sync_column,
            zoom_exp_pu_reference,
            zoom_model_pu_reference,
            form_values,
            result,
            is_per_unit=True,
            target_initial_pu=zoom_target,
        )

        figure.tight_layout(pad=0.8, h_pad=0.8, w_pad=0.8)
        return figure

    def _build_per_unit_data(
        self,
        result: ValidationResult,
        form_values: dict[str, str],
    ) -> tuple[dict[str, np.ndarray] | object, dict[str, np.ndarray] | object, dict[str, float]]:
        target_initial_pu = self._build_target_initial_pu_map(form_values)
        pu_experimental = result.scalar_experimental.copy()
        pu_model = result.scalar_model.copy()
        for column in SCALAR_COLUMN_ORDER:
            if column not in pu_experimental.columns or column not in pu_model.columns:
                continue
            target_pu = target_initial_pu.get(column, 1.0)
            exp_base = self._initial_dataset_mean(result.scalar_experimental, column, sample_count=3)
            model_base = self._initial_dataset_mean(result.scalar_model, column, sample_count=3)
            pu_experimental[column] = pu_experimental[column] / exp_base * target_pu
            pu_model[column] = pu_model[column] / model_base * target_pu

        return pu_experimental, pu_model, target_initial_pu

    def _build_reference_based_per_unit_data(
        self,
        result: ValidationResult,
        form_values: dict[str, str],
    ) -> tuple[dict[str, np.ndarray] | object, dict[str, np.ndarray] | object, dict[str, float]]:
        target_initial_pu = self._build_target_initial_pu_map(form_values)
        pu_experimental = result.scalar_experimental.copy()
        pu_model = result.scalar_model.copy()
        for column in SCALAR_COLUMN_ORDER:
            if column not in pu_experimental.columns or column not in pu_model.columns:
                continue
            target_pu = target_initial_pu.get(column, 1.0)
            exp_base = self._initial_dataset_mean(result.scalar_experimental, column, sample_count=3)
            pu_experimental[column] = pu_experimental[column] / exp_base * target_pu
            pu_model[column] = pu_model[column] / exp_base * target_pu

        return pu_experimental, pu_model, target_initial_pu

    def _selected_sync_analysis_mode(self) -> str:
        selected_mode = str(self.sync_analysis_mode_var.get()).strip()
        if selected_mode in {"real", "per_unit_independent", "per_unit_reference"}:
            return selected_mode
        return "per_unit_independent"

    def _active_scalar_datasets(
        self,
        result: ValidationResult,
    ) -> tuple[object, object, dict[str, float], str]:
        form_values = self._get_current_form_values()
        selected_mode = self._selected_sync_analysis_mode()
        if selected_mode == "real":
            return result.scalar_experimental.copy(), result.scalar_model.copy(), self._build_target_initial_pu_map(form_values), selected_mode
        if selected_mode == "per_unit_reference":
            exp_data, model_data, target_map = self._build_reference_based_per_unit_data(result, form_values)
            return exp_data, model_data, target_map, selected_mode
        exp_data, model_data, target_map = self._build_per_unit_data(result, form_values)
        return exp_data, model_data, target_map, "per_unit_independent"

    def _build_analysis_zoom_series(
        self,
        result: ValidationResult,
        selected_mode: str,
        target_initial_pu: dict[str, float],
    ) -> tuple[object, object]:
        if selected_mode == "real":
            return result.zoom_experimental.copy(), result.zoom_model.copy()

        zoom_target = target_initial_pu.get(result.sync_column, 1.0)
        exp_base = self._series_initial_mean(result.zoom_experimental, sample_count=3)
        if selected_mode == "per_unit_reference":
            model_base = exp_base
        else:
            model_base = self._series_initial_mean(result.zoom_model, sample_count=3)

        zoom_exp = result.zoom_experimental.copy()
        zoom_model = result.zoom_model.copy()
        zoom_exp[:] = zoom_exp.to_numpy(dtype=float) / exp_base * zoom_target
        zoom_model[:] = zoom_model.to_numpy(dtype=float) / model_base * zoom_target
        return zoom_exp, zoom_model

    def _build_active_validation_result(self, result: ValidationResult) -> ValidationResult:
        scalar_experimental, scalar_model, target_initial_pu, selected_mode = self._active_scalar_datasets(result)
        zoom_experimental, zoom_model = self._build_analysis_zoom_series(result, selected_mode, target_initial_pu)
        return ValidationResult(
            signal_type=result.signal_type,
            sync_label=result.sync_label,
            sync_column=result.sync_column,
            scalar_experimental=scalar_experimental,
            scalar_model=scalar_model,
            waveform_experimental=result.waveform_experimental,
            waveform_model=result.waveform_model,
            zoom_experimental=zoom_experimental,
            zoom_model=zoom_model,
            zoom_time_s=result.zoom_time_s,
        )

    def _current_result_for_analysis(self) -> ValidationResult | None:
        if self.current_analysis_result is not None:
            return self.current_analysis_result
        return self.last_result

    def _build_target_initial_pu_map(self, form_values: dict[str, str]) -> dict[str, float]:
        target_map = {column: 1.0 for column in SCALAR_COLUMN_ORDER}
        if self.current_test_key == "steady_state":
            test_power_fraction = self._safe_float(form_values.get("test_power_percent")) / 100.0
            if test_power_fraction > 0:
                target_map["active_power"] = test_power_fraction
                target_map["current"] = test_power_fraction
                target_map["reactive_power"] = test_power_fraction
                for column in SCALAR_COLUMN_ORDER:
                    if column.endswith("_unbalance"):
                        continue
                    if column.startswith("current_") or column.startswith("active_power_") or column.startswith("reactive_power_"):
                        target_map[column] = test_power_fraction
            return target_map

        pre_field_by_test = {
            "step_test": "pre_step_percent",
            "fault_test": "pre_fault_percent",
            "ramp_test": "pre_ramp_percent",
        }
        quantity_field_by_test = {
            "step_test": "step_quantity",
            "fault_test": "fault_quantity",
            "ramp_test": "ramp_quantity",
        }
        if self.current_test_key in pre_field_by_test:
            pre_value = self._safe_float(form_values.get(pre_field_by_test[self.current_test_key])) / 100.0
            quantity_label = form_values.get(quantity_field_by_test[self.current_test_key], "")
            quantity_column = TEST_QUANTITY_TO_INTERNAL.get(quantity_label)
            if quantity_column and pre_value > 0:
                target_map[quantity_column] = pre_value
        return target_map

    def _selected_transient_normalization_mode(self) -> str:
        selected_value = str(self.metric_special_options["transient_normalization_mode"].get()).strip()
        valid_modes = {mode_key for mode_key, _mode_label in self.TRANSIENT_NORMALIZATION_MODES}
        if selected_value in valid_modes:
            return selected_value
        label_to_key = {
            mode_label: mode_key for mode_key, mode_label in self.TRANSIENT_NORMALIZATION_MODES
        }
        if selected_value in label_to_key:
            return label_to_key[selected_value]
        return "event_amplitude"

    def _get_transient_event_fraction(self) -> float:
        if self.current_test_key not in ("step_test", "fault_test", "ramp_test"):
            return float("nan")

        form_values = self._get_current_form_values()
        field_map = {
            "step_test": ("pre_step_percent", "post_step_percent"),
            "fault_test": ("pre_fault_percent", "during_fault_percent"),
            "ramp_test": ("pre_ramp_percent", "post_ramp_percent"),
        }
        start_field, end_field = field_map[self.current_test_key]
        start_value = self._safe_float(form_values.get(start_field))
        end_value = self._safe_float(form_values.get(end_field))
        amplitude_fraction = abs(end_value - start_value) / 100.0
        if amplitude_fraction <= 0:
            return float("nan")
        return amplitude_fraction

    def _nominal_scalar_value(self, column: str) -> float:
        form_values = self._get_current_form_values()
        nominal_power = self._safe_float(form_values.get("nominal_power_w"))
        nominal_voltage = self._safe_float(form_values.get("nominal_voltage_v"))

        if column.endswith("_unbalance"):
            return 100.0
        if column == "voltage" or column.startswith("voltage_"):
            return nominal_voltage if nominal_voltage > 0 else float("nan")
        if column == "current" or column.startswith("current_"):
            if nominal_power > 0 and nominal_voltage > 0:
                return nominal_power / nominal_voltage
            return float("nan")
        if column == "frequency":
            if self.last_result is not None and "frequency" in self.last_result.scalar_experimental.columns:
                initial_frequency = self._initial_dataset_mean(
                    self.last_result.scalar_experimental,
                    "frequency",
                    sample_count=3,
                )
                nominal_frequency = self._infer_frequency_base(initial_frequency)
                if nominal_frequency > 0:
                    return nominal_frequency
            return 60.0
        if column in ("active_power", "reactive_power"):
            return nominal_power if nominal_power > 0 else float("nan")
        if column.startswith("active_power_") or column.startswith("reactive_power_"):
            return nominal_power / 3.0 if nominal_power > 0 else float("nan")
        return float("nan")

    def _origin_event_percent_denominator_pu(
        self,
        column: str,
        eps: float = 1e-12,
    ) -> float:
        if self.last_result is None:
            return float("nan")

        event_fraction = self._get_transient_event_fraction()
        if not np.isfinite(event_fraction) or event_fraction <= eps:
            return float("nan")

        nominal_value = self._nominal_scalar_value(column)
        if not np.isfinite(nominal_value) or abs(nominal_value) <= eps:
            return float("nan")

        base_value = self._initial_dataset_mean(self.last_result.scalar_experimental, column, sample_count=3)
        if not np.isfinite(base_value) or abs(base_value) <= eps:
            return float("nan")

        target_map = self._build_target_initial_pu_map(self._get_current_form_values())
        target_initial_pu = float(target_map.get(column, 1.0))
        denominator_pu = (event_fraction * nominal_value / abs(base_value)) * abs(target_initial_pu)
        if denominator_pu <= eps:
            return float("nan")
        return denominator_pu

    def _infer_frequency_base(self, initial_frequency: float) -> float:
        if not np.isfinite(initial_frequency) or initial_frequency <= 0:
            return 1.0
        if abs(initial_frequency - 50.0) <= abs(initial_frequency - 60.0):
            return 50.0
        return 60.0

    def _initial_dataset_value(self, dataset, column: str) -> float:
        if column not in dataset.columns or dataset.empty:
            return 1.0
        initial_value = float(dataset[column].iloc[0])
        if not np.isfinite(initial_value) or initial_value == 0:
            return 1.0
        return initial_value

    def _initial_dataset_mean(self, dataset, column: str, sample_count: int = 3) -> float:
        if column not in dataset.columns or dataset.empty:
            return 1.0
        values = dataset[column].head(sample_count).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 1.0
        mean_value = float(np.mean(values))
        if mean_value == 0:
            return 1.0
        return mean_value

    def _series_initial_value(self, series) -> float:
        if series.empty:
            return 1.0
        initial_value = float(series.iloc[0])
        if not np.isfinite(initial_value) or initial_value == 0:
            return 1.0
        return initial_value

    def _series_initial_mean(self, series, sample_count: int = 3) -> float:
        if series.empty:
            return 1.0
        values = series.head(sample_count).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 1.0
        mean_value = float(np.mean(values))
        if mean_value == 0:
            return 1.0
        return mean_value

    def _safe_float(self, value: str | None) -> float:
        if value is None:
            return 0.0
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return 0.0

    def _set_axis_margin(
        self,
        axis,
        column: str,
        experimental_values: np.ndarray,
        model_values: np.ndarray,
        form_values: dict[str, str],
        result: ValidationResult,
        is_per_unit: bool,
        target_initial_pu: float,
    ) -> None:
        combined = np.concatenate([experimental_values, model_values])
        combined = combined[np.isfinite(combined)]
        if combined.size == 0:
            return

        min_value = float(np.min(combined))
        max_value = float(np.max(combined))
        amplitude = max_value - min_value
        if is_per_unit:
            if amplitude == 0:
                padding = max(abs(target_initial_pu) * 0.15, 0.1)
            else:
                padding = max(amplitude * 0.15, abs(target_initial_pu) * 0.15, 0.1)
        else:
            padding = max(amplitude * 0.1, self._axis_padding_floor(column, form_values, result))
            if amplitude == 0:
                padding = max(padding, abs(max_value) * 0.1, 0.1)
        axis.set_ylim(min_value - padding, max_value + padding)

    def _set_time_axis_limits(
        self,
        axis,
        experimental_time: np.ndarray,
        model_time: np.ndarray,
    ) -> None:
        combined = np.concatenate([experimental_time, model_time])
        combined = combined[np.isfinite(combined)]
        if combined.size == 0:
            return
        axis.set_xlim(float(np.min(combined)), float(np.max(combined)))
        axis.margins(x=0)

    def _axis_padding_floor(
        self,
        column: str,
        form_values: dict[str, str],
        result: ValidationResult,
    ) -> float:
        nominal_power = self._safe_float(form_values.get("nominal_power_w"))
        nominal_voltage = self._safe_float(form_values.get("nominal_voltage_v"))

        if column == "frequency":
            return 5.0
        if column.endswith("_unbalance"):
            return 2.0
        if column == "voltage" or column.startswith("voltage_"):
            if nominal_voltage > 0:
                return nominal_voltage * 0.14
            return max(abs(self._initial_dataset_value(result.scalar_experimental, column)) * 0.14, 1.0)
        if column == "current" or column.startswith("current_"):
            if nominal_power > 0 and nominal_voltage > 0:
                return (nominal_power / nominal_voltage) * 0.15
            base_value = np.mean(
                [
                    self._initial_dataset_value(result.scalar_experimental, column),
                    self._initial_dataset_value(result.scalar_model, column),
                ]
            )
            return max(abs(base_value) * 0.15, 0.2)
        if column in ("active_power", "reactive_power") or column.startswith("active_power_") or column.startswith("reactive_power_"):
            if nominal_power > 0:
                divisor = 3.0 if "_" in column and column not in ("active_power", "reactive_power") else 1.0
                return (nominal_power / divisor) * 0.15
            base_value = np.mean(
                [
                    self._initial_dataset_value(result.scalar_experimental, column),
                    self._initial_dataset_value(result.scalar_model, column),
                ]
            )
            return max(abs(base_value) * 0.15, 0.1)
        return 0.1

    def _confirm_synchronization(self) -> None:
        if self.last_result is None or self.current_test_key is None:
            messagebox.showerror(
                "No data",
                "It was not possible to open the next step without synchronized data.",
            )
            return

        self.current_analysis_result = self._build_active_validation_result(self.last_result)
        self.current_window_segments = build_test_windows(
            test_key=self.current_test_key,
            form_values=self._get_current_form_values(),
            validation_result=self.current_analysis_result,
            tolerance_percent=self._get_window_tolerance_percent(),
            min_transition_percent=self._get_min_transition_percent(),
        )
        self._build_window_result_screen(self.current_analysis_result, self.current_window_segments)

    def _build_window_result_screen(
        self,
        result: ValidationResult,
        window_segments: dict[str, list[WindowSegment]],
    ) -> None:
        self._clear_screen()

        _, container = self._create_scrollable_screen(padx=18, pady=18)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 10))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=lambda: self._build_validation_result_screen(self.last_result or result),
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Test windows",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            container,
            text=(
                "The detected windows are highlighted in the plots below to guide the validation by segment. "
                "If needed, drag the dashed vertical lines to manually adjust the boundaries."
            ),
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1100,
        )
        subtitle.pack(anchor="w", pady=(4, 10))

        summary_lines = []
        for column, segments in window_segments.items():
            summary_lines.append(
                f"{SCALAR_DISPLAY_LABELS.get(column, column)}: "
                + " | ".join(
                    f"{window.name} ({window.start_s:.3f}s to {window.end_s:.3f}s)"
                    for window in segments
                )
            )
        summary_text = "\n".join(summary_lines)
        summary_label = tk.Label(
            container,
            text=summary_text,
            font=("Segoe UI", 9),
            bg="#f4f6fb",
            fg="#64748b",
            justify="left",
            wraplength=1120,
        )
        summary_label.pack(anchor="w", pady=(0, 12))

        chart_frame = tk.Frame(container, bg="#f4f6fb")
        chart_frame.pack(fill="both", expand=True)

        figure = self._build_window_figure(result, window_segments)
        self.current_figure = figure
        self.current_figure_canvas = FigureCanvasTkAgg(figure, master=chart_frame)
        self.current_figure_canvas.draw()
        self.current_figure_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._enable_window_boundary_drag(self.current_figure_canvas)

        actions_frame = tk.Frame(container, bg="#f4f6fb")
        actions_frame.pack(fill="x", pady=(12, 0))

        info = tk.Label(
            actions_frame,
            text="Check the highlighted windows. On the next screen you will be able to choose which comparative metrics to calculate.",
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#64748b",
            anchor="w",
        )
        info.pack(side="left")

        continue_button = tk.Button(
            actions_frame,
            text="Continue",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._build_metric_selection_screen,
        )
        continue_button.pack(side="right")

    def _build_window_figure(
        self,
        result: ValidationResult,
        window_segments: dict[str, list[WindowSegment]],
    ) -> Figure:
        form_values = self._get_current_form_values()
        pu_targets = self._build_target_initial_pu_map(form_values)
        selected_mode = self._selected_sync_analysis_mode()
        available_columns = [
            column
            for column in SCALAR_COLUMN_ORDER
            if column in result.scalar_experimental.columns and column in result.scalar_model.columns
        ]
        figure = Figure(figsize=(15.5, max(5.5, len(available_columns) * 2.0)), dpi=100)
        figure.patch.set_facecolor("#f4f6fb")
        self.window_boundary_lines = []

        window_colors = [
            "#dbeafe",
            "#fde68a",
            "#dcfce7",
            "#fecaca",
            "#e9d5ff",
        ]

        for plot_index, column in enumerate(available_columns, start=1):
            axis = figure.add_subplot(len(available_columns), 1, plot_index)
            exp_values = result.scalar_experimental[column].to_numpy(dtype=float)
            model_values = result.scalar_model[column].to_numpy(dtype=float)
            exp_plot_time, exp_plot_values = self._build_smooth_plot_series(
                result.scalar_experimental["time"].to_numpy(dtype=float),
                exp_values,
            )
            model_plot_time, model_plot_values = self._build_smooth_plot_series(
                result.scalar_model["time"].to_numpy(dtype=float),
                model_values,
            )

            column_windows = window_segments.get(column, [])
            for window_index, window in enumerate(column_windows):
                axis.axvspan(
                    window.start_s,
                    window.end_s,
                    color=window_colors[window_index % len(window_colors)],
                    alpha=0.28,
                    linewidth=0,
                )
            for boundary_index in range(1, len(column_windows)):
                boundary_time = column_windows[boundary_index].start_s
                boundary_line = axis.axvline(
                    boundary_time,
                    color="#334155",
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.75,
                    picker=6,
                )
                boundary_line._window_column = column
                boundary_line._window_boundary_index = boundary_index
                self.window_boundary_lines.append(boundary_line)

            axis.plot(
                exp_plot_time,
                exp_plot_values,
                label="Experimental",
                color="#1d4ed8",
                linewidth=1.5,
            )
            axis.plot(
                model_plot_time,
                model_plot_values,
                label="Model",
                color="#dc2626",
                linewidth=1.5,
                alpha=0.85,
            )
            axis.set_title(
                (
                    SCALAR_DISPLAY_LABELS[column]
                    if selected_mode == "real"
                    else (
                        f"{SCALAR_DISPLAY_LABELS[column]} in PU (experimental base)"
                        if selected_mode == "per_unit_reference"
                        else f"{SCALAR_DISPLAY_LABELS[column]} in PU"
                    )
                ),
                loc="left",
                fontsize=11,
                fontweight="bold",
            )
            if plot_index == len(available_columns):
                axis.set_xlabel("Time (s)")
            axis.grid(alpha=0.25)
            axis.legend(loc="upper right")
            self._set_time_axis_limits(
                axis,
                result.scalar_experimental["time"].to_numpy(dtype=float),
                result.scalar_model["time"].to_numpy(dtype=float),
            )
            self._set_axis_margin(
                axis,
                column,
                exp_values,
                model_values,
                form_values,
                result,
                is_per_unit=selected_mode != "real",
                target_initial_pu=pu_targets.get(column, 1.0),
            )

            ymin, ymax = axis.get_ylim()
            text_y = ymax - (ymax - ymin) * 0.08
            for window_index, window in enumerate(column_windows):
                middle = (window.start_s + window.end_s) / 2.0
                axis.text(
                    middle,
                    text_y,
                    window.name,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#334155",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": window_colors[window_index % len(window_colors)],
                        "edgecolor": "none",
                        "alpha": 0.85,
                    },
                )

        figure.tight_layout(pad=0.9, h_pad=1.0)
        return figure

    def _enable_window_boundary_drag(self, figure_canvas: FigureCanvasTkAgg) -> None:
        canvas = figure_canvas

        def on_pick(event) -> None:
            artist = getattr(event, "artist", None)
            if not hasattr(artist, "_window_boundary_index"):
                return
            self.window_drag_state = {
                "column": str(artist._window_column),
                "boundary_index": int(artist._window_boundary_index),
            }

        def on_motion(event) -> None:
            if self.window_drag_state is None or event.xdata is None:
                return
            column = str(self.window_drag_state["column"])
            boundary_index = int(self.window_drag_state["boundary_index"])
            new_time = self._clamp_window_boundary_time(column, boundary_index, float(event.xdata))
            for line in self.window_boundary_lines:
                if (
                    int(getattr(line, "_window_boundary_index", -1)) == boundary_index
                    and str(getattr(line, "_window_column", "")) == column
                ):
                    line.set_xdata([new_time, new_time])
            canvas.draw_idle()

        def on_release(event) -> None:
            if self.window_drag_state is None:
                return
            column = str(self.window_drag_state["column"])
            boundary_index = int(self.window_drag_state["boundary_index"])
            if event.xdata is not None:
                new_time = self._clamp_window_boundary_time(column, boundary_index, float(event.xdata))
                self._set_window_boundary_time(column, boundary_index, new_time)
            self.window_drag_state = None
            active_result = self._current_result_for_analysis()
            if active_result is not None:
                self._build_window_result_screen(active_result, self.current_window_segments)

        canvas.mpl_connect("pick_event", on_pick)
        canvas.mpl_connect("motion_notify_event", on_motion)
        canvas.mpl_connect("button_release_event", on_release)

    def _clamp_window_boundary_time(self, column: str, boundary_index: int, proposed_time: float) -> float:
        min_gap = 1e-6
        segments = self.current_window_segments.get(column, [])
        if boundary_index <= 0 or boundary_index >= len(segments):
            return proposed_time
        lower = float(segments[boundary_index - 1].start_s) + min_gap
        upper = float(segments[boundary_index].end_s) - min_gap
        if lower >= upper:
            return lower
        return min(max(proposed_time, lower), upper)

    def _set_window_boundary_time(self, column: str, boundary_index: int, new_time: float) -> None:
        updated_segments: dict[str, list[WindowSegment]] = {}
        for current_column, segments in self.current_window_segments.items():
            if current_column != column or boundary_index <= 0 or boundary_index >= len(segments):
                updated_segments[current_column] = segments
                continue
            column_segments = list(segments)
            previous_segment = column_segments[boundary_index - 1]
            next_segment = column_segments[boundary_index]
            column_segments[boundary_index - 1] = WindowSegment(
                previous_segment.name,
                previous_segment.start_s,
                new_time,
            )
            column_segments[boundary_index] = WindowSegment(
                next_segment.name,
                new_time,
                next_segment.end_s,
            )
            updated_segments[current_column] = column_segments
        self.current_window_segments = updated_segments

    def _build_metric_selection_screen(self) -> None:
        self._clear_screen()
        self.metric_threshold_entries = {}

        _, container = self._create_scrollable_screen(padx=20, pady=20)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 10))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=lambda: self._build_window_result_screen(
                self._current_result_for_analysis() or self.last_result,
                self.current_window_segments,
            ),
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Comparative Metric Selection",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            container,
            text="Select the metrics you want to calculate. When a metric is selected, the threshold fields for Good and Acceptable will be enabled beside it. Outside those ranges, the result will be considered poor.",
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1100,
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        defaults_button = tk.Button(
            container,
            text="Import default values",
            font=("Segoe UI", 10, "bold"),
            bg="#0f766e",
            fg="white",
            activebackground="#115e59",
            activeforeground="white",
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
            command=self._import_default_metric_limits,
        )
        defaults_button.pack(anchor="e", pady=(0, 12))

        scoring_card = tk.Frame(container, bg="#eef4fb", bd=1, relief="solid", padx=18, pady=18)
        scoring_card.pack(fill="x", pady=(0, 14))

        scoring_title = tk.Label(
            scoring_card,
            text="Scoring configuration",
            font=("Segoe UI", 13, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        scoring_title.pack(anchor="w")

        scoring_note = tk.Label(
            scoring_card,
            text=(
                "Scores are calculated from 0 to 100 for each category and quantity. "
                "You can choose the scoring method, the window aggregation mode, and the weight of each metric."
            ),
            font=("Segoe UI", 10),
            bg="#eef4fb",
            fg="#49657f",
            justify="left",
            wraplength=1040,
        )
        scoring_note.pack(anchor="w", pady=(4, 12))

        scoring_mode_row = tk.Frame(scoring_card, bg="#eef4fb")
        scoring_mode_row.pack(fill="x", pady=(0, 10))

        scoring_mode_label = tk.Label(
            scoring_mode_row,
            text="Scoring mode:",
            font=("Segoe UI", 10, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        scoring_mode_label.pack(side="left")

        scoring_mode_combo = ttk.Combobox(
            scoring_mode_row,
            textvariable=self.metric_special_options["score_mode"],
            state="readonly",
            width=36,
            values=[label for _key, label in self.SCORE_MODE_OPTIONS],
        )
        scoring_mode_combo.pack(side="left", padx=(10, 0))
        selected_score_mode = str(self.metric_special_options["score_mode"].get()).strip()
        score_mode_label = next(
            (label for key, label in self.SCORE_MODE_OPTIONS if key == selected_score_mode),
            self.SCORE_MODE_OPTIONS[0][1],
        )
        scoring_mode_combo.set(score_mode_label)

        aggregation_row = tk.Frame(scoring_card, bg="#eef4fb")
        aggregation_row.pack(fill="x")

        aggregation_label = tk.Label(
            aggregation_row,
            text="Window aggregation:",
            font=("Segoe UI", 10, "bold"),
            bg="#eef4fb",
            fg="#17324d",
        )
        aggregation_label.pack(side="left")

        aggregation_combo = ttk.Combobox(
            aggregation_row,
            textvariable=self.metric_special_options["window_aggregation_mode"],
            state="readonly",
            width=36,
            values=[label for _key, label in self.WINDOW_AGGREGATION_OPTIONS],
        )
        aggregation_combo.pack(side="left", padx=(10, 0))
        selected_aggregation = str(self.metric_special_options["window_aggregation_mode"].get()).strip()
        aggregation_mode_label = next(
            (label for key, label in self.WINDOW_AGGREGATION_OPTIONS if key == selected_aggregation),
            self.WINDOW_AGGREGATION_OPTIONS[0][1],
        )
        aggregation_combo.set(aggregation_mode_label)

        for group in self.METRIC_GROUP_DEFINITIONS:
            is_unavailable = False
            unavailable_message = ""

            if group.get("signal_type") == "sinusoidal" and self.current_signal_type != "sinusoidal":
                is_unavailable = True
                unavailable_message = (
                    "These metrics are unavailable because the data were imported "
                    "as scalar signals rather than sinusoidal waveforms."
                )

            allowed_test_keys = group.get("test_keys")
            if (
                not is_unavailable
                and allowed_test_keys is not None
                and self.current_test_key not in allowed_test_keys
            ):
                is_unavailable = True
                unavailable_message = (
                    "These metrics are unavailable because the selected test "
                    "does not have a transient regime for this step."
                )

            if is_unavailable:
                info_card = tk.Frame(container, bg="#eef4fb", bd=1, relief="solid", padx=18, pady=18)
                info_card.pack(fill="x", pady=(0, 14))
                group_title = tk.Label(
                    info_card,
                    text=group["title"],
                    font=("Segoe UI", 13, "bold"),
                    bg="#eef4fb",
                    fg="#17324d",
                )
                group_title.pack(anchor="w")
                group_note = tk.Label(
                    info_card,
                    text=unavailable_message,
                    font=("Segoe UI", 10),
                    bg="#eef4fb",
                    fg="#49657f",
                    justify="left",
                    wraplength=1040,
                )
                group_note.pack(anchor="w", pady=(6, 0))
                continue

            self._build_metric_group_card(container, group)

        footer = tk.Frame(container, bg="#f4f6fb")
        footer.pack(fill="x", pady=(8, 0))

        footer_text = tk.Label(
            footer,
            text="Select the desired metrics, adjust or import the classification thresholds, and click Continue to generate the comparative tables.",
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#64748b",
            anchor="w",
        )
        footer_text.pack(side="left")

        continue_button = tk.Button(
            footer,
            text="Continue",
            font=("Segoe UI", 11, "bold"),
            bg="#1d4ed8",
            fg="white",
            activebackground="#1e40af",
            activeforeground="white",
            bd=0,
            padx=18,
            pady=12,
            cursor="hand2",
            command=self._build_metric_results_screen,
        )
        continue_button.pack(side="right")

    def _build_metric_group_card(self, parent: tk.Frame, group_definition: dict[str, object]) -> None:
        card = tk.Frame(parent, bg="white", bd=1, relief="solid", padx=18, pady=18)
        card.pack(fill="x", pady=(0, 14))

        header = tk.Frame(card, bg="white")
        header.pack(fill="x")

        title = tk.Label(
            header,
            text=str(group_definition["title"]),
            font=("Segoe UI", 14, "bold"),
            bg="white",
            fg="#17324d",
        )
        title.pack(side="left", anchor="w")

        select_all_button = tk.Button(
            header,
            text="Select all",
            font=("Segoe UI", 9, "bold"),
            bg="#dbeafe",
            fg="#1e3a8a",
            activebackground="#bfdbfe",
            activeforeground="#1e3a8a",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            command=lambda group_id=str(group_definition["id"]): self._select_all_metrics_in_group(group_id),
        )
        select_all_button.pack(side="right")

        subtitle = tk.Label(
            card,
            text=str(group_definition["subtitle"]),
            font=("Segoe UI", 10),
            bg="white",
            fg="#49657f",
            justify="left",
            wraplength=1040,
        )
        subtitle.pack(anchor="w", pady=(4, 12))

        if str(group_definition["id"]) == "waveform":
            extra_option_frame = tk.Frame(card, bg="#f8fbff", bd=1, relief="solid", padx=14, pady=12)
            extra_option_frame.pack(fill="x", pady=(0, 12))

            extra_option = tk.Checkbutton(
                extra_option_frame,
                text="Additionally calculate metrics with phase delay adjustment (to correct cases where the model current has phase delay)",
                variable=self.metric_special_options["waveform_adjust_phase_error"],
                bg="#f8fbff",
                fg="#17324d",
                activebackground="#f8fbff",
                activeforeground="#17324d",
                font=("Segoe UI", 10),
                justify="left",
                wraplength=980,
                anchor="w",
            )
            extra_option.pack(anchor="w")

        if str(group_definition["id"]) == "transient_scalar":
            extra_option_frame = tk.Frame(card, bg="#f8fbff", bd=1, relief="solid", padx=14, pady=12)
            extra_option_frame.pack(fill="x", pady=(0, 12))

            extra_option = tk.Checkbutton(
                extra_option_frame,
                text="Additionally calculate metrics with delay-time adjustment (to correct cases where the model is only delayed)",
                variable=self.metric_special_options["transient_adjust_reaction_time"],
                bg="#f8fbff",
                fg="#17324d",
                activebackground="#f8fbff",
                activeforeground="#17324d",
                font=("Segoe UI", 10),
                justify="left",
                wraplength=980,
                anchor="w",
            )
            extra_option.pack(anchor="w")

            normalization_row = tk.Frame(extra_option_frame, bg="#f8fbff")
            normalization_row.pack(fill="x", pady=(10, 0))

            normalization_label = tk.Label(
                normalization_row,
                text="Normalization for normalized mean error and NRMSE in significant transients:",
                font=("Segoe UI", 10, "bold"),
                bg="#f8fbff",
                fg="#17324d",
            )
            normalization_label.pack(side="left")

            normalization_combo = ttk.Combobox(
                normalization_row,
                textvariable=self.metric_special_options["transient_normalization_mode"],
                state="readonly",
                width=64,
                values=[mode_label for _mode_key, mode_label in self.TRANSIENT_NORMALIZATION_MODES],
            )
            normalization_combo.pack(side="left", padx=(10, 0))
            selected_mode_key = self._selected_transient_normalization_mode()
            selected_mode_label = next(
                (
                    mode_label
                    for mode_key, mode_label in self.TRANSIENT_NORMALIZATION_MODES
                    if mode_key == selected_mode_key
                ),
                self.TRANSIENT_NORMALIZATION_MODES[1][1],
            )
            normalization_combo.set(selected_mode_label)

            normalization_help = tk.Label(
                extra_option_frame,
                text=(
                    "In nearly stationary segments, normalization continues to be performed "
                    "as steady-state."
                ),
                font=("Segoe UI", 9),
                bg="#f8fbff",
                fg="#64748b",
                justify="left",
                wraplength=980,
            )
            normalization_help.pack(anchor="w", pady=(8, 0))

        for section in group_definition["sections"]:
            section_label = tk.Label(
                card,
                text=str(section["title"]),
                font=("Segoe UI", 11, "bold"),
                bg="white",
                fg="#17324d",
            )
            section_label.pack(anchor="w", pady=(0, 8))

            for metric_id, metric_label in section["metrics"]:
                self._build_metric_row(card, str(group_definition["id"]), metric_id, metric_label)

    def _build_metric_row(
        self,
        parent: tk.Frame,
        group_id: str,
        metric_id: str,
        metric_label: str,
    ) -> None:
        metric_state = self.metric_selection_state[group_id][metric_id]
        selected_var = metric_state["selected"]
        weight_var = metric_state["weight"]
        limit_vars = metric_state["limits"]
        unit_label_text = self.METRIC_UNIT_LABELS.get(metric_id, "%")
        is_not_implemented = metric_id in self.NOT_IMPLEMENTED_METRICS
        if is_not_implemented:
            selected_var.set(False)

        row = tk.Frame(parent, bg="white")
        row.pack(fill="x", pady=(0, 8))

        left_frame = tk.Frame(row, bg="white")
        left_frame.pack(side="left", fill="x", expand=True)

        right_frame = tk.Frame(row, bg="white")
        right_frame.pack(side="right", anchor="e")

        checkbox = tk.Checkbutton(
            left_frame,
            text=(
                f"{metric_label} (not implemented yet)"
                if is_not_implemented
                else metric_label
            ),
            variable=selected_var,
            bg="white",
            fg="#94a3b8" if is_not_implemented else "#17324d",
            activebackground="white",
            activeforeground="#17324d",
            font=("Segoe UI", 10),
            anchor="w",
            state="disabled" if is_not_implemented else "normal",
            command=lambda current_group=group_id, current_metric=metric_id: self._update_metric_threshold_state(
                current_group,
                current_metric,
            ),
        )
        checkbox.pack(side="left", anchor="w")

        self.metric_threshold_entries.setdefault(group_id, {})

        for limit_key, label_text in (("good", "Good"), ("acceptable", "Acceptable")):
            limit_frame = tk.Frame(right_frame, bg="white")
            limit_frame.pack(side="left", padx=(12, 0), anchor="e")

            limit_label = tk.Label(
                limit_frame,
                text=label_text,
                font=("Segoe UI", 9, "bold"),
                bg="white",
                fg="#49657f",
            )
            limit_label.pack(anchor="w")

            entry_row = tk.Frame(limit_frame, bg="white")
            entry_row.pack(anchor="w")

            entry = tk.Entry(
                entry_row,
                textvariable=limit_vars[limit_key],
                font=("Segoe UI", 10),
                relief="solid",
                bd=1,
                width=10,
                state="disabled",
                disabledbackground="#eef2f7",
                disabledforeground="#7c8da1",
            )
            entry.pack(side="left", anchor="w", ipady=4)

            unit_label = tk.Label(
                entry_row,
                text=unit_label_text,
                font=("Segoe UI", 9),
                bg="white",
                fg="#64748b",
                padx=6,
            )
            unit_label.pack(side="left", anchor="w")
            self.metric_threshold_entries[group_id][f"{metric_id}:{limit_key}"] = entry

        weight_frame = tk.Frame(right_frame, bg="white")
        weight_frame.pack(side="left", padx=(12, 0), anchor="e")

        weight_label = tk.Label(
            weight_frame,
            text="Weight",
            font=("Segoe UI", 9, "bold"),
            bg="white",
            fg="#49657f",
        )
        weight_label.pack(anchor="w")

        weight_entry_row = tk.Frame(weight_frame, bg="white")
        weight_entry_row.pack(anchor="w")

        weight_entry = tk.Entry(
            weight_entry_row,
            textvariable=weight_var,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            width=8,
            state="disabled",
            disabledbackground="#eef2f7",
            disabledforeground="#7c8da1",
        )
        weight_entry.pack(side="left", anchor="w", ipady=4)

        weight_unit = tk.Label(
            weight_entry_row,
            text="x",
            font=("Segoe UI", 9),
            bg="white",
            fg="#64748b",
            padx=6,
        )
        weight_unit.pack(side="left", anchor="w")
        self.metric_threshold_entries[group_id][f"{metric_id}:weight"] = weight_entry

        self._update_metric_threshold_state(group_id, metric_id)

    def _update_metric_threshold_state(self, group_id: str, metric_id: str) -> None:
        metric_state = self.metric_selection_state[group_id][metric_id]
        is_selected = bool(metric_state["selected"].get()) and metric_id not in self.NOT_IMPLEMENTED_METRICS
        target_state = "normal" if is_selected else "disabled"

        for limit_key in ("good", "acceptable", "weight"):
            entry = self.metric_threshold_entries.get(group_id, {}).get(f"{metric_id}:{limit_key}")
            if entry is None:
                continue
            entry.configure(state=target_state)

    def _select_all_metrics_in_group(self, group_id: str) -> None:
        for metric_id, metric_state in self.metric_selection_state.get(group_id, {}).items():
            if metric_id in self.NOT_IMPLEMENTED_METRICS:
                metric_state["selected"].set(False)
                self._update_metric_threshold_state(group_id, metric_id)
                continue
            metric_state["selected"].set(True)
            self._update_metric_threshold_state(group_id, metric_id)

    def _import_default_metric_limits(self) -> None:
        for group_id, group_metrics in self.metric_selection_state.items():
            for metric_id, metric_state in group_metrics.items():
                good_limit, acceptable_limit = self.DEFAULT_LIMITS.get(metric_id, (5, 10))
                metric_state["limits"]["good"].set(str(good_limit).replace(".", ","))
                metric_state["limits"]["acceptable"].set(str(acceptable_limit).replace(".", ","))
                metric_state["weight"].set("1")

    def _build_metric_results_screen(self) -> None:
        if self.last_result is None:
            messagebox.showerror(
                "No data",
                "It was not possible to calculate the metrics without validated signals.",
            )
            return

        self._clear_screen()
        _, container = self._create_scrollable_screen(padx=20, pady=20)

        top_bar = tk.Frame(container, bg="#f4f6fb")
        top_bar.pack(fill="x", pady=(0, 10))

        back_button = tk.Button(
            top_bar,
            text="Back",
            font=("Segoe UI", 10, "bold"),
            bg="#dbe4f0",
            fg="#17324d",
            activebackground="#c9d7e8",
            activeforeground="#17324d",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._build_metric_selection_screen,
        )
        back_button.pack(side="left")

        title = tk.Label(
            container,
            text="Comparative Metric Results",
            font=("Segoe UI", 22, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            container,
            text="The tables below show the selected metrics organized by group and by test window.",
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#49657f",
            justify="left",
            wraplength=1100,
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        test_description = self._build_test_description()
        group_results = self._calculate_selected_metric_results()
        score_sections = self._build_score_sections(group_results)
        has_any_group_output = False
        transient_payload = group_results.get("transient_scalar", {})
        if transient_payload.get("tables"):
            debug_button = tk.Button(
                top_bar,
                text="Check plots",
                font=("Segoe UI", 10, "bold"),
                bg="#0f766e",
                fg="white",
                activebackground="#115e59",
                activeforeground="white",
                bd=0,
                padx=16,
                pady=8,
                cursor="hand2",
                command=self._show_transient_debug_plots,
            )
            debug_button.pack(side="right")

        if score_sections:
            self._build_score_summary_card(
                container=container,
                test_description=test_description,
                score_sections=score_sections,
            )
            has_any_group_output = True

        for group in self.METRIC_GROUP_DEFINITIONS:
            group_id = str(group["id"])
            result_payload = group_results.get(group_id)
            if result_payload is None:
                continue
            if not result_payload.get("tables") and not result_payload.get("pending_metrics"):
                continue
            has_any_group_output = True
            self._build_metric_result_group_card(
                container=container,
                group_definition=group,
                test_description=test_description,
                result_payload=result_payload,
            )

        if not has_any_group_output:
            empty_label = tk.Label(
                container,
                text="No metric has been selected for calculation yet.",
                font=("Segoe UI", 10),
                bg="#f4f6fb",
                fg="#64748b",
            )
            empty_label.pack(anchor="w")

    def _selected_score_mode(self) -> str:
        selected_value = str(self.metric_special_options["score_mode"].get()).strip()
        valid_modes = {mode_key for mode_key, _label in self.SCORE_MODE_OPTIONS}
        if selected_value in valid_modes:
            return selected_value
        label_to_key = {label: key for key, label in self.SCORE_MODE_OPTIONS}
        return label_to_key.get(selected_value, self.SCORE_MODE_OPTIONS[0][0])

    def _selected_window_aggregation_mode(self) -> str:
        selected_value = str(self.metric_special_options["window_aggregation_mode"].get()).strip()
        valid_modes = {mode_key for mode_key, _label in self.WINDOW_AGGREGATION_OPTIONS}
        if selected_value in valid_modes:
            return selected_value
        label_to_key = {label: key for key, label in self.WINDOW_AGGREGATION_OPTIONS}
        return label_to_key.get(selected_value, self.WINDOW_AGGREGATION_OPTIONS[0][0])

    def _metric_limits(self, metric_id: str) -> tuple[float, float]:
        metric_state = None
        for group_metrics in self.metric_selection_state.values():
            if metric_id in group_metrics:
                metric_state = group_metrics[metric_id]
                break
        default_good, default_acceptable = self.DEFAULT_LIMITS.get(metric_id, (5.0, 10.0))
        if metric_state is None:
            return float(default_good), float(default_acceptable)

        good_text = str(metric_state["limits"]["good"].get()).strip()
        acceptable_text = str(metric_state["limits"]["acceptable"].get()).strip()
        good_limit = self._safe_float(good_text) if good_text else float(default_good)
        acceptable_limit = self._safe_float(acceptable_text) if acceptable_text else float(default_acceptable)

        if metric_id in self.HIGHER_IS_BETTER_METRICS:
            if good_limit < acceptable_limit:
                good_limit, acceptable_limit = acceptable_limit, good_limit
        else:
            if good_limit > acceptable_limit:
                good_limit, acceptable_limit = acceptable_limit, good_limit
        return float(good_limit), float(acceptable_limit)

    def _metric_weight(self, metric_id: str) -> float:
        for group_metrics in self.metric_selection_state.values():
            if metric_id in group_metrics:
                weight_value = self._safe_float(group_metrics[metric_id]["weight"].get())
                return weight_value if weight_value > 0 else 1.0
        return 1.0

    def _score_metric_value(self, metric_id: str, value: float) -> float:
        if not np.isfinite(value):
            return float("nan")

        good_limit, acceptable_limit = self._metric_limits(metric_id)
        higher_is_better = metric_id in self.HIGHER_IS_BETTER_METRICS
        score_mode = self._selected_score_mode()

        if score_mode == "threshold":
            if higher_is_better:
                if value >= good_limit:
                    return 100.0
                if value >= acceptable_limit:
                    return 70.0
                return 0.0
            if value <= good_limit:
                return 100.0
            if value <= acceptable_limit:
                return 70.0
            return 0.0

        eps = 1e-12
        if higher_is_better:
            perfect_value = 1.0
            if value >= perfect_value:
                return 100.0
            if value >= good_limit:
                span = max(perfect_value - good_limit, eps)
                return float(85.0 + 15.0 * (value - good_limit) / span)
            if value >= acceptable_limit:
                span = max(good_limit - acceptable_limit, eps)
                return float(60.0 + 25.0 * (value - acceptable_limit) / span)
            decline_span = max(abs(acceptable_limit), abs(good_limit - acceptable_limit), eps)
            return float(max(0.0, 60.0 - 60.0 * (acceptable_limit - value) / decline_span))

        perfect_value = 0.0
        if value <= perfect_value + eps:
            return 100.0
        if value <= good_limit:
            span = max(good_limit - perfect_value, eps)
            return float(100.0 - 15.0 * (value - perfect_value) / span)
        if value <= acceptable_limit:
            span = max(acceptable_limit - good_limit, eps)
            return float(85.0 - 25.0 * (value - good_limit) / span)
        decline_span = max(abs(acceptable_limit), abs(acceptable_limit - good_limit), eps)
        return float(max(0.0, 60.0 - 60.0 * (value - acceptable_limit) / decline_span))

    def _window_duration_for_quantity(self, quantity: str, window_name: str) -> float:
        for window in self.current_window_segments.get(quantity, []):
            if window.name == window_name:
                return max(float(window.end_s - window.start_s), 1e-9)
        return 1.0

    def _aggregate_window_scores(self, quantity: str, window_scores: list[tuple[str, float]]) -> float:
        valid_scores = [(window_name, score) for window_name, score in window_scores if np.isfinite(score)]
        if not valid_scores:
            return float("nan")

        aggregation_mode = self._selected_window_aggregation_mode()
        if aggregation_mode == "worst":
            return float(min(score for _window_name, score in valid_scores))
        if aggregation_mode == "duration_weighted":
            weights = np.asarray(
                [self._window_duration_for_quantity(quantity, window_name) for window_name, _score in valid_scores],
                dtype=float,
            )
            scores = np.asarray([score for _window_name, score in valid_scores], dtype=float)
            if np.sum(weights) <= 0:
                return float(np.mean(scores))
            return float(np.average(scores, weights=weights))
        return float(np.mean([score for _window_name, score in valid_scores]))

    def _score_quantity_label(self, quantity: str) -> str:
        if quantity in SCALAR_DISPLAY_LABELS:
            return SCALAR_DISPLAY_LABELS[quantity]
        if quantity in SINUSOIDAL_DISPLAY_LABELS:
            return SINUSOIDAL_DISPLAY_LABELS[quantity]
        return quantity.replace("_", " ").title()

    def _infer_quantity_from_score_table(self, table_title: str) -> str | None:
        normalized_title = table_title.strip().lower()
        for key, label in SINUSOIDAL_DISPLAY_LABELS.items():
            if normalized_title.startswith(f"{label.lower()} waveform"):
                return key
        reverse_scalar_labels = {
            label.lower(): key
            for key, label in SCALAR_DISPLAY_LABELS.items()
        }
        return reverse_scalar_labels.get(normalized_title)

    def _extract_score_records(
        self,
        group_results: dict[str, dict[str, object]],
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for result_payload in group_results.values():
            for table in result_payload.get("tables", []):
                table_title = str(table["title"])
                quantity = self._infer_quantity_from_score_table(table_title)
                if quantity is None:
                    continue
                normalized_table_title = table_title.lower()
                table_variant = (
                    "phase_adjusted"
                    if "phase-adjusted" in normalized_table_title
                    or "phase-delay-adjusted" in normalized_table_title
                    else "original"
                )
                column_headers = [str(column_name) for column_name in list(table["columns"])[1:]]
                for row_data in table["rows"]:
                    if not bool(row_data.get("colorize", True)):
                        continue
                    metric_id = str(row_data.get("metric_id", ""))
                    row_label = str(row_data.get("label", ""))
                    row_variant = "delay_adjusted" if "delay-adjusted" in row_label.lower() else table_variant
                    for window_name, value in zip(column_headers, row_data.get("values", [])):
                        numeric_value = self._extract_metric_numeric_value(value)
                        if not np.isfinite(numeric_value):
                            continue
                        records.append(
                            {
                                "variant": row_variant,
                                "quantity": quantity,
                                "metric_id": metric_id,
                                "window_name": window_name,
                                "value": float(numeric_value),
                            }
                        )
        return records

    def _score_value_color(self, score: float) -> tuple[str, str]:
        if not np.isfinite(score):
            return "white", "#334155"
        if score >= 85.0:
            return "#dcfce7", "#166534"
        if score >= 60.0:
            return "#fef3c7", "#92400e"
        return "#fee2e2", "#991b1b"

    def _build_score_sections(
        self,
        group_results: dict[str, dict[str, object]],
    ) -> list[dict[str, object]]:
        records = self._extract_score_records(group_results)
        if not records:
            return []

        variants_to_build = [("original", "Original scores")]
        if self.metric_special_options["waveform_adjust_phase_error"].get():
            variants_to_build.append(("phase_adjusted", "Phase-delay-corrected scores"))
        if self.metric_special_options["transient_adjust_reaction_time"].get():
            variants_to_build.append(("delay_adjusted", "Delay-adjusted scores"))

        sections: list[dict[str, object]] = []
        for variant_key, variant_title in variants_to_build:
            rows: list[dict[str, object]] = []
            available_quantities: set[str] = set()
            for category_definition in self.SCORE_CATEGORY_DEFINITIONS:
                if variant_key != "original" and variant_key not in tuple(category_definition.get("adjusted_variants", ())):
                    continue
                quantity_scores: dict[str, float] = {}
                for quantity in SCALAR_COLUMN_ORDER:
                    metric_scores: list[tuple[float, float]] = []
                    for metric_id in category_definition["metrics"]:
                        matching_records = [
                            record
                            for record in records
                            if record["quantity"] == quantity
                            and record["metric_id"] == metric_id
                            and record["variant"] == variant_key
                        ]
                        if not matching_records:
                            continue
                        window_scores = [
                            (str(record["window_name"]), self._score_metric_value(metric_id, float(record["value"])))
                            for record in matching_records
                        ]
                        aggregated_metric_score = self._aggregate_window_scores(quantity, window_scores)
                        if np.isfinite(aggregated_metric_score):
                            metric_scores.append((aggregated_metric_score, self._metric_weight(metric_id)))
                    if metric_scores:
                        weighted_sum = float(sum(score * weight for score, weight in metric_scores))
                        weight_sum = float(sum(weight for _score, weight in metric_scores))
                        if weight_sum > 0:
                            quantity_scores[quantity] = weighted_sum / weight_sum
                            available_quantities.add(quantity)
                if quantity_scores:
                    rows.append(
                        {
                            "category": str(category_definition["title"]),
                            "values": quantity_scores,
                        }
                    )
            if rows and available_quantities:
                ordered_quantities = [
                    quantity
                    for quantity in SCALAR_COLUMN_ORDER
                    if quantity in available_quantities
                ]
                sections.append(
                    {
                        "variant": variant_key,
                        "title": variant_title,
                        "quantities": ordered_quantities,
                        "rows": rows,
                    }
                )
        return sections

    def _score_category_titles_in_order(self) -> list[str]:
        return [str(category["title"]) for category in self.SCORE_CATEGORY_DEFINITIONS]

    def _build_score_lookup(
        self,
        score_sections: list[dict[str, object]],
    ) -> dict[str, dict[str, dict[str, float]]]:
        lookup: dict[str, dict[str, dict[str, float]]] = {}
        for section in score_sections:
            variant = str(section.get("variant", "original"))
            variant_map = lookup.setdefault(variant, {})
            for row in section.get("rows", []):
                category_name = str(row["category"])
                category_map = variant_map.setdefault(category_name, {})
                for quantity, score in dict(row.get("values", {})).items():
                    try:
                        numeric_score = float(score)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(numeric_score):
                        category_map[str(quantity)] = numeric_score
        return lookup

    def _build_radar_series(
        self,
        labels: list[str],
        values: list[float],
        color: str,
        name: str,
        *,
        fill_alpha: float = 0.14,
    ) -> dict[str, object]:
        return {
            "labels": labels,
            "values": values,
            "color": color,
            "name": name,
            "fill_alpha": fill_alpha,
        }

    def _plot_radar_chart(
        self,
        axis,
        labels: list[str],
        series_list: list[dict[str, object]],
        title: str,
    ) -> None:
        labels, series_list = self._normalize_radar_axes(labels, series_list)
        axis.set_theta_offset(np.pi / 2.0)
        axis.set_theta_direction(-1)
        if len(labels) == 0:
            axis.set_axis_off()
            axis.text(
                0.5,
                0.5,
                "Insufficient data",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#64748b",
            )
            axis.set_title(title, fontsize=10, fontweight="bold", pad=18, color="#17324d")
            return
        angle_values = np.linspace(0, 2.0 * np.pi, len(labels), endpoint=False)
        closed_angles = np.concatenate([angle_values, [angle_values[0]]])

        axis.set_xticks(angle_values)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylim(0, 100)
        axis.set_yticks([20, 40, 60, 80, 100])
        axis.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="#475569")
        axis.grid(color="#cbd5e1", alpha=0.8)
        axis.spines["polar"].set_color("#94a3b8")
        axis.spines["polar"].set_linewidth(1.0)
        axis.set_facecolor("#fbfdff")

        has_series = False
        for series in series_list:
            values = np.asarray(series["values"], dtype=float)
            if values.size != len(labels):
                continue
            if not np.isfinite(values).any():
                continue
            closed_values = np.concatenate([values, [values[0]]])
            axis.plot(
                closed_angles,
                closed_values,
                color=str(series["color"]),
                linewidth=2.0,
                marker="o",
                markersize=4.0,
                label=str(series["name"]),
            )
            axis.fill(
                closed_angles,
                closed_values,
                color=str(series["color"]),
                alpha=float(series.get("fill_alpha", 0.14)),
            )
            has_series = True

        axis.set_title(title, fontsize=10, fontweight="bold", pad=18, color="#17324d")
        if has_series:
            axis.legend(loc="upper right", bbox_to_anchor=(1.28, 1.18), fontsize=7, frameon=False)

    def _normalize_radar_axes(
        self,
        labels: list[str],
        series_list: list[dict[str, object]],
    ) -> tuple[list[str], list[dict[str, object]]]:
        if not labels or not series_list:
            return [], series_list

        normalized_series: list[dict[str, object]] = []
        for series in series_list:
            values = np.asarray(series.get("values", []), dtype=float)
            if values.size != len(labels):
                continue
            normalized_series.append(
                {
                    **series,
                    "values": values.tolist(),
                }
            )
        if not normalized_series:
            return [], []

        valid_indices: list[int] = []
        for index in range(len(labels)):
            axis_values = [float(series["values"][index]) for series in normalized_series]
            if any(np.isfinite(axis_value) for axis_value in axis_values):
                valid_indices.append(index)

        if not valid_indices:
            return [], []

        filtered_labels = [labels[index] for index in valid_indices]
        filtered_series: list[dict[str, object]] = []
        for series in normalized_series:
            filtered_series.append(
                {
                    **series,
                    "values": [float(series["values"][index]) for index in valid_indices],
                }
            )

        if len(filtered_labels) == 1:
            filtered_labels = filtered_labels * 4
            filtered_series = [
                {
                    **series,
                    "values": [float(series["values"][0])] * 4,
                }
                for series in filtered_series
            ]
        elif len(filtered_labels) == 2:
            expanded_labels = [
                filtered_labels[0],
                filtered_labels[1],
                filtered_labels[0],
                filtered_labels[1],
            ]
            expanded_series: list[dict[str, object]] = []
            for series in filtered_series:
                first_value = float(series["values"][0])
                second_value = float(series["values"][1])
                expanded_series.append(
                    {
                        **series,
                        "values": [first_value, second_value, first_value, second_value],
                    }
                )
            filtered_labels = expanded_labels
            filtered_series = expanded_series

        return filtered_labels, filtered_series

    def _build_score_radar_figure(
        self,
        charts: list[dict[str, object]],
        title: str,
    ) -> Figure | None:
        if not charts:
            return None

        chart_count = len(charts)
        columns = 2 if chart_count > 1 else 1
        rows = int(np.ceil(chart_count / columns))
        figure = Figure(figsize=(12.8, max(4.8, rows * 5.0)), dpi=100)
        figure.patch.set_facecolor("white")
        figure.suptitle(title, fontsize=13, fontweight="bold", color="#17324d", y=0.995)

        for chart_index, chart in enumerate(charts, start=1):
            axis = figure.add_subplot(rows, columns, chart_index, projection="polar")
            self._plot_radar_chart(
                axis,
                labels=list(chart["labels"]),
                series_list=list(chart["series"]),
                title=str(chart["title"]),
            )

        figure.tight_layout(pad=1.2, rect=[0, 0, 1, 0.97])
        return figure

    def _build_quantity_radar_charts(
        self,
        score_sections: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        lookup = self._build_score_lookup(score_sections)
        original_map = lookup.get("original", {})
        if not original_map:
            return []

        categories = [
            category_name
            for category_name in self._score_category_titles_in_order()
            if category_name in original_map
        ]
        quantities = []
        for category_name in categories:
            for quantity in original_map.get(category_name, {}):
                if quantity not in quantities:
                    quantities.append(quantity)

        charts: list[dict[str, object]] = []
        for quantity in quantities:
            original_values = [
                float(original_map.get(category_name, {}).get(quantity, float("nan")))
                for category_name in categories
            ]
            if not np.isfinite(np.asarray(original_values, dtype=float)).any():
                continue

            series = [
                self._build_radar_series(
                    labels=categories,
                    values=original_values,
                    color="#2563eb",
                    name="Original",
                    fill_alpha=0.10,
                )
            ]

            if self.metric_special_options["waveform_adjust_phase_error"].get():
                phase_map = lookup.get("phase_adjusted", {})
                if phase_map:
                    phase_values = [
                        float(phase_map.get(category_name, {}).get(quantity, original_map.get(category_name, {}).get(quantity, float("nan"))))
                        for category_name in categories
                    ]
                    if np.isfinite(np.asarray(phase_values, dtype=float)).any():
                        series.append(
                            self._build_radar_series(
                                labels=categories,
                                values=phase_values,
                                color="#16a34a",
                                name="Phase-delay-corrected (where applicable)",
                                fill_alpha=0.08,
                            )
                        )

            if self.metric_special_options["transient_adjust_reaction_time"].get():
                delay_map = lookup.get("delay_adjusted", {})
                if delay_map:
                    delay_values = [
                        float(delay_map.get(category_name, {}).get(quantity, original_map.get(category_name, {}).get(quantity, float("nan"))))
                        for category_name in categories
                    ]
                    if np.isfinite(np.asarray(delay_values, dtype=float)).any():
                        series.append(
                            self._build_radar_series(
                                labels=categories,
                                values=delay_values,
                                color="#dc2626",
                                name="Delay-adjusted (where applicable)",
                                fill_alpha=0.08,
                            )
                        )

            charts.append(
                {
                    "title": self._score_quantity_label(quantity),
                    "labels": categories,
                    "series": series,
                }
            )
        return charts

    def _build_category_radar_charts(
        self,
        score_sections: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        lookup = self._build_score_lookup(score_sections)
        original_map = lookup.get("original", {})
        if not original_map:
            return []

        all_quantities = list(SCALAR_COLUMN_ORDER)
        charts: list[dict[str, object]] = []
        for category_definition in self.SCORE_CATEGORY_DEFINITIONS:
            category_name = str(category_definition["title"])
            if category_name not in original_map:
                continue
            labels = [
                self._score_quantity_label(quantity)
                for quantity in all_quantities
                if quantity in original_map.get(category_name, {})
            ]
            quantity_keys = [
                quantity
                for quantity in all_quantities
                if quantity in original_map.get(category_name, {})
            ]
            if not quantity_keys:
                continue

            series = [
                self._build_radar_series(
                    labels=labels,
                    values=[float(original_map[category_name].get(quantity, float("nan"))) for quantity in quantity_keys],
                    color="#2563eb",
                    name="Original",
                    fill_alpha=0.10,
                )
            ]

            if (
                self.metric_special_options["waveform_adjust_phase_error"].get()
                and "phase_adjusted" in tuple(category_definition.get("adjusted_variants", ()))
            ):
                phase_map = lookup.get("phase_adjusted", {})
                if category_name in phase_map:
                    series.append(
                        self._build_radar_series(
                            labels=labels,
                            values=[
                                float(
                                    phase_map[category_name].get(
                                        quantity,
                                        original_map[category_name].get(quantity, float("nan")),
                                    )
                                )
                                for quantity in quantity_keys
                            ],
                            color="#16a34a",
                            name="Phase-delay-corrected",
                            fill_alpha=0.08,
                        )
                    )

            if (
                self.metric_special_options["transient_adjust_reaction_time"].get()
                and "delay_adjusted" in tuple(category_definition.get("adjusted_variants", ()))
            ):
                delay_map = lookup.get("delay_adjusted", {})
                if category_name in delay_map:
                    series.append(
                        self._build_radar_series(
                            labels=labels,
                            values=[
                                float(
                                    delay_map[category_name].get(
                                        quantity,
                                        original_map[category_name].get(quantity, float("nan")),
                                    )
                                )
                                for quantity in quantity_keys
                            ],
                            color="#dc2626",
                            name="Delay-adjusted",
                            fill_alpha=0.08,
                        )
                    )

            charts.append(
                {
                    "title": category_name,
                    "labels": labels,
                    "series": series,
                }
            )
        return charts

    def _build_score_summary_card(
        self,
        container: tk.Frame,
        test_description: str,
        score_sections: list[dict[str, object]],
    ) -> None:
        card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=18, pady=18)
        card.pack(fill="x", pady=(0, 14))

        title = tk.Label(
            card,
            text="Model scores (0 to 100)",
            font=("Segoe UI", 14, "bold"),
            bg="white",
            fg="#17324d",
        )
        title.pack(anchor="w")

        test_label = tk.Label(
            card,
            text=test_description,
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#49657f",
        )
        test_label.pack(anchor="w", pady=(4, 8))

        mode_label = next(
            (label for key, label in self.SCORE_MODE_OPTIONS if key == self._selected_score_mode()),
            self.SCORE_MODE_OPTIONS[0][1],
        )
        aggregation_label = next(
            (label for key, label in self.WINDOW_AGGREGATION_OPTIONS if key == self._selected_window_aggregation_mode()),
            self.WINDOW_AGGREGATION_OPTIONS[0][1],
        )
        summary = tk.Label(
            card,
            text=f"Scoring mode: {mode_label} | Window aggregation: {aggregation_label}",
            font=("Segoe UI", 9),
            bg="white",
            fg="#64748b",
        )
        summary.pack(anchor="w", pady=(0, 10))

        for section in score_sections:
            section_frame = tk.Frame(card, bg="white")
            section_frame.pack(fill="x", pady=(0, 12))

            section_title = tk.Label(
                section_frame,
                text=str(section["title"]),
                font=("Segoe UI", 11, "bold"),
                bg="white",
                fg="#17324d",
            )
            section_title.pack(anchor="w", pady=(0, 6))

            table_frame = tk.Frame(section_frame, bg="white")
            table_frame.pack(fill="x")

            columns = ["Category"] + [self._score_quantity_label(quantity) for quantity in section["quantities"]]
            for column_index, column_name in enumerate(columns):
                header = tk.Label(
                    table_frame,
                    text=column_name,
                    font=("Segoe UI", 9, "bold"),
                    bg="#e8eef7",
                    fg="#17324d",
                    bd=1,
                    relief="solid",
                    padx=8,
                    pady=6,
                )
                header.grid(row=0, column=column_index, sticky="nsew")
                table_frame.grid_columnconfigure(column_index, weight=1 if column_index > 0 else 2)

            for row_index, row_data in enumerate(section["rows"], start=1):
                row_values = [str(row_data["category"])]
                for quantity in section["quantities"]:
                    score_value = float(row_data["values"].get(quantity, float("nan")))
                    row_values.append("-" if not np.isfinite(score_value) else f"{score_value:.3f}")
                for column_index, cell_value in enumerate(row_values):
                    bg_color = "white"
                    fg_color = "#334155"
                    if column_index > 0:
                        score_value = float(row_data["values"].get(section["quantities"][column_index - 1], float("nan")))
                        bg_color, fg_color = self._score_value_color(score_value)
                    cell = tk.Label(
                        table_frame,
                        text=cell_value,
                        font=("Segoe UI", 9),
                        bg=bg_color,
                        fg=fg_color,
                        bd=1,
                        relief="solid",
                        padx=8,
                        pady=6,
                        justify="left" if column_index == 0 else "center",
                        anchor="w" if column_index == 0 else "center",
                    )
                    cell.grid(row=row_index, column=column_index, sticky="nsew")

        quantity_radar_charts = self._build_quantity_radar_charts(score_sections)
        quantity_radar_figure = self._build_score_radar_figure(
            quantity_radar_charts,
            "Radar charts by quantity",
        )
        if quantity_radar_figure is not None:
            quantity_label = tk.Label(
                card,
                text="Radar charts by quantity",
                font=("Segoe UI", 11, "bold"),
                bg="white",
                fg="#17324d",
            )
            quantity_label.pack(anchor="w", pady=(6, 6))

            quantity_figure_frame = tk.Frame(card, bg="white")
            quantity_figure_frame.pack(fill="both", expand=True, pady=(0, 12))
            quantity_canvas = FigureCanvasTkAgg(quantity_radar_figure, master=quantity_figure_frame)
            quantity_canvas.draw()
            quantity_canvas.get_tk_widget().pack(fill="both", expand=True)

        category_radar_charts = self._build_category_radar_charts(score_sections)
        category_radar_figure = self._build_score_radar_figure(
            category_radar_charts,
            "Radar charts by score category",
        )
        if category_radar_figure is not None:
            category_label = tk.Label(
                card,
                text="Radar charts by score category",
                font=("Segoe UI", 11, "bold"),
                bg="white",
                fg="#17324d",
            )
            category_label.pack(anchor="w", pady=(6, 6))

            category_figure_frame = tk.Frame(card, bg="white")
            category_figure_frame.pack(fill="both", expand=True)
            category_canvas = FigureCanvasTkAgg(category_radar_figure, master=category_figure_frame)
            category_canvas.draw()
            category_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_metric_result_group_card(
        self,
        container: tk.Frame,
        group_definition: dict[str, object],
        test_description: str,
        result_payload: dict[str, object],
    ) -> None:
        card = tk.Frame(container, bg="white", bd=1, relief="solid", padx=18, pady=18)
        card.pack(fill="x", pady=(0, 14))

        title = tk.Label(
            card,
            text=str(group_definition["title"]),
            font=("Segoe UI", 14, "bold"),
            bg="white",
            fg="#17324d",
        )
        title.pack(anchor="w")

        test_label = tk.Label(
            card,
            text=test_description,
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#49657f",
        )
        test_label.pack(anchor="w", pady=(4, 8))

        pending_metrics = result_payload.get("pending_metrics", [])
        if pending_metrics:
            pending_label = tk.Label(
                card,
                text="Pending implementation at this stage: " + ", ".join(pending_metrics),
                font=("Segoe UI", 9),
                bg="white",
                fg="#9a6700",
                justify="left",
                wraplength=1040,
            )
            pending_label.pack(anchor="w", pady=(0, 10))

        tables = result_payload.get("tables", [])
        if not tables:
            empty_label = tk.Label(
                card,
                text="No calculable metric was selected in this group.",
                font=("Segoe UI", 10),
                bg="white",
                fg="#64748b",
            )
            empty_label.pack(anchor="w")
            return

        for table in tables:
            self._build_metric_table(
                parent=card,
                title=str(table["title"]),
                columns=list(table["columns"]),
                rows=list(table["rows"]),
            )

    def _build_metric_table(
        self,
        parent: tk.Frame,
        title: str,
        columns: list[str],
        rows: list[dict[str, object]],
    ) -> None:
        section = tk.Frame(parent, bg="white")
        section.pack(fill="x", pady=(0, 12))

        section_title = tk.Label(
            section,
            text=title,
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#17324d",
        )
        section_title.pack(anchor="w", pady=(0, 6))

        table_frame = tk.Frame(section, bg="white")
        table_frame.pack(fill="x")

        for column_index, column_name in enumerate(columns):
            header = tk.Label(
                table_frame,
                text=column_name,
                font=("Segoe UI", 9, "bold"),
                bg="#e8eef7",
                fg="#17324d",
                bd=1,
                relief="solid",
                padx=8,
                pady=6,
            )
            sticky = "nsew"
            header.grid(row=0, column=column_index, sticky=sticky)
            table_frame.grid_columnconfigure(column_index, weight=1 if column_index > 0 else 2)

        for row_index, row_data in enumerate(rows, start=1):
            metric_id = str(row_data["metric_id"])
            display_metric_id = str(row_data.get("display_metric_id", metric_id))
            colorize_cells = bool(row_data.get("colorize", True))
            row_values = [str(row_data["label"])]
            row_values.extend(
                self._format_metric_display(display_metric_id, value)
                for value in row_data["values"]
            )
            for column_index, cell_value in enumerate(row_values):
                bg_color = "white"
                fg_color = "#334155"
                if column_index > 0 and colorize_cells:
                    metric_value = self._extract_metric_numeric_value(row_data["values"][column_index - 1])
                    bg_color, fg_color = self._metric_cell_colors(display_metric_id, metric_value)
                cell = tk.Label(
                    table_frame,
                    text=cell_value,
                    font=("Segoe UI", 9),
                    bg=bg_color,
                    fg=fg_color,
                    bd=1,
                    relief="solid",
                    padx=8,
                    pady=6,
                    justify="left" if column_index == 0 else "center",
                    anchor="w" if column_index == 0 else "center",
                )
                cell.grid(row=row_index, column=column_index, sticky="nsew")

    def _show_transient_debug_plots(self) -> None:
        if not self.last_transient_debug_data:
            self._calculate_selected_metric_results()
        debug_entries = [
            entry
            for column in SCALAR_COLUMN_ORDER
            for entry in self.last_transient_debug_data.get(column, [])
        ]
        if not debug_entries:
            messagebox.showinfo(
                "No debug data",
                "There are no computed transient data available for plot review.",
            )
            return
            messagebox.showinfo(
                "Sem dados de conferência",
                "Nao ha dados transitórios calculados para conferir os plots.",
            )
            return

        debug_entries.sort(key=lambda entry: (float(entry["window_start_s"]), SCALAR_COLUMN_ORDER.index(str(entry["column"]))))

        debug_window = tk.Toplevel(self.root)
        debug_window.title("Transient metric point review")
        debug_window.title("Transient metric point review")
        debug_window.geometry("1460x900")
        debug_window.configure(bg="#f4f6fb")
        debug_window.title("Transient metric point review")

        container = tk.Frame(debug_window, bg="#f4f6fb")
        container.pack(fill="both", expand=True, padx=16, pady=16)

        title = tk.Label(
            container,
            text="Review of characteristic transient points",
            font=("Segoe UI", 18, "bold"),
            bg="#f4f6fb",
            fg="#17324d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            container,
            text="Markers indicate where the 10%, 90%, overshoot, and settling points were found for the reference and model.",
            font=("Segoe UI", 10),
            bg="#f4f6fb",
            fg="#49657f",
        )
        subtitle.pack(anchor="w", pady=(4, 10))

        chart_frame = tk.Frame(container, bg="#f4f6fb")
        chart_frame.pack(fill="both", expand=True)

        plot_canvas = tk.Canvas(chart_frame, bg="#f4f6fb", highlightthickness=0)
        scrollbar = ttk.Scrollbar(chart_frame, orient="vertical", command=plot_canvas.yview)
        plot_canvas.configure(yscrollcommand=scrollbar.set)
        plot_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        plot_holder = tk.Frame(plot_canvas, bg="#f4f6fb")
        plot_canvas.create_window((0, 0), window=plot_holder, anchor="nw")
        plot_holder.bind(
            "<Configure>",
            lambda _event: plot_canvas.configure(scrollregion=plot_canvas.bbox("all")),
        )

        figure = Figure(figsize=(15, max(4.5, len(debug_entries) * 2.8)), dpi=100)
        figure.patch.set_facecolor("#f4f6fb")

        for plot_index, entry in enumerate(debug_entries, start=1):
            axis = figure.add_subplot(len(debug_entries), 1, plot_index)
            time_values = np.asarray(entry["time"], dtype=float)
            reference_values = np.asarray(entry["y_real"], dtype=float)
            model_values = np.asarray(entry["y_model"], dtype=float)
            reference_plot_time, reference_plot_values = self._build_smooth_plot_series(
                time_values,
                reference_values,
            )
            model_plot_time, model_plot_values = self._build_smooth_plot_series(
                time_values,
                model_values,
            )
            reference_features = dict(entry["reference_features"])
            model_features = dict(entry["model_features"])

            axis.plot(reference_plot_time, reference_plot_values, color="#1d4ed8", linewidth=1.6, label="Experimental")
            model_line, = axis.plot(model_plot_time, model_plot_values, color="#dc2626", linewidth=1.4, label="Model")
            model_line.set_label("Model")

            reference_messages = reference_features.get("metric_messages", {})
            model_messages = model_features.get("metric_messages", {})
            if not (isinstance(reference_messages, dict) and len(reference_messages) >= 5):
                axis.axhline(float(reference_features["threshold_10"]), color="#93c5fd", linestyle="--", linewidth=0.8)
                axis.axhline(float(reference_features["threshold_90"]), color="#60a5fa", linestyle="--", linewidth=0.8)
                if np.isfinite(float(reference_features.get("threshold_95", float("nan")))):
                    axis.axhline(float(reference_features["threshold_95"]), color="#2563eb", linestyle=":", linewidth=0.8)
                axis.axhspan(
                    float(reference_features["settling_lower"]),
                    float(reference_features["settling_upper"]),
                    color="#dbeafe",
                    alpha=0.25,
                )
            self._plot_transition_feature_markers(
                axis=axis,
                time_values=time_values,
                signal_values=reference_values,
                features=reference_features,
                color="#1d4ed8",
                prefix="Exp",
            )
            self._plot_transition_feature_markers(
                axis=axis,
                time_values=time_values,
                signal_values=model_values,
                features=model_features,
                color="#dc2626",
                prefix="Mod",
            )

            if isinstance(reference_messages, dict) and reference_messages:
                info_text = " | ".join(sorted(set(reference_messages.values())))
                axis.text(
                    0.01,
                    0.92,
                    str(info_text),
                    transform=axis.transAxes,
                    fontsize=9,
                    color="#7c2d12",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffedd5", "edgecolor": "#fdba74"},
                )
            elif isinstance(model_messages, dict) and model_messages:
                info_text = " | ".join(sorted(set(model_messages.values())))
                axis.text(
                    0.01,
                    0.92,
                    str(info_text),
                    transform=axis.transAxes,
                    fontsize=9,
                    color="#7c2d12",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffedd5", "edgecolor": "#fdba74"},
                )

            axis.set_title(
                f"{SCALAR_DISPLAY_LABELS.get(str(entry['column']), str(entry['column']))} - {entry['window_name']}",
                loc="left",
                fontsize=12,
                fontweight="bold",
            )
            transient_duration = max(
                float(entry["window_end_s"]) - float(entry["window_start_s"]),
                0.0,
            )
            axis.set_xlim(0.0, transient_duration if transient_duration > 0 else float(time_values[-1]))
            axis.grid(alpha=0.25)
            axis.set_ylabel("PU")
            if plot_index == len(debug_entries):
                axis.set_xlabel("Tempo a partir do inicio do transitório (s)")
            axis.legend(loc="upper right", fontsize=8, ncol=2)
            if plot_index == len(debug_entries):
                axis.set_xlabel("Time from transient start (s)")

        figure.tight_layout(pad=1.4)

        canvas = FigureCanvasTkAgg(figure, master=plot_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_transition_feature_markers(
        self,
        axis,
        time_values: np.ndarray,
        signal_values: np.ndarray,
        features: dict[str, object],
        color: str,
        prefix: str,
    ) -> None:
        marker_map = {
            "reaction_marker_time_s": ("o", "10%"),
            "rise_end_time_s": ("s", "90%"),
            "response_marker_time_s": ("P", "95%"),
            "settling_marker_time_s": ("D", "Ts"),
            "overshoot_time_s": ("^", "Peak"),
        }
        for feature_key, (marker_style, label_suffix) in marker_map.items():
            feature_time = features.get(feature_key)
            try:
                numeric_time = float(feature_time)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric_time):
                continue
            y_value = float(np.interp(numeric_time, time_values, signal_values))
            axis.scatter(
                [numeric_time],
                [y_value],
                s=42,
                marker=marker_style,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
            axis.annotate(
                f"{prefix} {label_suffix}",
                xy=(numeric_time, y_value),
                xytext=(4, 8),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    def _build_smooth_plot_series(
        self,
        time_values: np.ndarray,
        signal_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        time_values = np.asarray(time_values, dtype=float)
        signal_values = np.asarray(signal_values, dtype=float)
        if len(time_values) < 4 or len(signal_values) < 4 or len(time_values) != len(signal_values):
            return time_values, signal_values

        unique_mask = np.ones(len(time_values), dtype=bool)
        unique_mask[1:] = np.diff(time_values) > 0
        unique_time = time_values[unique_mask]
        unique_signal = signal_values[unique_mask]
        if len(unique_time) < 4:
            return time_values, signal_values

        dense_count = min(max(len(unique_time) * 8, len(unique_time)), 4000)
        dense_time = np.linspace(float(unique_time[0]), float(unique_time[-1]), dense_count)
        try:
            interpolator = PchipInterpolator(unique_time, unique_signal)
            dense_signal = interpolator(dense_time)
        except Exception:
            return time_values, signal_values
        return dense_time, np.asarray(dense_signal, dtype=float)

    def _calculate_selected_metric_results(self) -> dict[str, dict[str, object]]:
        results: dict[str, dict[str, object]] = {}
        if self.last_result is None:
            return results
        self.last_transient_debug_data = {}
        self.transient_delay_cache = {}

        steady_metric_labels = {
            "steady_mean_error": "Normalized mean error",
            "steady_nrmse": "NRMSE",
            "steady_residual_rms": "Normalized residual RMS",
            "steady_ssi": "SSI error",
            "steady_std_error": "Relative standard deviation error",
            "steady_peak_to_peak_error": "Relative peak-to-peak error",
        }
        transient_metric_labels = {
            "transient_mean_error": "Normalized mean error",
            "transient_nrmse": "NRMSE",
            "transient_overshoot_undershoot": "Signal difference: Overshoot/Undershoot",
            "transient_relative_overshoot_undershoot": "Relative signal difference (%): Overshoot/Undershoot",
            "transient_rise_fall_time": "Signal difference: Rise/Fall time",
            "transient_relative_rise_fall_time": "Relative signal difference (%): Rise/Fall time",
            "transient_settling_time": "Signal difference: Settling time",
            "transient_relative_settling_time": "Relative signal difference (%): Settling time",
            "transient_response_time": "Signal difference: Response time",
            "transient_relative_response_time": "Relative signal difference (%): Response time",
            "transient_reaction_time": "Signal difference: Reaction time",
            "transient_relative_reaction_time": "Relative signal difference (%): Reaction time",
            "transient_delay_time": "Delay time",
            "transient_pearson_r": "Pearson r",
            "transient_r2": "Coefficient of determination (R²)",
        }

        steady_selected = [
            metric_id
            for metric_id in self.metric_selection_state["steady_scalar"]
            if self.metric_selection_state["steady_scalar"][metric_id]["selected"].get()
        ]
        transient_selected = [
            metric_id
            for metric_id in self.metric_selection_state["transient_scalar"]
            if self.metric_selection_state["transient_scalar"][metric_id]["selected"].get()
        ]
        waveform_selected = [
            metric_id
            for metric_id in self.metric_selection_state["waveform"]
            if self.metric_selection_state["waveform"][metric_id]["selected"].get()
        ]

        waveform_metric_labels = {
            "waveform_mean_error": "Normalized mean error",
            "waveform_nrmse": "NRMSE",
            "fundamental_amplitude_error": "Relative fundamental amplitude error",
            "phase_error": "Phase delay",
            "thd_total_error": "Total THD error",
            "harmonic_even_amplitude_mean_error": "Mean even-harmonic amplitude error",
            "harmonic_odd_amplitude_mean_error": "Mean odd-harmonic amplitude error",
            "frequency_mean_error": "Mean frequency error",
            "tve": "TVE",
            "max_harmonic_error": "Maximum harmonic error",
        }
        waveform_calculable = [metric_id for metric_id in waveform_selected if metric_id in waveform_metric_labels]
        waveform_pending = [self._metric_label_for_id("waveform", metric_id) for metric_id in waveform_selected if metric_id not in waveform_metric_labels]
        results["waveform"] = {
            "pending_metrics": waveform_pending,
            "tables": self._build_waveform_metric_tables(
                metric_ids=waveform_calculable,
                metric_labels=waveform_metric_labels,
            ),
        }

        steady_calculable = [metric_id for metric_id in steady_selected if metric_id in steady_metric_labels]
        steady_pending = [self._metric_label_for_id("steady_scalar", metric_id) for metric_id in steady_selected if metric_id not in steady_metric_labels]
        results["steady_scalar"] = {
            "pending_metrics": steady_pending,
            "tables": self._build_scalar_metric_tables(
                metric_ids=steady_calculable,
                metric_labels=steady_metric_labels,
                window_kind="steady",
            ),
        }

        transient_calculable = [metric_id for metric_id in transient_selected if metric_id in transient_metric_labels]
        transient_pending = [
            (
                self._metric_label_for_id("transient_scalar", metric_id)
                + " (to be implemented in a future version)"
            )
            for metric_id in transient_selected
            if metric_id not in transient_metric_labels
        ]
        transient_tables = self._build_scalar_metric_tables(
            metric_ids=transient_calculable,
            metric_labels=transient_metric_labels,
            window_kind="transient",
            include_delay_adjusted=self.metric_special_options["transient_adjust_reaction_time"].get(),
        )

        results["transient_scalar"] = {
            "pending_metrics": transient_pending,
            "tables": transient_tables,
        }

        return results

    def _build_scalar_metric_tables(
        self,
        metric_ids: list[str],
        metric_labels: dict[str, str],
        window_kind: str,
        include_delay_adjusted: bool = False,
    ) -> list[dict[str, object]]:
        active_result = self._current_result_for_analysis()
        if active_result is None or not metric_ids:
            return []

        experimental_dataset = active_result.scalar_experimental
        model_dataset = active_result.scalar_model
        tables: list[dict[str, object]] = []
        for column in SCALAR_COLUMN_ORDER:
            if (
                column not in experimental_dataset.columns
                or column not in model_dataset.columns
            ):
                continue

            column_windows = self._get_windows_for_kind(column, window_kind)
            if not column_windows:
                continue

            rows: list[dict[str, object]] = []
            for metric_id in metric_ids:
                if window_kind == "transient" and metric_id in self.TRANSIENT_FEATURE_METRIC_MAP:
                    base_metric_id = self._base_transient_feature_metric_id(metric_id)
                    difference_values: list[object] = []
                    reference_values: list[object] = []
                    model_values: list[object] = []
                    for window in column_windows:
                        try:
                                metric_bundle = self._calculate_transient_metric_bundle_for_window(
                                experimental_dataset=experimental_dataset,
                                model_dataset=model_dataset,
                                column=column,
                                metric_id=metric_id,
                                window=window,
                                delay_adjusted=False,
                            )
                        except Exception as exc:
                            metric_bundle = {
                                "difference": {"message": f"Calculation failed: {exc}"},
                                "reference": {"message": f"Calculation failed: {exc}"},
                                "model": {"message": f"Calculation failed: {exc}"},
                            }
                        difference_values.append(metric_bundle["difference"])
                        reference_values.append(metric_bundle["reference"])
                        model_values.append(metric_bundle["model"])

                    if not self._should_suppress_metric_row(difference_values):
                        rows.append(
                            {
                                "metric_id": metric_id,
                                "display_metric_id": metric_id,
                                "label": metric_labels[metric_id],
                                "values": difference_values,
                                "colorize": True,
                            }
                        )
                        rows.append(
                            {
                                "metric_id": f"{metric_id}_reference",
                                "display_metric_id": base_metric_id,
                                "label": f"Reference: {self.TRANSIENT_FEATURE_METRIC_MAP[metric_id][1]}",
                                "values": reference_values,
                                "colorize": False,
                            }
                        )
                        rows.append(
                            {
                                "metric_id": f"{metric_id}_model",
                                "display_metric_id": base_metric_id,
                                "label": f"Model: {self.TRANSIENT_FEATURE_METRIC_MAP[metric_id][1]}",
                                "values": model_values,
                                "colorize": False,
                            }
                        )
                else:
                    row_values: list[object] = []
                    for window in column_windows:
                        try:
                                metric_value = self._calculate_metric_for_window(
                                experimental_dataset=experimental_dataset,
                                model_dataset=model_dataset,
                                column=column,
                                metric_id=metric_id,
                                window=window,
                                window_kind=window_kind,
                            )
                        except Exception as exc:
                            metric_value = {"message": f"Calculation failed: {exc}"}
                        row_values.append(metric_value)
                    rows.append(
                        {
                            "metric_id": metric_id,
                            "display_metric_id": metric_id,
                            "label": metric_labels[metric_id],
                            "values": row_values,
                            "colorize": True,
                        }
                    )

                if (
                    include_delay_adjusted
                    and window_kind == "transient"
                    and metric_id != "transient_delay_time"
                ):
                    if metric_id in self.TRANSIENT_FEATURE_METRIC_MAP:
                        base_metric_id = self._base_transient_feature_metric_id(metric_id)
                        adjusted_difference_values: list[object] = []
                        adjusted_reference_values: list[object] = []
                        adjusted_model_values: list[object] = []
                        for window in column_windows:
                            try:
                                metric_bundle = self._calculate_transient_metric_bundle_for_window(
                                    experimental_dataset=experimental_dataset,
                                    model_dataset=model_dataset,
                                    column=column,
                                    metric_id=metric_id,
                                    window=window,
                                    delay_adjusted=True,
                                )
                            except Exception as exc:
                                metric_bundle = {
                                    "difference": {"message": f"Calculation failed: {exc}"},
                                    "reference": {"message": f"Calculation failed: {exc}"},
                                    "model": {"message": f"Calculation failed: {exc}"},
                                }
                            adjusted_difference_values.append(metric_bundle["difference"])
                            adjusted_reference_values.append(metric_bundle["reference"])
                            adjusted_model_values.append(metric_bundle["model"])
                        if not self._should_suppress_metric_row(adjusted_difference_values):
                            rows.append(
                                {
                                    "metric_id": metric_id,
                                    "display_metric_id": metric_id,
                                    "label": f"{metric_labels[metric_id]} (delay-adjusted)",
                                    "values": adjusted_difference_values,
                                    "colorize": True,
                                }
                            )
                            rows.append(
                                {
                                    "metric_id": f"{metric_id}_reference_adjusted",
                                    "display_metric_id": base_metric_id,
                                    "label": f"Reference: {self.TRANSIENT_FEATURE_METRIC_MAP[metric_id][1]} (delay-adjusted)",
                                    "values": adjusted_reference_values,
                                    "colorize": False,
                                }
                            )
                            rows.append(
                                {
                                    "metric_id": f"{metric_id}_model_adjusted",
                                    "display_metric_id": base_metric_id,
                                    "label": f"Model: {self.TRANSIENT_FEATURE_METRIC_MAP[metric_id][1]} (delay-adjusted)",
                                    "values": adjusted_model_values,
                                    "colorize": False,
                                }
                            )
                    else:
                        adjusted_values: list[object] = []
                        for window in column_windows:
                            try:
                                metric_value = self._calculate_metric_for_window(
                                    experimental_dataset=experimental_dataset,
                                    model_dataset=model_dataset,
                                    column=column,
                                    metric_id=metric_id,
                                    window=window,
                                    window_kind=window_kind,
                                    delay_adjusted=True,
                                )
                            except Exception as exc:
                                metric_value = {"message": f"Calculation failed: {exc}"}
                            adjusted_values.append(metric_value)
                        rows.append(
                            {
                                "metric_id": metric_id,
                                "display_metric_id": metric_id,
                                "label": f"{metric_labels[metric_id]} (delay-adjusted)",
                                "values": adjusted_values,
                                "colorize": True,
                            }
                        )

            tables.append(
                {
                    "title": SCALAR_DISPLAY_LABELS[column],
                    "columns": ["Index"] + [window.name for window in column_windows],
                    "rows": rows,
                }
            )

        return tables

    def _build_waveform_metric_tables(
        self,
        metric_ids: list[str],
        metric_labels: dict[str, str],
    ) -> list[dict[str, object]]:
        if (
            self.last_result is None
            or self.last_result.waveform_experimental is None
            or self.last_result.waveform_model is None
            or not metric_ids
        ):
            return []

        tables: list[dict[str, object]] = []
        cycle_metrics = self._build_waveform_cycle_metrics(phase_adjust_current=False)
        adjusted_current_cycle_metrics = None
        if self.metric_special_options["waveform_adjust_phase_error"].get():
            adjusted_current_cycle_metrics = self._build_waveform_cycle_metrics(phase_adjust_current=True)

        signal_items = [
            (signal_key, SINUSOIDAL_DISPLAY_LABELS.get(signal_key, signal_key))
            for signal_key in SINUSOIDAL_DISPLAY_LABELS
            if signal_key in cycle_metrics
        ]

        for signal_key, signal_label in signal_items:
            signal_cycles = cycle_metrics.get(signal_key, [])
            if not signal_cycles:
                continue

            rows: list[dict[str, object]] = []
            for metric_id in metric_ids:
                metric_value = self._aggregate_waveform_metric(signal_cycles, metric_id)
                rows.append(
                    {
                        "metric_id": metric_id,
                        "label": metric_labels[metric_id],
                        "values": [metric_value],
                    }
                )

            tables.append(
                {
                    "title": f"{signal_label} waveform",
                    "columns": ["Index", "Average across complete non-transient cycles"],
                    "rows": rows,
                }
            )

            if (
                signal_key.startswith("current")
                and self.metric_special_options["waveform_adjust_phase_error"].get()
                and adjusted_current_cycle_metrics is not None
                and signal_key in adjusted_current_cycle_metrics
            ):
                adjusted_rows: list[dict[str, object]] = []
                for metric_id in metric_ids:
                    adjusted_rows.append(
                        {
                            "metric_id": metric_id,
                            "label": metric_labels[metric_id],
                            "values": [self._aggregate_waveform_metric(adjusted_current_cycle_metrics[signal_key], metric_id)],
                        }
                    )

                tables.append(
                    {
                        "title": f"{signal_label} waveform (phase-delay-adjusted)",
                        "columns": ["Index", "Average across complete non-transient cycles"],
                        "rows": adjusted_rows,
                    }
                )

        return tables

    def _build_waveform_cycle_metrics(
        self,
        phase_adjust_current: bool,
    ) -> dict[str, list[dict[str, object]]]:
        if (
            self.last_result is None
            or self.last_result.waveform_experimental is None
            or self.last_result.waveform_model is None
        ):
            return {"voltage": [], "current": []}

        exp_waveform = self.last_result.waveform_experimental
        model_waveform = self.last_result.waveform_model.copy()
        exp_reference_column = "voltage" if "voltage" in exp_waveform.columns else "voltage_a"
        model_reference_column = "voltage" if "voltage" in model_waveform.columns else "voltage_a"
        if exp_reference_column not in exp_waveform.columns or model_reference_column not in model_waveform.columns:
            return {"voltage": [], "current": []}

        exp_voltage_boundaries = detect_cycle_boundaries(
            exp_waveform["time"].to_numpy(dtype=float),
            exp_waveform[exp_reference_column].to_numpy(dtype=float),
        )
        model_voltage_boundaries = detect_cycle_boundaries(
            model_waveform["time"].to_numpy(dtype=float),
            model_waveform[model_reference_column].to_numpy(dtype=float),
        )
        cycle_count = min(len(exp_voltage_boundaries), len(model_voltage_boundaries)) - 1
        if cycle_count < 1:
            return {"voltage": [], "current": []}

        signal_keys = [
            signal_key
            for signal_key in SINUSOIDAL_DISPLAY_LABELS
            if signal_key in exp_waveform.columns and signal_key in model_waveform.columns
        ]
        metrics_by_signal = {signal_key: [] for signal_key in signal_keys}
        for cycle_index in range(cycle_count):
            exp_start = float(exp_voltage_boundaries[cycle_index]["time"])
            exp_end = float(exp_voltage_boundaries[cycle_index + 1]["time"])
            model_start = float(model_voltage_boundaries[cycle_index]["time"])
            model_end = float(model_voltage_boundaries[cycle_index + 1]["time"])
            exp_duration = exp_end - exp_start
            model_duration = model_end - model_start
            if exp_duration <= 0 or model_duration <= 0:
                continue

            cycle_allowed_by_signal = {
                signal_key: self._waveform_cycle_is_non_transient(
                    signal_key=signal_key,
                    exp_start=exp_start,
                    exp_end=exp_end,
                    model_start=model_start,
                    model_end=model_end,
                )
                for signal_key in signal_keys
            }
            if not any(cycle_allowed_by_signal.values()):
                continue

            exp_cycle_frequency = 1.0 / exp_duration
            model_cycle_frequency = 1.0 / model_duration

            for signal_key in signal_keys:
                if not cycle_allowed_by_signal.get(signal_key, False):
                    continue

                exp_cycle = self._extract_cycle_samples(
                    exp_waveform["time"].to_numpy(dtype=float),
                    exp_waveform[signal_key].to_numpy(dtype=float),
                    exp_start,
                    exp_end,
                )
                model_cycle = self._extract_cycle_samples(
                    model_waveform["time"].to_numpy(dtype=float),
                    model_waveform[signal_key].to_numpy(dtype=float),
                    model_start,
                    model_end,
                )
                if exp_cycle is None or model_cycle is None:
                    continue

                exp_harmonics = self._compute_harmonic_features(
                    exp_cycle["time"],
                    exp_cycle["signal"],
                    exp_cycle_frequency,
                    max_harmonic=50,
                )
                model_harmonics = self._compute_harmonic_features(
                    model_cycle["time"],
                    model_cycle["signal"],
                    model_cycle_frequency,
                    max_harmonic=50,
                )
                if exp_harmonics is None or model_harmonics is None:
                    continue

                model_cycle_signal = model_cycle["signal"]
                if phase_adjust_current and signal_key.startswith("current"):
                    model_harmonics = self._phase_adjust_harmonic_features(
                        exp_harmonics=exp_harmonics,
                        model_harmonics=model_harmonics,
                    )
                    model_cycle_signal = self._phase_adjust_model_signal(
                        exp_time=exp_cycle["time"],
                        exp_signal=exp_cycle["signal"],
                        model_time=model_cycle["time"],
                        model_signal=model_cycle["signal"],
                        base_frequency=exp_cycle_frequency,
                    )
                harmonic_errors = self._normalized_harmonic_errors(
                    exp_harmonics["rms"],
                    model_harmonics["rms"],
                    harmonics=range(2, 34),
                )
                metrics_by_signal[signal_key].append(
                    {
                        "waveform_mean_error": self._waveform_mean_error(
                            exp_time=exp_cycle["time"],
                            exp_signal=exp_cycle["signal"],
                            model_time=model_cycle["time"],
                            model_signal=model_cycle_signal,
                        ),
                        "waveform_nrmse": self._waveform_nrmse(
                            exp_time=exp_cycle["time"],
                            exp_signal=exp_cycle["signal"],
                            model_time=model_cycle["time"],
                            model_signal=model_cycle_signal,
                        ),
                        "fundamental_amplitude_error": self._relative_error(
                            exp_harmonics["rms"][1],
                            model_harmonics["rms"][1],
                        ) * 100.0,
                        "phase_error": self._phase_error_degrees(
                            exp_harmonics["phasor"][1],
                            model_harmonics["phasor"][1],
                        ),
                        "thd_total_error": abs(exp_harmonics["thd"] - model_harmonics["thd"]) * 100.0,
                        "harmonic_even_amplitude_mean_error": self._mean_harmonic_error_from_normalized(
                            harmonic_errors,
                            harmonics=range(2, 34, 2),
                        ),
                        "harmonic_odd_amplitude_mean_error": self._mean_harmonic_error_from_normalized(
                            harmonic_errors,
                            harmonics=range(3, 34, 2),
                        ),
                        "frequency_mean_error": self._relative_error(
                            exp_cycle_frequency,
                            model_cycle_frequency,
                        ) * 100.0,
                        "tve": self._tve_error(
                            exp_harmonics["phasor"][1],
                            model_harmonics["phasor"][1],
                        ) * 100.0,
                        "harmonic_errors": harmonic_errors,
                    }
                )

        return metrics_by_signal

    def _waveform_cycle_is_non_transient(
        self,
        signal_key: str,
        exp_start: float,
        exp_end: float,
        model_start: float,
        model_end: float,
    ) -> bool:
        windows = self.current_window_segments.get(signal_key, [])
        if not windows:
            return True

        cycle_start = min(float(exp_start), float(model_start))
        cycle_end = max(float(exp_end), float(model_end))
        if cycle_end <= cycle_start:
            return False

        tolerance_s = 1e-9
        for window in windows:
            if self._is_transient_window_name(window.name):
                continue
            if window.end_s <= window.start_s:
                continue
            if (
                cycle_start >= float(window.start_s) - tolerance_s
                and cycle_end <= float(window.end_s) + tolerance_s
            ):
                return True
        return False

    def _compute_harmonic_features(
        self,
        time_values: np.ndarray,
        signal_values: np.ndarray,
        base_frequency: float,
        max_harmonic: int,
    ) -> dict[str, object] | None:
        if len(time_values) < 8 or base_frequency <= 0:
            return None

        harmonic_rms: dict[int, float] = {}
        harmonic_phasor: dict[int, complex] = {}
        relative_time = time_values - float(time_values[0])
        duration = float(relative_time[-1])
        if duration <= 0:
            return None

        for harmonic in range(1, max_harmonic + 1):
            angular_frequency = 2.0 * np.pi * base_frequency * harmonic
            kernel = np.exp(-1j * angular_frequency * relative_time)
            harmonic_phasor[harmonic] = (
                np.sqrt(2.0) / duration
            ) * np.trapezoid(signal_values * kernel, relative_time)
            harmonic_rms[harmonic] = float(abs(harmonic_phasor[harmonic]))

        fundamental_rms = harmonic_rms.get(1, 0.0)
        if fundamental_rms <= 0:
            thd = float("nan")
        else:
            thd = float(
                np.sqrt(sum(harmonic_rms[h] ** 2 for h in range(2, max_harmonic + 1))) / fundamental_rms
            )

        return {
            "rms": harmonic_rms,
            "phasor": harmonic_phasor,
            "thd": thd,
        }

    def _phase_adjust_harmonic_features(
        self,
        exp_harmonics: dict[str, object],
        model_harmonics: dict[str, object],
    ) -> dict[str, object]:
        exp_phasors = exp_harmonics["phasor"]
        model_phasors = model_harmonics["phasor"]
        exp_fundamental = exp_phasors.get(1, 0j)
        model_fundamental = model_phasors.get(1, 0j)
        if abs(exp_fundamental) == 0 or abs(model_fundamental) == 0:
            return model_harmonics

        phase_delta = np.angle(exp_fundamental) - np.angle(model_fundamental)
        phase_delta = float(np.arctan2(np.sin(phase_delta), np.cos(phase_delta)))
        adjusted_phasors = {
            harmonic: phasor * np.exp(1j * harmonic * phase_delta)
            for harmonic, phasor in model_phasors.items()
        }
        return {
            "rms": dict(model_harmonics["rms"]),
            "phasor": adjusted_phasors,
            "thd": model_harmonics["thd"],
        }

    def _phase_adjust_model_signal(
        self,
        exp_time: np.ndarray,
        exp_signal: np.ndarray,
        model_time: np.ndarray,
        model_signal: np.ndarray,
        base_frequency: float,
    ) -> np.ndarray:
        exp_harmonics = self._compute_harmonic_features(exp_time, exp_signal, base_frequency, max_harmonic=1)
        model_harmonics = self._compute_harmonic_features(model_time, model_signal, base_frequency, max_harmonic=1)
        if exp_harmonics is None or model_harmonics is None:
            return model_signal
        exp_phasor = exp_harmonics["phasor"][1]
        model_phasor = model_harmonics["phasor"][1]
        if abs(exp_phasor) == 0 or abs(model_phasor) == 0 or base_frequency <= 0:
            return model_signal
        phase_delta = np.angle(model_phasor) - np.angle(exp_phasor)
        phase_delta = np.arctan2(np.sin(phase_delta), np.cos(phase_delta))
        time_shift = phase_delta / (2.0 * np.pi * base_frequency)
        return np.interp(
            model_time - time_shift,
            model_time,
            model_signal,
            left=model_signal[0],
            right=model_signal[-1],
        )

    def _extract_cycle_samples(
        self,
        time_values: np.ndarray,
        signal_values: np.ndarray,
        start_time: float,
        end_time: float,
    ) -> dict[str, np.ndarray] | None:
        if end_time <= start_time:
            return None
        mask = (time_values > start_time) & (time_values < end_time)
        inner_time = time_values[mask]
        if inner_time.size < 4:
            return None
        cycle_time = np.concatenate(([start_time], inner_time, [end_time]))
        cycle_signal = np.concatenate(
            (
                [float(np.interp(start_time, time_values, signal_values))],
                signal_values[mask],
                [float(np.interp(end_time, time_values, signal_values))],
            )
        )
        relative_time = cycle_time - float(cycle_time[0])
        return {"time": relative_time, "signal": cycle_signal}

    def _infer_frequency_from_waveform(self, time_values: np.ndarray, voltage_values: np.ndarray) -> float:
        boundaries = detect_cycle_boundaries(time_values, voltage_values)
        if len(boundaries) < 2:
            return 60.0
        durations = np.diff([float(boundary["time"]) for boundary in boundaries])
        valid_durations = durations[durations > 0]
        if valid_durations.size == 0:
            return 60.0
        return float(1.0 / np.median(valid_durations))

    def _normalized_harmonic_errors(
        self,
        true_rms: dict[int, float],
        hat_rms: dict[int, float],
        harmonics,
        eps: float = 1e-12,
    ) -> dict[int, float]:
        fundamental_true = float(true_rms.get(1, 0.0))
        fundamental_hat = float(hat_rms.get(1, 0.0))
        if fundamental_true <= eps or fundamental_hat <= eps:
            return {}

        errors: dict[int, float] = {}
        for harmonic in harmonics:
            reference = float(true_rms.get(harmonic, 0.0)) / fundamental_true
            estimate = float(hat_rms.get(harmonic, 0.0)) / fundamental_hat
            errors[int(harmonic)] = abs(reference - estimate) * 100.0
        return errors

    def _mean_harmonic_error_from_normalized(
        self,
        harmonic_errors: dict[int, float],
        harmonics,
    ) -> float:
        values = [float(harmonic_errors[h]) for h in harmonics if h in harmonic_errors]
        if not values:
            return float("nan")
        return float(np.mean(values))

    def _aggregate_waveform_metric(
        self,
        cycle_metrics: list[dict[str, object]],
        metric_id: str,
    ) -> object:
        if not cycle_metrics:
            return float("nan")
        if metric_id == "max_harmonic_error":
            harmonic_buckets: dict[int, list[float]] = {}
            for cycle_metric in cycle_metrics:
                harmonic_errors = cycle_metric.get("harmonic_errors")
                if not isinstance(harmonic_errors, dict):
                    continue
                for harmonic, value in harmonic_errors.items():
                    harmonic_buckets.setdefault(int(harmonic), []).append(float(value))
            if not harmonic_buckets:
                return {"value": float("nan"), "harmonic": "-"}
            harmonic_means = {
                harmonic: float(np.mean(values))
                for harmonic, values in harmonic_buckets.items()
                if values
            }
            if not harmonic_means:
                return {"value": float("nan"), "harmonic": "-"}
            max_harmonic = max(harmonic_means, key=harmonic_means.get)
            return {"value": harmonic_means[max_harmonic], "harmonic": max_harmonic}

        values = []
        for cycle_metric in cycle_metrics:
            current_value = cycle_metric.get(metric_id)
            try:
                numeric_value = float(current_value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                values.append(numeric_value)
        if not values:
            return float("nan")
        return float(np.mean(values))

    def _align_cycle_waveforms(
        self,
        exp_time: np.ndarray,
        exp_signal: np.ndarray,
        model_time: np.ndarray,
        model_signal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if len(exp_time) < 2 or len(model_time) < 2:
            return None
        exp_duration = float(exp_time[-1] - exp_time[0])
        model_duration = float(model_time[-1] - model_time[0])
        if exp_duration <= 0 or model_duration <= 0:
            return None
        exp_axis = (exp_time - float(exp_time[0])) / exp_duration
        model_axis = (model_time - float(model_time[0])) / model_duration
        aligned_model = np.interp(exp_axis, model_axis, model_signal)
        return np.asarray(exp_signal, dtype=float), np.asarray(aligned_model, dtype=float)

    def _waveform_reference_rms(self, signal_values: np.ndarray, eps: float = 1e-12) -> float:
        reference_rms = float(np.sqrt(np.mean(np.square(signal_values))))
        return reference_rms if reference_rms > eps else float("nan")

    def _waveform_mean_error(
        self,
        exp_time: np.ndarray,
        exp_signal: np.ndarray,
        model_time: np.ndarray,
        model_signal: np.ndarray,
    ) -> float:
        aligned = self._align_cycle_waveforms(exp_time, exp_signal, model_time, model_signal)
        if aligned is None:
            return float("nan")
        exp_values, model_values = aligned
        reference_rms = self._waveform_reference_rms(exp_values)
        if not np.isfinite(reference_rms):
            return float("nan")
        return float(np.mean(np.abs(exp_values - model_values)) / reference_rms * 100.0)

    def _waveform_nrmse(
        self,
        exp_time: np.ndarray,
        exp_signal: np.ndarray,
        model_time: np.ndarray,
        model_signal: np.ndarray,
    ) -> float:
        aligned = self._align_cycle_waveforms(exp_time, exp_signal, model_time, model_signal)
        if aligned is None:
            return float("nan")
        exp_values, model_values = aligned
        reference_rms = self._waveform_reference_rms(exp_values)
        if not np.isfinite(reference_rms):
            return float("nan")
        rmse = float(np.sqrt(np.mean(np.square(exp_values - model_values))))
        return float(rmse / reference_rms * 100.0)

    def _relative_error(self, y_true: float, y_hat: float, eps: float = 1e-12) -> float:
        if not np.isfinite(y_true) or abs(y_true) <= eps:
            return float("nan")
        return float(abs(y_true - y_hat) / abs(y_true))

    def _phase_error_degrees(self, phasor_true: complex, phasor_hat: complex) -> float:
        if abs(phasor_true) == 0 or abs(phasor_hat) == 0:
            return float("nan")
        delta = np.angle(phasor_hat) - np.angle(phasor_true)
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        return float(abs(np.degrees(delta)))

    def _frequency_mean_error(self, window: WindowSegment, eps: float = 1e-12) -> float:
        if self.last_result is None or "frequency" not in self.last_result.scalar_experimental.columns:
            return float("nan")
        exp_dataset = self.last_result.scalar_experimental
        model_dataset = self.last_result.scalar_model
        mask = (exp_dataset["time"] >= window.start_s) & (exp_dataset["time"] <= window.end_s)
        if np.count_nonzero(mask) < 1:
            return float("nan")
        exp_mean = float(np.mean(exp_dataset.loc[mask, "frequency"].to_numpy(dtype=float)))
        model_mean = float(np.mean(model_dataset.loc[mask, "frequency"].to_numpy(dtype=float)))
        if abs(exp_mean) <= eps:
            return float("nan")
        return float(abs(exp_mean - model_mean) / abs(exp_mean))

    def _tve_error(self, phasor_true: complex, phasor_hat: complex, eps: float = 1e-12) -> float:
        denominator = abs(phasor_true)
        if denominator <= eps:
            return float("nan")
        return float(abs(phasor_hat - phasor_true) / denominator)

    def _max_harmonic_error(
        self,
        true_rms: dict[int, float],
        hat_rms: dict[int, float],
        harmonics,
        eps: float = 1e-12,
    ) -> dict[str, object]:
        max_error = float("nan")
        max_harmonic = None
        for harmonic in harmonics:
            reference = float(true_rms.get(harmonic, 0.0))
            estimate = float(hat_rms.get(harmonic, 0.0))
            if abs(reference) <= eps:
                continue
            current_error = abs(reference - estimate) / abs(reference)
            if not np.isfinite(max_error) or current_error > max_error:
                max_error = float(current_error)
                max_harmonic = harmonic
        if max_harmonic is None:
            return {"value": float("nan"), "harmonic": "-"}
        return {"value": max_error * 100.0, "harmonic": max_harmonic}

    def _get_windows_for_kind(self, column: str, window_kind: str) -> list[WindowSegment]:
        windows = self.current_window_segments.get(column, [])
        if window_kind == "steady":
            return [window for window in windows if not self._is_transient_window_name(window.name)]
        return [window for window in windows if self._is_transient_window_name(window.name)]

    def _is_transient_window_name(self, window_name: str) -> bool:
        normalized = window_name.strip().lower()
        return "transient" in normalized or normalized == "ramp"

    def _calculate_metric_for_window(
        self,
        experimental_dataset,
        model_dataset,
        column: str,
        metric_id: str,
        window: WindowSegment,
        window_kind: str,
        delay_adjusted: bool = False,
    ) -> float:
        if self.last_result is None:
            return float("nan")

        time_values = experimental_dataset["time"].to_numpy(dtype=float)
        mask = (time_values >= window.start_s) & (time_values <= window.end_s)
        if np.count_nonzero(mask) < 2:
            return float("nan")

        y_true = experimental_dataset.loc[mask, column].to_numpy(dtype=float)
        y_hat = model_dataset.loc[mask, column].to_numpy(dtype=float)

        if delay_adjusted and metric_id in ("transient_mean_error", "transient_nrmse"):
            delay_time = self._get_global_transient_delay(
                experimental_dataset=experimental_dataset,
                model_dataset=model_dataset,
                target_window=window,
            )
            if not np.isfinite(delay_time):
                return float("nan")
            window_time = time_values[mask]
            y_true = experimental_dataset.loc[mask, column].to_numpy(dtype=float)
            y_hat = np.interp(
                window_time + delay_time,
                time_values,
                model_dataset[column].to_numpy(dtype=float),
                left=float(model_dataset[column].iloc[0]),
                right=float(model_dataset[column].iloc[-1]),
            )
            transient_context = self._build_transient_metric_context(
                experimental_dataset=experimental_dataset,
                model_dataset=model_dataset,
                column=column,
                window=window,
                mask=mask,
                delay_adjusted=True,
            )
            if transient_context is None:
                return float("nan")
            if metric_id == "transient_mean_error":
                return self._mean_relative_error_transient(
                    y_true,
                    y_hat,
                    column,
                    transient_context["y_init_real"],
                    transient_context["y_final_real"],
                ) * 100.0
            return self._nrmse_transient(
                y_true,
                y_hat,
                column,
                transient_context["y_init_real"],
                transient_context["y_final_real"],
            ) * 100.0

        if metric_id in ("steady_mean_error", "transient_mean_error"):
            if window_kind == "steady":
                return self._mean_relative_error_steady(y_true, y_hat) * 100.0
            transient_context = self._build_transient_metric_context(
                experimental_dataset=experimental_dataset,
                model_dataset=model_dataset,
                column=column,
                window=window,
                mask=mask,
                delay_adjusted=delay_adjusted,
            )
            if transient_context is None:
                return float("nan")
            return self._mean_relative_error_transient(
                y_true,
                y_hat,
                column,
                transient_context["y_init_real"],
                transient_context["y_final_real"],
            ) * 100.0

        if metric_id in ("steady_nrmse", "transient_nrmse"):
            if window_kind == "steady":
                return self._nrmse_steady(y_true, y_hat) * 100.0
            transient_context = self._build_transient_metric_context(
                experimental_dataset=experimental_dataset,
                model_dataset=model_dataset,
                column=column,
                window=window,
                mask=mask,
                delay_adjusted=delay_adjusted,
            )
            if transient_context is None:
                return float("nan")
            return self._nrmse_transient(
                y_true,
                y_hat,
                column,
                transient_context["y_init_real"],
                transient_context["y_final_real"],
            ) * 100.0

        if metric_id == "steady_residual_rms":
            return self._steady_residual_rms(y_true, y_hat) * 100.0

        if metric_id == "steady_ssi":
            return self._stationary_similarity_error(y_true, y_hat) * 100.0

        if metric_id == "steady_std_error":
            return self._relative_std_error(y_true, y_hat) * 100.0

        if metric_id == "steady_peak_to_peak_error":
            return self._relative_peak_to_peak_error(y_true, y_hat) * 100.0

        if metric_id.startswith("transient_") and metric_id not in ("transient_mean_error", "transient_nrmse"):
            transient_context = self._build_transient_metric_context(
                experimental_dataset=experimental_dataset,
                model_dataset=model_dataset,
                column=column,
                window=window,
                mask=mask,
                delay_adjusted=delay_adjusted,
            )
            if transient_context is None:
                return float("nan")

            t_values = transient_context["time"]
            y_model = transient_context["y_model"]
            y_real = transient_context["y_real"]
            y_init_real = transient_context["y_init_real"]
            y_final_real = transient_context["y_final_real"]
            y_init_model = transient_context["y_init_model"]
            y_final_model = transient_context["y_final_model"]

            reference_features = self._compute_transition_characteristics(
                t=t_values,
                y=y_real,
                y_init=y_init_real,
                y_final=y_final_real,
                band=0.05,
                max_settling_time_s=max(float(window.end_s - window.start_s), 0.0),
            )
            model_features = self._compute_transition_characteristics(
                t=t_values,
                y=y_model,
                y_init=y_init_model,
                y_final=y_final_model,
                band=0.05,
                max_settling_time_s=max(float(window.end_s - window.start_s), 0.0),
            )
            self._store_transient_debug_entry(
                column=column,
                window=window,
                context=transient_context,
                reference_features=reference_features,
                model_features=model_features,
            )

            if metric_id == "transient_overshoot_undershoot":
                return self._metric_error_from_features(
                    metric_name="overshoot_pct",
                    model_features=model_features,
                    reference_features=reference_features,
                )
            if metric_id == "transient_rise_fall_time":
                return self._convert_metric_display_value(
                    metric_id,
                    self._metric_error_from_features(
                        metric_name="rise_fall_time_s",
                        model_features=model_features,
                        reference_features=reference_features,
                    ),
                )
            if metric_id == "transient_settling_time":
                return self._convert_metric_display_value(
                    metric_id,
                    self._metric_error_from_features(
                        metric_name="settling_time_s",
                        model_features=model_features,
                        reference_features=reference_features,
                    ),
                )
            if metric_id == "transient_response_time":
                return self._convert_metric_display_value(
                    metric_id,
                    self._metric_error_from_features(
                        metric_name="response_time_s",
                        model_features=model_features,
                        reference_features=reference_features,
                    ),
                )
            if metric_id == "transient_reaction_time":
                return self._convert_metric_display_value(
                    metric_id,
                    self._metric_error_from_features(
                        metric_name="reaction_time_s",
                        model_features=model_features,
                        reference_features=reference_features,
                    ),
                )
            if metric_id == "transient_delay_time":
                if delay_adjusted:
                    return {
                        "value": 0.0,
                        "direction": "adjusted",
                    }
                delay_value = self._get_global_transient_delay(
                    experimental_dataset=experimental_dataset,
                    model_dataset=model_dataset,
                    target_window=window,
                )
                try:
                    numeric_delay = float(delay_value)
                except (TypeError, ValueError):
                    return {"message": "Could not determine the delay time"}
                if not np.isfinite(numeric_delay):
                    return {"message": "Could not determine the delay time"}
                direction_label = "model delayed" if numeric_delay > 0 else "model advanced"
                if abs(numeric_delay) <= 1e-12:
                    direction_label = "no delay"
                return {
                    "value": abs(numeric_delay) * 1000.0,
                    "direction": direction_label,
                }
            if metric_id == "transient_pearson_r":
                if self._has_insufficient_variation(reference_features):
                    return {"message": "There is no significant variation in the segment"}
                pearson_value = self._pearson_r(y_real, y_model)
                if not np.isfinite(pearson_value):
                    return {"message": "Insufficient y variation to normalize the signal"}
                return max(pearson_value, 0.0)
            if metric_id == "transient_r2":
                if self._has_insufficient_variation(reference_features):
                    return {"message": "There is no significant variation in the segment"}
                r2_value = self._r2_score(y_real, y_model)
                if not np.isfinite(r2_value):
                    return {"message": "Variacao de y insuficiente para normalizar o sinal"}
                return max(r2_value, 0.0)

        return float("nan")

    def _calculate_transient_metric_bundle_for_window(
        self,
        experimental_dataset,
        model_dataset,
        column: str,
        metric_id: str,
        window: WindowSegment,
        delay_adjusted: bool,
    ) -> dict[str, object]:
        time_values = experimental_dataset["time"].to_numpy(dtype=float)
        mask = (time_values >= window.start_s) & (time_values <= window.end_s)
        transient_context = self._build_transient_metric_context(
            experimental_dataset=experimental_dataset,
            model_dataset=model_dataset,
            column=column,
            window=window,
            mask=mask,
            delay_adjusted=delay_adjusted,
        )
        if transient_context is None:
            message = {"message": "Could not determine the characteristic points"}
            return {"difference": message, "reference": message, "model": message}

        t_values = transient_context["time"]
        y_model = transient_context["y_model"]
        y_real = transient_context["y_real"]
        y_init_real = transient_context["y_init_real"]
        y_final_real = transient_context["y_final_real"]
        y_init_model = transient_context["y_init_model"]
        y_final_model = transient_context["y_final_model"]

        reference_features = self._compute_transition_characteristics(
            t=t_values,
            y=y_real,
            y_init=y_init_real,
            y_final=y_final_real,
            band=0.05,
            max_settling_time_s=max(float(window.end_s - window.start_s), 0.0),
        )
        model_features = self._compute_transition_characteristics(
            t=t_values,
            y=y_model,
            y_init=y_init_model,
            y_final=y_final_model,
            band=0.05,
            max_settling_time_s=max(float(window.end_s - window.start_s), 0.0),
        )
        self._store_transient_debug_entry(
            column=column,
            window=window,
            context=transient_context,
            reference_features=reference_features,
            model_features=model_features,
        )

        feature_name, _feature_label = self.TRANSIENT_FEATURE_METRIC_MAP[metric_id]
        difference = self._metric_error_from_features(
            metric_name=feature_name,
            model_features=model_features,
            reference_features=reference_features,
            relative_to_reference=metric_id in self.RELATIVE_TRANSIENT_FEATURE_METRICS,
        )
        display_metric_id = self._base_transient_feature_metric_id(metric_id)
        reference_value = self._feature_value_for_display(display_metric_id, feature_name, reference_features)
        model_value = self._feature_value_for_display(display_metric_id, feature_name, model_features)
        return {
            "difference": self._convert_metric_display_value(metric_id, difference),
            "reference": reference_value,
            "model": model_value,
        }

    def _build_transient_metric_context(
        self,
        experimental_dataset,
        model_dataset,
        column: str,
        window: WindowSegment,
        mask: np.ndarray,
        delay_adjusted: bool = False,
    ) -> dict[str, object] | None:
        if self.last_result is None:
            return None

        column_windows = self.current_window_segments.get(column, [])
        window_index = next(
            (
                index
                for index, current_window in enumerate(column_windows)
                if current_window.name == window.name
                and abs(current_window.start_s - window.start_s) <= 1e-9
                and abs(current_window.end_s - window.end_s) <= 1e-9
            ),
            None,
        )
        if window_index is None:
            return None
        if window_index == 0 or window_index >= len(column_windows) - 1:
            return None

        previous_window = column_windows[window_index - 1]
        next_window = column_windows[window_index + 1]
        time_axis = experimental_dataset["time"].to_numpy(dtype=float)
        previous_mask = (time_axis >= previous_window.start_s) & (time_axis <= previous_window.end_s)
        next_mask = (time_axis >= next_window.start_s) & (time_axis <= next_window.end_s)
        analysis_mask = (time_axis >= window.start_s) & (time_axis <= next_window.end_s)
        if np.count_nonzero(previous_mask) < 1 or np.count_nonzero(next_mask) < 1:
            return None
        if np.count_nonzero(analysis_mask) < 3:
            return None

        y_init_real = float(np.mean(experimental_dataset.loc[previous_mask, column].to_numpy(dtype=float)))
        y_final_real = float(np.mean(experimental_dataset.loc[next_mask, column].to_numpy(dtype=float)))
        y_init_model = float(np.mean(model_dataset.loc[previous_mask, column].to_numpy(dtype=float)))
        y_final_model = float(np.mean(model_dataset.loc[next_mask, column].to_numpy(dtype=float)))
        if (
            not np.isfinite(y_init_real)
            or not np.isfinite(y_final_real)
            or not np.isfinite(y_init_model)
            or not np.isfinite(y_final_model)
        ):
            return None

        time_values = experimental_dataset.loc[analysis_mask, "time"].to_numpy(dtype=float)
        relative_time = time_values - float(time_values[0])
        y_real = experimental_dataset.loc[analysis_mask, column].to_numpy(dtype=float)
        y_model = model_dataset.loc[analysis_mask, column].to_numpy(dtype=float)
        if delay_adjusted:
            delay_time = self._get_global_transient_delay(
                experimental_dataset=experimental_dataset,
                model_dataset=model_dataset,
                target_window=window,
            )
        else:
            delay_time = self._estimate_delay_time_from_signals(
                t=relative_time,
                y_real=y_real,
                y_model=y_model,
                y_init_real=y_init_real,
                y_final_real=y_final_real,
                y_init_model=y_init_model,
                y_final_model=y_final_model,
            )
        if delay_adjusted and np.isfinite(delay_time):
            y_model = np.interp(
                relative_time + delay_time,
                relative_time,
                y_model,
                left=float(y_model[0]),
                right=float(y_model[-1]),
            )
            y_init_model = y_init_real
            y_final_model = y_final_real

        return {
            "time": relative_time,
            "y_real": y_real,
            "y_model": y_model,
            "y_init_real": y_init_real,
            "y_final_real": y_final_real,
            "y_init_model": y_init_model,
            "y_final_model": y_final_model,
            "delay_time_s": delay_time,
            "delay_adjusted": delay_adjusted,
            "window_name": window.name,
            "analysis_end_s": next_window.end_s,
        }

    def _get_global_transient_delay(
        self,
        experimental_dataset,
        model_dataset,
        target_window: WindowSegment,
    ) -> float:
        if self.last_result is None:
            return float("nan")

        sync_column = self.last_result.sync_column
        if sync_column not in experimental_dataset.columns or sync_column not in model_dataset.columns:
            return float("nan")

        sync_windows = self.current_window_segments.get(sync_column, [])
        sync_window = next(
            (
                current_window
                for current_window in sync_windows
                if current_window.name == target_window.name
            ),
            None,
        )
        if sync_window is None:
            return float("nan")

        cache_key = f"{sync_column}|{sync_window.name}|{sync_window.start_s:.9f}|{sync_window.end_s:.9f}"
        if cache_key in self.transient_delay_cache:
            return self.transient_delay_cache[cache_key]

        time_values = experimental_dataset["time"].to_numpy(dtype=float)
        mask = (time_values >= sync_window.start_s) & (time_values <= sync_window.end_s)
        context = self._build_transient_metric_context(
            experimental_dataset=experimental_dataset,
            model_dataset=model_dataset,
            column=sync_column,
            window=sync_window,
            mask=mask,
            delay_adjusted=False,
        )
        if context is None:
            delay_time = float("nan")
        else:
            delay_time = float(context.get("delay_time_s", float("nan")))
        self.transient_delay_cache[cache_key] = delay_time
        return delay_time

    def _mean_relative_error_steady(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        denom = np.where(np.abs(y_true) > eps, y_true, np.nan)
        rel = np.abs((y_true - y_hat) / denom)
        return float(np.nanmean(rel))

    def _mean_relative_error_transient(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        column: str,
        y_init: float,
        y_final: float,
        eps: float = 1e-12,
    ) -> float:
        denominator = self._transient_error_denominator(y_true, column, y_init, y_final, eps=eps)
        if denominator <= eps:
            return float("nan")
        return float(np.mean(np.abs(y_true - y_hat)) / denominator)

    def _nrmse_steady(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        y_bar = float(np.mean(y_true))
        if abs(y_bar) <= eps:
            return float("nan")
        rmse = np.sqrt(np.mean((y_true - y_hat) ** 2))
        return float(rmse / y_bar)

    def _nrmse_transient(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        column: str,
        y_init: float,
        y_final: float,
        eps: float = 1e-12,
    ) -> float:
        rmse = np.sqrt(np.mean((y_true - y_hat) ** 2))
        denominator = self._transient_error_denominator(y_true, column, y_init, y_final, eps=eps)
        if denominator <= eps:
            return float("nan")
        return float(rmse / denominator)

    def _transient_error_denominator(
        self,
        y_true: np.ndarray,
        column: str,
        y_init: float,
        y_final: float,
        eps: float = 1e-12,
    ) -> float:
        if self._should_use_steady_normalization_for_transient(y_true, y_init, y_final, eps=eps):
            mean_value = abs(float(np.mean(y_true)))
            return mean_value if mean_value > eps else float("nan")

        selected_mode = self._selected_transient_normalization_mode()
        if selected_mode == "mean_value":
            mean_value = abs(float(np.mean(y_true)))
            return mean_value if mean_value > eps else float("nan")

        if selected_mode == "origin_event_percent":
            event_percent_denominator = self._origin_event_percent_denominator_pu(column, eps=eps)
            if np.isfinite(event_percent_denominator) and event_percent_denominator > eps:
                return event_percent_denominator

        reference_level = max(abs(y_init), abs(y_final), 1e-6)
        step_amplitude = abs(y_final - y_init)
        transient_excursion = float(np.max(np.abs(y_true - y_final))) if len(y_true) else 0.0
        return max(step_amplitude, transient_excursion, reference_level * 0.002, eps)

    def _should_use_steady_normalization_for_transient(
        self,
        y_true: np.ndarray,
        y_init: float,
        y_final: float,
        eps: float = 1e-12,
    ) -> bool:
        if len(y_true) == 0:
            return True
        reference_level = max(abs(float(np.mean(y_true))), abs(y_init), abs(y_final), 1e-6)
        step_amplitude = abs(y_final - y_init)
        oscillation_amplitude = float(np.max(y_true) - np.min(y_true))
        insignificance_threshold = max(reference_level * 0.01, 0.01, eps)
        return step_amplitude <= insignificance_threshold and oscillation_amplitude <= insignificance_threshold

    def _stationary_similarity_error(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        y_bar = float(np.mean(y_true))
        yhat_bar = float(np.mean(y_hat))
        if abs(y_bar) <= eps:
            return float("nan")
        ssi = float(1.0 - abs(y_bar - yhat_bar) / abs(y_bar))
        return float(1.0 - ssi)

    def _steady_residual_rms(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        mean_value = float(np.mean(y_true))
        if abs(mean_value) <= eps:
            return float("nan")
        residual_rms = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
        return residual_rms / abs(mean_value)

    def _relative_std_error(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        std_true = float(np.std(y_true))
        std_hat = float(np.std(y_hat))
        mean_true = float(np.mean(y_true))
        if abs(mean_true) <= eps:
            return float("nan")
        return float(abs(std_true - std_hat) / abs(mean_true))

    def _relative_peak_to_peak_error(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        ptp_true = float(np.ptp(y_true))
        ptp_hat = float(np.ptp(y_hat))
        mean_true = float(np.mean(y_true))
        if abs(mean_true) <= eps:
            return float("nan")
        return float(abs(ptp_true - ptp_hat) / abs(mean_true))

    def _absolute_metric_error(self, value_model: object, value_reference: object) -> object:
        try:
            numeric_model = float(value_model)
            numeric_reference = float(value_reference)
        except (TypeError, ValueError):
            return {"message": "Could not determine the characteristic points"}
        if not np.isfinite(numeric_model) or not np.isfinite(numeric_reference):
            return {"message": "Could not determine the characteristic points"}
        return float(abs(numeric_model - numeric_reference))

    def _metric_error_from_features(
        self,
        metric_name: str,
        model_features: dict[str, object],
        reference_features: dict[str, object],
        relative_to_reference: bool = False,
        eps: float = 1e-12,
    ) -> object:
        model_messages = model_features.get("metric_messages", {})
        reference_messages = reference_features.get("metric_messages", {})
        if isinstance(model_messages, dict) and metric_name in model_messages:
            return {"message": str(model_messages[metric_name])}
        if isinstance(reference_messages, dict) and metric_name in reference_messages:
            return {"message": str(reference_messages[metric_name])}
        if relative_to_reference:
            return self._relative_metric_error(
                model_features.get(metric_name),
                reference_features.get(metric_name),
                eps=eps,
            )
        return self._absolute_metric_error(
            model_features.get(metric_name),
            reference_features.get(metric_name),
        )

    def _relative_metric_error(
        self,
        value_model: object,
        value_reference: object,
        eps: float = 1e-12,
    ) -> object:
        try:
            numeric_model = float(value_model)
            numeric_reference = float(value_reference)
        except (TypeError, ValueError):
            return {"message": "Could not determine the characteristic points"}
        if not np.isfinite(numeric_model) or not np.isfinite(numeric_reference):
            return {"message": "Could not determine the characteristic points"}
        if abs(numeric_reference) <= eps:
            if abs(numeric_model) <= eps:
                return 0.0
            return {"message": "Does not apply: reference metric equals zero"}
        return float(abs(numeric_model - numeric_reference) / abs(numeric_reference) * 100.0)

    def _feature_value_for_display(
        self,
        metric_id: str,
        feature_name: str,
        features: dict[str, object],
    ) -> object:
        metric_messages = features.get("metric_messages", {})
        if isinstance(metric_messages, dict) and feature_name in metric_messages:
            return {"message": str(metric_messages[feature_name])}
        return self._convert_metric_display_value(metric_id, features.get(feature_name, float("nan")))

    def _convert_metric_display_value(self, metric_id: str, value: object) -> object:
        if metric_id not in self.TIME_METRIC_IDS:
            return value
        if isinstance(value, dict):
            if "message" in value:
                return value
            converted = dict(value)
            if "value" in converted:
                try:
                    converted["value"] = float(converted["value"]) * 1000.0
                except (TypeError, ValueError):
                    pass
            return converted
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value
        if not np.isfinite(numeric_value):
            return numeric_value
        return numeric_value * 1000.0

    def _has_insufficient_variation(self, features: dict[str, object]) -> bool:
        messages = features.get("metric_messages", {})
        if not isinstance(messages, dict):
            return False
        return any(
            str(message) == self.NO_SIGNIFICANT_OSCILLATION_MESSAGE
            for message in messages.values()
        )

    def _should_suppress_metric_row(self, values: list[object]) -> bool:
        messages: list[str] = []
        for value in values:
            if not isinstance(value, dict) or "message" not in value:
                return False
            messages.append(str(value["message"]))
        return bool(messages) and all(message == self.EQUIVALENT_INITIAL_FINAL_MESSAGE for message in messages)

    def _base_transient_feature_metric_id(self, metric_id: str) -> str:
        return self.RELATIVE_TRANSIENT_FEATURE_BASE_METRIC.get(metric_id, metric_id)

    def _estimate_delay_time_from_signals(
        self,
        t: np.ndarray,
        y_real: np.ndarray,
        y_model: np.ndarray,
        y_init_real: float,
        y_final_real: float,
        y_init_model: float,
        y_final_model: float,
    ) -> float:
        real_start = self._event_start_time(
            t=t,
            y=y_real,
            y_init=y_init_real,
            y_final=y_final_real,
        )
        model_start = self._event_start_time(
            t=t,
            y=y_model,
            y_init=y_init_model,
            y_final=y_final_model,
        )
        if not np.isfinite(real_start) or not np.isfinite(model_start):
            return float("nan")
        return float(model_start - real_start)

    def _event_start_time(
        self,
        t: np.ndarray,
        y: np.ndarray,
        y_init: float,
        y_final: float,
    ) -> float:
        delta = y_final - y_init
        if abs(delta) > 0.01:
            direction = "up" if delta > 0 else "down"
            level_10 = y_init + 0.10 * delta
            return self._find_threshold_crossing_time(t, y, level_10, direction)

        threshold = max(abs(y_init) * 0.01, 0.005)
        indexes = np.where(np.abs(y - y_init) >= threshold)[0]
        if len(indexes) == 0:
            return float("nan")
        return float(t[indexes[0]])

    def _compute_transition_characteristics(
        self,
        t: np.ndarray,
        y: np.ndarray,
        y_init: float,
        y_final: float,
        band: float = 0.05,
        max_settling_time_s: float | None = None,
        eps: float = 1e-9,
    ) -> dict[str, object]:
        delta = y_final - y_init
        baseline_magnitude = max(abs(y_init), abs(y_final), 1e-6)
        effective_delta = abs(delta)
        metric_messages: dict[str, str] = {}
        transient_mask = np.ones_like(t, dtype=bool)
        if max_settling_time_s is not None and np.isfinite(max_settling_time_s):
            transient_mask = t <= float(max_settling_time_s)
            if not np.any(transient_mask):
                transient_mask = np.ones_like(t, dtype=bool)
        transient_t = t[transient_mask]
        transient_y = y[transient_mask]

        if effective_delta <= max(eps, 0.01):
            baseline = (y_init + y_final) / 2.0
            band_delta = max(abs(baseline) * 0.002, float(np.std(y)) * 2.5, 0.0005)
            positive_peak = float(np.max(transient_y))
            negative_peak = float(np.min(transient_y))
            positive_dev = positive_peak - baseline
            negative_dev = baseline - negative_peak
            oscillation_amplitude = max(positive_dev, negative_dev)
            if oscillation_amplitude <= max(0.5 * band_delta, 0.01):
                message = self.NO_SIGNIFICANT_OSCILLATION_MESSAGE
                return {
                    "metric_messages": {
                        "overshoot_pct": message,
                        "settling_time_s": message,
                        "reaction_time_s": message,
                        "rise_fall_time_s": message,
                        "response_time_s": message,
                    }
                }

            direction = "up" if positive_dev >= negative_dev else "down"
            overshoot_time = self._find_peak_time(transient_t, transient_y, direction)
            settling_entry = self._find_settling_entry_time_absolute(
                t=t,
                y=y,
                lower=baseline - band_delta,
                upper=baseline + band_delta,
                start_time=overshoot_time,
            )
            settling_time = settling_entry
            if not np.isfinite(settling_time):
                settling_time = float(t[-1] - t[0])
                settling_entry = float(t[-1])
            if max_settling_time_s is not None and np.isfinite(max_settling_time_s):
                settling_time = min(float(settling_time), float(max_settling_time_s))
                settling_entry = min(float(settling_entry), float(max_settling_time_s))
            metric_messages.update(
                {
                    "reaction_time_s": self.EQUIVALENT_INITIAL_FINAL_MESSAGE,
                    "rise_fall_time_s": self.EQUIVALENT_INITIAL_FINAL_MESSAGE,
                    "response_time_s": self.EQUIVALENT_INITIAL_FINAL_MESSAGE,
                }
            )
            return {
                "reaction_time_s": float("nan"),
                "response_time_s": float("nan"),
                "rise_fall_time_s": float("nan"),
                "settling_time_s": settling_time,
                "overshoot_pct": float(oscillation_amplitude / baseline_magnitude * 100.0),
                "threshold_10": float("nan"),
                "threshold_90": float("nan"),
                "threshold_95": float("nan"),
                "settling_lower": float(baseline - band_delta),
                "settling_upper": float(baseline + band_delta),
                "reaction_marker_time_s": float("nan"),
                "rise_end_time_s": float("nan"),
                "response_marker_time_s": float("nan"),
                "settling_marker_time_s": settling_entry,
                "overshoot_time_s": overshoot_time,
                "y_init": float(y_init),
                "y_final": float(y_final),
                "direction": direction,
                "metric_messages": metric_messages,
            }

        direction = "up" if delta > 0 else "down"
        level_10 = y_init + 0.10 * delta
        level_90 = y_init + 0.90 * delta
        level_95 = y_init + 0.95 * delta

        t10 = self._find_threshold_crossing_time(t, y, level_10, direction)
        t90 = self._find_threshold_crossing_time(t, y, level_90, direction)
        t95 = self._find_threshold_crossing_time(t, y, level_95, direction)

        reaction_time = t10
        rise_fall_time = float("nan")
        if np.isfinite(t10) and np.isfinite(t90) and t90 >= t10:
            rise_fall_time = float(t90 - t10)

        response_time = float("nan")
        if np.isfinite(t10) and np.isfinite(t95) and t95 >= t10:
            response_time = float(t95 - t10)

        settling_band_delta = max(effective_delta * band, abs(y_final) * 0.002, 0.0005)
        settling_lower = float(y_final - settling_band_delta)
        settling_upper = float(y_final + settling_band_delta)
        settling_entry = self._find_settling_entry_time_absolute(
            t=t,
            y=y,
            lower=settling_lower,
            upper=settling_upper,
            start_time=t95 if np.isfinite(t95) else (t10 if np.isfinite(t10) else float(t[0])),
        )
        settling_time = float("nan")
        if np.isfinite(t10) and np.isfinite(settling_entry) and settling_entry >= t10:
            settling_time = float(settling_entry - t10)
        elif not np.isfinite(settling_entry):
            settling_entry = float(t[-1])
            settling_time = float(t[-1] - t[0])
        if max_settling_time_s is not None and np.isfinite(max_settling_time_s):
            settling_time = min(float(settling_time), float(max_settling_time_s))
            if np.isfinite(t10):
                settling_entry = min(float(settling_entry), float(max_settling_time_s))
            else:
                settling_entry = min(float(settling_entry), float(max_settling_time_s))

        overshoot_base = max(abs(y_final), 1e-6)
        if direction == "up":
            peak = float(np.max(transient_y))
            overshoot_pct = max((peak - y_final) / overshoot_base, 0.0) * 100.0
        else:
            peak = float(np.min(transient_y))
            overshoot_pct = max((y_final - peak) / overshoot_base, 0.0) * 100.0

        if not np.isfinite(reaction_time):
            metric_messages["reaction_time_s"] = "Could not determine the reaction time"
        if not np.isfinite(rise_fall_time):
            metric_messages["rise_fall_time_s"] = "Could not determine the rise/fall time"
        if not np.isfinite(response_time):
            metric_messages["response_time_s"] = "Could not determine the response time"
        if not np.isfinite(settling_time):
            metric_messages["settling_time_s"] = "Could not determine the settling time"

        return {
            "reaction_time_s": reaction_time,
            "response_time_s": response_time,
            "rise_fall_time_s": rise_fall_time,
            "settling_time_s": settling_time,
            "overshoot_pct": float(overshoot_pct),
            "threshold_10": float(level_10),
            "threshold_90": float(level_90),
            "threshold_95": float(level_95),
            "settling_lower": settling_lower,
            "settling_upper": settling_upper,
            "reaction_marker_time_s": reaction_time,
            "rise_end_time_s": t90,
            "response_marker_time_s": t95,
            "settling_marker_time_s": settling_entry,
            "overshoot_time_s": self._find_peak_time(transient_t, transient_y, direction),
            "y_init": float(y_init),
            "y_final": float(y_final),
            "direction": direction,
            "metric_messages": metric_messages,
        }

    def _find_threshold_crossing_time(
        self,
        t: np.ndarray,
        y: np.ndarray,
        level: float,
        direction: str,
    ) -> float:
        if len(t) < 2:
            return float("nan")
        if direction == "up" and y[0] >= level:
            return float(t[0])
        if direction == "down" and y[0] <= level:
            return float(t[0])

        for index in range(len(y) - 1):
            y0 = float(y[index])
            y1 = float(y[index + 1])
            t0 = float(t[index])
            t1 = float(t[index + 1])
            if direction == "up":
                crossed = y0 <= level <= y1 or y0 < level and y1 > level
            else:
                crossed = y0 >= level >= y1 or y0 > level and y1 < level
            if not crossed:
                continue
            if abs(y1 - y0) <= 1e-12:
                return t0
            ratio = (level - y0) / (y1 - y0)
            ratio = min(max(ratio, 0.0), 1.0)
            return float(t0 + ratio * (t1 - t0))
        return float("nan")

    def _find_settling_entry_time_absolute(
        self,
        t: np.ndarray,
        y: np.ndarray,
        lower: float,
        upper: float,
        start_time: float,
    ) -> float:
        if len(t) < 2:
            return float("nan")
        inside_band = (y >= lower) & (y <= upper)
        hold_points = min(max(3, len(y) // 10), len(y))
        start_index = 0
        if np.isfinite(start_time):
            candidate_indexes = np.where(t >= start_time)[0]
            if len(candidate_indexes) > 0:
                start_index = int(candidate_indexes[0])

        for index in range(start_index, len(y)):
            if not inside_band[index]:
                continue
            tail = inside_band[index:]
            if tail.size == 0:
                continue
            if np.all(tail):
                if index == 0:
                    return float(t[0])
                previous_value = float(y[index - 1])
                current_value = float(y[index])
                if previous_value < lower:
                    target_level = lower
                elif previous_value > upper:
                    target_level = upper
                else:
                    target_level = current_value
                if lower <= previous_value <= upper:
                    return float(t[index])
                if abs(current_value - previous_value) <= 1e-12:
                    return float(t[index])
                ratio = (target_level - previous_value) / (current_value - previous_value)
                ratio = min(max(ratio, 0.0), 1.0)
                return float(t[index - 1] + ratio * (t[index] - t[index - 1]))

        return float("nan")

    def _find_peak_time(self, t: np.ndarray, y: np.ndarray, direction: str) -> float:
        if len(t) == 0:
            return float("nan")
        if direction == "up":
            peak_index = int(np.argmax(y))
        else:
            peak_index = int(np.argmin(y))
        return float(t[peak_index])

    def _store_transient_debug_entry(
        self,
        column: str,
        window: WindowSegment,
        context: dict[str, object],
        reference_features: dict[str, object],
        model_features: dict[str, object],
    ) -> None:
        adjusted_suffix = "|adjusted" if context.get("delay_adjusted") else "|original"
        key = f"{column}|{window.name}|{window.start_s:.9f}|{window.end_s:.9f}{adjusted_suffix}"
        entry = {
            "key": key,
            "column": column,
            "window_name": (
                f"{window.name} (delay-adjusted)"
                if context.get("delay_adjusted")
                else window.name
            ),
            "window_start_s": float(window.start_s),
            "window_end_s": float(window.end_s),
            "time": np.asarray(context["time"], dtype=float),
            "y_real": np.asarray(context["y_real"], dtype=float),
            "y_model": np.asarray(context["y_model"], dtype=float),
            "reference_features": reference_features,
            "model_features": model_features,
        }
        current_entries = self.last_transient_debug_data.setdefault(column, [])
        for index, current_entry in enumerate(current_entries):
            if current_entry["key"] == key:
                current_entries[index] = entry
                return
        current_entries.append(entry)

    def _pearson_r(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        centered_true = y_true - np.mean(y_true)
        centered_hat = y_hat - np.mean(y_hat)
        numerator = float(np.sum(centered_true * centered_hat))
        denominator = float(
            np.sqrt(np.sum(centered_true ** 2)) * np.sqrt(np.sum(centered_hat ** 2))
        )
        if denominator <= eps:
            return float("nan")
        return float(numerator / denominator)

    def _r2_score(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        ss_res = float(np.sum((y_true - y_hat) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot <= eps:
            return float("nan")
        return float(1.0 - ss_res / ss_tot)

    def _format_metric_display(self, metric_id: str, value: object) -> str:
        if isinstance(value, dict):
            if "message" in value:
                return str(value["message"])
            metric_value = self._extract_metric_numeric_value(value)
            if not np.isfinite(metric_value):
                return "-"
            if "harmonic" in value:
                unit_label = self.METRIC_UNIT_LABELS.get(metric_id, "%")
                return f"{metric_value:.3f} {unit_label} (h{value['harmonic']})"
            if "direction" in value:
                return f"{metric_value:.3f} {self.METRIC_UNIT_LABELS.get(metric_id, '%')} ({value['direction']})"
            return f"{metric_value:.3f} {self.METRIC_UNIT_LABELS.get(metric_id, '%')}"
        return self._format_metric_value(metric_id, float(value))

    def _format_metric_value(self, metric_id: str, value: float) -> str:
        if not np.isfinite(value):
            return "-"
        unit_label = self.METRIC_UNIT_LABELS.get(metric_id, "%")
        return f"{value:.3f} {unit_label}"

    def _extract_metric_numeric_value(self, value: object) -> float:
        if isinstance(value, dict):
            if "message" in value:
                return float("nan")
            return float(value.get("value", float("nan")))
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return numeric_value

    def _metric_cell_colors(self, metric_id: str, value: float) -> tuple[str, str]:
        if not np.isfinite(value):
            return "white", "#334155"

        metric_state = None
        for group_id, group_metrics in self.metric_selection_state.items():
            if metric_id in group_metrics:
                metric_state = group_metrics[metric_id]
                break
        if metric_state is None:
            return "white", "#334155"

        good_text = metric_state["limits"]["good"].get().strip()
        acceptable_text = metric_state["limits"]["acceptable"].get().strip()
        if not any((good_text, acceptable_text)):
            return "white", "#334155"

        good_limit = self._safe_float(good_text) if good_text else None
        acceptable_limit = self._safe_float(acceptable_text) if acceptable_text else None

        if metric_id in self.HIGHER_IS_BETTER_METRICS:
            if good_limit is not None and value >= good_limit:
                return "#dcfce7", "#166534"
            if acceptable_limit is not None and value >= acceptable_limit:
                return "#fef3c7", "#92400e"
            return "#fee2e2", "#991b1b"

        if good_limit is not None and value <= good_limit:
            return "#dcfce7", "#166534"
        if acceptable_limit is not None and value <= acceptable_limit:
            return "#fef3c7", "#92400e"
        return "#fee2e2", "#991b1b"

    def _metric_label_for_id(self, group_id: str, metric_id: str) -> str:
        for group in self.METRIC_GROUP_DEFINITIONS:
            if str(group["id"]) != group_id:
                continue
            for section in group["sections"]:
                for current_metric_id, metric_label in section["metrics"]:
                    if current_metric_id == metric_id:
                        return metric_label
        return metric_id

    def _build_test_description(self) -> str:
        form_values = self._get_current_form_values()
        if self.current_test_key == "steady_state":
            power_percent = form_values.get("test_power_percent", "")
            return f"Steady-state test at {power_percent}% of nominal power"
        if self.current_test_key == "step_test":
            quantity = form_values.get("step_quantity", "").lower()
            pre_value = form_values.get("pre_step_percent", "")
            post_value = form_values.get("post_step_percent", "")
            return f"Step test in {quantity} from {pre_value}% to {post_value}%"
        if self.current_test_key == "fault_test":
            quantity = form_values.get("fault_quantity", "").lower()
            pre_value = form_values.get("pre_fault_percent", "")
            during_value = form_values.get("during_fault_percent", "")
            return f"Transient disturbance test in {quantity} from {pre_value}% to {during_value}%"
        if self.current_test_key == "ramp_test":
            quantity = form_values.get("ramp_quantity", "").lower()
            pre_value = form_values.get("pre_ramp_percent", "")
            post_value = form_values.get("post_ramp_percent", "")
            return f"Ramp test in {quantity} from {pre_value}% to {post_value}%"
        return "Comparative test"

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ModelComparisonApp().run()
