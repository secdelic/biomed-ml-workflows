from __future__ import annotations

import ast
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from biomed_ml_workflows import figures
from examples.figure_gallery.run_all_figures import build_demo_cases


ROOT = Path(__file__).resolve().parents[1]


def assert_unchanged(before: object, after: object) -> None:
    if isinstance(before, np.ndarray):
        assert isinstance(after, np.ndarray)
        np.testing.assert_array_equal(before, after)
    elif isinstance(before, dict):
        assert set(before) == set(after)
        for key in before:
            assert_unchanged(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after)
        for first, second in zip(before, after, strict=True):
            assert_unchanged(first, second)
    else:
        assert before == after


def test_registry_covers_all_52_public_functions_once() -> None:
    cases = build_demo_cases()
    names = [case.name for case in cases]
    assert len(figures.__all__) == 52
    assert len(names) == len(set(names)) == 52
    assert set(figures.__all__) == set(names)


def test_every_function_returns_a_figure_without_mutation() -> None:
    for case in build_demo_cases():
        before_args = deepcopy(case.args)
        before_kwargs = deepcopy(case.kwargs)
        result = case.function(*case.args, **case.kwargs)
        assert isinstance(result, tuple)
        assert isinstance(result[0], Figure)
        assert result[0].axes
        assert_unchanged(before_args, case.args)
        assert_unchanged(before_kwargs, case.kwargs)
        plt.close(result[0])


def test_every_function_rejects_an_empty_primary_input() -> None:
    for case in build_demo_cases():
        invalid_args = (np.asarray([]), *case.args[1:])
        try:
            case.function(*invalid_args, **case.kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{case.name} accepted an empty primary input")


def test_headless_serialization() -> None:
    case = build_demo_cases()[0]
    figure, _ = case.function(*case.args, **case.kwargs)
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    assert len(buffer.getvalue()) > 1000
    plt.close(figure)


def test_plotting_modules_have_no_io_network_training_or_save_calls() -> None:
    prohibited: list[str] = []
    for path in (ROOT / "biomed_ml_workflows" / "figures").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name in {"open", "savefig", "fit", "fit_transform", "train", "urlopen", "post"}:
                prohibited.append(f"{path.name}:{node.lineno}:{name}")
    assert prohibited == []


def test_visual_contracts_and_known_binary_curves() -> None:
    cases = build_demo_cases()
    for function_name in ("plot_bar", "plot_class_distribution_bar"):
        case = next(item for item in cases if item.name == function_name)
        _, ax = case.function(*case.args, **case.kwargs)
        assert ax.get_axisbelow()

    panel = next(item for item in cases if item.name == "plot_image_mask_prediction")
    _, axes = panel.function(*panel.args, **panel.kwargs)
    assert np.asarray(axes).size == 3

    with np.testing.assert_raises(ValueError):
        figures.plot_survival_curves([0, 1, 2], np.ones((2, 2)))

    observed = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    _, roc_ax = figures.plot_roc_curve(observed, scores, class_names=["Positive"])
    assert roc_ax.lines[0].get_label() == "Positive (AUC=0.750)"
    np.testing.assert_allclose(roc_ax.lines[0].get_xdata(), [0, 0, 0.5, 0.5, 1])
    np.testing.assert_allclose(roc_ax.lines[0].get_ydata(), [0, 0.5, 0.5, 1, 1])
    _, pr_ax = figures.plot_precision_recall_curve(observed, scores, class_names=["Positive"])
    assert pr_ax.lines[0].get_xdata()[0] == 0
    assert pr_ax.lines[0].get_ydata()[0] == 1
    assert pr_ax.lines[0].get_xdata()[-1] == 1
    plt.close("all")
