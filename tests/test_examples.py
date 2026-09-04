from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.figure_gallery import run_all_figures as gallery
from examples.quick_start.run_quick_start import run


def test_quick_start_runs_all_workflows(tmp_path: Path) -> None:
    result = run(tmp_path / "quick")
    assert result["status"] == "SYNTHETIC_TECHNICAL_DEMONSTRATION_ONLY"
    assert result["scientific_claim"] == "NONE"
    assert set(result) >= {"classification", "segmentation", "survival", "environment"}
    for directory in ("metrics", "predictions", "figures", "logs"):
        assert (tmp_path / "quick" / directory).is_dir()
    loaded = json.loads((tmp_path / "quick" / "logs" / "quick_start.json").read_text(encoding="utf-8"))
    assert loaded["status"] == result["status"]
    assert len(list((tmp_path / "quick" / "figures").glob("*.png"))) == 3
    plt.close("all")


def test_complete_gallery_generates_and_reopens_52_pngs(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "gallery"
    monkeypatch.setattr(gallery, "OUTPUT_ROOT", destination)
    generated = gallery.generate_pack()
    assert len(generated) == 52
    assert len({path.name for path in generated}) == 52
    assert all(path.is_file() and path.stat().st_size > 1000 for path in generated)
    assert {path.parent.name for path in generated} == {
        "statistical",
        "classification",
        "segmentation",
        "survival",
        "interpretation",
    }
