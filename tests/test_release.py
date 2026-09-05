from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib

import yaml

import biomed_ml_workflows


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", "output", "__pycache__", ".pytest_cache", "build", "dist"}


def public_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_PARTS or part.startswith(".venv") or part.endswith(".egg-info") for part in path.parts)
        and path.suffix != ".pyc"
    ]


def test_version_and_required_public_files() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert biomed_ml_workflows.__version__ == "0.1.1"
    assert configuration["project"]["version"] == "0.1.1"
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "0.1.1"
    required = (
        "README.md",
        "AGENTS.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "CITATION.cff",
        ".zenodo.json",
        "CHANGELOG.md",
        "RELEASE_NOTES_v0.1.1.md",
        "PUBLIC_RELEASE_STATUS.md",
        "docs/USER_GUIDE_ZH_CN.md",
        "docs/FIGURE_CAPABILITIES.md",
        "docs/CODEX_USAGE.md",
        "docs/REPRODUCIBILITY.md",
        "docs/KNOWN_LIMITATIONS.md",
    )
    assert all((ROOT / relative).is_file() for relative in required)


def test_document_links_resolve() -> None:
    missing: list[str] = []
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for document in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        for target in link_pattern.findall(document.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            path_text = target.split("#", 1)[0]
            if path_text and not (document.parent / path_text).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert missing == []


def test_citation_and_archive_metadata_are_valid() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["version"] == "0.1.1"
    assert citation["license"] == "MIT"
    assert citation["authors"] == [{
        "given-names": "xuankun",
        "family-names": "zheng",
        "orcid": "https://orcid.org/0009-0006-7036-7811",
    }]
    assert citation["doi"] == "10.5281/zenodo.22346004"

    archive = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert archive["version"] == "0.1.1"
    assert archive["license"] == "MIT"
    assert archive["upload_type"] == "software"
    assert archive["creators"] == [{"name": "zheng, xuankun", "orcid": "0009-0006-7036-7811"}]
    assert "doi" not in {str(key).lower() for key in archive}


def test_public_tree_contains_no_internal_trace_private_path_or_binary_asset() -> None:
    files = public_files()
    relative_names = [path.relative_to(ROOT).as_posix() for path in files]
    forbidden_directories = {
        "audit" + "_snapshot",
        "pro" + "venance",
        "plotting" + "_audit",
        "plotting" + "_migration",
        "showcase" + "_figure_pack",
    }
    assert not any(set(Path(name).parts) & forbidden_directories for name in relative_names)

    text_files = [path for path in files if path.suffix.lower() in {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".cff", ".txt"}]
    joined = "\n".join(path.read_text(encoding="utf-8") for path in text_files)
    historic_patterns = (
        re.compile(r"\bM\d{4}\b"),
        re.compile(r"\bA\d{4}\b"),
        re.compile(r"\bPLOT-[A-Z0-9-]+\b"),
        re.compile(r"Source\s+Archive", re.IGNORECASE),
    )
    assert all(pattern.search(joined) is None for pattern in historic_patterns)
    assert re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", joined) is None
    assert re.search(r"/(?:home|Users)/[^\s]+", joined) is None

    prohibited_suffixes = {
        ".dcm", ".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".h5", ".hdf5",
        ".pt", ".pth", ".ckpt", ".onnx", ".pkl", ".pickle", ".joblib",
    }
    assert not any(any(name.lower().endswith(suffix) for suffix in prohibited_suffixes) for name in relative_names)
    assert not any(path.stat().st_size > 2_000_000 for path in files)

    secret_signatures = (
        "-----BEGIN " + "PRIVATE KEY-----",
        "gh" + "p_",
        "sk-" + "proj-",
        "AKIA" + "[0-9A-Z]{16}",
    )
    assert secret_signatures[0] not in joined
    assert secret_signatures[1] not in joined
    assert secret_signatures[2] not in joined
    assert re.search(secret_signatures[3], joined) is None
