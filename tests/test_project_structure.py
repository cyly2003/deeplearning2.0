from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_authoritative_project_documents_exist() -> None:
    required_files = [
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "docs" / "统一多任务残差QSAR架构说明.md",
    ]

    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_files if not path.is_file()]
    assert not missing, (
        "Missing authoritative project documents: "
        + ", ".join(missing)
        + ". Read these first before adding code or experiments."
    )


def test_curated_sqlite_entrypoint_exists() -> None:
    sqlite_path = PROJECT_ROOT / "ecotox_clean.sqlite"
    assert sqlite_path.is_file(), (
        "Expected curated SQLite entrypoint at 'ecotox_clean.sqlite' as defined by the project architecture."
    )
    assert sqlite_path.stat().st_size > 0, "The curated SQLite entrypoint exists but is empty."


def test_architecture_assets_directory_is_present() -> None:
    assets_dir = PROJECT_ROOT / "docs" / "assets"
    assert assets_dir.is_dir(), "Expected 'docs/assets' to store architecture and manuscript-ready figure assets."

    svg_assets = list(assets_dir.glob("*.svg"))
    assert svg_assets, "Expected at least one SVG architecture asset under 'docs/assets'."


def test_scaffold_contract_tests_are_checked_in() -> None:
    expected_tests = [
        PROJECT_ROOT / "tests" / "test_project_structure.py",
        PROJECT_ROOT / "tests" / "test_configs.py",
    ]

    missing = [str(path.relative_to(PROJECT_ROOT)) for path in expected_tests if not path.is_file()]
    assert not missing, (
        "Missing scaffold contract tests: "
        + ", ".join(missing)
        + ". Keep these tests in-tree so new workers can validate the repository skeleton quickly."
    )
