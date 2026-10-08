"""Story I5: the documentation says what the code does."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import click
import pytest
import typer

import pv_geom
from pv_geom.cli import app
from pv_geom.sample import write_demo
from pv_geom.schema import output_schema

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guide"
PAGES = sorted(GUIDE.rglob("*.md"))
HAND_WRITTEN = [p for p in PAGES if p.parent.name != "reference"]


def test_generated_reference_pages_are_current() -> None:
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_docs.py"), "--check"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_page_is_in_the_navigation() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    missing = [str(p.relative_to(GUIDE).as_posix()) for p in PAGES
               if p.relative_to(GUIDE).as_posix() not in nav]
    assert not missing, f"pages not in mkdocs.yml: {missing}"


def test_internal_links_resolve() -> None:
    broken = []
    for page in PAGES:
        for target in re.findall(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)", page.read_text("utf-8")):
            if not (page.parent / target).resolve().exists():
                broken.append(f"{page.relative_to(GUIDE)} -> {target}")
    assert not broken, broken


def _commands() -> dict[str, set[str]]:
    group = typer.main.get_command(app)
    assert isinstance(group, click.Group)
    return {name: {opt for p in cmd.params if isinstance(p, click.Option)
                   for opt in [*p.opts, *p.secondary_opts]}
            for name, cmd in group.commands.items()}


def test_commands_and_options_in_the_pages_exist() -> None:
    commands = _commands()
    problems = []
    for page in HAND_WRITTEN:
        text = page.read_text(encoding="utf-8").replace("\\\n", " ")
        # Commands are checked where they are shown as commands: in code blocks.
        blocks = re.findall(r"```(?:bash)?\n(.*?)```", text, flags=re.S)
        for line in "\n".join(blocks).splitlines():
            m = re.match(r"\s*(?:uv run )?pv-geom ([a-z][a-z-]+)(.*)", line)
            if not m:
                continue
            name = m.group(1)
            if name not in commands:
                problems.append(f"{page.name}: unknown command '{name}'")
                continue
            for opt in re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", m.group(2)):
                if opt not in commands[name]:
                    problems.append(f"{page.name}: pv-geom {name} has no option {opt}")
    assert not problems, problems


def test_columns_named_in_the_pages_exist() -> None:
    known = set(output_schema(False).names) | {"region", "facet_index", "facet_id"}
    text = "\n".join(p.read_text(encoding="utf-8") for p in HAND_WRITTEN)
    named = set(re.findall(r"`((?:tilt|azimuth|fit|n|height|angle|grid|geometry|open|roof)"
                           r"_[a-z_0-9]+)`", text))
    assert not named - known, sorted(named - known)


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> pv_geom.RunResult:
    return pv_geom.run(write_demo(tmp_path_factory.mktemp("demo")), use_dask=False)


def test_the_tutorial_tells_the_truth(demo) -> None:
    rows = pv_geom.load(demo.output)
    shown = ["polygon_id", "status", "geometry_basis", "tilt_deg", "azimuth_deg",
             "tilt_unc_deg", "recommended"]
    assert set(shown) <= set(rows.columns)
    # "The first array was generated at a tilt of 20 degrees facing 180 degrees ...
    #  You should see about 20.00 and 180.03."
    assert rows.loc[0, "tilt_deg"] == pytest.approx(20.00, abs=0.02)
    assert rows.loc[0, "azimuth_deg"] == pytest.approx(180.03, abs=0.03)
    # The layout the tutorial draws.
    out = Path(demo.output)
    assert (out / "manifest.json").exists() and list(out.glob("part-*.parquet"))
    assert (out.parent / "demo.yaml").exists() and (out.parent / "inputs").is_dir()
    report = demo.report()
    for name in ("report.html", "report.md", "methods.md", "tables", "figures", "dataset"):
        assert (report.out_dir / name).exists(), name
    assert "n_fitted" in report.summary
