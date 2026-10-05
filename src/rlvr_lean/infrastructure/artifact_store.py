"""JSONL artifact store for resumable phases (spec §7, "Resumable phases").

One directory per run (`<store>/runs/<profile>/`). A phase writes its rows with `write_rows` and then marks
itself done with a small summary; a rerun sees the marker and skips the phase. Rows are keyed by stable,
content-derived ids chosen by the caller, so nothing depends on write order.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable


class ArtifactStore:
    def __init__(self, root: Path, mirror_directory: Path | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        # A job runner collects a task's out/ directory; every JSONL written here is copied there too, so a
        # durable copy exists off the GPU box even if its container is re-created.
        self.mirror_directory = mirror_directory

    def path(self, name: str) -> Path:
        return self.root / name

    def write_rows(self, name: str, rows: Iterable[dict[str, Any]]) -> int:
        temporary = self.path(name + ".tmp")
        count = 0
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        temporary.replace(self.path(name))
        if self.mirror_directory is not None:
            self.mirror_directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.path(name), self.mirror_directory / name)
        return count

    def mirror(self, name: str) -> bool:
        """Copy a file that is already in the store to the mirror again. A rerun is another task with an output
        directory of its own: what an earlier task's step stored must reach it too, or it is never
        collected (a task that fails delivers no step files, and its rerun skips the steps that are done)."""
        if self.mirror_directory is None or not self.path(name).exists():
            return False
        self.mirror_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path(name), self.mirror_directory / name)
        return True

    def read_rows(self, name: str) -> list[dict[str, Any]]:
        path = self.path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist; has the phase that writes it run?")
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def is_done(self, phase: str) -> bool:
        return self.path(f"{phase}.done.json").exists()

    def mark_done(self, phase: str, summary: dict[str, Any]) -> None:
        self.path(f"{phase}.done.json").write_text(json.dumps(summary, indent=2, default=str))
        if self.mirror_directory is not None:
            shutil.copy2(self.path(f"{phase}.done.json"), self.mirror_directory / f"{phase}.done.json")

    def done_summary(self, phase: str) -> dict[str, Any]:
        return json.loads(self.path(f"{phase}.done.json").read_text())
