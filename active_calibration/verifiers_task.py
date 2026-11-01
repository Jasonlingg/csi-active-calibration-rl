"""Prepare local CSI Harbor bundles and grader images for Verifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .data import TRAIN_RECEIVERS, default_project_root
from .harbor_task import build_task

VERIFIERS_REVISION = "c4e2f24a9ae0434ab55f5627c9086633b9391ad6"


def prepare_tasks(
    output: Path,
    *,
    split: str = "train",
    receivers=None,
    seed: int = 0,
    project_root: Path | str | None = None,
) -> Path:
    """Prepare one split into a new directory; build images locally, without pushing."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if split not in ("train", "eval"):
        raise ValueError("split must be 'train' or 'eval'")
    allowed = tuple(TRAIN_RECEIVERS) if split == "train" else tuple("ABCDEFG")
    selected = list(allowed if receivers is None else receivers)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(r not in allowed for r in selected)
    ):
        raise ValueError(f"{split} receivers must be distinct members of {allowed}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".csi-verifiers-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        entries = []
        for receiver in selected:
            name = f"csi-{split}-{receiver}-seed{seed}"
            task = build_task(
                staging / "tasks" / name, project_root, receiver, seed, split=split
            )
            image_file = Path(temporary) / "image-id.txt"
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--quiet",
                    "--iidfile",
                    str(image_file),
                    str(task / "tests"),
                ],
                check=True,
            )
            image = image_file.read_text().strip()
            if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
                raise ValueError("Docker did not return an immutable grader image ID")
            config_path = task / "task.toml"
            config_path.write_text(
                config_path.read_text().replace(
                    "[verifier.environment]\n",
                    f"[verifier.environment]\ndocker_image = {json.dumps(image)}\n",
                    1,
                )
            )
            manifest_path = task / "build_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["sha256"]["task.toml"] = hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest()
            manifest["verifiers"] = {
                "revision": VERIFIERS_REVISION,
                "verifier_image": image,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            entries.append(
                {
                    "path": str(task.relative_to(staging)),
                    "manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                }
            )
        root = (
            Path(project_root) if project_root is not None else default_project_root()
        )
        shutil.copy2(root / "provenance/README.md", staging / "PROVENANCE.md")
        (staging / "index.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "split": split,
                    "tasks": entries,
                    "verifiers_revision": VERIFIERS_REVISION,
                    "scope": "Local Docker integration; images have not been published",
                },
                indent=2,
            )
            + "\n"
        )
        staging.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--receivers", choices=tuple("ABCDEFG"), nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    print(
        prepare_tasks(
            args.out,
            split=args.split,
            receivers=args.receivers,
            seed=args.seed,
            project_root=args.project_root,
        )
    )


if __name__ == "__main__":
    main()
