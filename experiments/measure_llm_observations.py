"""CPU-only observation-size audit, not an agent runner or learning experiment.

Run from the project root after installing the project. Optional token counts
require `tokenizers` and an explicitly supplied tokenizer.json; this script
does not download anything, call a model, or reconstruct training token IDs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from active_calibration.harbor_client import compact_state
from active_calibration.harbor_grader import grade
from active_calibration.harbor_service import CalibrationService
from active_calibration.harbor_task import build_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    results = []
    # Compare identical action sequences: formatting must not change rewards.
    # This uses full-precision margin choices, not decisions made by an LLM.
    with tempfile.TemporaryDirectory(prefix="csi-observation-audit-") as directory:
        root = Path(directory)
        for split, receivers in (("train", "ABCDE"), ("eval", "ABCDEFG")):
            for receiver in receivers:
                task = build_task(root / f"{split}-{receiver}", receiver=receiver, split=split)
                bundle = task / "environment/calibration/episode.npz"
                history = task / "journal.json"
                service = CalibrationService(bundle, history)
                state = service.state()
                texts = {name: [] for name in ("full", "minified_full", "compact")}
                actions = []
                while True:
                    texts["full"].append(json.dumps(state, indent=2, allow_nan=False) + "\n")
                    texts["minified_full"].append(json.dumps(state, separators=(",", ":"), allow_nan=False) + "\n")
                    texts["compact"].append(json.dumps(
                        compact_state(state, include_features=not actions),
                        separators=(",", ":"), allow_nan=False,
                    ) + "\n")
                    if state["terminated"]:
                        break
                    observation = state["observation"]
                    def margin(index):
                        scores = sorted(observation["scores"][index])
                        return scores[-1] - scores[-2], index
                    action = min(
                        (i for i, valid in enumerate(observation["action_mask"]) if valid),
                        key=margin,
                    )
                    actions.append(action)
                    state = service.query(action)
                report = grade(bundle, history)
                assert report["valid"] and report["queries"] == 16
                assert report["reward"] == state["reward"]
                sizes = {}
                for name, responses in texts.items():
                    size = {"bytes": sum(len(t.encode()) for t in responses)}
                    if tokenizer:
                        counts = [len(tokenizer.encode(t, add_special_tokens=False).ids) for t in responses]
                        size["text_tokens"] = sum(counts)
                        # Observation text only; repeated inputs during sequential
                        # decisions also contain all previous observations.
                        size["cumulative_text_input_tokens"] = sum(
                            sum(counts[:i + 1]) for i in range(len(counts))
                        )
                    sizes[name] = size
                results.append({
                    "split": split, "receiver": receiver, "seed": 0,
                    "actions": actions, "report": report, "sizes": sizes,
                    "journal": json.loads(history.read_text()),
                })
    output = {
        "scope": "CPU formatting audit on identical full-precision margin actions; no LLM calls or learning",
        "excludes": "prompts, chat templates, tool schemas, model outputs; counts are not training capture",
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() if args.tokenizer else None,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path("active_calibration/harbor_client.py"),
                         Path("active_calibration/harbor_task.py"))
        },
        "episodes": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name in ("full", "minified_full", "compact"):
        sizes = [r["sizes"][name] for r in results]
        print(name, {key: [min(s[key] for s in sizes), max(s[key] for s in sizes)] for key in sizes[0]})


if __name__ == "__main__":
    main()
