"""Compare query strategies on the same development evaluation episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Callable

import gymnasium
import numpy as np

from .data import EXPECTED_SHA256, load_episodes
from .env import ActiveCalibrationEnv


Observation = dict[str, np.ndarray]


def margin_action(observation: Observation) -> int:
    """Query the smallest gap between the classifier's two largest scores."""
    scores = np.sort(observation["scores"], axis=1)
    margin = scores[:, -1] - scores[:, -2]
    return int(np.argmin(np.where(observation["action_mask"], margin, np.inf)))


def diversity_action(observation: Observation) -> int:
    """Start nearest the pool centroid, then maximize distance to queried rows."""
    x = observation["features"].astype(np.float64)
    available = observation["action_mask"].astype(bool)
    queried = observation["queried_mask"].astype(bool)
    valid = available | queried
    if not queried.any():
        distance = ((x - x[valid].mean(axis=0)) ** 2).sum(axis=1)
        return int(np.argmin(np.where(available, distance, np.inf)))
    distance = ((x[:, None, :] - x[queried][None, :, :]) ** 2).sum(axis=2).min(axis=1)
    return int(np.argmax(np.where(available, distance, -np.inf)))


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def evaluate(project_root: Path | str | None = None, budget: int = 16,
             seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
             checkpoint: Path | None = None) -> dict:
    episodes = load_episodes(project_root, split="eval")
    env = ActiveCalibrationEnv(episodes, budget=budget)
    policies: dict[str, Callable[[Observation], int] | None] = {
        "random": None, "uncertainty": margin_action, "diversity": diversity_action,
    }
    if checkpoint is not None:
        from .policy import NeuralPolicy
        policies["rl"] = NeuralPolicy(checkpoint)
    records = []
    for name, policy in policies.items():
        for episode_index, episode in enumerate(episodes):
            for seed in seeds:
                observation, _ = env.reset(seed=seed, options={"episode_index": episode_index})
                rng = np.random.default_rng(seed)
                actions, labels = [], []
                terminated = False
                while not terminated:
                    if policy is None:
                        action = int(rng.choice(np.flatnonzero(observation["action_mask"])))
                    else:
                        action = int(policy(observation))
                    observation, reward, terminated, truncated, info = env.step(action)
                    if truncated:
                        raise RuntimeError("Unexpected truncated episode")
                    actions.append(action)
                    labels.append(info["label"])
                records.append({
                    "policy": name, "receiver": episode.receiver, "seed": seed,
                    "reward_examples": len(episode.eval_y),
                    "initial_accuracy": info["initial_accuracy"],
                    "final_accuracy": info["final_accuracy"], "reward": reward,
                    "query_indices": actions, "acquired_labels": labels,
                })
    summary = {}
    for name in policies:
        selected = [row for row in records if row["policy"] == name]
        groups = {"all_receivers": selected,
                  "seen_receivers": [row for row in selected if row["receiver"] in "ABCDE"],
                  "unseen_receivers": [row for row in selected if row["receiver"] in "FG"]}
        entry = {}
        for group, rows in groups.items():
            entry[group] = {
                "source_accuracy": float(np.mean([row["initial_accuracy"] for row in rows])),
                "final_accuracy": _mean_std([row["final_accuracy"] for row in rows]),
                "accuracy_gain": _mean_std([row["reward"] for row in rows]),
            }
        entry["per_receiver"] = {
            episode.receiver: {
                "source_accuracy": next(row["initial_accuracy"] for row in selected
                                        if row["receiver"] == episode.receiver),
                "final_accuracy": _mean_std([row["final_accuracy"] for row in selected
                                            if row["receiver"] == episode.receiver]),
            }
            for episode in episodes
        }
        # These are paired seed repeats, not additional independent recordings.
        entry["seed_repeat_mean_accuracy"] = {
            str(seed): float(np.mean([row["final_accuracy"] for row in selected
                                     if row["seed"] == seed])) for seed in seeds
        }
        summary[name] = entry
    env.close()
    result = {
        "protocol": "Development evaluation, 16 of 24 labels by default; terminal accuracy gain",
        "budget": budget, "seeds": list(seeds),
        "unique_reward_events": len(set().union(*(set(ep.eval_event_ids) for ep in episodes))),
        "receiver_event_pairs": sum(len(ep.eval_y) for ep in episodes),
        "limitations": [
            "Receivers observe synchronized physical events; do not treat them as independent participants or rooms.",
            "Seeds repeat the same finite recordings; standard deviations are descriptive, not confidence intervals.",
            "These recordings were already used in author development; this is not a prospective scientific holdout.",
        ],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "gymnasium": gymnasium.__version__},
        "fixture_sha256": EXPECTED_SHA256,
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
        "summary": summary, "episodes": records,
    }
    if checkpoint is not None:
        import torch
        result["runtime"]["torch"] = torch.__version__
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", "--repo-root", dest="project_root", type=Path)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("provide distinct evaluation seeds")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = evaluate(args.project_root, args.budget, tuple(args.seeds), args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("policy          source      final       gain")
    for policy, values in result["summary"].items():
        all_receivers = values["all_receivers"]
        print(f"{policy:14s} {all_receivers['source_accuracy']:7.2%}  "
              f"{all_receivers['final_accuracy']['mean']:8.2%}  "
              f"{all_receivers['accuracy_gain']['mean']:+8.2%}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
