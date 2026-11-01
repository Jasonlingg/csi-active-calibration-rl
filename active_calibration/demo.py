"""Watch a baseline or learned policy request activity labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import RECEIVERS, load_episodes
from .env import ActiveCalibrationEnv
from .evaluate import diversity_action, margin_action


ACTIVITIES = ("empty room", "jumping", "standing", "walking", "running",
              "squat", "clapping", "waving arms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", "--repo-root", dest="project_root", type=Path)
    parser.add_argument("--receiver", choices=RECEIVERS, default="F")
    parser.add_argument("--policy", choices=("random", "uncertainty", "diversity", "rl"),
                        default="uncertainty")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.policy == "rl" and args.checkpoint is None:
        parser.error("--policy rl requires --checkpoint")
    policy = {"uncertainty": margin_action, "diversity": diversity_action}.get(args.policy)
    if args.policy == "rl":
        from .policy import NeuralPolicy
        policy = NeuralPolicy(args.checkpoint)
    env = ActiveCalibrationEnv(
        load_episodes(args.project_root, split="eval", receivers=[args.receiver]),
        budget=args.budget,
    )
    observation, _ = env.reset(seed=args.seed, options={"episode_index": 0})
    rng = np.random.default_rng(args.seed)
    print(f"Receiver {args.receiver}; {args.policy}; {args.budget} label requests")
    for step in range(1, args.budget + 1):
        action = (policy(observation) if policy else
                  int(rng.choice(np.flatnonzero(observation["action_mask"]))))
        observation, reward, terminated, _, info = env.step(action)
        print(f"{step:2d}. Query row {action:2d}: {ACTIVITIES[info['label']]}")
        if terminated:
            print(f"Accuracy: {info['initial_accuracy']:.1%} -> {info['final_accuracy']:.1%}; "
                  f"terminal reward: {reward:+.4f}")
    env.close()


if __name__ == "__main__":
    main()
