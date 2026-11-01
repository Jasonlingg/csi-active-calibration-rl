"""Train the small query policy on training events only.

Terminal-return REINFORCE uses a learned state-value baseline, an entropy
bonus, and small batches of complete episodes. The learner and episode
reward are owned by the environment; neither is optimized here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from .policy import PolicyConfig, SetActorCritic


def train_policy(
    episodes: Sequence[Any], *, episode_count: int = 400, seed: int = 0,
    budget: int = 16, hidden_dim: int = 64, learning_rate: float = 3e-4,
    batch_episodes: int = 8, entropy_coefficient: float = 0.005,
    value_coefficient: float = 0.5, log_every: int = 50,
) -> tuple[SetActorCritic, list[dict[str, Any]]]:
    """Train on supplied ``split='train'`` episodes, without evaluation reads."""
    from .env import ActiveCalibrationEnv

    if not episodes:
        raise ValueError("at least one training episode is required")
    if any(episode.split != "train" for episode in episodes):
        raise ValueError("training refuses episodes from a non-training split")
    if min(episode_count, budget, hidden_dim, batch_episodes, log_every) < 1:
        raise ValueError("episode count, budget, model size, batch size and log interval must be positive")
    if learning_rate <= 0 or min(entropy_coefficient, value_coefficient) < 0:
        raise ValueError("learning rate must be positive and loss coefficients nonnegative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    env = ActiveCalibrationEnv(episodes, budget=budget)
    observation, _ = env.reset(seed=seed, options={"episode_index": 0})
    model = SetActorCritic(PolicyConfig(
        feature_dim=int(observation["features"].shape[1]),
        n_classes=int(observation["scores"].shape[1]), hidden_dim=hidden_dim,
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    pending_episodes: list[tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], float]] = []
    started = time.perf_counter()

    for number in range(1, episode_count + 1):
        episode_index = int(rng.integers(len(episodes)))
        observation, _ = env.reset(seed=seed + number, options={"episode_index": episode_index})
        log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        episode_return = 0.0
        actions: list[int] = []

        while True:
            logits, value = model(observation)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_probabilities.append(distribution.log_prob(action))
            values.append(value)
            entropies.append(distribution.entropy())
            action_number = int(action.item())
            actions.append(action_number)
            observation, reward, terminated, truncated, _ = env.step(action_number)
            episode_return += float(reward)
            if terminated or truncated:
                break
        if truncated:
            raise RuntimeError("training requires complete episodes, not truncated returns")
        pending_episodes.append((log_probabilities, values, entropies, episode_return))
        history.append({
            "episode": number,
            "episode_index": episode_index,
            "receiver": str(episodes[episode_index].receiver),
            "return": episode_return,
            "query_count": len(actions),
            "actions": actions,
        })

        if len(pending_episodes) == batch_episodes or number == episode_count:
            actor_losses, value_losses, entropy_terms = [], [], []
            for log_probs, state_values, state_entropies, final_return in pending_episodes:
                stacked_values = torch.stack(state_values)
                targets = torch.full_like(stacked_values, final_return)
                advantages = targets - stacked_values.detach()
                actor_losses.append(-(torch.stack(log_probs) * advantages).mean())
                value_losses.append(F.mse_loss(stacked_values, targets))
                entropy_terms.append(torch.stack(state_entropies).mean())
            actor_loss = torch.stack(actor_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy = torch.stack(entropy_terms).mean()
            loss = actor_loss + value_coefficient * value_loss - entropy_coefficient * entropy
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("training produced a non-finite gradient")
            optimizer.step()
            history[-1]["update"] = {
                "actor_loss": float(actor_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "gradient_norm": float(gradient_norm),
            }
            pending_episodes.clear()

        if number % log_every == 0 or number == episode_count:
            recent = history[-min(log_every, number):]
            mean_return = float(np.mean([row["return"] for row in recent]))
            print(f"episode {number}/{episode_count} train_mean_return={mean_return:.5f} "
                  f"elapsed={time.perf_counter() - started:.1f}s", flush=True)

    model.eval()
    return model, history


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=400, help="number of training episodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--project-root", "--repo-root", dest="project_root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="new output directory")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args(argv)
    output = args.out or Path(__file__).resolve().parents[1] / "runs" / f"train-seed-{args.seed}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"training output must be a new or empty directory: {output}")

    # Intentionally load only training receivers and training physical events.
    from .data import load_episodes
    episodes = load_episodes(project_root=args.project_root, split="train")
    model, history = train_policy(
        episodes, episode_count=args.episodes, seed=args.seed, budget=args.budget,
        hidden_dim=args.hidden_dim, learning_rate=args.learning_rate,
        batch_episodes=args.batch_episodes, entropy_coefficient=args.entropy_coefficient,
        value_coefficient=args.value_coefficient, log_every=args.log_every,
    )
    configuration = {
        "algorithm": "terminal-return actor-critic REINFORCE",
        "split": "train", "receivers": [str(episode.receiver) for episode in episodes],
        "episode_count": args.episodes, "seed": args.seed, "budget": args.budget,
        "learning_rate": args.learning_rate, "batch_episodes": args.batch_episodes,
        "entropy_coefficient": args.entropy_coefficient,
        "value_coefficient": args.value_coefficient, "model": asdict(model.config),
        "project_root": str(args.project_root.resolve()) if args.project_root else None,
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "training_events": [{
            "receiver": str(episode.receiver),
            "pool_event_ids": np.asarray(episode.pool_event_ids).tolist(),
            "reward_event_ids": np.asarray(episode.eval_event_ids).tolist(),
        } for episode in episodes],
    }
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = model.checkpoint()
    checkpoint["training_config"] = configuration
    torch.save(checkpoint, output / "policy.pt")
    (output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    (output / "learning_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"saved policy and training history to {output}", flush=True)


if __name__ == "__main__":
    main()
