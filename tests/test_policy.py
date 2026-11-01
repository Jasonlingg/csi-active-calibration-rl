"""Check policy symmetries, legal actions, and safe persistence without training."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from active_calibration.env import ActiveCalibrationEnv  # noqa: E402
from active_calibration.policy import NeuralPolicy, PolicyConfig, SetActorCritic  # noqa: E402
from active_calibration.train import train_policy  # noqa: E402


def policy_and_observation(episode):
    torch.manual_seed(620)
    model = SetActorCritic(PolicyConfig(feature_dim=5, hidden_dim=16)).eval()
    env = ActiveCalibrationEnv([episode], budget=3, max_pool_size=9)
    env.reset(seed=91)
    observation, _, _, _, _ = env.step(2)
    return model, observation


def test_actor_is_permutation_equivariant_and_value_is_invariant(episode):
    """Reordering candidates must reorder the policy, without changing value."""
    model, observation = policy_and_observation(episode)
    permutation = np.random.default_rng(43).permutation(9)
    permuted = {
        key: value.copy() if key == "budget" else value[permutation].copy()
        for key, value in observation.items()
    }
    with torch.no_grad():
        original_logits, original_value = model(observation)
        permuted_logits, permuted_value = model(permuted)
    torch.testing.assert_close(
        permuted_logits, original_logits[permutation], rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(permuted_value, original_value, rtol=1e-6, atol=1e-7)


def test_queried_and_padded_candidates_have_no_probability_or_context(episode):
    """Padding cannot affect the legal distribution or the state-value estimate."""
    model, observation = policy_and_observation(episode)
    available = observation["action_mask"].astype(bool)
    padding = ~(available | observation["queried_mask"].astype(bool))
    assert padding.sum() == 3
    with torch.no_grad():
        logits, value = model(observation)
    assert torch.isneginf(logits[~available]).all()
    assert torch.isfinite(logits[available]).all()
    probabilities = logits.softmax(dim=0)
    assert torch.equal(probabilities[~available], torch.zeros_like(probabilities[~available]))
    assert probabilities.sum().item() == pytest.approx(1.0)

    corrupted_padding = {key: value.copy() for key, value in observation.items()}
    corrupted_padding["features"][padding] = 1e6
    corrupted_padding["scores"][padding] = -1e6
    compact = {
        key: value.copy() if key == "budget" else value[~padding].copy()
        for key, value in observation.items()
    }
    with torch.no_grad():
        corrupted_logits, corrupted_value = model(corrupted_padding)
        compact_logits, compact_value = model(compact)
    torch.testing.assert_close(corrupted_logits, logits, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(corrupted_value, value, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(compact_logits, logits[~padding], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(compact_value, value, rtol=1e-6, atol=1e-7)


def test_checkpoint_safely_round_trips_model_and_legal_action(episode, tmp_path, monkeypatch):
    model, observation = policy_and_observation(episode)
    checkpoint_path = tmp_path / "policy.pt"
    torch.save(model.checkpoint(), checkpoint_path)
    actual_load = torch.load
    load_options = []

    def audited_load(*args, **kwargs):
        load_options.append(kwargs.copy())
        return actual_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", audited_load)
    restored = NeuralPolicy(checkpoint_path)
    assert len(load_options) == 1
    assert load_options[0]["weights_only"] is True
    assert load_options[0]["map_location"] == "cpu"
    assert restored.model.config == model.config
    with torch.no_grad():
        expected_logits, expected_value = model(observation)
        actual_logits, actual_value = restored.model(observation)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    action = restored(observation)
    assert action == int(expected_logits.argmax())
    assert observation["action_mask"][action] == 1


@pytest.mark.parametrize("include_train", [False, True])
def test_trainer_refuses_evaluation_episodes_before_creating_env(episode, monkeypatch, include_train):
    """The trainer must reject even a mixed list before any episode is run."""
    import active_calibration.env as environment_module

    def forbidden_environment(*args, **kwargs):
        raise AssertionError("Evaluation data reached environment construction")

    monkeypatch.setattr(environment_module, "ActiveCalibrationEnv", forbidden_environment)
    episodes = ([episode] if include_train else []) + [replace(episode, split="eval")]
    with pytest.raises(ValueError, match="non-training split"):
        train_policy(episodes, episode_count=1, budget=2)
