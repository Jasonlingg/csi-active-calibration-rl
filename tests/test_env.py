"""Check the decision problem, reward, and separation of hidden labels."""

from dataclasses import replace

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from active_calibration.env import ActiveCalibrationEnv
from active_calibration.learner import KernelRidgeLearner
from conftest import assert_observations_equal


def test_gymnasium_contract(episode):
    check_env(ActiveCalibrationEnv([episode], budget=3), skip_render_check=True)


def test_reward_is_terminal_accuracy_delta(episode):
    """Buying every pool label must score the resulting full fit, with no shaping."""
    baseline = KernelRidgeLearner().fit(
        episode.source_f, episode.source_y, None, None, episode.pool_f
    )
    final = KernelRidgeLearner().fit(
        episode.source_f,
        episode.source_y,
        episode.pool_f,
        episode.pool_y,
        episode.pool_f,
    )
    initial_accuracy = np.mean(baseline.predict(episode.eval_f) == episode.eval_y)
    final_accuracy = np.mean(final.predict(episode.eval_f) == episode.eval_y)
    env = ActiveCalibrationEnv([episode], budget=len(episode.pool_f))
    env.reset(seed=20)
    rewards = []
    for action in range(len(episode.pool_f)):
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        assert terminated == (action == len(episode.pool_f) - 1)
        assert not truncated
    assert rewards[:-1] == [0.0] * (len(episode.pool_f) - 1)
    assert rewards[-1] == pytest.approx(final_accuracy - initial_accuracy)
    assert info["initial_accuracy"] == pytest.approx(initial_accuracy)
    assert info["final_accuracy"] == pytest.approx(final_accuracy)
    assert info["accuracy_gain"] == pytest.approx(rewards[-1])
    assert info["queries"] == len(episode.pool_f)


def test_hidden_evaluation_labels_change_only_terminal_reward(episode):
    """Opposite hidden answers must produce identical public query trajectories."""
    # Source predicts 0; nearby, acquired target examples teach class 1.
    # This gives an observable reward change, unlike an unchanged classifier.
    common = replace(
        episode,
        source_f=np.full((8, 5), -10.0),
        source_y=np.zeros(8, dtype=np.int64),
        pool_f=np.repeat(np.arange(6)[:, None], 5, axis=1).astype(float),
        pool_y=np.ones(6, dtype=np.int64),
        eval_f=np.zeros((7, 5)),
    )
    zero = ActiveCalibrationEnv(
        [replace(common, eval_y=np.zeros(7, dtype=np.int64))], budget=3
    )
    one = ActiveCalibrationEnv(
        [replace(common, eval_y=np.ones(7, dtype=np.int64))], budget=3
    )
    obs_zero, info_zero = zero.reset(seed=19)
    obs_one, info_one = one.reset(seed=19)
    assert_observations_equal(obs_zero, obs_one)
    assert info_zero == info_one
    for action in range(3):
        result_zero = zero.step(action)
        result_one = one.step(action)
        assert_observations_equal(result_zero[0], result_one[0])
        assert result_zero[2:4] == result_one[2:4]
        if action < 2:
            assert result_zero[1] == result_one[1] == 0.0
            assert result_zero[4] == result_one[4]
            assert set(result_zero[4]) <= {"query_index", "label"}
        else:
            assert result_zero[1] == pytest.approx(-1.0)
            assert result_one[1] == pytest.approx(1.0)


def test_unrequested_pool_labels_are_masked(episode):
    env = ActiveCalibrationEnv([episode], budget=3, max_pool_size=9)
    obs, info = env.reset(seed=41)
    assert not info
    assert set(obs) == {
        "features", "scores", "labels", "action_mask", "queried_mask", "budget"
    }
    assert np.all(obs["labels"] == -1)
    for step, action in enumerate([2, 0, 4], start=1):
        obs, _, _, _, info = env.step(action)
        queried = obs["queried_mask"].astype(bool)
        assert queried.sum() == step
        assert np.all(obs["labels"][~queried] == -1)
        assert obs["labels"][action] == info.get("label", obs["labels"][action])
        assert np.all((obs["labels"][queried] >= 0) & (obs["labels"][queried] < 8))
        assert not np.any(obs["action_mask"][queried])
        assert np.all(obs["features"][6:] == 0)
        assert np.all(obs["scores"][6:] == 0)
        assert not np.any(obs["action_mask"][6:])
        assert obs["budget"][0] == pytest.approx((3 - step) / 3)


def test_unrequested_pool_labels_do_not_influence_scores(episode):
    """Changing labels that have not been bought must not affect the policy."""
    pool_f = np.repeat(np.arange(6)[:, None], 5, axis=1).astype(float)
    common = replace(episode, pool_f=pool_f, pool_y=np.zeros(6, dtype=np.int64))
    changed_y = np.ones(6, dtype=np.int64)
    changed_y[0] = 0
    original = ActiveCalibrationEnv([common], budget=3)
    changed = ActiveCalibrationEnv([replace(common, pool_y=changed_y)], budget=3)
    first, _ = original.reset(seed=16)
    second, _ = changed.reset(seed=16)
    assert_observations_equal(first, second)
    # Standardizing preserves the ordering of this deliberately monotonic pool.
    actions = np.argsort(first["features"][:, 0])
    original_step = original.step(int(actions[0]))
    changed_step = changed.step(int(actions[0]))
    assert_observations_equal(original_step[0], changed_step[0])
    assert original_step[1:] == changed_step[1:]
    assert original_step[4]["label"] == 0
    assert original.step(int(actions[1]))[4]["label"] == 0
    assert changed.step(int(actions[1]))[4]["label"] == 1


@pytest.mark.parametrize("bad_action", [1, 6, -1, 9])
def test_invalid_actions_do_not_spend_budget_or_change_fit(episode, bad_action):
    """A rejected query must leave the next valid transition unchanged."""
    reference = ActiveCalibrationEnv([episode], budget=3, max_pool_size=9)
    tested = ActiveCalibrationEnv([episode], budget=3, max_pool_size=9)
    reference.reset(seed=4)
    tested.reset(seed=4)
    reference.step(1)
    tested.step(1)
    with pytest.raises((ValueError, IndexError)):
        tested.step(bad_action)
    assert np.array_equal(reference.action_masks(), tested.action_masks())
    expected = reference.step(2)
    actual = tested.step(2)
    assert_observations_equal(expected[0], actual[0])
    assert expected[1:] == actual[1:]


def test_seed_reproduces_episode_order_and_queries(episode):
    env = ActiveCalibrationEnv([episode], budget=2)
    first_obs, first_info = env.reset(seed=123)
    first_step = env.step(0)
    second_obs, second_info = env.reset(seed=123)
    second_step = env.step(0)
    assert_observations_equal(first_obs, second_obs)
    assert first_info == second_info
    assert_observations_equal(first_step[0], second_step[0])
    assert first_step[1:] == second_step[1:]
    orders = {
        env.reset(seed=seed)[0]["features"].tobytes() for seed in range(5)
    }
    assert len(orders) > 1, "Reset must shuffle away the original pool row order"


def test_action_masks_are_copies_and_exhausted_episode_is_closed(episode):
    env = ActiveCalibrationEnv([episode], budget=1)
    obs, _ = env.reset(seed=6)
    mask = env.action_masks()
    mask[:] = False
    assert env.action_masks().any()
    obs["action_mask"][:] = 0
    assert env.action_masks().any()
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated
    assert not env.action_masks().any()
    with pytest.raises(RuntimeError):
        env.step(1)
