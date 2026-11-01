"""Fixed, receiver-agnostic downstream learner for label-query episodes.

The policy chooses which target examples to label; it cannot replace this
learner. Normalization uses the full, unlabeled query pool and stays fixed as
labels are acquired. Evaluation features are used only to compute predictions.
"""

from __future__ import annotations

import numpy as np

N_FFT_BINS = 16
N_ACF_LAGS = 8
FEATURE_DIM = 3 + N_FFT_BINS + N_ACF_LAGS
EPS = 1e-9


def features(windows: np.ndarray) -> np.ndarray:
    """Convert ``(examples, time, subcarriers)`` amplitudes to 27 features.

    At least 32 time positions are required so every window has the same 16
    non-DC Fourier bins. Inputs must be finite, with at least one subcarrier.
    Empty batches are supported.
    """
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 2 * N_FFT_BINS or x.shape[2] == 0:
        raise ValueError("windows must have shape (n, time >= 32, subcarriers > 0)")
    if not np.isfinite(x).all():
        raise ValueError("windows must contain only finite measurements")
    static = x.mean(axis=1, keepdims=True)
    dynamic = x - static
    envelope = np.sqrt((dynamic ** 2).mean(axis=2))

    level = np.log(np.maximum(x.mean(axis=(1, 2)), EPS))
    energy = np.log(np.maximum((dynamic ** 2).mean(axis=(1, 2)), EPS))
    jitter = envelope.std(axis=1) / np.maximum(envelope.mean(axis=1), EPS)

    centred = envelope - envelope.mean(axis=1, keepdims=True)
    spectrum = np.log(np.maximum(
        np.abs(np.fft.rfft(centred, axis=1))[:, 1:N_FFT_BINS + 1], EPS
    ))

    acf = np.empty((len(x), N_ACF_LAGS))
    for lag in range(1, N_ACF_LAGS + 1):
        a, b = centred[:, :-lag], centred[:, lag:]
        denom = np.maximum(
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), EPS
        )
        acf[:, lag - 1] = (a * b).sum(axis=1) / denom

    return np.column_stack([level, energy, jitter, spectrum, np.nan_to_num(acf)])


def _feature_matrix(value: np.ndarray, name: str, width: int | None = None,
                    *, allow_empty: bool = False) -> np.ndarray:
    x = np.array(value, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] == 0 or (not allow_empty and not len(x)):
        raise ValueError(f"{name} must be a nonempty 2-D feature matrix")
    if width is not None and x.shape[1] != width:
        raise ValueError(f"{name} has a different feature dimension")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must contain only finite features")
    return x


def _labels(value: np.ndarray, length: int, n_classes: int,
            name: str) -> np.ndarray:
    y = np.asarray(value)
    if y.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.issubdtype(y.dtype, np.number) or not np.isfinite(y).all():
        raise ValueError(f"{name} must contain integer class IDs")
    if not np.equal(y, np.floor(y)).all() or ((y < 0) | (y >= n_classes)).any():
        raise ValueError(f"{name} must contain class IDs from 0 to {n_classes - 1}")
    return np.array(y, dtype=np.int64, copy=True)


def _check_parameters(n_classes: int, gamma: float, ridge: float,
                      target_weight: float) -> None:
    if (not isinstance(n_classes, (int, np.integer)) or isinstance(n_classes, bool)
            or n_classes <= 0):
        raise ValueError("n_classes must be a positive integer")
    for name, value in (("gamma", gamma), ("ridge", ridge),
                        ("target_weight", target_weight)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    d2 = (np.square(a).sum(1)[:, None] + np.square(b).sum(1)[None, :]
          - 2.0 * a @ b.T)
    return np.exp(-gamma * np.maximum(d2, 0.0) / a.shape[1])


class KernelRidgeLearner:
    """Original weighted RBF kernel ridge classifier on one-hot labels.

    Target labels have weight eight by default. This implementation is also a
    simple reference against which the incremental episode learner is checked.
    """

    def __init__(self, n_classes: int = 8, gamma: float = 0.25,
                 ridge: float = 0.05, target_weight: float = 8.0) -> None:
        _check_parameters(n_classes, gamma, ridge, target_weight)
        self.n_classes = n_classes
        self.gamma = gamma
        self.ridge = ridge
        self.target_weight = target_weight
        self._fitted = False

    def fit(self, source_f: np.ndarray, source_y: np.ndarray,
            target_f: np.ndarray | None, target_y: np.ndarray | None,
            pool_f: np.ndarray) -> "KernelRidgeLearner":
        self._fitted = False
        source = _feature_matrix(source_f, "source_f")
        pool = _feature_matrix(pool_f, "pool_f", source.shape[1])
        self._mu = pool.mean(axis=0)
        self._sd = np.maximum(pool.std(axis=0), 1e-6)
        blocks = [self._z(source)]
        labels = [_labels(source_y, len(source), self.n_classes, "source_y")]
        weights = [np.ones(len(source))]
        if target_f is not None:
            target = _feature_matrix(target_f, "target_f", source.shape[1],
                                     allow_empty=True)
            if len(target):
                if target_y is None:
                    raise ValueError("target_y is required for target_f")
                blocks.append(self._z(target))
                labels.append(_labels(target_y, len(target), self.n_classes, "target_y"))
                weights.append(np.full(len(target), self.target_weight))
        elif target_y is not None and len(target_y):
            raise ValueError("target_f is required for target_y")

        self._train = np.vstack(blocks)
        y = np.concatenate(labels)
        w = np.concatenate(weights)
        onehot = np.eye(self.n_classes)[y]
        k = self._kernel(self._train, self._train)
        self._alpha = np.linalg.solve(k + np.diag(self.ridge / w), onehot)
        self._fitted = True
        return self

    def scores(self, f: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("learner is not fitted")
        x = _feature_matrix(f, "f", self._train.shape[1], allow_empty=True)
        return self._kernel(self._z(x), self._train) @ self._alpha

    def predict(self, f: np.ndarray) -> np.ndarray:
        return self.scores(f).argmax(axis=1)

    def _z(self, f: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(f, dtype=np.float64) - self._mu) / self._sd,
                       -8.0, 8.0)

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _kernel(a, b, self.gamma)


class PreparedLearner:
    """Exactly the same fit, with a small solve after each acquired label.

    Eliminating the fixed source block with a Schur complement leaves only a
    ``queries x queries`` system to solve. Construction uses source labels and
    unlabeled pool/evaluation features, never pool or evaluation labels.
    """

    def __init__(self, source_f: np.ndarray, source_y: np.ndarray,
                 pool_f: np.ndarray, eval_f: np.ndarray, n_classes: int = 8,
                 gamma: float = 0.25, ridge: float = 0.05,
                 target_weight: float = 8.0) -> None:
        _check_parameters(n_classes, gamma, ridge, target_weight)
        source = _feature_matrix(source_f, "source_f")
        pool = _feature_matrix(pool_f, "pool_f", source.shape[1])
        evaluation = _feature_matrix(eval_f, "eval_f", source.shape[1])
        source_y = _labels(source_y, len(source), n_classes, "source_y")
        self.n_classes = n_classes
        self.ridge = ridge
        self.target_weight = target_weight
        self.pool_size = len(pool)
        mu = pool.mean(axis=0)
        sd = np.maximum(pool.std(axis=0), 1e-6)
        source = np.clip((source - mu) / sd, -8.0, 8.0)
        pool = np.clip((pool - mu) / sd, -8.0, 8.0)
        evaluation = np.clip((evaluation - mu) / sd, -8.0, 8.0)
        self.pool_features = pool
        self.pool_features.setflags(write=False)

        kss = _kernel(source, source, gamma)
        ksp = _kernel(source, pool, gamma)
        kes = _kernel(evaluation, source, gamma)
        rhs = np.column_stack([np.eye(n_classes)[source_y], ksp])
        solved = np.linalg.solve(kss + ridge * np.eye(len(source)), rhs)
        alpha_source = solved[:, :n_classes]
        source_to_pool = solved[:, n_classes:]
        self._base_pool = ksp.T @ alpha_source
        self._base_eval = kes @ alpha_source
        conditional_pool = _kernel(pool, pool, gamma) - ksp.T @ source_to_pool
        self._conditional_pool = (conditional_pool + conditional_pool.T) / 2.0
        self._conditional_eval = (
            _kernel(evaluation, pool, gamma) - kes @ source_to_pool
        )

    def scores(self, selected_indices: np.ndarray | list[int],
               selected_y: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return query-pool and reward-set scores in their original row order."""
        raw_indices = np.asarray(selected_indices)
        if raw_indices.ndim != 1:
            raise ValueError("selected_indices must be one-dimensional")
        if len(raw_indices) and (not np.issubdtype(raw_indices.dtype, np.integer)
                                 or np.issubdtype(raw_indices.dtype, np.bool_)):
            raise ValueError("selected_indices must contain integer indices")
        indices = raw_indices.astype(np.int64)
        if ((indices < 0) | (indices >= self.pool_size)).any():
            raise ValueError("selected index is outside the query pool")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("selected_indices must not contain duplicates")
        y = _labels(np.asarray(selected_y), len(indices), self.n_classes, "selected_y")
        if not len(indices):
            return self._base_pool.copy(), self._base_eval.copy()
        conditional = self._conditional_pool[np.ix_(indices, indices)]
        system = conditional + (self.ridge / self.target_weight) * np.eye(len(indices))
        residual = np.eye(self.n_classes)[y] - self._base_pool[indices]
        alpha_target = np.linalg.solve(system, residual)
        return (
            self._base_pool + self._conditional_pool[:, indices] @ alpha_target,
            self._base_eval + self._conditional_eval[:, indices] @ alpha_target,
        )
