# src/fds_pokemon/utils.py
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation

from .features import ratio


def sweep_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    lo: float = 0.2,
    hi: float = 0.8,
    step: float = 0.0005,
) -> Tuple[float, float]:
    """Return (best_accuracy, best_threshold) on OOF."""
    thr_grid = np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)
    best_thr, best_acc = 0.5, 0.0
    y_true_bin = y_true.astype(int)
    for thr in thr_grid:
        acc = accuracy_score(y_true_bin, (probs >= thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_acc, best_thr


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Convert scores to [0,1] via rank; handy for blending."""
    return (rankdata(x, method="average") - 1) / (len(x) - 1 + 1e-12)


def drop_flawed_row(train_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop known-bad row 4877, robust to index/position."""
    if 4877 in train_df.index:
        return train_df.drop(index=4877)
    if len(train_df) > 4877:
        return train_df.drop(train_df.index[4877])
    return train_df


def prune_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Remove constant & exact-duplicate columns using train only; mirror to test."""
    feature_cols = [c for c in train_df.columns if c != target_col]
    const_cols = [c for c in feature_cols if train_df[c].std(ddof=0) == 0.0]
    dup_cols: List[str] = []
    seen = {}
    for c in feature_cols:
        if c in const_cols:
            continue
        sig = (train_df[c].values.tobytes(),
               float(train_df[c].mean()),
               float(train_df[c].std(ddof=0)))
        if sig in seen:
            dup_cols.append(c)
        else:
            seen[sig] = c
    drop_cols = list(set(const_cols + dup_cols))
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")
    kept_features = [c for c in train_df.columns if c != target_col]
    return train_df, test_df, kept_features


def winsorize(
    df: pd.DataFrame,
    cols: List[str],
    lower: float = 0.005,
    upper: float = 0.995,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Winsorize df[cols] in-place; return df and (q_low, q_hi)."""
    q_low = df[cols].quantile(lower)
    q_hi = df[cols].quantile(upper)
    df[cols] = df[cols].clip(lower=q_low, upper=q_hi, axis=1)
    return df, q_low, q_hi


def adversarial_weights(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    random_state: int = 42,
) -> np.ndarray:
    """
    Adversarial reweighting: upweight train rows that look like test rows.
    """
    trX = train_df.drop(columns=[target_col]).copy()
    teX = test_df.copy()
    trX["__is_test__"] = 0
    teX["__is_test__"] = 1
    ALL = pd.concat([trX, teX], axis=0)
    y_adv = ALL["__is_test__"].values
    X_adv = ALL.drop(columns=["__is_test__"]).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    p_test = np.zeros(len(ALL), dtype=float)
    params = dict(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=4,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="binary",
        n_jobs=-1,
        verbosity=-1,
    )
    for tr, va in skf.split(X_adv, y_adv):
        m = LGBMClassifier(random_state=random_state, **params)
        m.fit(
            X_adv[tr],
            y_adv[tr],
            eval_set=[(X_adv[va], y_adv[va])],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(100), log_evaluation(period=0)],
        )
        p_test[va] = m.predict_proba(X_adv[va])[:, 1]

    p_train = p_test[: len(train_df)]
    eps = 1e-6
    w = (1.0 - p_train) / np.clip(p_train, eps, 1 - eps)
    w = w / np.mean(w)
    return w


def tta_predict_prob(
    predict_fn,
    X_test: np.ndarray,
    feat_std: np.ndarray,
    tta_draws: int,
    tta_alpha: float,
    clip_lo: np.ndarray | None,
    clip_hi: np.ndarray | None,
) -> np.ndarray:
    """
    Test-time augmentation with Gaussian noise; clipped to train quantile bounds.
    """
    base = predict_fn(X_test)
    if tta_draws <= 0:
        return base
    tta_sum = np.zeros_like(base)
    for _ in range(tta_draws):
        noise = np.random.normal(0.0, tta_alpha * feat_std, size=X_test.shape)
        X_noisy = X_test + noise
        if clip_lo is not None and clip_hi is not None:
            X_noisy = np.clip(X_noisy, clip_lo, clip_hi)
        tta_sum += predict_fn(X_noisy)
    return 0.5 * (base + tta_sum / tta_draws)
