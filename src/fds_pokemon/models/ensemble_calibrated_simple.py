import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ..data import load_dataset
from ..utils import sweep_best_threshold, rank_normalize


def train_and_predict(
    train_path: str = "/kaggle/input/fds-pokemon-battles-prediction-2025/train.jsonl",
    test_path: str = "/kaggle/input/fds-pokemon-battles-prediction-2025/test.jsonl",
    n_splits: int = 7,
    random_state: int = 42,
    submission_path: str = "submission_ensemble_platt.csv",
    repeats: int = 3,
    use_platt: bool = True,
    choose_rank_avg: bool = True,
) -> None:
    # ===== Load & align =====
    train_df = load_dataset(train_path, is_train=True)
    test_df = load_dataset(test_path, is_train=False)

    target_col = "player_won"
    feature_cols = [c for c in train_df.columns if c != target_col]

    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0.0
    for c in list(test_df.columns):
        if c not in feature_cols:
            test_df.drop(columns=[c], inplace=True, errors="ignore")

    train_df = train_df[feature_cols + [target_col]]
    test_df = test_df[feature_cols]

    train_df.fillna(0.0, inplace=True)
    test_df.fillna(0.0, inplace=True)

    X = train_df[feature_cols].values
    y = train_df[target_col].values.astype(int)
    X_test = test_df[feature_cols].values

    n_models = n_splits * repeats
    oof_preds = np.zeros(len(train_df), dtype=float)
    test_preds = np.zeros(len(test_df), dtype=float)

    model_params = dict(
        n_estimators=20000,
        learning_rate=0.0095,
        num_leaves=160,
        max_depth=4,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        min_child_samples=60,
        min_split_gain=0.05,
        reg_alpha=0.5,
        reg_lambda=0.5,
        objective="binary",
        n_jobs=-1,
        verbosity=-1,
    )

    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + 1000 * rep)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X[tr_idx], X[val_idx]
            y_train, y_val = y[tr_idx], y[val_idx]

            model = LGBMClassifier(random_state=(random_state + 1000 * rep + fold), **model_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=0)],
            )

            val_proba = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] += val_proba / repeats
            test_preds += model.predict_proba(X_test)[:, 1] / n_models

    # ===== Build streams: raw, calibrated(Platt), rank-average =====
    streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    streams["raw"] = (oof_preds.copy(), test_preds.copy())

    if use_platt:
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(oof_preds.reshape(-1, 1), y)
        oof_cal = lr.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
        test_cal = lr.predict_proba(test_preds.reshape(-1, 1))[:, 1]
        streams["platt"] = (oof_cal, test_cal)

    if choose_rank_avg:
        oof_rank = rank_normalize(oof_preds)
        test_rank = rank_normalize(test_preds)
        streams["rank"] = (oof_rank, test_rank)

    # ===== Choose best stream by OOF accuracy after threshold sweep =====
    best_name, best_oof_acc, best_thr = "raw", -1.0, 0.5
    for name, (oof_s, _) in streams.items():
        acc, thr = sweep_best_threshold(y, oof_s, lo=0.3, hi=0.7, step=0.001)
        if acc > best_oof_acc:
            best_oof_acc, best_thr, best_name = acc, thr, name

    _, test_best = streams[best_name]

    print("=" * 56)
    print(f" Selected stream      : {best_name}")
    print(f" Best OOF Threshold   : {best_thr:.3f}")
    print(f" Final OOF Accuracy   : {best_oof_acc:.5f}")
    print("=" * 56)

    # ===== Submission =====
    test_labels = (test_best >= best_thr).astype(int)
    submission = pd.DataFrame({"battle_id": test_df.index, "player_won": test_labels}).reset_index(drop=True)
    submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    train_and_predict()
