import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ..data import load_dataset
from ..utils import (
    sweep_best_threshold,
    rank_normalize,
    drop_flawed_row,
    prune_features,
)


def train_and_predict(
    train_path: str = "/kaggle/input/fds-pokemon-battles-prediction-2025/train.jsonl",
    test_path: str = "/kaggle/input/fds-pokemon-battles-prediction-2025/test.jsonl",
    n_splits: int = 7,
    random_state: int = 42,
    submission_path: str = "submission_ensemble_platt_iso_stack.csv",
    repeats: int = 3,
    use_platt: bool = True,
    choose_rank_avg: bool = True,
    do_isotonic: bool = True,
    do_stack_meta: bool = True,
) -> None:
    # ===== Load & align =====
    train_df = load_dataset(train_path, is_train=True)
    test_df = load_dataset(test_path, is_train=False)

    target_col = "player_won"
    train_df = drop_flawed_row(train_df, target_col)

    feature_cols_all = [c for c in train_df.columns if c != target_col]
    for c in feature_cols_all:
        if c not in test_df.columns:
            test_df[c] = 0.0
    extra_test_cols = [c for c in test_df.columns if c not in feature_cols_all]
    if extra_test_cols:
        test_df.drop(columns=extra_test_cols, inplace=True, errors="ignore")

    train_df = train_df[[c for c in feature_cols_all] + [target_col]]
    test_df = test_df[[c for c in feature_cols_all]]

    train_df = train_df.fillna(0.0)
    test_df = test_df.fillna(0.0)

    # Feature pruning
    train_df, test_df, feature_cols = prune_features(train_df, test_df, target_col)

    X = train_df[feature_cols].values
    y = train_df[target_col].values.astype(int)
    X_test = test_df[feature_cols].values

    n_models = n_splits * repeats

    # Storage for streams
    oof_raw = np.zeros(len(train_df), dtype=float)
    test_raw = np.zeros(len(test_df), dtype=float)

    # Fold-wise calibrated accumulators
    oof_platt_cv = np.zeros_like(oof_raw)
    test_platt_cv = np.zeros_like(test_raw)
    oof_iso_cv = np.zeros_like(oof_raw)
    test_iso_cv = np.zeros_like(test_raw)

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
            test_proba = model.predict_proba(X_test)[:, 1]

            # Raw streams (average across all models)
            oof_raw[val_idx] += val_proba / repeats
            test_raw += test_proba / n_models

            # Platt (Logistic) per-fold calibration
            if use_platt:
                lr = LogisticRegression(solver="lbfgs")
                lr.fit(val_proba.reshape(-1, 1), y_val)
                oof_platt_cv[val_idx] += lr.predict_proba(val_proba.reshape(-1, 1))[:, 1] / repeats
                test_platt_cv += lr.predict_proba(test_proba.reshape(-1, 1))[:, 1] / n_models

            # Isotonic per-fold calibration
            if do_isotonic:
                try:
                    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                    iso.fit(val_proba, y_val)
                    oof_iso_cv[val_idx] += iso.transform(val_proba) / repeats
                    test_iso_cv += iso.transform(test_proba) / n_models
                except Exception:
                    oof_iso_cv[val_idx] += val_proba / repeats
                    test_iso_cv += test_proba / n_models

    # ===== Global calibration on OOF -> Test =====
    streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    streams["raw"] = (oof_raw.copy(), test_raw.copy())

    if use_platt:
        lr_global = LogisticRegression(solver="lbfgs")
        lr_global.fit(oof_raw.reshape(-1, 1), y)
        oof_platt_global = lr_global.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
        test_platt_global = lr_global.predict_proba(test_raw.reshape(-1, 1))[:, 1]
        streams["platt_global"] = (oof_platt_global, test_platt_global)
        streams["platt_cv"] = (oof_platt_cv, test_platt_cv)

    if do_isotonic:
        try:
            iso_global = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso_global.fit(oof_raw, y)
            oof_iso_global = iso_global.transform(oof_raw)
            test_iso_global = iso_global.transform(test_raw)
        except Exception:
            oof_iso_global, test_iso_global = oof_raw.copy(), test_raw.copy()
        streams["iso_global"] = (oof_iso_global, test_iso_global)
        streams["iso_cv"] = (oof_iso_cv, test_iso_cv)

    if choose_rank_avg:
        oof_rank = rank_normalize(oof_raw)
        test_rank = rank_normalize(test_raw)
        streams["rank"] = (oof_rank, test_rank)

    # ===== Stacked meta-calibrator over streams =====
    if do_stack_meta:
        feat_list = []
        names = []
        feat_candidates = [
            ("raw", streams.get("raw")),
            ("rank", streams.get("rank")),
            ("platt_cv", streams.get("platt_cv") if use_platt else None),
            ("platt_global", streams.get("platt_global") if use_platt else None),
            ("iso_cv", streams.get("iso_cv") if do_isotonic else None),
            ("iso_global", streams.get("iso_global") if do_isotonic else None),
        ]
        for nm, pair in feat_candidates:
            if pair is not None:
                feat_list.append(pair[0])
                names.append(nm)
        if feat_list:
            X_meta = np.vstack(feat_list).T
            meta = LogisticRegression(solver="lbfgs", max_iter=1000)
            meta.fit(X_meta, y)
            test_feats = [streams[nm][1] for nm in names]
            X_meta_test = np.vstack(test_feats).T
            oof_stack = meta.predict_proba(X_meta)[:, 1]
            test_stack = meta.predict_proba(X_meta_test)[:, 1]
            streams["stack"] = (oof_stack, test_stack)

    # ===== Choose best stream by OOF accuracy =====
    best_name, best_oof_acc, best_thr = None, -1.0, 0.5
    for name, (oof_s, _) in streams.items():
        acc, thr = sweep_best_threshold(y, oof_s, lo=0.2, hi=0.8, step=0.0005)
        if acc > best_oof_acc:
            best_oof_acc, best_thr, best_name = acc, thr, name

    _, test_best = streams[best_name]

    print("=" * 64)
    print(f" Selected stream      : {best_name}")
    print(f" Best OOF Threshold   : {best_thr:.4f}")
    print(f" Final OOF Accuracy   : {best_oof_acc:.6f}")
    print(f" Features kept        : {len(feature_cols)}")
    print("=" * 64)

    # ===== Submission =====
    test_labels = (test_best >= best_thr).astype(int)
    submission = pd.DataFrame({"battle_id": test_df.index, "player_won": test_labels}).reset_index(drop=True)
    submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    train_and_predict()
