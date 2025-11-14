import warnings
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Optional learners (guarded)
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier, Pool

    HAS_CAT = True
except Exception:
    HAS_CAT = False

from ..data import load_dataset
from ..utils import (
    sweep_best_threshold,
    rank_normalize,
    drop_flawed_row,
    prune_features,
    winsorize,
    adversarial_weights,
    tta_predict_prob,
)


def train_and_predict(
    train_path: str = "/kaggle/input/fds-pokemon-battles-prediction-2025/train.jsonl",
    test_path: str = "/kaggle/input/fds-pokemon-battles-prediction-2025/test.jsonl",
    n_splits: int = 7,
    random_state: int = 42,
    submission_path: str = "submission_advstack.csv",
    repeats: int = 3,
    use_platt: bool = True,
    choose_rank_avg: bool = True,
    do_isotonic: bool = True,
    do_stack_meta: bool = True,
    tta_draws: int = 5,
    tta_alpha: float = 0.0025,
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

    # Prune constant/duplicate features
    train_df, test_df, feature_cols = prune_features(train_df, test_df, target_col)

    # Winsorize train; clip test to same quantiles
    train_df, q_low, q_hi = winsorize(train_df, feature_cols, lower=0.005, upper=0.995)
    test_df[feature_cols] = test_df[feature_cols].clip(lower=q_low, upper=q_hi, axis=1)

    # Bounds for TTA clipping
    clip_lo = q_low.values.astype(float)
    clip_hi = q_hi.values.astype(float)

    # Adversarial reweighting
    adv_weights = adversarial_weights(train_df, test_df, target_col, random_state=random_state)

    X = train_df[feature_cols].values
    y = train_df[target_col].values.astype(int)
    X_test = test_df[feature_cols].values

    n_models = n_splits * repeats

    # Streams storage
    streams: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # LGBM accumulators
    oof_raw = np.zeros(len(train_df), dtype=float)
    test_raw = np.zeros(len(test_df), dtype=float)
    oof_platt_cv = np.zeros_like(oof_raw)
    test_platt_cv = np.zeros_like(test_raw)
    oof_iso_cv = np.zeros_like(oof_raw)
    test_iso_cv = np.zeros_like(test_raw)

    # Optional model space
    if HAS_XGB:
        oof_xgb = np.zeros_like(oof_raw)
        test_xgb = np.zeros_like(test_raw)
        oof_xgb_platt = np.zeros_like(oof_raw)
        test_xgb_platt = np.zeros_like(test_raw)
        oof_xgb_iso = np.zeros_like(oof_raw)
        test_xgb_iso = np.zeros_like(test_raw)

    if HAS_CAT:
        oof_cat = np.zeros_like(oof_raw)
        test_cat = np.zeros_like(test_raw)
        oof_cat_platt = np.zeros_like(oof_raw)
        test_cat_platt = np.zeros_like(test_raw)
        oof_cat_iso = np.zeros_like(oof_raw)
        test_cat_iso = np.zeros_like(test_raw)

    # Params (keep LGBM HPs intact)
    lgb_params = dict(
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
    xgb_params = dict(
        n_estimators=20000,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=6,
        reg_alpha=0.5,
        reg_lambda=0.5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )
    cat_params = dict(
        iterations=20000,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        subsample=0.7,
        rsm=0.7,
        random_strength=10.0,
        od_type="Iter",
        od_wait=200,
        verbose=False,
    )

    feat_std = X.std(axis=0) + 1e-12  # for TTA noise scale

    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + 1000 * rep)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X[tr_idx], X[val_idx]
            y_train, y_val = y[tr_idx], y[val_idx]
            w_train = adv_weights[tr_idx]

            # ---- LGBM
            model = LGBMClassifier(random_state=(random_state + 1000 * rep + fold), **lgb_params)
            model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=0)],
            )

            val_proba = model.predict_proba(X_val)[:, 1]
            test_proba = tta_predict_prob(
                predict_fn=lambda Z: model.predict_proba(Z)[:, 1],
                X_test=X_test,
                feat_std=feat_std,
                tta_draws=tta_draws,
                tta_alpha=tta_alpha,
                clip_lo=clip_lo,
                clip_hi=clip_hi,
            )

            oof_raw[val_idx] += val_proba / repeats
            test_raw += test_proba / n_models

            if use_platt:
                lr = LogisticRegression(solver="lbfgs")
                lr.fit(val_proba.reshape(-1, 1), y_val)
                oof_platt_cv[val_idx] += lr.predict_proba(val_proba.reshape(-1, 1))[:, 1] / repeats
                test_platt_cv += lr.predict_proba(test_proba.reshape(-1, 1))[:, 1] / n_models

            if do_isotonic:
                try:
                    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                    iso.fit(val_proba, y_val)
                    oof_iso_cv[val_idx] += iso.transform(val_proba) / repeats
                    test_iso_cv += iso.transform(test_proba) / n_models
                except Exception:
                    oof_iso_cv[val_idx] += val_proba / repeats
                    test_iso_cv += test_proba / n_models

            # ---- XGBoost (optional)
            if HAS_XGB:
                xgb = XGBClassifier(**xgb_params, random_state=(random_state + 777 * rep + fold))
                xgb.fit(
                    X_train,
                    y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    early_stopping_rounds=200,
                )
                x_val = xgb.predict_proba(X_val)[:, 1]
                x_tst = tta_predict_prob(
                    predict_fn=lambda Z: xgb.predict_proba(Z)[:, 1],
                    X_test=X_test,
                    feat_std=feat_std,
                    tta_draws=tta_draws,
                    tta_alpha=tta_alpha,
                    clip_lo=clip_lo,
                    clip_hi=clip_hi,
                )
                oof_xgb[val_idx] += x_val / repeats
                test_xgb += x_tst / n_models
                if use_platt:
                    lr = LogisticRegression(solver="lbfgs")
                    lr.fit(x_val.reshape(-1, 1), y_val)
                    oof_xgb_platt[val_idx] += lr.predict_proba(x_val.reshape(-1, 1))[:, 1] / repeats
                    test_xgb_platt += lr.predict_proba(x_tst.reshape(-1, 1))[:, 1] / n_models
                if do_isotonic:
                    try:
                        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                        iso.fit(x_val, y_val)
                        oof_xgb_iso[val_idx] += iso.transform(x_val) / repeats
                        test_xgb_iso += iso.transform(x_tst) / n_models
                    except Exception:
                        oof_xgb_iso[val_idx] += x_val / repeats
                        test_xgb_iso += x_tst / n_models

            # ---- CatBoost (optional)
            if HAS_CAT:
                cat = CatBoostClassifier(**cat_params, random_seed=(random_state + 555 * rep + fold))
                cat.fit(Pool(X_train, y_train, weight=w_train), eval_set=Pool(X_val, y_val), verbose=False)
                c_val = cat.predict_proba(X_val)[:, 1]
                c_tst = tta_predict_prob(
                    predict_fn=lambda Z: cat.predict_proba(Z)[:, 1],
                    X_test=X_test,
                    feat_std=feat_std,
                    tta_draws=tta_draws,
                    tta_alpha=tta_alpha,
                    clip_lo=clip_lo,
                    clip_hi=clip_hi,
                )
                oof_cat[val_idx] += c_val / repeats
                test_cat += c_tst / n_models
                if use_platt:
                    lr = LogisticRegression(solver="lbfgs")
                    lr.fit(c_val.reshape(-1, 1), y_val)
                    oof_cat_platt[val_idx] += lr.predict_proba(c_val.reshape(-1, 1))[:, 1] / repeats
                    test_cat_platt += lr.predict_proba(c_tst.reshape(-1, 1))[:, 1] / n_models
                if do_isotonic:
                    try:
                        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                        iso.fit(c_val, y_val)
                        oof_cat_iso[val_idx] += iso.transform(c_val) / repeats
                        test_cat_iso += iso.transform(c_tst) / n_models
                    except Exception:
                        oof_cat_iso[val_idx] += c_val / repeats
                        test_cat_iso += c_tst / n_models

    # Base streams
    streams["lgb_raw"] = (oof_raw.copy(), test_raw.copy())
    if use_platt:
        streams["lgb_platt_cv"] = (oof_platt_cv.copy(), test_platt_cv.copy())
    if do_isotonic:
        streams["lgb_iso_cv"] = (oof_iso_cv.copy(), test_iso_cv.copy())
    if choose_rank_avg:
        streams["lgb_rank"] = (rank_normalize(oof_raw), rank_normalize(test_raw))

    if HAS_XGB:
        streams["xgb_raw"] = (oof_xgb.copy(), test_xgb.copy())
        if use_platt:
            streams["xgb_platt_cv"] = (oof_xgb_platt.copy(), test_xgb_platt.copy())
        if do_isotonic:
            streams["xgb_iso_cv"] = (oof_xgb_iso.copy(), test_xgb_iso.copy())
        if choose_rank_avg:
            streams["xgb_rank"] = (rank_normalize(oof_xgb), rank_normalize(test_xgb))

    if HAS_CAT:
        streams["cat_raw"] = (oof_cat.copy(), test_cat.copy())
        if use_platt:
            streams["cat_platt_cv"] = (oof_cat_platt.copy(), test_cat_platt.copy())
        if do_isotonic:
            streams["cat_iso_cv"] = (oof_cat_iso.copy(), test_cat_iso.copy())
        if choose_rank_avg:
            streams["cat_rank"] = (rank_normalize(oof_cat), rank_normalize(test_cat))

    # Global calibration for each *_raw stream
    for key in list(streams.keys()):
        if key.endswith("_raw"):
            oof_s, tst_s = streams[key]
            if use_platt:
                lr = LogisticRegression(solver="lbfgs")
                lr.fit(oof_s.reshape(-1, 1), y)
                streams[key.replace("_raw", "_platt_global")] = (
                    lr.predict_proba(oof_s.reshape(-1, 1))[:, 1],
                    lr.predict_proba(tst_s.reshape(-1, 1))[:, 1],
                )
            if do_isotonic:
                try:
                    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                    iso.fit(oof_s, y)
                    streams[key.replace("_raw", "_iso_global")] = (
                        iso.transform(oof_s),
                        iso.transform(tst_s),
                    )
                except Exception:
                    streams[key.replace("_raw", "_iso_global")] = (oof_s.copy(), tst_s.copy())

    # Meta stack
    if do_stack_meta:
        names_order = []
        feat_list = []
        for nm in [
            "lgb_raw",
            "lgb_rank",
            "lgb_platt_cv",
            "lgb_platt_global",
            "lgb_iso_cv",
            "lgb_iso_global",
            "xgb_raw",
            "xgb_rank",
            "xgb_platt_cv",
            "xgb_platt_global",
            "xgb_iso_cv",
            "xgb_iso_global",
            "cat_raw",
            "cat_rank",
            "cat_platt_cv",
            "cat_platt_global",
            "cat_iso_cv",
            "cat_iso_global",
        ]:
            if nm in streams:
                feat_list.append(streams[nm][0])
                names_order.append(nm)
        if feat_list:
            X_meta = np.vstack(feat_list).T
            meta = LogisticRegression(solver="lbfgs", max_iter=1000)
            meta.fit(X_meta, y)
            X_meta_test = np.vstack([streams[nm][1] for nm in names_order]).T
            oof_stack = meta.predict_proba(X_meta)[:, 1]
            test_stack = meta.predict_proba(X_meta_test)[:, 1]
            streams["stack"] = (oof_stack, test_stack)

    # Choose best by OOF accuracy
    best_name, best_oof_acc, best_thr = None, -1.0, 0.5
    for name, (oof_s, _) in streams.items():
        acc, thr = sweep_best_threshold(y, oof_s, lo=0.15, hi=0.85, step=0.0005)
        if acc > best_oof_acc:
            best_oof_acc, best_thr, best_name = acc, thr, name

    _, test_best = streams[best_name]

    print("=" * 72)
    print(f" Selected stream      : {best_name}")
    print(f" Best OOF Threshold   : {best_thr:.4f}")
    print(f" Final OOF Accuracy   : {best_oof_acc:.6f}")
    print(f" Features kept        : {len(feature_cols)}")
    print(f" XGBoost enabled      : {HAS_XGB}")
    print(f" CatBoost enabled     : {HAS_CAT}")
    print(f" TTA draws            : {tta_draws}, alpha={tta_alpha}")
    print("=" * 72)

    # Submission
    test_labels = (test_best >= best_thr).astype(int)
    submission = pd.DataFrame({"battle_id": test_df.index, "player_won": test_labels}).reset_index(drop=True)
    submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    train_and_predict()
