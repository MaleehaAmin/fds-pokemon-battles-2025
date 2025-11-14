# src/fds_pokemon/features.py
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Dict, List, Tuple

# =========================================================
# Type chart & helpers
# =========================================================
TYPE_CHART: Dict[str, Dict[str, float]] = {
    "NORMAL": {"ROCK": 0.5, "GHOST": 0.0},
    "FIRE": {
        "GRASS": 2.0, "ICE": 2.0, "BUG": 2.0,
        "ROCK": 0.5, "FIRE": 0.5, "WATER": 0.5, "DRAGON": 0.5,
    },
    "WATER": {
        "FIRE": 2.0, "GROUND": 2.0, "ROCK": 2.0,
        "WATER": 0.5, "GRASS": 0.5, "DRAGON": 0.5,
    },
    "ELECTRIC": {
        "WATER": 2.0, "FLYING": 2.0,
        "ELECTRIC": 0.5, "GRASS": 0.5, "DRAGON": 0.5,
        "GROUND": 0.0,
    },
    "GRASS": {
        "WATER": 2.0, "GROUND": 2.0, "ROCK": 2.0,
        "FIRE": 0.5, "GRASS": 0.5, "POISON": 0.5,
        "FLYING": 0.5, "BUG": 0.5, "DRAGON": 0.5,
    },
    "ICE": {
        "GRASS": 2.0, "GROUND": 2.0, "FLYING": 2.0, "DRAGON": 2.0,
        "FIRE": 0.5, "WATER": 0.5, "ICE": 0.5,
    },
    "FIGHTING": {
        "NORMAL": 2.0, "ICE": 2.0, "ROCK": 2.0,
        "POISON": 0.5, "FLYING": 0.5, "PSYCHIC": 0.5,
        "BUG": 0.5, "GHOST": 0.0,
    },
    "POISON": {
        "GRASS": 2.0,
        "POISON": 0.5, "GROUND": 0.5, "ROCK": 0.5, "GHOST": 0.5,
    },
    "GROUND": {
        "FIRE": 2.0, "ELECTRIC": 2.0, "POISON": 2.0, "ROCK": 2.0,
        "BUG": 0.5, "GRASS": 0.5, "FLYING": 0.0,
    },
    "FLYING": {"GRASS": 2.0, "FIGHTING": 2.0, "BUG": 2.0, "ELECTRIC": 0.5, "ROCK": 0.5},
    "PSYCHIC": {"FIGHTING": 2.0, "POISON": 2.0, "PSYCHIC": 0.5},
    "BUG": {"GRASS": 2.0, "POISON": 2.0, "FIRE": 0.5, "FIGHTING": 0.5, "FLYING": 0.5, "GHOST": 0.5},
    "ROCK": {"FIRE": 2.0, "ICE": 2.0, "FLYING": 2.0, "BUG": 2.0, "FIGHTING": 0.5, "GROUND": 0.5},
    "GHOST": {"GHOST": 2.0, "NORMAL": 0.0, "PSYCHIC": 0.0},
    "DRAGON": {"DRAGON": 2.0},
}

TOP_OU_MON = {
    "TAUROS", "CHANSEY", "SNORLAX", "STARMIE", "ALAKAZAM", "EXEGGUTOR",
    "JOLTEON", "ZAPDOS", "GOLEM", "RHYDON", "JYNX", "CLOYSTER", "GENGAR",
}


def type_effectiveness(attack_type: str, defend_types: List[str]) -> float:
    if not attack_type or not defend_types:
        return 1.0
    atk = attack_type.upper()
    eff = 1.0
    for dt in defend_types:
        eff *= TYPE_CHART.get(atk, {}).get(dt.upper(), 1.0)
    return eff


def safe_mean(x, default=0.0) -> float:
    arr = np.asarray(list(x) if isinstance(x, (list, tuple)) else x)
    return float(np.mean(arr)) if arr.size else default


def safe_std(x, default=0.0) -> float:
    arr = np.asarray(list(x) if isinstance(x, (list, tuple)) else x)
    return float(np.std(arr)) if arr.size else default


def safe_quantile(x, q: float, default=0.0) -> float:
    arr = np.asarray(list(x) if isinstance(x, (list, tuple)) else x)
    return float(np.quantile(arr, q)) if arr.size else default


def entropy(counter: Counter) -> float:
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    p = np.array([c / tot for c in counter.values()], dtype=float)
    return float(-(p * np.log(p + 1e-12)).sum())


def ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_bst(p: Dict[str, Any]) -> int:
    return (
        p["base_hp"]
        + p["base_atk"]
        + p["base_def"]
        + p["base_spa"]
        + p["base_spd"]
        + p["base_spe"]
    )


def offensive_profile(atk_types: List[str], def_types: List[str]) -> Tuple[float, float, float]:
    effs = [type_effectiveness(t, def_types) for t in atk_types] or [1.0]
    mmax = float(max(effs))
    mmed = float(np.median(effs))
    mtop2 = float(np.mean(sorted(effs, reverse=True)[:2]))
    return mmax, mmed, mtop2


# =========================================================
# Main feature extractor
# =========================================================
def extract_features_from_battle(b: Dict[str, Any]) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    p1_team = b["p1_team_details"]
    p2_lead = b["p2_lead_details"]
    timeline = b["battle_timeline"]

    p2_lead_types = [t.upper() for t in p2_lead["types"]]

    p1_bsts, p1_levels, p1_speeds = [], [], []
    p1_type_counts = Counter()
    p1_synergy_off, p1_synergy_off_med, p1_synergy_off_top2 = [], [], []
    p1_synergy_def = []
    p1_weak_count = p1_resist_count = p1_immune_count = 0
    mono = dual = 0
    flagship_flags = []
    faster_than_p2 = 0

    for mon in p1_team:
        mon_types = [t.upper() for t in mon["types"]]
        p1_bsts.append(compute_bst(mon))
        p1_levels.append(mon["level"])
        p1_speeds.append(mon["base_spe"])
        for t in mon_types:
            p1_type_counts[t] += 1

        mono += int(len(mon_types) == 1)
        dual += int(len(mon_types) >= 2)

        mmax, mmed, mtop2 = offensive_profile(mon_types, p2_lead_types)
        p1_synergy_off.append(mmax)
        p1_synergy_off_med.append(mmed)
        p1_synergy_off_top2.append(mtop2)

        mon_def_eff = (
            min(type_effectiveness(t2, mon_types) for t2 in p2_lead_types)
            if mon_types and p2_lead_types
            else 1.0
        )
        p1_synergy_def.append(mon_def_eff)

        weak = any(type_effectiveness(t2, mon_types) > 1.0 for t2 in p2_lead_types)
        resist = any(0.0 < type_effectiveness(t2, mon_types) < 1.0 for t2 in p2_lead_types)
        immune = any(type_effectiveness(t2, mon_types) == 0.0 for t2 in p2_lead_types)
        p1_weak_count += int(weak)
        p1_resist_count += int(resist)
        p1_immune_count += int(immune)

        faster_than_p2 += int(mon["base_spe"] > p2_lead["base_spe"])

        flagship_flags.append(int(mon["name"].upper() in TOP_OU_MON))

    feats["p1_bst_mean"] = safe_mean(p1_bsts)
    feats["p1_bst_max"] = float(max(p1_bsts))
    feats["p1_bst_min"] = float(min(p1_bsts))
    feats["p1_bst_std"] = safe_std(p1_bsts)

    feats["p1_level_mean"] = safe_mean(p1_levels)
    feats["p1_level_max"] = float(max(p1_levels))
    feats["p1_level_min"] = float(min(p1_levels))
    feats["p1_level_std"] = safe_std(p1_levels)

    feats["p1_spe_mean"] = safe_mean(p1_speeds)
    feats["p1_spe_max"] = float(max(p1_speeds))
    feats["p1_spe_min"] = float(min(p1_speeds))
    feats["p1_spe_std"] = safe_std(p1_speeds)

    feats["p1_type_unique_count"] = float(len(p1_type_counts))
    feats["p1_type_entropy"] = entropy(p1_type_counts)
    feats["p1_mono_type_ratio"] = ratio(mono, len(p1_team))
    feats["p1_dual_type_ratio"] = ratio(dual, len(p1_team))
    for t, c in p1_type_counts.items():
        feats[f"p1_type_count_{t}"] = float(c)

    feats["p1_team_off_synergy_mean"] = safe_mean(p1_synergy_off)
    feats["p1_team_off_synergy_median"] = safe_mean(p1_synergy_off_med)
    feats["p1_team_off_synergy_top2_mean"] = safe_mean(p1_synergy_off_top2)
    feats["p1_team_off_synergy_max"] = float(max(p1_synergy_off))
    feats["p1_team_def_synergy_mean"] = safe_mean(p1_synergy_def)
    feats["p1_team_def_synergy_min"] = float(min(p1_synergy_def))

    feats["p1_team_weak_count_vs_p2_lead"] = float(p1_weak_count)
    feats["p1_team_resist_count_vs_p2_lead"] = float(p1_resist_count)
    feats["p1_team_immunity_count_vs_p2_lead"] = float(p1_immune_count)

    off_max = p1_synergy_off or [1.0]
    feats["p1_cov_ge2_count"] = float(sum(x >= 2.0 for x in off_max))
    feats["p1_cov_eq1_count"] = float(sum(np.isclose(x, 1.0) for x in off_max))
    feats["p1_cov_le0_5_count"] = float(sum(x <= 0.5 for x in off_max))
    feats["p1_cov_eq0_count"] = float(sum(np.isclose(x, 0.0) for x in off_max))

    feats["p1_any_faster_than_p2lead"] = float(faster_than_p2 > 0)
    feats["p1_frac_faster_than_p2lead"] = ratio(faster_than_p2, len(p1_team))

    feats["p1_ou_flagship_count"] = float(sum(flagship_flags))

    # ---- Leads
    p1_lead = p1_team[0]
    p1_lead_types = [t.upper() for t in p1_lead["types"]]
    p1_lead_bst = compute_bst(p1_lead)
    p2_bst = compute_bst(p2_lead)

    feats["p2_lead_bst"] = float(p2_bst)
    feats["p2_lead_level"] = float(p2_lead["level"])
    feats["p2_lead_spe"] = float(p2_lead["base_spe"])
    for t in p2_lead_types:
        feats[f"p2_lead_type_{t}"] = 1.0

    feats["p1_lead_bst"] = float(p1_lead_bst)
    feats["p1_lead_level"] = float(p1_lead["level"])
    feats["p1_lead_spe"] = float(p1_lead["base_spe"])

    p1_off_vs_p2 = max(type_effectiveness(t, p2_lead_types) for t in p1_lead_types) if p1_lead_types else 1.0
    p2_off_vs_p1 = max(type_effectiveness(t, p1_lead_types) for t in p2_lead_types) if p2_lead_types else 1.0
    feats["p1_lead_off_eff_vs_p2"] = float(p1_off_vs_p2)
    feats["p2_lead_off_eff_vs_p1"] = float(p2_off_vs_p1)
    feats["lead_off_eff_diff"] = float(p1_off_vs_p2 - p2_off_vs_p1)

    feats["lead_speed_diff"] = float(p1_lead["base_spe"] - p2_lead["base_spe"])
    feats["lead_level_diff"] = float(p1_lead["level"] - p2_lead["level"])
    feats["lead_bst_diff"] = float(p1_lead_bst - p2_bst)
    feats["p1_team_mean_spe_adv_vs_p2_lead"] = feats["p1_spe_mean"] - feats["p2_lead_spe"]

    feats["p1_lead_has_SE_STAB_vs_p2"] = float(
        any(type_effectiveness(t, p2_lead_types) > 1.0 for t in p1_lead_types)
    )
    feats["p2_lead_has_SE_STAB_vs_p1"] = float(
        any(type_effectiveness(t, p1_lead_types) > 1.0 for t in p2_lead_types)
    )

    # ---- Timeline
    n_turns = len(timeline)
    feats["n_turns"] = float(n_turns)

    if n_turns == 0:
        zero_keys = [
            "p1_hp_mean",
            "p1_hp_max",
            "p1_hp_min",
            "p1_hp_last",
            "p2_hp_mean",
            "p2_hp_max",
            "p2_hp_min",
            "p2_hp_last",
            "hp_diff_mean",
            "hp_diff_max",
            "hp_diff_min",
            "hp_diff_last",
            "p1_moves_used",
            "p2_moves_used",
            "p1_ahead_ratio",
            "p1_early_ahead_ratio",
            "p1_mid_ahead_ratio",
            "p1_late_ahead_ratio",
            "hp_diff_auc",
            "hp_diff_std",
            "lead_change_count",
            "first_lead_turn",
            "p1_damage_total",
            "p2_damage_total",
            "p1_damage_mean",
            "p2_damage_mean",
            "p1_damage_max",
            "p2_damage_max",
            "net_damage_total",
            "net_damage_rate",
            "p1_priority_moves",
            "p2_priority_moves",
            "p1_bp_mean",
            "p2_bp_mean",
            "p1_acc_mean",
            "p2_acc_mean",
            "p1_total_boost_mean",
            "p2_total_boost_mean",
            "boost_total_diff_mean",
            "p1_total_boost_max",
            "p2_total_boost_max",
            "boost_total_diff_max",
            "p1_total_boost_std",
            "p2_total_boost_std",
            "p1_total_boost_sum",
            "p2_total_boost_sum",
            "p1_switches",
            "p2_switches",
            "switch_diff",
            "p1_switch_rate",
            "p2_switch_rate",
            "p1_ko_like",
            "p2_ko_like",
            "ko_like_diff",
            "first_ko_turn_p1",
            "first_ko_turn_p2",
            "first_ko_turn_diff",
            "early_hp_diff_mean",
            "early_hp_diff_last",
            "early_p1_hp_mean",
            "early_p2_hp_mean",
            "early_hp_diff_slope",
            "mid_hp_diff_mean",
            "mid_hp_diff_last",
            "mid_p1_hp_mean",
            "mid_p2_hp_mean",
            "mid_hp_diff_slope",
            "late_hp_diff_mean",
            "late_hp_diff_last",
            "late_p1_hp_mean",
            "late_p2_hp_mean",
            "late_hp_diff_slope",
            "major_status_ratio_p1",
            "major_status_ratio_p2",
            "hp_diff_pos_delta_sum",
            "hp_diff_neg_delta_sum",
            "hp_diff_net_delta",
            "hp_diff_sign_changes",
            "longest_ahead_streak",
            "comeback_flag",
            "p1_dpt_q25",
            "p1_dpt_q50",
            "p1_dpt_q75",
            "p2_dpt_q25",
            "p2_dpt_q50",
            "p2_dpt_q75",
            "p1_last_segment_dpt",
            "p2_last_segment_dpt",
            "p1_status_moves",
            "p2_status_moves",
            "p1_damaging_moves_count",
            "p2_damaging_moves_count",
            "p1_super_effective_hits",
            "p2_super_effective_hits",
            "se_hits_diff",
        ]
        for k in zero_keys:
            feats[k] = 0.0
        return feats

    p1_hp_list, p2_hp_list, hp_diff_list = [], [], []
    seg_ahead = {"early": 0, "mid": 0, "late": 0}
    seg_counts = {"early": 0, "mid": 0, "late": 0}
    p1_status_counts, p2_status_counts = Counter(), Counter()

    p1_damaging_bp, p2_damaging_bp = [], []
    p1_status_moves = p2_status_moves = 0
    p1_moves_used = p2_moves_used = 0
    p1_super_eff_hits = p2_super_eff_hits = 0
    p1_priority_moves = p2_priority_moves = 0
    p1_acc_list, p2_acc_list = [], []
    p1_total_boosts, p2_total_boosts = [], []
    p1_switches = p2_switches = 0
    p1_ko_like = p2_ko_like = 0
    first_ko_turn_p1 = first_ko_turn_p2 = 0.0
    prev_p1_name = prev_p2_name = None
    prev_p1_hp = prev_p2_hp = None

    prev_diff_sign = 0
    lead_change_count = 0
    first_lead_turn = 0.0
    total_ahead_turns = 0

    p1_damage_total = p2_damage_total = 0.0
    pos_delta_sum = neg_delta_sum = 0.0
    longest_ahead_streak = curr_streak = 0

    segs = {s: {"p1_hp": [], "p2_hp": [], "hp_diff": []} for s in ("early", "mid", "late")}

    for t in timeline:
        turn = int(t["turn"])
        seg = "early" if turn <= 10 else ("mid" if turn <= 20 else "late")
        p1s = t["p1_pokemon_state"]
        p2s = t["p2_pokemon_state"]
        p1_name, p2_name = p1s["name"], p2s["name"]
        p1_hp, p2_hp = float(p1s["hp_pct"]), float(p2s["hp_pct"])
        p1_hp_list.append(p1_hp)
        p2_hp_list.append(p2_hp)
        diff = p1_hp - p2_hp
        hp_diff_list.append(diff)

        segs[seg]["p1_hp"].append(p1_hp)
        segs[seg]["p2_hp"].append(p2_hp)
        segs[seg]["hp_diff"].append(diff)

        seg_counts[seg] += 1
        if diff > 0:
            total_ahead_turns += 1
            seg_ahead[seg] += 1
            curr_streak += 1
            longest_ahead_streak = max(longest_ahead_streak, curr_streak)
        else:
            curr_streak = 0

        if len(hp_diff_list) >= 2:
            delta = hp_diff_list[-1] - hp_diff_list[-2]
            if delta > 0:
                pos_delta_sum += delta
            elif delta < 0:
                neg_delta_sum += -delta

        sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
        if prev_diff_sign != 0 and sign != 0 and sign != prev_diff_sign:
            lead_change_count += 1
        if first_lead_turn == 0.0 and diff > 0:
            first_lead_turn = float(turn)
        prev_diff_sign = sign if sign != 0 else prev_diff_sign

        p1_status_counts[p1s.get("status", "nostatus")] += 1
        p2_status_counts[p2s.get("status", "nostatus")] += 1

        if prev_p1_name is not None and p1_name != prev_p1_name:
            p1_switches += 1
        if prev_p2_name is not None and p2_name != prev_p2_name:
            p2_switches += 1

        if prev_p1_hp is not None and prev_p1_hp > 0 and p1_hp <= 0:
            p1_ko_like += 1
            if first_ko_turn_p1 == 0.0:
                first_ko_turn_p1 = float(turn)
        if prev_p2_hp is not None and prev_p2_hp > 0 and p2_hp <= 0:
            p2_ko_like += 1
            if first_ko_turn_p2 == 0.0:
                first_ko_turn_p2 = float(turn)

        if prev_p2_hp is not None:
            p1_damage_total += max(prev_p2_hp - p2_hp, 0.0)
        if prev_p1_hp is not None:
            p2_damage_total += max(prev_p1_hp - p1_hp, 0.0)
        prev_p1_hp, prev_p2_hp = p1_hp, p2_hp
        prev_p1_name, prev_p2_name = p1_name, p2_name

        p1b, p2b = p1s.get("boosts") or {}, p2s.get("boosts") or {}
        p1_total_boosts.append(sum(p1b.values()))
        p2_total_boosts.append(sum(p2b.values()))
        m1, m2 = t.get("p1_move_details"), t.get("p2_move_details")

        if m1 is not None:
            p1_moves_used += 1
            bp = m1.get("base_power") or 0
            acc = m1.get("accuracy") or 100
            prio = m1.get("priority") or 0
            if bp > 0:
                p1_damaging_bp.append(bp)
                p1_acc_list.append(acc)
                eff = type_effectiveness(m1.get("type"), p2s.get("types") or p2_lead_types)
                if eff > 1.0:
                    p1_super_eff_hits += 1
            else:
                p1_status_moves += 1
            if prio > 0:
                p1_priority_moves += 1

        if m2 is not None:
            p2_moves_used += 1
            bp = m2.get("base_power") or 0
            acc = m2.get("accuracy") or 100
            prio = m2.get("priority") or 0
            if bp > 0:
                p2_damaging_bp.append(bp)
                p2_acc_list.append(acc)
                eff = type_effectiveness(m2.get("type"), p1s.get("types") or p1_lead_types)
                if eff > 1.0:
                    p2_super_eff_hits += 1
            else:
                p2_status_moves += 1
            if prio > 0:
                p2_priority_moves += 1

    # hp summaries
    feats["p1_hp_mean"] = safe_mean(p1_hp_list)
    feats["p1_hp_max"] = float(max(p1_hp_list))
    feats["p1_hp_min"] = float(min(p1_hp_list))
    feats["p1_hp_last"] = float(p1_hp_list[-1])
    feats["p2_hp_mean"] = safe_mean(p2_hp_list)
    feats["p2_hp_max"] = float(max(p2_hp_list))
    feats["p2_hp_min"] = float(min(p2_hp_list))
    feats["p2_hp_last"] = float(p2_hp_list[-1])
    feats["hp_diff_mean"] = safe_mean(hp_diff_list)
    feats["hp_diff_max"] = float(max(hp_diff_list))
    feats["hp_diff_min"] = float(min(hp_diff_list))
    feats["hp_diff_last"] = float(hp_diff_list[-1])

    feats["hp_diff_auc"] = float(np.sum(hp_diff_list))
    feats["hp_diff_std"] = safe_std(hp_diff_list)
    feats["lead_change_count"] = float(lead_change_count)
    feats["first_lead_turn"] = float(first_lead_turn)

    # momentum
    feats["hp_diff_pos_delta_sum"] = float(pos_delta_sum)
    feats["hp_diff_neg_delta_sum"] = float(neg_delta_sum)
    feats["hp_diff_net_delta"] = float(pos_delta_sum - neg_delta_sum)
    feats["hp_diff_sign_changes"] = float(lead_change_count)
    feats["longest_ahead_streak"] = float(longest_ahead_streak)
    feats["comeback_flag"] = float((safe_mean(segs["early"]["hp_diff"]) < 0) and (hp_diff_list[-1] > 0))

    # damage dynamics
    feats["p1_damage_total"] = float(p1_damage_total)
    feats["p2_damage_total"] = float(p2_damage_total)
    feats["p1_damage_mean"] = ratio(p1_damage_total, n_turns)
    feats["p2_damage_mean"] = ratio(p2_damage_total, n_turns)
    p1_dpt = np.maximum(0.0, -np.diff([*p2_hp_list, p2_hp_list[-1]])) if len(p2_hp_list) else np.array([])
    p2_dpt = np.maximum(0.0, -np.diff([*p1_hp_list, p1_hp_list[-1]])) if len(p1_hp_list) else np.array([])
    feats["p1_dpt_q25"] = safe_quantile(p1_dpt, 0.25)
    feats["p1_dpt_q50"] = safe_quantile(p1_dpt, 0.50)
    feats["p1_dpt_q75"] = safe_quantile(p1_dpt, 0.75)
    feats["p2_dpt_q25"] = safe_quantile(p2_dpt, 0.25)
    feats["p2_dpt_q50"] = safe_quantile(p2_dpt, 0.50)
    feats["p2_dpt_q75"] = safe_quantile(p2_dpt, 0.75)
    last_seg = slice(max(0, n_turns - 5), n_turns)
    feats["p1_last_segment_dpt"] = safe_mean(p1_dpt[last_seg]) if p1_dpt.size else 0.0
    feats["p2_last_segment_dpt"] = safe_mean(p2_dpt[last_seg]) if p2_dpt.size else 0.0

    # ahead ratios
    feats["p1_ahead_ratio"] = ratio(total_ahead_turns, n_turns)
    for seg in ["early", "mid", "late"]:
        feats[f"p1_{seg}_ahead_ratio"] = ratio(seg_ahead[seg], seg_counts[seg])

    # segment means/slopes
    for seg_name, seg_data in segs.items():
        p1_seg, p2_seg, diff_seg = seg_data["p1_hp"], seg_data["p2_hp"], seg_data["hp_diff"]
        if diff_seg:
            feats[f"{seg_name}_hp_diff_mean"] = safe_mean(diff_seg)
            feats[f"{seg_name}_hp_diff_last"] = float(diff_seg[-1])
            feats[f"{seg_name}_p1_hp_mean"] = safe_mean(p1_seg)
            feats[f"{seg_name}_p2_hp_mean"] = safe_mean(p2_seg)
            feats[f"{seg_name}_hp_diff_slope"] = float(diff_seg[-1] - diff_seg[0])
        else:
            feats[f"{seg_name}_hp_diff_mean"] = 0.0
            feats[f"{seg_name}_hp_diff_last"] = 0.0
            feats[f"{seg_name}_p1_hp_mean"] = 0.0
            feats[f"{seg_name}_p2_hp_mean"] = 0.0
            feats[f"{seg_name}_hp_diff_slope"] = 0.0

    # moves & aggression
    feats["p1_moves_used"] = float(p1_moves_used)
    feats["p2_moves_used"] = float(p2_moves_used)
    feats["p1_damaging_moves_count"] = float(len(p1_damaging_bp))
    feats["p2_damaging_moves_count"] = float(len(p2_damaging_bp))
    feats["p1_status_moves"] = float(p1_status_moves)
    feats["p2_status_moves"] = float(p2_status_moves)
    feats["p1_damaging_ratio"] = ratio(len(p1_damaging_bp), p1_moves_used)
    feats["p2_damaging_ratio"] = ratio(len(p2_damaging_bp), p2_moves_used)
    feats["p1_super_effective_hits"] = float(p1_super_eff_hits)
    feats["p2_super_effective_hits"] = float(p2_super_eff_hits)
    feats["se_hits_diff"] = float(p1_super_eff_hits - p2_super_eff_hits)
    feats["p1_priority_moves"] = float(p1_priority_moves)
    feats["p2_priority_moves"] = float(p2_priority_moves)
    feats["p1_bp_mean"] = safe_mean(p1_damaging_bp)
    feats["p2_bp_mean"] = safe_mean(p2_damaging_bp)
    feats["p1_acc_mean"] = safe_mean(p1_acc_list, default=100.0)
    feats["p2_acc_mean"] = safe_mean(p2_acc_list, default=100.0)

    # boosts
    feats["p1_total_boost_mean"] = safe_mean(p1_total_boosts)
    feats["p2_total_boost_mean"] = safe_mean(p2_total_boosts)
    feats["boost_total_diff_mean"] = feats["p1_total_boost_mean"] - feats["p2_total_boost_mean"]
    feats["p1_total_boost_max"] = float(max(p1_total_boosts))
    feats["p2_total_boost_max"] = float(max(p2_total_boosts))
    feats["boost_total_diff_max"] = feats["p1_total_boost_max"] - feats["p2_total_boost_max"]
    feats["p1_total_boost_std"] = safe_std(p1_total_boosts)
    feats["p2_total_boost_std"] = safe_std(p2_total_boosts)
    feats["p1_total_boost_sum"] = float(np.sum(p1_total_boosts))
    feats["p2_total_boost_sum"] = float(np.sum(p2_total_boosts))

    # status distributions + majors and key statuses
    def add_status(side: str, counter: Counter):
        total = sum(counter.values()) or 1
        for status, cnt in counter.items():
            feats[f"{side}_status_{status}"] = float(cnt) / total
        feats[f"major_status_ratio_{side}"] = 1.0 - (counter.get("nostatus", 0) / total)
        feats[f"{side}_sleep_freeze_ratio"] = ratio(counter.get("slp", 0) + counter.get("frz", 0), total)
        feats[f"{side}_toxic_burn_ratio"] = ratio(counter.get("tox", 0) + counter.get("brn", 0), total)

    add_status("p1", p1_status_counts)
    add_status("p2", p2_status_counts)

    # switches & KOs
    feats["p1_switches"] = float(p1_switches)
    feats["p2_switches"] = float(p2_switches)
    feats["switch_diff"] = float(p1_switches - p2_switches)
    feats["p1_switch_rate"] = ratio(p1_switches, n_turns)
    feats["p2_switch_rate"] = ratio(p2_switches, n_turns)
    feats["p1_ko_like"] = float(p1_ko_like)
    feats["p2_ko_like"] = float(p2_ko_like)
    feats["ko_like_diff"] = float(p1_ko_like - p2_ko_like)
    feats["first_ko_turn_p1"] = float(first_ko_turn_p1)
    feats["first_ko_turn_p2"] = float(first_ko_turn_p2)
    feats["first_ko_turn_diff"] = float((first_ko_turn_p2 or 0.0) - (first_ko_turn_p1 or 0.0))

    # stall index
    feats["stall_index"] = ratio(p1_status_moves + p2_status_moves, n_turns)

    return feats