# src/fds_pokemon/data.py
import json
import pandas as pd
from typing import Any, Dict

from .features import extract_features_from_battle


def load_dataset(path: str, is_train: bool) -> pd.DataFrame:
    """
    Load the JSONL dataset at `path` and return a feature DataFrame
    indexed by battle_id. If `is_train=True`, includes 'player_won'.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            battle: Dict[str, Any] = json.loads(line)
            feat = extract_features_from_battle(battle)
            feat["battle_id"] = battle["battle_id"]
            if is_train:
                feat["player_won"] = int(battle["player_won"])
            rows.append(feat)
    df = pd.DataFrame(rows).set_index("battle_id").sort_index()
    return df
