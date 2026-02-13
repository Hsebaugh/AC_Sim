import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")

DEFAULTS = {
    "fps": 60,
    "grid": {"x": 32, "y": 32, "z": 16},
    "room": {"width": 3.0, "length": 3.0, "height": 3.0},
    "temperature": {"indoor": 25.0, "outdoor": 35.0},
    "units": "standard",
}


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            user_cfg = json.load(f)
        merged = {**DEFAULTS, **user_cfg}
        return merged
    return dict(DEFAULTS)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
