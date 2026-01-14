import yaml
from pathlib import Path


def load_config_by_id(config_path: str, exp_id: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config_list = yaml.safe_load(f)
    for config in config_list:
        if config["id"] == exp_id:
            return config
    raise ValueError(f"Experiment ID '{exp_id}' not found in {config_path}.")

