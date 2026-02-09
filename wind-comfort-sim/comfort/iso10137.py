import yaml

def load_iso_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_bands(cfg):
    bands = [(b["label"], b["max_mg"], b.get("color", "#eeeeee")) for b in cfg["limits"]]
    return bands, cfg.get("window_sec", 60)