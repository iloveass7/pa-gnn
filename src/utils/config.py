"""
Configuration loader utility.
Loads YAML configs with dot-notation access and merge support.
"""

import yaml
from pathlib import Path
from copy import deepcopy


class ConfigDict(dict):
    """Dictionary subclass with dot-notation attribute access."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __repr__(self):
        return f"ConfigDict({dict.__repr__(self)})"


def _convert_to_config_dict(d):
    """Recursively convert nested dicts to ConfigDict."""
    if isinstance(d, dict):
        return ConfigDict({k: _convert_to_config_dict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_convert_to_config_dict(item) for item in d]
    return d


def load_config(path):
    """
    Load a YAML configuration file.
    
    Args:
        path: Path to YAML file (str or Path).
        
    Returns:
        ConfigDict with dot-notation access.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    
    if raw is None:
        return ConfigDict()
    
    return _convert_to_config_dict(raw)


def merge_configs(base, override):
    """
    Deep-merge override into base config. Override values take precedence.
    
    Args:
        base: Base ConfigDict.
        override: Override ConfigDict (values from this take priority).
        
    Returns:
        New merged ConfigDict.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return ConfigDict(merged)


def save_config(config, path):
    """
    Save a ConfigDict to a YAML file.
    
    Args:
        config: ConfigDict to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert ConfigDict back to regular dict for YAML serialization
    def to_dict(d):
        if isinstance(d, dict):
            return {k: to_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [to_dict(item) for item in d]
        return d
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(to_dict(config), f, default_flow_style=False, sort_keys=False)
