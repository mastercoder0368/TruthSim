"""Configuration loading and management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Global config storage
_config: Optional[Dict[str, Any]] = None
_models_config: Optional[Dict[str, Any]] = None


def find_project_root() -> Path:
    """Find the project root directory by looking for config/ folder."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "config").exists():
            return parent
    return Path.cwd()


def load_config(
        config_path: Optional[str] = None,
        models_path: Optional[str] = None,
        env_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load configuration from YAML files and environment variables.

    Args:
        config_path: Path to config.yaml (default: config/config.yaml)
        models_path: Path to models.yaml (default: config/models.yaml)
        env_path: Path to .env file (default: .env)

    Returns:
        Dictionary containing merged configuration
    """
    global _config, _models_config

    project_root = find_project_root()

    # Load environment variables
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv(project_root / ".env")

    # Load main config
    if config_path is None:
        config_path = project_root / "config" / "config.yaml"

    with open(config_path, "r") as f:
        _config = yaml.safe_load(f)

    # Load models config
    if models_path is None:
        models_path = project_root / "config" / "models.yaml"

    with open(models_path, "r") as f:
        _models_config = yaml.safe_load(f)

    # Merge configs
    merged = {**_config, "models": _models_config}

    # Override with environment variables where applicable
    merged = _apply_env_overrides(merged)

    return merged


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config."""

    # API Keys
    config["api_keys"] = {
        "umls": os.getenv("UMLS_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "together": os.getenv("TOGETHER_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "huggingface": os.getenv("HF_TOKEN"),
    }

    # Path overrides
    if os.getenv("DATA_DIR"):
        config["paths"]["data_dir"] = os.getenv("DATA_DIR")
    if os.getenv("OUTPUT_DIR"):
        config["paths"]["output_dir"] = os.getenv("OUTPUT_DIR")
    if os.getenv("CACHE_DIR"):
        config["paths"]["umls_cache"] = os.getenv("CACHE_DIR")

    return config


def get_config() -> Dict[str, Any]:
    """Get the loaded configuration. Loads default config if not already loaded."""
    global _config
    if _config is None:
        return load_config()
    return _config


def get_api_key(service: str) -> Optional[str]:
    """Get API key for a specific service."""
    config = get_config()
    return config.get("api_keys", {}).get(service)


def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific model."""
    config = get_config()
    models = config.get("models", {}).get("doctor_models", [])

    for model in models:
        if model.get("name") == model_name:
            return model

    return None


def get_prompt_path(prompt_name: str) -> Path:
    """Get the path to a prompt file."""
    project_root = find_project_root()
    return project_root / "prompts" / f"{prompt_name}.txt"


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from file."""
    prompt_path = get_prompt_path(prompt_name)

    if not prompt_path.exists():
        # Try .yaml extension
        prompt_path = prompt_path.with_suffix(".yaml")

    with open(prompt_path, "r") as f:
        return f.read()


def load_noise_behaviors() -> Dict[str, Any]:
    """Load noise behavior descriptions."""
    project_root = find_project_root()
    noise_path = project_root / "prompts" / "noise_behaviors.yaml"

    with open(noise_path, "r") as f:
        return yaml.safe_load(f)
