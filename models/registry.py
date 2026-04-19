REGISTRY = {}

def register(name, input_type: str, task_type: str):
    def decorator(cls):
        assert name not in REGISTRY, f"Model name '{name}' is already registered."
        REGISTRY[name] = {
            "class": cls,
            "input_type": input_type,
            "task_type": task_type
        }
        return cls
    return decorator


def class_to_config(model_class):
    for name, cfg in REGISTRY.items():
        if cfg["class"] == model_class:
            return name, cfg
    raise ValueError(f"Model class {model_class} not found in registry.")


def load_registered_model(path, map_location=None):
    """Load either a legacy pickled model or a registry/state_dict checkpoint."""
    import torch

    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(loaded, dict) or "state_dict" not in loaded:
        return loaded

    model_name = loaded.get("model")
    if model_name not in REGISTRY:
        raise ValueError(f"Checkpoint model '{model_name}' not found in registry.")

    model = REGISTRY[model_name]["class"]()
    model.load_state_dict(loaded["state_dict"])
    return model
