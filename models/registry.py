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
