import importlib

from ppmat.trainer.base_trainer import BaseTrainer

__all__ = ["BaseTrainer", "build_trainer"]


def build_trainer(cfg, **kwargs):
    """Build trainer from a ``Trainer`` config block.

    ``__class_name__`` may be a class exported by this module (default
    ``BaseTrainer``) or a dotted path such as
    ``ppmat.models.matinvent.trainer.MatInventTrainer``, which keeps
    model-specific trainers inside their own model package.
    """
    class_name = cfg.get("__class_name__", "BaseTrainer")
    init_params = cfg.get("__init_params__", {})
    if "." in class_name:
        module_path, _, attribute = class_name.rpartition(".")
        trainer_cls = getattr(importlib.import_module(module_path), attribute)
    else:
        trainer_cls = eval(class_name)
    return trainer_cls(cfg, **init_params, **kwargs)
