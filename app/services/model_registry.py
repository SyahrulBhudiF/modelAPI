"""
Model Registry

Manages the mapping between model names and their handler instances.
Parses MODEL_REGISTRY environment variable and instantiates handlers.
"""

from app.services.base import BaseHandler
from app.services.handlers.tabr_handler import TabRHandler

# Handler type mapping
HANDLER_TYPES: dict[str, type[BaseHandler]] = {
    "tabr": TabRHandler,
}


class ModelRegistry:
    """
    Registry for model handlers.

    Parses the MODEL_REGISTRY config and instantiates appropriate handlers.
    Format: model_name:handler_key:model_dir[,model_name:handler_key:model_dir,...]
    """

    def __init__(self):
        self._handlers: dict[str, BaseHandler] = {}

    def register(self, model_name: str, handler: BaseHandler) -> None:
        """Register a handler instance with a model name."""
        self._handlers[model_name] = handler

    def get(self, model_name: str) -> BaseHandler | None:
        """Get a handler by model name."""
        return self._handlers.get(model_name)

    def has(self, model_name: str) -> bool:
        """Check if a model is registered."""
        return model_name in self._handlers

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._handlers.keys())

    def load_from_config(self, registry_config: str) -> None:
        """
        Parse MODEL_REGISTRY config and load handlers.

        Args:
            registry_config: Comma-separated entries of model_name:handler_key:model_dir
        """
        if not registry_config:
            return

        entries = registry_config.split(",")
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            parts = entry.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid registry entry: {entry}. "
                    "Expected format: model_name:handler_key:model_dir"
                )

            model_name, handler_key, model_dir = parts

            if handler_key not in HANDLER_TYPES:
                raise ValueError(
                    f"Unknown handler key: {handler_key}. "
                    f"Available handlers: {list(HANDLER_TYPES.keys())}"
                )

            handler_class = HANDLER_TYPES[handler_key]
            handler = handler_class(model_dir=model_dir)
            self.register(model_name, handler)

    def load_all(self) -> None:
        """Load all registered models into memory."""
        for model_name, handler in self._handlers.items():
            print(f"Loading model: {model_name}")
            handler.load()
            print(f"Model loaded: {model_name}")

    def unload_all(self) -> None:
        """Unload all models from memory."""
        for model_name, handler in self._handlers.items():
            print(f"Unloading model: {model_name}")
            handler.unload()


# Global registry instance
registry = ModelRegistry()
