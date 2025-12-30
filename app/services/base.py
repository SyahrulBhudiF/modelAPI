"""
Base handler interface for model services.

All model handlers must inherit from BaseHandler and implement
the required methods for loading, preprocessing, and prediction.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseHandler(ABC):
    """
    Abstract base class for model handlers.

    Each handler is responsible for:
    - Loading model checkpoints and artifacts
    - Performing input preprocessing
    - Running inference
    """

    def __init__(self, model_dir: str):
        """
        Initialize the handler.

        Args:
            model_dir: Directory containing model artifacts (checkpoints, scalers, etc.)
        """
        self.model_dir = model_dir
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """
        Load model checkpoints and preprocessing artifacts.

        This method is called once during application startup.
        After successful loading, set self._is_loaded = True.
        """
        pass

    @abstractmethod
    def preprocess(self, input_data: list[bytes] | bytes | Any) -> Any:
        """
        Preprocess raw input data into model-ready format.

        Args:
            input_data: Raw input data. Can be:
                - list[bytes]: Multiple video segments to be merged
                - bytes: Single video file
                - Any: Other input formats supported by the handler

        Returns:
            Preprocessed data ready for model inference (single tensor).
        """
        pass

    @abstractmethod
    def predict(self, preprocessed_data: Any) -> dict[str, Any]:
        """
        Run inference on preprocessed data.

        Args:
            preprocessed_data: Data returned from preprocess()

        Returns:
            Dictionary containing prediction results.
        """
        pass

    def infer(self, input_data: list[bytes] | bytes | Any) -> dict[str, Any]:
        """
        Full inference pipeline: preprocess + predict.

        Args:
            input_data: Raw input data (segments list or single file)

        Returns:
            Dictionary containing single prediction result.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() before inference.")

        preprocessed = self.preprocess(input_data)
        return self.predict(preprocessed)

    def unload(self) -> None:
        """
        Optional: Unload model from memory.

        Override this method if your handler needs cleanup logic.
        """
        self._is_loaded = False
