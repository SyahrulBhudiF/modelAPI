"""
API Routes

Defines the inference endpoints for the model serving API.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.security import verify_token
from app.services.model_registry import registry

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/models", dependencies=[Depends(verify_token)])
async def list_models() -> dict[str, list[str]]:
    """
    List all available models.

    Returns:
        Dictionary containing list of registered model names.
    """
    return {"models": registry.list_models()}


@router.post("/infer/{model_name}", dependencies=[Depends(verify_token)])
async def infer(
    model_name: str,
    files: Annotated[
        list[UploadFile], File(description="Video segment files for inference")
    ],
) -> dict[str, Any]:
    """
    Run inference on the specified model.

    Multiple video segment files are merged and processed as a single input,
    producing one prediction result.

    Args:
        model_name: Name of the model to use for inference.
        files: Video segment files (will be merged into single prediction).

    Returns:
        Dictionary containing single prediction result.

    Raises:
        HTTPException: 404 if model not found, 500 on inference error.
    """
    # Check if model exists
    handler = registry.get(model_name)
    if handler is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found. Available models: {registry.list_models()}",
        )

    # Check if model is loaded
    if not handler.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_name}' is not loaded. Please try again later.",
        )

    try:
        # Read all file segments
        segments = []
        filenames = []
        for file in files:
            content = await file.read()
            segments.append(content)
            filenames.append(file.filename)

        # Run inference with all segments merged
        result = handler.infer(segments)

        return {
            "model": model_name,
            "segments": filenames,
            "total_segments": len(segments),
            "prediction": result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@router.get("/models/{model_name}/status", dependencies=[Depends(verify_token)])
async def model_status(model_name: str) -> dict[str, Any]:
    """
    Get status of a specific model.

    Args:
        model_name: Name of the model to check.

    Returns:
        Dictionary containing model status information.
    """
    handler = registry.get(model_name)
    if handler is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found.",
        )

    return {
        "model": model_name,
        "loaded": handler.is_loaded,
        "model_dir": handler.model_dir,
    }
