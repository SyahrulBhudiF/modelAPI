"""
AI Model Serving API - Main Application

FastAPI-based stateless model inference service.
Entry point for the application with lifespan management for model loading.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Handles model loading on startup and cleanup on shutdown.
    """
    # Startup: Load models
    settings = get_settings()

    # Import here to avoid circular imports
    from app.services.model_registry import registry

    print("=" * 50)
    print("Starting AI Model Serving API")
    print("=" * 50)

    # Parse and load models from registry config
    if settings.MODEL_REGISTRY:
        print(f"Loading models from registry: {settings.MODEL_REGISTRY}")
        registry.load_from_config(settings.MODEL_REGISTRY)
        registry.load_all()
        print(f"Loaded models: {registry.list_models()}")
    else:
        print("Warning: No models configured in MODEL_REGISTRY")

    print("=" * 50)
    print("API Ready")
    print("=" * 50)

    yield

    # Shutdown: Cleanup
    print("Shutting down API...")
    registry.unload_all()
    print("All models unloaded. Goodbye!")


def create_app() -> FastAPI:
    """
    Application factory for creating the FastAPI app.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="AI Model Serving API",
        description="Stateless model inference service for video-based predictions",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.DEBUG,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=settings.get_cors_methods(),
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)

    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "AI Model Serving API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
