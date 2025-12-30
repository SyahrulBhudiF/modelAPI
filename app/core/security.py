"""
Security module for Bearer token authentication.

Provides FastAPI dependency for validating static Bearer tokens.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings: Settings = Depends(get_settings),
) -> str:
    """
    Validate Bearer token from request header.

    Args:
        credentials: HTTP Authorization credentials from request
        settings: Application settings containing the valid token

    Returns:
        The validated token string

    Raises:
        HTTPException: 401 if token is invalid or missing
    """
    token = credentials.credentials

    if token != settings.AI_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token


# Dependency alias for cleaner route definitions
require_auth = Depends(verify_token)
