"""Authentication middleware."""

from typing import Optional
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings


security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health and docs endpoints
        if request.url.path in ["/health", "/health/live", "/health/ready", "/docs", "/redoc", "/openapi.json", "/"]:
            return await call_next(request)
        
        return await call_next(request)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = None,
) -> dict:
    """Validate JWT token and return user info.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        User dictionary
        
    Raises:
        HTTPException: If token is invalid
    """
    # If no API key is configured, skip authentication
    if not settings.api_key:
        return {"id": "anonymous", "username": "anonymous", "role": "user"}
    
    if credentials is None:
        # Try to get API key from header
        raise HTTPException(
            status_code=401,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid token: missing subject",
            )
        
        return {
            "id": user_id,
            "username": payload.get("username", "unknown"),
            "role": payload.get("role", "user"),
        }
        
    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_api_key(request: Request) -> dict:
    """Verify API key from header.
    
    Args:
        request: FastAPI request
        
    Returns:
        User dictionary
    """
    # If no API key is configured, skip authentication
    if not settings.api_key:
        return {"id": "anonymous", "username": "anonymous", "role": "user"}
    
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
        )
    
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
    
    return {"id": "api_user", "username": "api", "role": "user"}


def create_access_token(data: dict, expires_delta: Optional[int] = None) -> str:
    """Create a JWT access token.
    
    Args:
        data: Token payload data
        expires_delta: Optional expiration in minutes
        
    Returns:
        JWT token string
    """
    from datetime import timedelta, datetime
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expiration_minutes)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    
    return encoded_jwt