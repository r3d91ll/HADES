"""
Minimal OAuth 2.0 Provider for FastAPI-MCP

Implements the minimum OAuth 2.0 endpoints required by FastAPI-MCP.
This is a simplified implementation for lab environments.

⚠️  WARNING: LAB ENVIRONMENT ONLY ⚠️
"""

import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Union
from fastapi.responses import Response
from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from .token_manager import TokenManager

logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class TokenRequest(BaseModel):
    grant_type: str
    code: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    redirect_uri: Optional[str] = None
    code_verifier: Optional[str] = None  # PKCE parameter

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    scope: Optional[str] = None

class TokenIntrospectionRequest(BaseModel):
    token: str
    token_type_hint: Optional[str] = None

class TokenIntrospectionResponse(BaseModel):
    active: bool
    client_id: Optional[str] = None
    username: Optional[str] = None
    scope: Optional[str] = None
    exp: Optional[int] = None
    iat: Optional[int] = None

class ClientRegistrationRequest(BaseModel):
    client_name: str
    client_uri: Optional[str] = None
    redirect_uris: Optional[list] = None
    grant_types: Optional[list] = None
    response_types: Optional[list] = None
    scope: Optional[str] = None

class ClientRegistrationResponse(BaseModel):
    client_id: str
    client_secret: str
    client_name: str
    redirect_uris: list
    grant_types: list
    response_types: list

class MinimalOAuthProvider:
    """
    Minimal OAuth 2.0 Provider that satisfies FastAPI-MCP requirements.
    
    This implements a simplified OAuth 2.0 flow:
    1. Authorization endpoint returns pre-approved codes
    2. Token endpoint exchanges codes for API keys
    3. Introspection validates tokens
    4. Revocation deactivates tokens
    """
    
    def __init__(self, token_manager: TokenManager, base_url: str = "http://localhost:8595"):
        """
        Initialize the OAuth provider.
        
        Args:
            token_manager: Token management instance
            base_url: Base URL for OAuth endpoints
        """
        self.token_manager = token_manager
        self.base_url = base_url.rstrip('/')
        self.security = HTTPBearer(auto_error=False)
        
        # OAuth metadata
        self.metadata = {
            "issuer": self.base_url,
            "authorization_endpoint": f"{self.base_url}/oauth/authorize",
            "token_endpoint": f"{self.base_url}/oauth/token",
            "revocation_endpoint": f"{self.base_url}/oauth/revoke",
            "introspection_endpoint": f"{self.base_url}/oauth/introspect",
            "registration_endpoint": f"{self.base_url}/oauth/register",
            "scopes_supported": ["openid", "profile", "email", "api"],
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "client_credentials"],
            "token_endpoint_auth_methods_supported": ["none", "client_secret_basic"],
            "code_challenge_methods_supported": ["S256"],
        }
    
    def setup_routes(self, app: FastAPI) -> None:
        """Add OAuth routes to FastAPI app."""
        
        @app.get("/.well-known/oauth-authorization-server")
        async def oauth_metadata() -> JSONResponse:
            """OAuth 2.0 Authorization Server Metadata (RFC 8414)."""
            return JSONResponse(content=self.metadata)
        
        @app.get("/oauth/authorize")
        async def authorize(
            response_type: str,
            client_id: str,
            redirect_uri: Optional[str] = None,
            scope: Optional[str] = None,
            state: Optional[str] = None,
            code_challenge: Optional[str] = None,
            code_challenge_method: Optional[str] = None
        ) -> Union[RedirectResponse, JSONResponse]:
            """
            Authorization endpoint.
            
            In a real OAuth flow, this would show a login page.
            For our minimal implementation, we auto-approve if the client_id
            corresponds to a valid agent.
            """
            logger.info(f"Authorization request for client_id: {client_id}")
            
            if response_type != "code":
                raise HTTPException(status_code=400, detail="unsupported_response_type")
            
            # Check if client_id is a valid agent
            agent_info = self.token_manager.get_agent_info(client_id)
            if not agent_info or not agent_info["is_active"]:
                raise HTTPException(status_code=400, detail="invalid_client")
            
            # Generate authorization code
            try:
                auth_code = self.token_manager.create_auth_code(client_id)
                
                # In a real implementation, we'd redirect to redirect_uri
                # For simplicity, return the code directly
                response_data = {
                    "code": auth_code,
                    "state": state
                }
                
                if redirect_uri:
                    # Build redirect URL
                    redirect_url = f"{redirect_uri}?code={auth_code}"
                    if state:
                        redirect_url += f"&state={state}"
                    return RedirectResponse(url=redirect_url)
                else:
                    return JSONResponse(content=response_data)
                    
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/oauth/token", response_model=TokenResponse)
        async def token_endpoint(
            grant_type: str = Form(...),
            code: Optional[str] = Form(None),
            client_id: Optional[str] = Form(None),
            client_secret: Optional[str] = Form(None),
            redirect_uri: Optional[str] = Form(None),
            code_verifier: Optional[str] = Form(None)
        ) -> TokenResponse:
            """
            Token endpoint.
            
            Exchanges authorization codes for access tokens.
            """
            logger.info(f"Token request: grant_type={grant_type}")
            
            if grant_type == "authorization_code":
                if not code:
                    raise HTTPException(status_code=400, detail="missing_code")
                
                # Note: For simplicity, we're not validating PKCE code_verifier
                # In production, you should validate code_verifier against code_challenge
                if code_verifier:
                    logger.info("PKCE code_verifier received (validation skipped in lab environment)")
                
                # Exchange auth code for agent ID
                agent_id = self.token_manager.exchange_auth_code(code)
                if not agent_id:
                    raise HTTPException(status_code=400, detail="invalid_grant")
                
                # Generate access token
                access_token = self.token_manager.create_token(agent_id, expires_in_days=30)
                
                return TokenResponse(
                    access_token=access_token,
                    token_type="Bearer",
                    expires_in=30 * 24 * 3600,  # 30 days in seconds
                    scope="api"
                )
                
            elif grant_type == "client_credentials":
                # Direct token for service accounts
                if not client_id:
                    raise HTTPException(status_code=400, detail="missing_client_id")
                
                # Verify client exists
                agent_info = self.token_manager.get_agent_info(client_id)
                if not agent_info or not agent_info["is_active"]:
                    raise HTTPException(status_code=400, detail="invalid_client")
                
                # Generate access token
                access_token = self.token_manager.create_token(client_id, expires_in_days=30)
                
                return TokenResponse(
                    access_token=access_token,
                    token_type="Bearer",
                    expires_in=30 * 24 * 3600,
                    scope="api"
                )
            else:
                raise HTTPException(status_code=400, detail="unsupported_grant_type")
        
        @app.post("/oauth/introspect", response_model=TokenIntrospectionResponse)
        async def introspect_token(request: TokenIntrospectionRequest) -> TokenIntrospectionResponse:
            """
            Token introspection endpoint (RFC 7662).
            
            Validates tokens and returns metadata.
            """
            logger.info("Token introspection request")
            
            agent_id = self.token_manager.validate_token(request.token)
            
            if agent_id:
                agent_info = self.token_manager.get_agent_info(agent_id)
                return TokenIntrospectionResponse(
                    active=True,
                    client_id=agent_id,
                    username=agent_info["name"] if agent_info else agent_id,
                    scope="api",
                    exp=int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp()),
                    iat=int(datetime.now(timezone.utc).timestamp())
                )
            else:
                return TokenIntrospectionResponse(active=False)
        
        @app.post("/oauth/revoke")
        async def revoke_token(
            token: str = Form(...),
            token_type_hint: Optional[str] = Form(None)
        ) -> JSONResponse:
            """
            Token revocation endpoint (RFC 7009).
            
            Revokes access tokens.
            """
            logger.info("Token revocation request")
            
            success = self.token_manager.revoke_token(token)
            
            # Always return 200 OK per RFC 7009
            return JSONResponse(content={"revoked": success})
        
        @app.post("/oauth/register", response_model=ClientRegistrationResponse)
        async def register_client(request: ClientRegistrationRequest) -> ClientRegistrationResponse:
            """
            Dynamic Client Registration endpoint (RFC 7591).
            
            This is a simplified implementation for FastAPI-MCP compatibility.
            """
            logger.info(f"Client registration request: {request.client_name}")
            
            # Generate client credentials
            # Generate URL-safe tokens
            # Generate URL-safe tokens
            client_id = f"client_{self.token_manager.generate_token(8).replace('+', '-').replace('/', '_').replace('=', '')}"
            client_secret = self.token_manager.generate_token(32)
            
            # Create agent for this client
            try:
                self.token_manager.create_agent(
                    agent_id=client_id,
                    name=request.client_name,
                    description=f"Auto-registered client: {request.client_name}"
                )
                
                return ClientRegistrationResponse(
                    client_id=client_id,
                    client_secret=client_secret,
                    client_name=request.client_name,
                    redirect_uris=request.redirect_uris or [],
                    grant_types=request.grant_types or ["authorization_code"],
                    response_types=request.response_types or ["code"]
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    def validate_bearer_token(self, credentials: Optional[HTTPAuthorizationCredentials]) -> str:
        """
        Validate bearer token and return agent_id.
        
        Use this as a FastAPI dependency for protected endpoints.
        """
        if not credentials:
            raise HTTPException(status_code=401, detail="Missing authorization header")
        
        if credentials.scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        agent_id = self.token_manager.validate_token(credentials.credentials)
        if not agent_id:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        return agent_id
    
    def get_current_agent(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> str:
        """FastAPI dependency to get current authenticated agent."""
        return self.validate_bearer_token(credentials)