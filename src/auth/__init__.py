"""
Minimal OAuth 2.0 Authentication System for HADES

⚠️  WARNING: LAB ENVIRONMENT ONLY ⚠️
This is a minimal OAuth 2.0 implementation designed for lab/development environments.
DO NOT USE IN PRODUCTION without proper security review and hardening.

For production environments, use:
- Auth0, Keycloak, or other enterprise OAuth providers
- Proper rate limiting and brute force protection
- Secure token storage and rotation
- SSL/TLS termination
- Proper audit logging
"""

from .token_manager import TokenManager
from .oauth_provider import MinimalOAuthProvider

__all__ = ["TokenManager", "MinimalOAuthProvider"]