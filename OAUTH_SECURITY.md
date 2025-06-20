# HADES OAuth 2.0 Security Documentation

⚠️ **CRITICAL SECURITY WARNING** ⚠️

**This OAuth 2.0 implementation is designed EXCLUSIVELY for LAB and DEVELOPMENT environments.**

**DO NOT USE IN PRODUCTION without proper security review and hardening.**

## Current Implementation: Lab Environment Only

### What We've Built

HADES includes a minimal OAuth 2.0 provider that satisfies the FastAPI-MCP requirements while providing basic authentication for multiple agents. This implementation includes:

- **Token Management**: SQLite-based storage with secure token generation
- **OAuth 2.0 Endpoints**: Authorization, token, introspection, and revocation endpoints
- **Admin Tools**: Simple CLI script for agent/token management
- **FastAPI Integration**: Works seamlessly with FastAPI-MCP

### Security Features (Lab Level)

✅ **Bearer Token Authentication**: API endpoints require valid tokens
✅ **Token Validation**: Proper token hashing and expiration
✅ **Agent Isolation**: Each agent has separate tokens and logging
✅ **OAuth 2.0 Compliance**: Meets FastAPI-MCP specification requirements
✅ **Audit Trail**: Token usage logging and agent activity tracking

### Security Limitations (Why Not Production Ready)

❌ **No Rate Limiting**: Vulnerable to brute force attacks
❌ **No HTTPS Enforcement**: Tokens transmitted in plaintext over HTTP
❌ **Simple Token Storage**: SQLite database without encryption at rest
❌ **No Session Management**: No refresh tokens or session invalidation
❌ **Minimal Input Validation**: Basic validation only
❌ **No OAuth Scopes**: All tokens have full API access
❌ **No PKCE Enforcement**: Code exchange without proof key
❌ **No Client Secret Verification**: Simplified client authentication
❌ **No Audit Logging**: No security event logging for production standards
❌ **No DoS Protection**: No request throttling or resource limits

## Lab Environment Usage

### Quick Start

1. **Create an Agent**:
   ```bash
   ./scripts/manage_agents.sh create my-agent "My Test Agent" "Testing HADES API"
   ```

2. **Get Token** (displayed after creation):
   ```
   Token: abc123xyz789...
   ```

3. **Use in MCP Client**:
   ```json
   {
     "mcpServers": {
       "hades-my-agent": {
         "command": "npx",
         "args": ["@modelcontextprotocol/server-fetch"],
         "env": {
           "FETCH_BASE_URL": "http://localhost:8595",
           "FETCH_API_KEY": "abc123xyz789..."
         }
       }
     }
   }
   ```

### Admin Commands

```bash
# List all agents
./scripts/manage_agents.sh list

# Get agent details
./scripts/manage_agents.sh info my-agent

# Generate new token
./scripts/manage_agents.sh token my-agent

# Revoke all tokens for agent
./scripts/manage_agents.sh revoke my-agent

# Deactivate agent
./scripts/manage_agents.sh deactivate my-agent
```

### API Usage

```bash
# Authenticated request
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "max_results": 5}' \
     http://localhost:8595/query

# OAuth metadata (for MCP clients)
curl http://localhost:8595/.well-known/oauth-authorization-server
```

## Production Migration Guide

For production or enterprise environments, replace the minimal OAuth provider with enterprise-grade solutions:

### Recommended Production OAuth Providers

1. **Auth0** (Managed Service)
   - Full OAuth 2.0/OpenID Connect support
   - Enterprise security features
   - Multi-tenant support
   - Advanced threat detection

2. **Keycloak** (Self-Hosted)
   - Open source identity management
   - SAML and OAuth 2.0 support
   - Role-based access control
   - Integration with LDAP/Active Directory

3. **AWS Cognito** (Cloud Native)
   - Serverless identity management
   - Integration with AWS services
   - Built-in security features
   - Scalable and managed

4. **Okta** (Enterprise)
   - Identity and access management
   - Single sign-on (SSO)
   - Multi-factor authentication
   - Compliance certifications

### Production Security Requirements

When migrating to production, ensure your OAuth provider includes:

#### Authentication Security
- [ ] **HTTPS Everywhere**: All endpoints use TLS 1.2+
- [ ] **Strong Password Policies**: Minimum complexity requirements
- [ ] **Multi-Factor Authentication**: Required for all users
- [ ] **Account Lockout**: Protection against brute force attacks
- [ ] **Session Management**: Secure session handling and timeout

#### Authorization Security  
- [ ] **OAuth 2.0 Scopes**: Fine-grained permission control
- [ ] **PKCE (RFC 7636)**: Proof Key for Code Exchange
- [ ] **Client Authentication**: Proper client secret management
- [ ] **Token Security**: Short-lived access tokens with refresh tokens
- [ ] **Role-Based Access Control**: Proper permission hierarchies

#### Infrastructure Security
- [ ] **Rate Limiting**: Protection against DoS attacks
- [ ] **Input Validation**: Comprehensive input sanitization
- [ ] **SQL Injection Protection**: Parameterized queries only
- [ ] **CORS Configuration**: Restrictive cross-origin policies
- [ ] **Security Headers**: HSTS, CSP, X-Frame-Options, etc.

#### Monitoring & Compliance
- [ ] **Audit Logging**: Complete security event logging
- [ ] **Threat Detection**: Real-time security monitoring
- [ ] **Vulnerability Scanning**: Regular security assessments
- [ ] **Compliance Certifications**: SOC 2, ISO 27001, etc.
- [ ] **Data Encryption**: At rest and in transit

### Migration Steps

1. **Choose Production OAuth Provider**: Select based on your infrastructure and compliance needs

2. **Update FastAPI-MCP Configuration**:
   ```python
   from fastapi_mcp.types import AuthConfig
   
   auth_config = AuthConfig(
       issuer="https://your-auth-provider.com",
       authorize_url="https://your-auth-provider.com/oauth/authorize",
       # Configure according to your provider
   )
   ```

3. **Remove Minimal OAuth Code**:
   - Delete `src/auth/` directory
   - Remove OAuth routes from `server.py`
   - Update dependencies

4. **Configure Client Applications**: Update MCP clients to use production OAuth flow

5. **Test Security**: Perform penetration testing and security review

## Database Schema

The minimal OAuth implementation uses the following SQLite schema:

```sql
-- Agents (users/applications)
CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL,
    last_used TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- API Tokens
CREATE TABLE tokens (
    token_hash TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
);

-- OAuth Authorization Codes (temporary)
CREATE TABLE auth_codes (
    code TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    is_used BOOLEAN DEFAULT 0,
    FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
);
```

## Troubleshooting

### Common Issues

1. **"Not authenticated" errors**: Check token format and validity
2. **OAuth validation errors**: Verify OAuth metadata endpoints are accessible
3. **FastAPI-MCP mount failures**: Ensure all OAuth endpoints return proper responses

### Debug Commands

```bash
# Check OAuth metadata
curl http://localhost:8595/.well-known/oauth-authorization-server

# Validate token
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8595/health

# Check server logs
tail -f logs/hades-service.log

# View agent database
sqlite3 auth/auth.db "SELECT * FROM agents;"
```

## Support

For production migration assistance or security questions:

1. Review this documentation thoroughly
2. Consult with your security team
3. Consider hiring OAuth/security specialists
4. Perform thorough security testing

Remember: **Security is not optional in production environments.**

---

**Generated by HADES Development Team**  
**Last Updated**: 2025-06-19  
**Version**: 1.0.0 (Lab Environment Only)