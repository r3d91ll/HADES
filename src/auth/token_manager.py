"""
Token Management System for Minimal OAuth 2.0

Manages API tokens and agent registration using SQLite for simplicity.
"""

import sqlite3
import secrets
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class Agent:
    """Represents an agent in the system."""
    agent_id: str
    name: str
    description: str
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True

@dataclass
class Token:
    """Represents an API token."""
    token_hash: str
    agent_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

class TokenManager:
    """Manages API tokens and agent registration for minimal OAuth 2.0."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the token manager.
        
        Args:
            db_path: Path to SQLite database file. Defaults to auth.db in project root.
        """
        if db_path is None:
            # Default to project root/auth/auth.db
            project_root = Path(__file__).parent.parent.parent
            auth_dir = project_root / "auth"
            auth_dir.mkdir(exist_ok=True)
            db_path = auth_dir / "auth.db"
        
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Agents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL,
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Tokens table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    token_hash TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
                )
            """)
            
            # Authorization codes table (for OAuth flow)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS auth_codes (
                    code TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    is_used BOOLEAN DEFAULT 0,
                    FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
                )
            """)
            
            conn.commit()
            logger.info(f"Initialized token database at {self.db_path}")
    
    def generate_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_token(self, token: str) -> str:
        """Hash a token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def create_agent(self, agent_id: str, name: str, description: str = "") -> Agent:
        """
        Create a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Optional description
            
        Returns:
            Created Agent object
            
        Raises:
            ValueError: If agent_id already exists
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if agent already exists
            cursor.execute("SELECT agent_id FROM agents WHERE agent_id = ?", (agent_id,))
            if cursor.fetchone():
                raise ValueError(f"Agent {agent_id} already exists")
            
            # Create agent
            now = datetime.now(timezone.utc)
            cursor.execute("""
                INSERT INTO agents (agent_id, name, description, created_at)
                VALUES (?, ?, ?, ?)
            """, (agent_id, name, description, now))
            
            conn.commit()
            
            agent = Agent(
                agent_id=agent_id,
                name=name,
                description=description,
                created_at=now
            )
            
            logger.info(f"Created agent: {agent_id} ({name})")
            return agent
    
    def create_token(self, agent_id: str, expires_in_days: Optional[int] = None) -> str:
        """
        Create a new API token for an agent.
        
        Args:
            agent_id: Agent to create token for
            expires_in_days: Token expiration in days (None for no expiration)
            
        Returns:
            The generated token (plain text - store securely!)
            
        Raises:
            ValueError: If agent doesn't exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Verify agent exists
            cursor.execute("SELECT agent_id FROM agents WHERE agent_id = ? AND is_active = 1", (agent_id,))
            if not cursor.fetchone():
                raise ValueError(f"Agent {agent_id} not found or inactive")
            
            # Generate token
            token = self.generate_token()
            token_hash = self.hash_token(token)
            
            now = datetime.now(timezone.utc)
            expires_at = None
            if expires_in_days:
                expires_at = now + timedelta(days=expires_in_days)
            
            # Store token
            cursor.execute("""
                INSERT INTO tokens (token_hash, agent_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (token_hash, agent_id, now, expires_at))
            
            conn.commit()
            
            logger.info(f"Created token for agent {agent_id} (expires: {expires_at})")
            return token
    
    def validate_token(self, token: str) -> Optional[str]:
        """
        Validate a token and return the associated agent_id.
        
        Args:
            token: Token to validate
            
        Returns:
            Agent ID if token is valid, None otherwise
        """
        token_hash = self.hash_token(token)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check token validity
            cursor.execute("""
                SELECT t.agent_id, t.expires_at, a.is_active
                FROM tokens t
                JOIN agents a ON t.agent_id = a.agent_id
                WHERE t.token_hash = ? AND t.is_active = 1
            """, (token_hash,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            agent_id, expires_at, agent_active = result
            
            # Check if agent is active
            if not agent_active:
                return None
            
            # Check expiration
            if expires_at:
                expires_dt = datetime.fromisoformat(expires_at)
                if datetime.now(timezone.utc) > expires_dt:
                    logger.warning(f"Token for agent {agent_id} has expired")
                    return None
            
            # Update last_used
            cursor.execute("""
                UPDATE agents SET last_used = ? WHERE agent_id = ?
            """, (datetime.now(timezone.utc), agent_id))
            
            conn.commit()
            return agent_id
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token.
        
        Args:
            token: Token to revoke
            
        Returns:
            True if token was revoked, False if not found
        """
        token_hash = self.hash_token(token)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE tokens SET is_active = 0 WHERE token_hash = ?
            """, (token_hash,))
            
            affected = cursor.rowcount > 0
            conn.commit()
            
            if affected:
                logger.info("Token revoked successfully")
            
            return affected
    
    def revoke_agent_tokens(self, agent_id: str) -> int:
        """
        Revoke all tokens for an agent.
        
        Args:
            agent_id: Agent whose tokens to revoke
            
        Returns:
            Number of tokens revoked
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE tokens SET is_active = 0 WHERE agent_id = ?
            """, (agent_id,))
            
            count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Revoked {count} tokens for agent {agent_id}")
            return count
    
    def deactivate_agent(self, agent_id: str) -> bool:
        """
        Deactivate an agent and revoke all its tokens.
        
        Args:
            agent_id: Agent to deactivate
            
        Returns:
            True if agent was deactivated, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Deactivate agent
            cursor.execute("""
                UPDATE agents SET is_active = 0 WHERE agent_id = ?
            """, (agent_id,))
            
            if cursor.rowcount == 0:
                return False
            
            # Revoke all tokens
            cursor.execute("""
                UPDATE tokens SET is_active = 0 WHERE agent_id = ?
            """, (agent_id,))
            
            conn.commit()
            
            logger.info(f"Deactivated agent {agent_id} and revoked all tokens")
            return True
    
    def list_agents(self) -> List[Agent]:
        """List all agents."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT agent_id, name, description, created_at, last_used, is_active
                FROM agents ORDER BY created_at DESC
            """)
            
            agents = []
            for row in cursor.fetchall():
                agent_id, name, description, created_at, last_used, is_active = row
                
                agents.append(Agent(
                    agent_id=agent_id,
                    name=name,
                    description=description,
                    created_at=datetime.fromisoformat(created_at),
                    last_used=datetime.fromisoformat(last_used) if last_used else None,
                    is_active=bool(is_active)
                ))
            
            return agents
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an agent."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get agent info
            cursor.execute("""
                SELECT agent_id, name, description, created_at, last_used, is_active
                FROM agents WHERE agent_id = ?
            """, (agent_id,))
            
            agent_row = cursor.fetchone()
            if not agent_row:
                return None
            
            agent_id, name, description, created_at, last_used, is_active = agent_row
            
            # Get token count
            cursor.execute("""
                SELECT COUNT(*) FROM tokens WHERE agent_id = ? AND is_active = 1
            """, (agent_id,))
            
            active_tokens = cursor.fetchone()[0]
            
            return {
                "agent_id": agent_id,
                "name": name,
                "description": description,
                "created_at": created_at,
                "last_used": last_used,
                "is_active": bool(is_active),
                "active_tokens": active_tokens
            }
    
    # OAuth 2.0 specific methods
    
    def create_auth_code(self, agent_id: str, expires_in_minutes: int = 10) -> str:
        """
        Create an authorization code for OAuth flow.
        
        Args:
            agent_id: Agent to create code for
            expires_in_minutes: Code expiration in minutes
            
        Returns:
            Authorization code
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Verify agent exists
            cursor.execute("SELECT agent_id FROM agents WHERE agent_id = ? AND is_active = 1", (agent_id,))
            if not cursor.fetchone():
                raise ValueError(f"Agent {agent_id} not found or inactive")
            
            # Generate code
            code = self.generate_token(16)  # Shorter for auth codes
            
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(minutes=expires_in_minutes)
            
            # Store code
            cursor.execute("""
                INSERT INTO auth_codes (code, agent_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (code, agent_id, now, expires_at))
            
            conn.commit()
            
            logger.info(f"Created auth code for agent {agent_id}")
            return code
    
    def exchange_auth_code(self, code: str) -> Optional[str]:
        """
        Exchange an authorization code for an agent ID.
        
        Args:
            code: Authorization code
            
        Returns:
            Agent ID if code is valid, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check code validity
            cursor.execute("""
                SELECT agent_id, expires_at, is_used
                FROM auth_codes
                WHERE code = ?
            """, (code,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            agent_id, expires_at, is_used = result
            
            # Check if already used
            if is_used:
                return None
            
            # Check expiration
            expires_dt = datetime.fromisoformat(expires_at)
            if datetime.now(timezone.utc) > expires_dt:
                return None
            
            # Mark as used
            cursor.execute("""
                UPDATE auth_codes SET is_used = 1 WHERE code = ?
            """, (code,))
            
            conn.commit()
            
            logger.info(f"Exchanged auth code for agent {agent_id}")
            return agent_id