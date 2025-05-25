"""
XnX Query Parameters and Identity Tokens for PathRAG integration.

This module defines the parameters used to configure XnX-enhanced path queries
and identity tokens for access control optimization.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Union
import uuid


@dataclass
class XnXQueryParams:
    """Parameters for XnX-enhanced path queries."""
    
    # Weight threshold for path selection (0.0 to 1.0)
    min_weight: float = 0.0
    
    # Maximum distance (number of hops) to consider
    max_distance: int = 3
    
    # Direction filter: +1 (inbound), -1 (outbound), or None (both)
    direction: Optional[int] = None
    
    # Temporal constraints for time-bound relationships
    temporal_constraint: Optional[Union[datetime, str]] = None
    
    # Identity token for assumed identity operations
    identity_token: Optional['XnXIdentityToken'] = None
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.min_weight < 0.0 or self.min_weight > 1.0:
            raise ValueError("min_weight must be between 0.0 and 1.0")
        
        if self.max_distance < 1:
            raise ValueError("max_distance must be at least 1")
        
        if self.direction is not None and self.direction not in [-1, 1]:
            raise ValueError("direction must be -1 (outbound), 1 (inbound), or None")


@dataclass
class XnXIdentityToken:
    """Token for XnX identity assumption operations."""
    
    # User ID assuming the identity
    user_id: str
    
    # Object ID whose identity is being assumed
    object_id: str
    
    # Weight of the relationship between user and object
    relationship_weight: float
    
    # When the token was created
    created_at: datetime = datetime.now()
    
    # How long the token is valid for
    expiration_minutes: int = 60
    
    # Unique token identifier
    token_id: str = str(uuid.uuid4())
    
    @property
    def is_valid(self) -> bool:
        """Check if the token is still valid."""
        expiration_time = self.created_at + timedelta(minutes=self.expiration_minutes)
        return datetime.now() < expiration_time
    
    @property
    def effective_weight(self) -> float:
        """Get the effective weight of this identity token.
        
        The effective weight decreases over time to model
        diminishing trust as the token ages.
        """
        if not self.is_valid:
            return 0.0
            
        total_lifetime = timedelta(minutes=self.expiration_minutes)
        elapsed = datetime.now() - self.created_at
        remaining_ratio = 1 - (elapsed / total_lifetime)
        
        # Weight decay formula - linear decay in this simple implementation
        return self.relationship_weight * remaining_ratio
