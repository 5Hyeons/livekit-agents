"""MongoDB-based user profile management."""

import logging
from datetime import datetime
from typing import Dict, Any

# MongoDB store will be passed as parameter

logger = logging.getLogger("user-profile")


class UserProfileManager:
    """MongoDB Store 기반 사용자 프로필 관리."""
    
    @staticmethod
    def get_or_create_profile(participant_id: str, store) -> Dict[str, Any]:
        """
        Get existing user profile or create a new one.
        
        Args:
            participant_id: The participant's identity
            store: MongoDB store instance
        
        Returns:
            User profile data dictionary
        """
        if not participant_id:
            return {"name": "Unknown"}
        
        try:
            
            # Try to get existing profile
            profile_data = store.get(
                namespace=("user_profile", participant_id),
                key="basic_info"
            )
            
            if profile_data and profile_data.value:
                logger.info(f"[UserProfile] Loaded existing profile for: {participant_id}")
                return profile_data.value
            else:
                # Create new profile with default values
                new_profile = {
                    "name": "Unknown",
                    "created_at": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "token_info": {
                        "total_granted": 10000,
                        "total_used": 0,
                        "remaining": 10000,
                        "status": "normal"
                    }
                }
                
                # Save to MongoDB Store
                store.put(
                    namespace=("user_profile", participant_id),
                    key="basic_info",
                    value=new_profile
                )
                
                logger.info(f"[UserProfile] Created new profile for: {participant_id}")
                return new_profile
                
        except Exception as e:
            logger.error(f"[UserProfile] Failed to get/create profile for {participant_id}: {e}")
            # Return minimal fallback profile
            return {"name": "Unknown"}
    
    @staticmethod
    def update_user_name(participant_id: str, name: str, store) -> bool:
        """
        Update user's name in their profile.
        
        Args:
            participant_id: The participant's identity
            name: The new name to save
            store: MongoDB store instance
        
        Returns:
            True if successful, False otherwise
        """
        if not participant_id:
            return False
        
        try:
            
            # Get existing profile
            profile_data = store.get(
                namespace=("user_profile", participant_id),
                key="basic_info"
            )
            
            if profile_data and profile_data.value:
                # Update existing profile
                updated_profile = profile_data.value
                old_name = updated_profile.get("name", "Unknown")
                updated_profile["name"] = name
                updated_profile["last_seen"] = datetime.now().isoformat()
                
                # Save updated profile to MongoDB Store
                store.put(
                    namespace=("user_profile", participant_id),
                    key="basic_info",
                    value=updated_profile
                )
                
                logger.info(f"[UserProfile] Updated user name: {participant_id} ({old_name} -> {name})")
                return True
            else:
                logger.warning(f"[UserProfile] Profile not found for update: {participant_id}")
                return False
                
        except Exception as e:
            logger.error(f"[UserProfile] Failed to update user name for {participant_id}: {e}")
            return False

    @staticmethod
    def update_token_balance(participant_id: str, user_profile_data: Dict[str, Any], store) -> bool:
        """
        Update user's token balance in MongoDB store.
        
        Args:
            participant_id: The participant's identity
            user_profile_data: Complete user profile data with updated token_info
            store: MongoDB store instance
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update the entire user profile with new token balance
            store.put(
                namespace=("user_profile", participant_id),
                key="basic_info",
                value=user_profile_data
            )
            
            # Log the token update
            token_info = user_profile_data.get("token_info", {})
            logger.info(
                f"[UserProfile] Token balance updated for: {participant_id} "
                f"(Remaining: {token_info.get('remaining', 0)}, "
                f"Used: {token_info.get('total_used', 0)})"
            )
            return True
            
        except Exception as e:
            logger.error(f"[UserProfile] Failed to update token balance for {participant_id}: {e}")
            return False